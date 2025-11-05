"""
실시간 잡음 제거 추론 스크립트

학습된 모델을 사용하여 잡음이 섞인 오디오에서 깨끗한 음성 추출
"""

import torch
from pathlib import Path
import argparse
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import SpectrogramUNet, WaveformUNet
from models.preprocessing import MicrophoneNoisePreprocessor


def load_model(checkpoint_path: str, device: torch.device):
    """체크포인트에서 모델과 설정 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 모델 생성
    model_config = config.model
    if model_config.model_type == "spectrogram_unet":
        model = SpectrogramUNet(
            n_fft=model_config.n_fft,
            hop_length=model_config.hop_length,
            n_channels=model_config.n_channels,
            output_mode=model_config.output_mode
        )
    elif model_config.model_type == "waveform_unet":
        model = WaveformUNet(n_channels=model_config.n_channels)
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 전처리기
    preprocessor = None
    if model_config.use_preprocessing:
        preprocessor = MicrophoneNoisePreprocessor(
            sample_rate=config.data.sample_rate,
            apply_proximity_correction=model_config.apply_proximity_correction,
            apply_pop_suppression=model_config.apply_pop_suppression,
            apply_hum_removal=model_config.apply_hum_removal,
            hum_freq=model_config.hum_freq
        ).to(device)
    
    sample_rate = config.data.sample_rate
    
    return model, preprocessor, sample_rate


@torch.no_grad()
def denoise_audio(
    audio: np.ndarray,
    model: torch.nn.Module,
    preprocessor,
    device: torch.device,
    chunk_size: int = None
) -> np.ndarray:
    """
    오디오에서 잡음 제거
    
    Args:
        audio: 입력 오디오 (numpy array)
        model: 학습된 모델
        preprocessor: 전처리기
        device: 디바이스
        chunk_size: 청크 크기 (None이면 전체 처리, 큰 파일은 청크로 나눔)
        
    Returns:
        향상된 오디오
    """
    # 청크 처리 (메모리 절약)
    if chunk_size is not None and len(audio) > chunk_size:
        return denoise_audio_chunked(audio, model, preprocessor, device, chunk_size)
    
    # Tensor로 변환
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    
    # 전처리
    if preprocessor is not None:
        audio_tensor = preprocessor(audio_tensor)
    
    # 모델 추론
    enhanced_tensor = model(audio_tensor)
    
    # NumPy로 변환
    enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
    
    return enhanced


def denoise_audio_chunked(
    audio: np.ndarray,
    model: torch.nn.Module,
    preprocessor,
    device: torch.device,
    chunk_size: int,
    overlap: int = None
) -> np.ndarray:
    """
    긴 오디오를 청크로 나누어 처리 (메모리 효율적)
    
    Args:
        audio: 입력 오디오
        model: 모델
        preprocessor: 전처리기
        device: 디바이스
        chunk_size: 청크 크기
        overlap: 청크 간 오버랩 (None이면 chunk_size의 25%)
        
    Returns:
        향상된 오디오
    """
    if overlap is None:
        overlap = chunk_size // 4
    
    hop_size = chunk_size - overlap
    num_chunks = int(np.ceil((len(audio) - overlap) / hop_size))
    
    enhanced_chunks = []
    
    for i in tqdm(range(num_chunks), desc="청크 처리"):
        start = i * hop_size
        end = min(start + chunk_size, len(audio))
        
        # 청크 추출
        chunk = audio[start:end]
        
        # 제로 패딩 (마지막 청크가 짧을 경우)
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        
        # 잡음 제거
        enhanced_chunk = denoise_audio(chunk, model, preprocessor, device, chunk_size=None)
        
        # 원래 길이로 자르기
        enhanced_chunk = enhanced_chunk[:end - start]
        
        # 오버랩 영역 처리 (크로스페이드)
        if i > 0 and overlap > 0:
            # 이전 청크의 끝부분과 현재 청크의 시작부분을 블렌딩
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            
            # 이전 청크의 오버랩 부분
            prev_overlap = enhanced_chunks[-1][-overlap:]
            # 현재 청크의 오버랩 부분
            curr_overlap = enhanced_chunk[:overlap]
            
            # 크로스페이드
            blended = prev_overlap * fade_out + curr_overlap * fade_in
            
            # 이전 청크의 오버랩 제거하고 블렌딩된 부분 추가
            enhanced_chunks[-1] = enhanced_chunks[-1][:-overlap]
            enhanced_chunk = np.concatenate([blended, enhanced_chunk[overlap:]])
        
        enhanced_chunks.append(enhanced_chunk)
    
    # 모든 청크 결합
    enhanced = np.concatenate(enhanced_chunks)
    
    # 원본 길이로 자르기
    enhanced = enhanced[:len(audio)]
    
    return enhanced


def process_file(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    device: torch.device,
    chunk_size: int = None
):
    """
    단일 파일 처리
    
    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로
        checkpoint_path: 모델 체크포인트 경로
        device: 디바이스
        chunk_size: 청크 크기 (긴 파일 처리용)
    """
    print(f"모델 로딩: {checkpoint_path}")
    model, preprocessor, sample_rate = load_model(checkpoint_path, device)
    print(f"모델 로드 완료 (샘플레이트: {sample_rate}Hz)")
    
    print(f"\n오디오 로딩: {input_path}")
    audio, _ = librosa.load(input_path, sr=sample_rate)
    duration = len(audio) / sample_rate
    print(f"   길이: {duration:.2f}초 ({len(audio)} 샘플)")
    
    print(f"\n잡음 제거 중...")
    enhanced = denoise_audio(audio, model, preprocessor, device, chunk_size)
    
    print(f"\n저장 중: {output_path}")
    sf.write(output_path, enhanced, sample_rate)
    
    print(f"완료!")


def process_directory(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    device: torch.device,
    chunk_size: int = None
):
    """
    디렉토리 내 모든 오디오 파일 처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        checkpoint_path: 모델 체크포인트
        device: 디바이스
        chunk_size: 청크 크기
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 오디오 파일 목록
    audio_files = (
        list(input_path.glob("*.wav")) +
        list(input_path.glob("*.flac")) +
        list(input_path.glob("*.mp3"))
    )
    
    if len(audio_files) == 0:
        print(f"{input_dir}에서 오디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"{len(audio_files)}개 파일 발견")
    
    # 모델 로드 (한 번만)
    print(f"모델 로딩: {checkpoint_path}")
    model, preprocessor, sample_rate = load_model(checkpoint_path, device)
    print(f"모델 로드 완료\n")
    
    # 각 파일 처리
    for audio_file in tqdm(audio_files, desc="파일 처리"):
        try:
            # 오디오 로드
            audio, _ = librosa.load(audio_file, sr=sample_rate)
            
            # 잡음 제거
            enhanced = denoise_audio(audio, model, preprocessor, device, chunk_size)
            
            # 저장
            output_file = output_path / f"{audio_file.stem}_enhanced.wav"
            sf.write(output_file, enhanced, sample_rate)
            
        except Exception as e:
            print(f"\n오류 ({audio_file.name}): {e}")
            continue
    
    print(f"\n모든 파일 처리 완료!")
    print(f"   출력 디렉토리: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="마이크 잡음 제거 - 추론")
    parser.add_argument("--input", type=str, required=True, 
                       help="입력 파일 또는 디렉토리")
    parser.add_argument("--output", type=str, required=True,
                       help="출력 파일 또는 디렉토리")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="모델 체크포인트 경로")
    parser.add_argument("--chunk_size", type=int, default=None,
                       help="청크 크기 (긴 오디오 처리용, 예: 160000)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="디바이스")
    
    args = parser.parse_args()
    
    # Device 설정
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}\n")
    
    input_path = Path(args.input)
    
    # 디렉토리 처리
    if input_path.is_dir():
        process_directory(
            input_dir=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            device=device,
            chunk_size=args.chunk_size
        )
    
    # 단일 파일 처리
    elif input_path.is_file():
        process_file(
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            device=device,
            chunk_size=args.chunk_size
        )
    
    else:
        print(f"입력 경로를 찾을 수 없습니다: {args.input}")


if __name__ == "__main__":
    main()

