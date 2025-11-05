"""
모델 평가 스크립트
테스트 데이터셋에 대해 PESQ, STOI, SI-SDR 등 계산
"""

import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import SpectrogramUNet, WaveformUNet
from models.preprocessing import MicrophoneNoisePreprocessor
from evaluation.metrics import MetricsCalculator, format_metrics
from data.dataset import SpeechEnhancementDataset
import librosa
import soundfile as sf


def load_model(checkpoint_path: str, device: torch.device):
    """체크포인트에서 모델 로드"""
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
    
    return model, preprocessor, config


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    test_noisy_dir: str,
    test_clean_dir: str,
    sample_rate: int,
    device: torch.device,
    preprocessor = None,
    save_output: bool = False,
    output_dir: str = None
):
    """
    테스트 데이터셋 전체에 대해 평가
    
    Args:
        model: 학습된 모델
        test_noisy_dir: 잡음 테스트 데이터
        test_clean_dir: 깨끗한 테스트 데이터
        sample_rate: 샘플링 레이트
        device: 디바이스
        preprocessor: 전처리기 (선택)
        save_output: 출력 오디오 저장 여부
        output_dir: 출력 디렉토리
    """
    # 데이터셋
    dataset = SpeechEnhancementDataset(
        noisy_dir=test_noisy_dir,
        clean_dir=test_clean_dir,
        sample_rate=sample_rate,
        segment_length=None,  # 전체 길이 사용
        augment=False
    )
    
    # 메트릭 계산기
    calculator = MetricsCalculator(sample_rate=sample_rate)
    
    # 결과 저장
    results = []
    
    # 출력 디렉토리
    if save_output and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"평가 시작: {len(dataset)}개 샘플\n")
    
    for idx in tqdm(range(len(dataset)), desc="평가 중"):
        # 데이터 로드
        noisy, clean = dataset[idx]
        
        # 배치 차원 추가
        noisy_batch = noisy.unsqueeze(0).to(device)
        clean_batch = clean.unsqueeze(0).to(device)
        
        # 전처리
        if preprocessor is not None:
            noisy_batch = preprocessor(noisy_batch)
        
        # 추론
        enhanced_batch = model(noisy_batch)
        
        # CPU로 이동
        enhanced = enhanced_batch.squeeze(0).cpu()
        clean = clean.cpu()
        noisy = noisy.cpu()
        
        # 지표 계산
        metrics = calculator.calculate_all(enhanced, clean)
        metrics['filename'] = dataset.noisy_files[idx].stem
        
        # Noisy 신호의 지표도 계산 (비교용)
        noisy_metrics = calculator.calculate_all(noisy, clean)
        metrics['noisy_si_sdr'] = noisy_metrics.get('si_sdr', None)
        metrics['noisy_pesq'] = noisy_metrics.get('pesq', None)
        metrics['noisy_stoi'] = noisy_metrics.get('stoi', None)
        
        # 개선량 계산
        if metrics['si_sdr'] is not None and metrics['noisy_si_sdr'] is not None:
            metrics['si_sdr_improvement'] = metrics['si_sdr'] - metrics['noisy_si_sdr']
        
        results.append(metrics)
        
        # 출력 저장
        if save_output and output_dir:
            output_file = output_path / f"{metrics['filename']}_enhanced.wav"
            sf.write(output_file, enhanced.numpy(), sample_rate)
    
    # DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # 통계 출력
    print("\n" + "="*60)
    print("평가 결과 요약")
    print("="*60)
    
    print("\n향상된 음성 (Enhanced):")
    enhanced_stats = {
        'si_sdr': df['si_sdr'].mean(),
        'pesq': df['pesq'].mean(),
        'stoi': df['stoi'].mean()
    }
    print(format_metrics(enhanced_stats))
    
    print("\n원본 잡음 신호 (Noisy):")
    noisy_stats = {
        'si_sdr': df['noisy_si_sdr'].mean(),
        'pesq': df['noisy_pesq'].mean(),
        'stoi': df['noisy_stoi'].mean()
    }
    print(format_metrics(noisy_stats))
    
    if 'si_sdr_improvement' in df.columns:
        print(f"\nSI-SDR 개선량: {df['si_sdr_improvement'].mean():.2f} dB")
    
    print("\n" + "="*60)
    
    return df


def evaluate_single_file(
    model: torch.nn.Module,
    noisy_path: str,
    clean_path: str,
    sample_rate: int,
    device: torch.device,
    preprocessor = None,
    output_path: str = None
):
    """
    단일 파일 평가
    
    Args:
        model: 학습된 모델
        noisy_path: 잡음 오디오 파일
        clean_path: 깨끗한 오디오 파일 (reference)
        sample_rate: 샘플링 레이트
        device: 디바이스
        preprocessor: 전처리기
        output_path: 출력 파일 경로 (선택)
    """
    # 오디오 로드
    noisy, _ = librosa.load(noisy_path, sr=sample_rate)
    clean, _ = librosa.load(clean_path, sr=sample_rate)
    
    # Tensor로 변환
    noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).to(device)
    clean_tensor = torch.FloatTensor(clean)
    
    # 전처리
    if preprocessor is not None:
        with torch.no_grad():
            noisy_tensor = preprocessor(noisy_tensor)
    
    # 추론
    with torch.no_grad():
        enhanced_tensor = model(noisy_tensor)
    
    # CPU로 이동
    enhanced = enhanced_tensor.squeeze(0).cpu()
    
    # 지표 계산
    calculator = MetricsCalculator(sample_rate=sample_rate)
    
    print("\n" + "="*60)
    print(f"파일: {Path(noisy_path).name}")
    print("="*60)
    
    print("\n향상된 음성 (Enhanced):")
    enhanced_metrics = calculator.calculate_all(enhanced, clean_tensor)
    print(format_metrics(enhanced_metrics))
    
    print("\n원본 잡음 신호 (Noisy):")
    noisy_metrics = calculator.calculate_all(noisy_tensor.squeeze(0).cpu(), clean_tensor)
    print(format_metrics(noisy_metrics))
    
    if enhanced_metrics['si_sdr'] and noisy_metrics['si_sdr']:
        improvement = enhanced_metrics['si_sdr'] - noisy_metrics['si_sdr']
        print(f"\nSI-SDR 개선량: {improvement:.2f} dB")
    
    print("="*60)
    
    # 출력 저장
    if output_path:
        sf.write(output_path, enhanced.numpy(), sample_rate)
        print(f"\n출력 저장: {output_path}")
    
    return enhanced_metrics


def main():
    parser = argparse.ArgumentParser(description="음성 향상 모델 평가")
    parser.add_argument("--checkpoint", type=str, required=True, help="모델 체크포인트 경로")
    parser.add_argument("--test_noisy_dir", type=str, help="테스트 잡음 데이터 디렉토리")
    parser.add_argument("--test_clean_dir", type=str, help="테스트 깨끗한 데이터 디렉토리")
    parser.add_argument("--noisy_file", type=str, help="단일 잡음 파일 (단일 파일 모드)")
    parser.add_argument("--clean_file", type=str, help="단일 깨끗한 파일 (단일 파일 모드)")
    parser.add_argument("--save_output", action="store_true", help="출력 오디오 저장")
    parser.add_argument("--output_dir", type=str, default="evaluation/outputs", help="출력 디렉토리")
    parser.add_argument("--output_file", type=str, help="출력 파일 (단일 파일 모드)")
    parser.add_argument("--csv_output", type=str, default="evaluation/results.csv", 
                       help="결과 CSV 파일 경로")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 로드
    print(f"모델 로딩: {args.checkpoint}")
    model, preprocessor, config = load_model(args.checkpoint, device)
    print(f"모델 로드 완료")
    
    sample_rate = config.data.sample_rate
    
    # 단일 파일 모드
    if args.noisy_file and args.clean_file:
        evaluate_single_file(
            model=model,
            noisy_path=args.noisy_file,
            clean_path=args.clean_file,
            sample_rate=sample_rate,
            device=device,
            preprocessor=preprocessor,
            output_path=args.output_file
        )
    
    # 데이터셋 모드
    elif args.test_noisy_dir and args.test_clean_dir:
        df = evaluate_dataset(
            model=model,
            test_noisy_dir=args.test_noisy_dir,
            test_clean_dir=args.test_clean_dir,
            sample_rate=sample_rate,
            device=device,
            preprocessor=preprocessor,
            save_output=args.save_output,
            output_dir=args.output_dir
        )
        
        # CSV 저장
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n결과 CSV 저장: {csv_path}")
    
    else:
        print("--test_noisy_dir와 --test_clean_dir 또는 --noisy_file과 --clean_file을 지정해주세요.")


if __name__ == "__main__":
    main()

