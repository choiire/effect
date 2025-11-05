"""
마이크 특화 잡음 합성기 (Microphone-Specific Noise Synthesizer)

근접효과, 팝노이즈, 전기적 잡음을 시뮬레이션하여
깨끗한 음성에 추가하는 데이터 증강 파이프라인
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional
import random
from tqdm import tqdm
import argparse


class MicrophoneNoiseSimulator:
    """마이크 특화 잡음 시뮬레이터"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def simulate_proximity_effect(
        self, 
        audio: np.ndarray, 
        boost_db: float = None
    ) -> np.ndarray:
        """
        근접효과 시뮬레이션: 80-250Hz 대역 저주파 부스트
        
        Args:
            audio: 입력 오디오 신호
            boost_db: 부스트 강도 (dB). None이면 0-12dB 랜덤
            
        Returns:
            근접효과가 적용된 오디오
        """
        if boost_db is None:
            boost_db = np.random.uniform(0, 12)  # 0-12dB 랜덤 부스트
        
        # 80-250Hz 대역통과 필터 설계
        low_freq = 80
        high_freq = 250
        
        # Butterworth 대역통과 필터
        sos = signal.butter(
            4, 
            [low_freq, high_freq], 
            btype='bandpass', 
            fs=self.sr, 
            output='sos'
        )
        
        # 저주파 성분 추출
        low_freq_component = signal.sosfilt(sos, audio)
        
        # dB를 선형 스케일로 변환
        boost_linear = 10 ** (boost_db / 20)
        
        # 부스트된 저주파 성분을 원본에 추가
        boosted_audio = audio + low_freq_component * (boost_linear - 1)
        
        # 클리핑 방지
        max_val = np.max(np.abs(boosted_audio))
        if max_val > 1.0:
            boosted_audio = boosted_audio / max_val * 0.99
        
        return boosted_audio
    
    def add_pop_noise(
        self, 
        audio: np.ndarray, 
        pop_frequency: float = None,
        intensity: float = None
    ) -> np.ndarray:
        """
        팝노이즈 추가: 순간적인 저주파 고에너지 펄스
        
        Args:
            audio: 입력 오디오 신호
            pop_frequency: 초당 팝 발생 횟수 (None이면 0-3회 랜덤)
            intensity: 팝 강도 (0-1, None이면 랜덤)
            
        Returns:
            팝노이즈가 추가된 오디오
        """
        if pop_frequency is None:
            pop_frequency = np.random.uniform(0, 3)  # 0-3 pops/second
        
        duration = len(audio) / self.sr
        num_pops = int(pop_frequency * duration)
        
        output = audio.copy()
        
        for _ in range(num_pops):
            # 랜덤 위치에 팝 삽입
            pop_position = np.random.randint(0, len(audio) - self.sr // 10)
            
            # 팝 지속시간: 50-150ms
            pop_duration_samples = np.random.randint(
                int(0.05 * self.sr), 
                int(0.15 * self.sr)
            )
            
            # 팝 강도
            if intensity is None:
                intensity = np.random.uniform(0.3, 0.8)
            
            # 저주파 사인파 펄스 생성 (40-120Hz)
            pop_freq = np.random.uniform(40, 120)
            t = np.arange(pop_duration_samples) / self.sr
            
            # 감쇠 엔벨로프 (빠른 어택, 느린 디케이)
            envelope = np.exp(-t * 15)  # 지수 감쇠
            pop_signal = intensity * np.sin(2 * np.pi * pop_freq * t) * envelope
            
            # 팝 삽입
            end_position = min(pop_position + pop_duration_samples, len(output))
            actual_duration = end_position - pop_position
            output[pop_position:end_position] += pop_signal[:actual_duration]
        
        # 클리핑 방지
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.99
        
        return output
    
    def add_electrical_noise(
        self, 
        audio: np.ndarray, 
        hum_freq: int = 60,
        hum_snr_db: float = None,
        white_noise_snr_db: float = None
    ) -> np.ndarray:
        """
        전기적 잡음 추가: 험(Hum) + 화이트 노이즈
        
        Args:
            audio: 입력 오디오 신호
            hum_freq: 험 주파수 (50 or 60 Hz)
            hum_snr_db: 험의 SNR (dB, None이면 20-50dB 랜덤)
            white_noise_snr_db: 화이트 노이즈 SNR (dB, None이면 30-60dB 랜덤)
            
        Returns:
            전기적 잡음이 추가된 오디오
        """
        output = audio.copy()
        
        # 1. 험(Hum) 추가 - 주기적 저주파 노이즈
        if hum_snr_db is None:
            hum_snr_db = np.random.uniform(20, 50)
        
        t = np.arange(len(audio)) / self.sr
        
        # 기본 험 주파수 + 고조파
        hum = np.sin(2 * np.pi * hum_freq * t)
        hum += 0.5 * np.sin(2 * np.pi * hum_freq * 2 * t)  # 2차 고조파
        hum += 0.25 * np.sin(2 * np.pi * hum_freq * 3 * t)  # 3차 고조파
        
        # SNR 기반으로 험 레벨 조정
        signal_power = np.mean(audio ** 2)
        hum_power = np.mean(hum ** 2)
        hum_scale = np.sqrt(signal_power / (10 ** (hum_snr_db / 10)) / hum_power)
        output += hum * hum_scale
        
        # 2. 화이트 노이즈 추가
        if white_noise_snr_db is None:
            white_noise_snr_db = np.random.uniform(30, 60)
        
        white_noise = np.random.randn(len(audio))
        noise_power = np.mean(white_noise ** 2)
        noise_scale = np.sqrt(signal_power / (10 ** (white_noise_snr_db / 10)) / noise_power)
        output += white_noise * noise_scale
        
        # 클리핑 방지
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.99
        
        return output
    
    def apply_all_noise(
        self,
        audio: np.ndarray,
        proximity_boost_db: float = None,
        pop_frequency: float = None,
        hum_freq: int = 60,
        hum_snr_db: float = None,
        white_noise_snr_db: float = None
    ) -> np.ndarray:
        """
        모든 마이크 잡음을 한번에 적용
        
        Args:
            audio: 입력 오디오
            (각 잡음 타입별 파라미터는 개별 함수 참조)
            
        Returns:
            모든 잡음이 적용된 오디오
        """
        # 1. 근접효과
        audio = self.simulate_proximity_effect(audio, proximity_boost_db)
        
        # 2. 팝노이즈
        audio = self.add_pop_noise(audio, pop_frequency)
        
        # 3. 전기적 잡음
        audio = self.add_electrical_noise(audio, hum_freq, hum_snr_db, white_noise_snr_db)
        
        return audio


def normalize_audio(audio: np.ndarray, target_rms_db: float = -25) -> np.ndarray:
    """
    오디오를 목표 RMS 레벨로 정규화
    
    Args:
        audio: 입력 오디오
        target_rms_db: 목표 RMS 레벨 (dBFS)
        
    Returns:
        정규화된 오디오
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_rms_db / 20)
        audio = audio * (target_rms / rms)
    return audio


def synthesize_dataset(
    clean_dir: Path,
    output_dir: Path,
    num_samples: int = None,
    sample_rate: int = 16000
):
    """
    깨끗한 음성 디렉토리로부터 잡음 데이터셋 생성
    
    Args:
        clean_dir: 깨끗한 음성 파일 디렉토리
        output_dir: 출력 디렉토리
        num_samples: 생성할 샘플 수 (None이면 모든 파일 처리)
        sample_rate: 샘플링 레이트
    """
    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    
    # 출력 디렉토리 생성
    noisy_dir = output_dir / "noisy"
    clean_output_dir = output_dir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 음성 파일 목록
    audio_files = list(clean_dir.glob("*.wav")) + list(clean_dir.glob("*.flac"))
    
    if num_samples:
        audio_files = audio_files[:num_samples]
    
    simulator = MicrophoneNoiseSimulator(sample_rate=sample_rate)
    
    print("마이크 잡음 데이터셋 합성 시작...")
    print(f"   입력: {len(audio_files)}개 파일")
    print(f"   출력: {output_dir}")
    
    for idx, audio_path in enumerate(tqdm(audio_files, desc="합성 중")):
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # 정규화
            audio = normalize_audio(audio, target_rms_db=np.random.uniform(-35, -15))
            
            # 잡음 적용 (랜덤 파라미터)
            noisy_audio = simulator.apply_all_noise(audio)
            
            # 저장
            base_name = audio_path.stem
            noisy_path = noisy_dir / f"{base_name}_noisy.wav"
            clean_path = clean_output_dir / f"{base_name}_clean.wav"
            
            sf.write(noisy_path, noisy_audio, sample_rate)
            sf.write(clean_path, audio, sample_rate)
            
        except Exception as e:
            print(f"오류 발생 ({audio_path.name}): {e}")
            continue
    
    print("데이터셋 합성 완료!")
    print(f"   잡음 음성: {noisy_dir}")
    print(f"   깨끗한 음성: {clean_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="마이크 잡음 데이터셋 합성")
    parser.add_argument("--clean_dir", type=str, required=True, help="깨끗한 음성 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 디렉토리")
    parser.add_argument("--num_samples", type=int, default=None, help="생성할 샘플 수")
    parser.add_argument("--sample_rate", type=int, default=16000, help="샘플링 레이트")
    
    args = parser.parse_args()
    
    synthesize_dataset(
        clean_dir=Path(args.clean_dir),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        sample_rate=args.sample_rate
    )

