"""
신호 처리 기반 전처리 필터
딥러닝 모델 전에 적용하여 마이크 잡음을 1차적으로 감쇠
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from typing import Optional


class ProximityEffectCorrector(nn.Module):
    """
    근접효과 보정 필터
    80Hz 이하 저주파를 감쇠시켜 근접효과를 보정
    """
    
    def __init__(self, sample_rate: int = 16000, cutoff_freq: int = 80):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        
        # High-pass filter 계수 계산 (Butterworth)
        sos = signal.butter(4, cutoff_freq, 'hp', fs=sample_rate, output='sos')
        
        # SOS (Second-Order Sections) 형식을 직접 계수로 변환
        # PyTorch에서 사용하기 위해 저장
        self.register_buffer('sos', torch.FloatTensor(sos))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch, samples] 또는 [samples]
            
        Returns:
            필터링된 오디오
        """
        # NumPy로 변환하여 scipy 필터 적용
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = audio.shape[0]
        output = []
        
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            filtered = signal.sosfilt(self.sos.cpu().numpy(), audio_np)
            output.append(torch.FloatTensor(filtered))
        
        output = torch.stack(output).to(audio.device)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class PopNoiseDetector(nn.Module):
    """
    팝노이즈 감지 및 억제
    짧은 시간 동안의 에너지 급증을 감지하고 soft clipping 적용
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        window_size: int = 512,
        threshold: float = 3.0
    ):
        """
        Args:
            sample_rate: 샘플링 레이트
            window_size: 에너지 계산 윈도우 크기
            threshold: 팝 감지 임계값 (평균 에너지의 배수)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.threshold = threshold
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch, samples] 또는 [samples]
            
        Returns:
            팝노이즈가 억제된 오디오
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_samples = audio.shape
        output = audio.clone()
        
        for i in range(batch_size):
            # 에너지 계산 (sliding window)
            audio_squared = audio[i] ** 2
            energy = torch.nn.functional.avg_pool1d(
                audio_squared.unsqueeze(0).unsqueeze(0),
                kernel_size=self.window_size,
                stride=1,
                padding=self.window_size // 2
            ).squeeze()
            
            # 평균 에너지
            mean_energy = energy.mean()
            
            # 팝 감지: 임계값을 초과하는 구간
            pop_mask = energy > (mean_energy * self.threshold)
            
            # 팝 구간에 soft clipping 적용
            if pop_mask.any():
                # 마스크를 원본 길이에 맞게 조정
                pop_mask_full = torch.zeros(num_samples, dtype=torch.bool, device=audio.device)
                valid_length = min(len(pop_mask), num_samples)
                pop_mask_full[:valid_length] = pop_mask[:valid_length]
                
                # Soft clipping (tanh)
                output[i, pop_mask_full] = torch.tanh(output[i, pop_mask_full] * 2) * 0.5
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class ElectricalNoiseFilter(nn.Module):
    """
    전기적 잡음 필터
    50/60Hz 험(hum) 제거를 위한 notch filter
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        hum_freq: int = 60,
        quality_factor: float = 30.0
    ):
        """
        Args:
            sample_rate: 샘플링 레이트
            hum_freq: 험 주파수 (50 or 60 Hz)
            quality_factor: 노치 필터 Q 팩터 (높을수록 좁은 대역)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hum_freq = hum_freq
        
        # Notch filter 계수 계산 (기본 주파수 + 고조파)
        freqs_to_remove = [hum_freq, hum_freq * 2, hum_freq * 3]
        
        sos_list = []
        for freq in freqs_to_remove:
            if freq < sample_rate / 2:  # Nyquist 주파수 이하만
                sos = signal.iirnotch(freq, quality_factor, sample_rate)
                # sos 형식으로 변환
                sos_cascade = signal.tf2sos(sos[0], sos[1])
                sos_list.append(sos_cascade)
        
        # 모든 notch filter를 cascaded SOS로 결합
        if sos_list:
            combined_sos = np.vstack(sos_list)
            self.register_buffer('sos', torch.FloatTensor(combined_sos))
        else:
            self.register_buffer('sos', torch.FloatTensor([]))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch, samples] 또는 [samples]
            
        Returns:
            험이 제거된 오디오
        """
        if len(self.sos) == 0:
            return audio
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = audio.shape[0]
        output = []
        
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            filtered = signal.sosfilt(self.sos.cpu().numpy(), audio_np)
            output.append(torch.FloatTensor(filtered))
        
        output = torch.stack(output).to(audio.device)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class MicrophoneNoisePreprocessor(nn.Module):
    """
    모든 마이크 잡음 전처리 필터를 통합한 모듈
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        apply_proximity_correction: bool = True,
        apply_pop_suppression: bool = True,
        apply_hum_removal: bool = True,
        hum_freq: int = 60
    ):
        super().__init__()
        
        self.apply_proximity_correction = apply_proximity_correction
        self.apply_pop_suppression = apply_pop_suppression
        self.apply_hum_removal = apply_hum_removal
        
        if apply_proximity_correction:
            self.proximity_corrector = ProximityEffectCorrector(sample_rate)
        
        if apply_pop_suppression:
            self.pop_detector = PopNoiseDetector(sample_rate)
        
        if apply_hum_removal:
            self.hum_filter = ElectricalNoiseFilter(sample_rate, hum_freq)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        모든 전처리 필터를 순차적으로 적용
        
        Args:
            audio: [batch, samples] 또는 [samples]
            
        Returns:
            전처리된 오디오
        """
        # 1. 근접효과 보정
        if self.apply_proximity_correction:
            audio = self.proximity_corrector(audio)
        
        # 2. 전기 잡음 제거
        if self.apply_hum_removal:
            audio = self.hum_filter(audio)
        
        # 3. 팝노이즈 억제 (마지막에 적용)
        if self.apply_pop_suppression:
            audio = self.pop_detector(audio)
        
        return audio


# 테스트 코드
if __name__ == "__main__":
    print("🎛️ 전처리 필터 테스트...")
    
    # 테스트 신호 생성 (1초, 16kHz)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 음성 시뮬레이션 (여러 주파수 혼합)
    audio = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200Hz
    audio += 0.3 * np.sin(2 * np.pi * 400 * t)  # 400Hz
    
    # 60Hz 험 추가
    audio += 0.1 * np.sin(2 * np.pi * 60 * t)
    
    audio_tensor = torch.FloatTensor(audio)
    
    # 전처리기 생성
    preprocessor = MicrophoneNoisePreprocessor(sample_rate=sr, hum_freq=60)
    
    # 필터 적용
    filtered = preprocessor(audio_tensor)
    
    print(f"   입력 shape: {audio_tensor.shape}")
    print(f"   출력 shape: {filtered.shape}")
    print(f"   입력 RMS: {audio_tensor.pow(2).mean().sqrt():.4f}")
    print(f"   출력 RMS: {filtered.pow(2).mean().sqrt():.4f}")
    print("✅ 전처리 필터 테스트 완료")

