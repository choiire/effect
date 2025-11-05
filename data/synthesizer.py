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
    
    def _design_low_shelf_biquad(
        self, 
        gain_db: float, 
        fc_hz: float, 
        q: float = 0.707
    ) -> np.ndarray:
        """
        Audio EQ Cookbook 공식에 따라 로우 쉘프 바이쿼드 필터 계수를 설계
        
        Args:
            gain_db: 쉘프의 게인 (dB)
            fc_hz: 코너 주파수 (Hz)
            q: 필터의 Q 팩터 (0.707은 6dB/octave 기울기)
            
        Returns:
            SOS 형식의 계수 배열 [b0, b1, b2, a0, a1, a2]
        """
        if gain_db == 0.0:
            # 통과 필터 (no-op)
            return np.array([[1, 0, 0, 1, 0, 0]])
        
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * fc_hz / self.sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        
        # alpha 계산
        alpha = sin_w0 / (2 * q)
        
        # 중간 계산
        beta = 2 * np.sqrt(A) * alpha
        
        # 계수 계산 (Cookbook 공식)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + beta)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - beta)
        a0 = (A + 1) + (A - 1) * cos_w0 + beta
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - beta
        
        # a0으로 정규화하여 SOS 형식으로 반환
        return np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])
    
    def _calculate_proximity_gain(
        self, 
        distance_cm: float, 
        pattern: str = 'cardioid',
        reference_distance_cm: float = 100.0
    ) -> Tuple[float, float]:
        """
        거리와 지향성 패턴에 따른 근접 효과 게인 계산
        
        Args:
            distance_cm: 음원과 마이크 사이의 거리 (cm)
            pattern: 마이크 지향성 패턴 ('cardioid' 또는 'figure8')
            reference_distance_cm: 기준 거리 (cm)
            
        Returns:
            (shelf_gain_db, level_gain_linear): 쉘프 필터 게인과 레벨 게인
        """
        # 평면 음원이거나 거리가 1m 이상이면 근접 효과 없음
        if distance_cm >= 100:
            return 0.0, 1.0
        
        # 패턴별 보정 상수 (실제 마이크 데이터 기반)
        # 5cm 거리에서 figure8은 약 20dB, cardioid는 약 10dB 부스트
        C_cardioid = 60.0
        C_figure8 = 120.0
        
        c_val = C_figure8 if pattern == 'figure8' else C_cardioid
        
        # 저주파 부스트 게인 (음색 변화)
        shelf_gain_db = c_val / max(distance_cm, 1.0)
        
        # 역제곱 법칙에 따른 레벨 증가
        level_gain_linear = reference_distance_cm / distance_cm
        
        return shelf_gain_db, level_gain_linear
    
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
        intensity: float = None,
        pattern: str = 'cardioid'
    ) -> np.ndarray:
        """
        물리 기반 팝노이즈 추가: 압력 경도 변환기의 근접 효과 시뮬레이션
        
        팝노이즈는 호흡이나 입술이 마이크에 매우 가까워질 때(1-5cm) 발생하는
        순간적인 근접 효과입니다. 이를 로우 쉘프 필터와 역제곱 법칙으로 모델링합니다.
        
        Args:
            audio: 입력 오디오 신호
            pop_frequency: 초당 팝 발생 횟수 (None이면 0-1.5회 랜덤)
            intensity: 팝 강도 (0-1, None이면 랜덤) - 최대 근접 거리에 영향
            pattern: 마이크 지향성 패턴 ('cardioid' 또는 'figure8')
            
        Returns:
            팝노이즈가 추가된 오디오
        """
        if pop_frequency is None:
            pop_frequency = np.random.uniform(0, 1.5)  # 0-1.5 pops/second
        
        duration = len(audio) / self.sr
        num_pops = int(pop_frequency * duration)
        
        if num_pops == 0:
            return audio.copy()
        
        output = audio.copy()
        
        for _ in range(num_pops):
            # 랜덤 위치에 팝 발생
            pop_position = np.random.randint(0, len(audio) - self.sr // 5)
            
            # 팝 지속시간: 80-200ms (호흡이 다가왔다가 멀어지는 시간)
            pop_duration_samples = np.random.randint(
                int(0.08 * self.sr), 
                int(0.20 * self.sr)
            )
            
            # 팝 강도 (최대 근접 거리에 영향)
            if intensity is None:
                intensity = np.random.uniform(0.4, 0.7)
            
            # 시간에 따른 거리 변화 시뮬레이션
            # 거리: 100cm -> min_distance -> 100cm (종형 곡선)
            t = np.linspace(0, 1, pop_duration_samples)
            
            # 종형 거리 프로파일 (가우시안 기반)
            # 중앙에서 가장 가까워짐
            distance_profile = 100.0 - (100.0 - (1.0 + intensity * 4.0)) * np.exp(-((t - 0.5) ** 2) / 0.05)
            
            # 추출할 세그먼트 (팝이 발생하는 구간의 원본 오디오)
            end_position = min(pop_position + pop_duration_samples, len(audio))
            actual_duration = end_position - pop_position
            audio_segment = audio[pop_position:end_position].copy()
            
            # 시간에 따라 변하는 근접 효과 적용
            pop_segment = np.zeros_like(audio_segment)
            
            # 코너 주파수 (근접 효과 시작 주파수)
            fc_hz = 150.0
            
            # 작은 윈도우로 나누어 각각에 다른 거리의 근접 효과 적용
            # 계산 효율을 위해 hop_size 사용
            hop_size = max(1, self.sr // 200)  # 약 5ms
            
            for i in range(0, actual_duration, hop_size):
                window_end = min(i + hop_size, actual_duration)
                window_length = window_end - i
                
                if window_length == 0:
                    continue
                
                # 현재 시간의 거리
                t_idx = min(i, len(distance_profile) - 1)
                current_distance = distance_profile[t_idx]
                
                # 거리에 따른 근접 효과 게인 계산
                shelf_gain_db, level_gain = self._calculate_proximity_gain(
                    current_distance, 
                    pattern=pattern,
                    reference_distance_cm=100.0
                )
                
                # 로우 쉘프 필터 설계
                sos = self._design_low_shelf_biquad(shelf_gain_db, fc_hz, q=0.707)
                
                # 현재 윈도우에 필터 적용
                window_segment = audio_segment[i:window_end]
                
                # 필터 적용 (음색 변화)
                filtered_segment = signal.sosfilt(sos, window_segment)
                
                # 레벨 증가 적용 (역제곱 법칙)
                pop_segment[i:window_end] = filtered_segment * level_gain
            
            # 팝 세그먼트를 원본에 혼합 (가산 합성)
            # 원본 신호는 유지하고 근접 효과만 추가
            output[pop_position:end_position] = audio_segment + (pop_segment - audio_segment) * intensity
        
        # 클리핑 방지
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.99
        
        return output
    
    def _generate_thermal_noise(self, n_samples: int, amplitude: float) -> np.ndarray:
        """
        열 노이즈 (Thermal/Johnson-Nyquist Noise) 생성
        
        가우시안 백색 노이즈로 모델링. 마이크 내부 도체의 열적 교란으로 발생.
        
        Args:
            n_samples: 샘플 수
            amplitude: RMS 진폭 (선형 스케일)
            
        Returns:
            열 노이즈 신호
        """
        # 표준 정규 분포 (평균=0, 표준편차=1)
        noise = np.random.normal(0.0, 1.0, n_samples)
        
        # RMS를 1로 정규화
        noise = noise / np.sqrt(np.mean(noise ** 2))
        
        # 목표 진폭으로 스케일링
        return noise * amplitude
    
    def _generate_flicker_noise(self, n_samples: int, amplitude: float) -> np.ndarray:
        """
        플리커 노이즈 (Flicker/1/f Noise) 생성
        
        "핑크 노이즈"로도 불리며, 반도체 결함으로 인한 저주파 노이즈.
        IIR 필터를 사용하여 백색 노이즈를 1/f 스펙트럼으로 성형.
        
        Args:
            n_samples: 샘플 수
            amplitude: RMS 진폭 (선형 스케일)
            
        Returns:
            플리커 노이즈 신호
        """
        # 백색 노이즈 생성
        white_noise = np.random.normal(0.0, 1.0, n_samples)
        
        # 핑크 노이즈 IIR 필터 계수 (Audio EQ Cookbook)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        
        # 필터 적용
        pink_noise = signal.lfilter(b, a, white_noise)
        
        # RMS를 1로 정규화
        pink_noise = pink_noise / np.sqrt(np.mean(pink_noise ** 2))
        
        # 목표 진폭으로 스케일링
        return pink_noise * amplitude
    
    def _generate_shot_noise(self, n_samples: int, amplitude: float, rate: float = 100.0) -> np.ndarray:
        """
        샷 노이즈 (Shot/Poisson Noise) 생성
        
        전하의 불연속적 이동으로 인한 "크래클" 사운드.
        푸아송 프로세스 기반 임펄스 트레인으로 모델링.
        
        Args:
            n_samples: 샘플 수
            amplitude: 임펄스 진폭
            rate: 초당 임펄스 발생 빈도 (Hz)
            
        Returns:
            샷 노이즈 신호
        """
        # 임펄스 트레인 생성
        crackle = np.zeros(n_samples)
        
        # 푸아송 프로세스: 임펄스 간 시간 간격은 지수 분포
        duration = n_samples / self.sr
        n_events = int(rate * duration * 2)  # 여유있게 생성
        
        if n_events > 0:
            # 지수 분포로 임펄스 간 간격 생성
            inter_arrival_times = np.random.exponential(1.0 / rate, n_events)
            
            # 샘플 인덱스로 변환
            arrival_samples = np.cumsum(inter_arrival_times * self.sr).astype(int)
            
            # 범위 내 임펄스만 배치
            valid_indices = arrival_samples[arrival_samples < n_samples]
            
            for idx in valid_indices:
                # 무작위 진폭의 임펄스
                crackle[idx] = np.random.uniform(-amplitude, amplitude)
        
        return crackle
    
    def _generate_mains_hum(
        self, 
        n_samples: int, 
        base_freq: float = 60.0,
        n_harmonics: int = 10,
        amplitude: float = 0.01,
        phase_mode: str = 'random'
    ) -> np.ndarray:
        """
        전원 험 (Mains Hum) 생성
        
        50/60Hz AC 전력선의 전자기 유도로 인한 노이즈.
        고조파의 가산 합성으로 모델링.
        
        Args:
            n_samples: 샘플 수
            base_freq: 기본 주파수 (50.0 or 60.0 Hz)
            n_harmonics: 생성할 고조파 수
            amplitude: 기본 주파수의 진폭
            phase_mode: 'random' (부드러운 Hum) 또는 'fixed' (날카로운 Buzz)
            
        Returns:
            험 노이즈 신호
        """
        t = np.arange(n_samples) / self.sr
        hum_signal = np.zeros(n_samples)
        
        for i in range(1, n_harmonics + 1):
            freq = base_freq * i
            
            # 2차 고조파 강조 (정류기 효과)
            if i == 2:
                harmonic_amplitude = amplitude * 1.5
            else:
                # 고조파 감쇠 (1/i^2)
                harmonic_amplitude = amplitude / (i ** 2)
            
            # 위상 설정
            if phase_mode == 'random':
                phase = np.random.rand() * 2 * np.pi  # 무작위 위상 (부드러운 Hum)
            else:
                phase = 0.0  # 고정 위상 (날카로운 Buzz)
            
            hum_signal += harmonic_amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # RMS 정규화 및 스케일링
        hum_signal = hum_signal / np.sqrt(np.mean(hum_signal ** 2)) * amplitude
        
        return hum_signal
    
    def _generate_rfi_noise(
        self,
        n_samples: int,
        carrier_freq: float = 8000.0,
        modulator_freq: float = 100.0,
        amplitude: float = 0.01,
        use_noise_modulator: bool = False
    ) -> np.ndarray:
        """
        RFI/EMI 노이즈 (Radio/Electromagnetic Interference) 생성
        
        무선 주파수 간섭 (Wi-Fi, 휴대폰 등)으로 인한 "와인" 및 "데이터 버즈".
        진폭 변조 (AM)로 모델링.
        
        Args:
            n_samples: 샘플 수
            carrier_freq: 반송파(Whine) 주파수 (Hz)
            modulator_freq: 변조파(Buzz) 주파수 (Hz)
            amplitude: 최종 신호 진폭
            use_noise_modulator: True시 노이즈로 변조 (Wi-Fi 데이터)
            
        Returns:
            RFI 노이즈 신호
        """
        t = np.arange(n_samples) / self.sr
        
        # 1. 반송파 (Carrier - Whine)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        # 2. 변조 신호 (Modulator - Data Buzz)
        if use_noise_modulator:
            # 핑크 노이즈로 복잡한 데이터 전송 시뮬레이션
            modulator_raw = self._generate_flicker_noise(n_samples, amplitude=1.0)
            modulator = (modulator_raw / np.max(np.abs(modulator_raw)) + 1.0) / 2.0  # 0~1 범위
        else:
            # 사각파로 주기적인 데이터 버즈 시뮬레이션
            modulator = 0.5 * signal.square(2 * np.pi * modulator_freq * t) + 0.5  # 0~1 범위
        
        # 3. 진폭 변조 (AM)
        rfi_signal = carrier * modulator
        
        # 4. RMS 정규화 및 스케일링
        rfi_signal = rfi_signal / np.sqrt(np.mean(rfi_signal ** 2)) * amplitude
        
        return rfi_signal
    
    def _mix_at_snr(self, clean_signal: np.ndarray, noise_signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        두 신호를 지정된 SNR(dB)로 정확하게 혼합
        
        Args:
            clean_signal: 원본 신호
            noise_signal: 노이즈 신호
            snr_db: 목표 SNR (dB)
            
        Returns:
            혼합된 신호
        """
        # 1. RMS 계산
        clean_rms = np.sqrt(np.mean(clean_signal ** 2))
        noise_rms = np.sqrt(np.mean(noise_signal ** 2))
        
        # 2. 목표 노이즈 RMS 계산
        snr_linear = 10 ** (snr_db / 20.0)
        target_noise_rms = clean_rms / snr_linear
        
        # 3. 스케일링 팩터 계산
        if noise_rms < 1e-10:
            scaling_factor = 0.0
        else:
            scaling_factor = target_noise_rms / noise_rms
        
        # 4. 노이즈 스케일링 및 믹싱
        adjusted_noise = noise_signal * scaling_factor
        mixed_signal = clean_signal + adjusted_noise
        
        return mixed_signal
    
    def add_electrical_noise(
        self, 
        audio: np.ndarray,
        thermal_amplitude: float = None,
        flicker_amplitude: float = None,
        shot_rate: float = None,
        hum_freq: float = 60.0,
        hum_amplitude: float = None,
        rfi_carrier_freq: float = None,
        rfi_amplitude: float = None,
        global_snr_db: float = None
    ) -> np.ndarray:
        """
        전기적 잡음 추가 (물리 기반 알고리즘 합성)
        
        5가지 유형의 전기적 노이즈를 생성하고 SNR 기반으로 혼합:
        1. 열 노이즈 (Thermal) - 가우시안 백색 노이즈
        2. 플리커 노이즈 (Flicker) - 1/f 핑크 노이즈
        3. 샷 노이즈 (Shot) - 푸아송 크래클
        4. 전원 험 (Mains Hum) - 고조파 가산 합성
        5. RFI/EMI - 진폭 변조
        
        Args:
            audio: 입력 오디오 신호
            thermal_amplitude: 열 노이즈 진폭 (None이면 랜덤)
            flicker_amplitude: 플리커 노이즈 진폭 (None이면 랜덤)
            shot_rate: 샷 노이즈 발생 빈도 (Hz, None이면 랜덤)
            hum_freq: 험 주파수 (50.0 or 60.0 Hz)
            hum_amplitude: 험 노이즈 진폭 (None이면 랜덤)
            rfi_carrier_freq: RFI 반송파 주파수 (None이면 랜덤)
            rfi_amplitude: RFI 노이즈 진폭 (None이면 랜덤)
            global_snr_db: 전체 노이즈의 목표 SNR (None이면 30-50dB 랜덤)
            
        Returns:
            전기적 잡음이 추가된 오디오
        """
        n_samples = len(audio)
        
        # 파라미터 랜덤 설정
        if thermal_amplitude is None:
            thermal_amplitude = np.random.uniform(0.0003, 0.0015)  # -70~-56 dBFS
        
        if flicker_amplitude is None:
            flicker_amplitude = np.random.uniform(0.0007, 0.0025)  # -63~-52 dBFS
        
        if shot_rate is None:
            shot_rate = np.random.uniform(40, 150)  # 40-150 Hz
        
        if hum_amplitude is None:
            hum_amplitude = np.random.uniform(0.003, 0.012)  # -50~-38 dBFS
        
        if rfi_carrier_freq is None:
            rfi_carrier_freq = np.random.uniform(6000, 12000)  # 6-12 kHz
        
        if rfi_amplitude is None:
            rfi_amplitude = np.random.uniform(0.001, 0.006)  # -60~-44 dBFS
        
        if global_snr_db is None:
            global_snr_db = np.random.uniform(30, 50)  # 30-50 dB SNR
        
        # "노이즈 칵테일" 생성
        noise_thermal = self._generate_thermal_noise(n_samples, thermal_amplitude)
        noise_flicker = self._generate_flicker_noise(n_samples, flicker_amplitude)
        noise_shot = self._generate_shot_noise(n_samples, hum_amplitude * 0.1, shot_rate)
        noise_hum = self._generate_mains_hum(
            n_samples, 
            base_freq=hum_freq,
            n_harmonics=np.random.randint(8, 15),
            amplitude=hum_amplitude,
            phase_mode='random'
        )
        noise_rfi = self._generate_rfi_noise(
            n_samples,
            carrier_freq=rfi_carrier_freq,
            modulator_freq=100.0,
            amplitude=rfi_amplitude,
            use_noise_modulator=np.random.rand() > 0.5
        )
        
        # 모든 노이즈 합산
        total_noise = noise_thermal + noise_flicker + noise_shot + noise_hum + noise_rfi
        
        # SNR 기반 정확한 믹싱
        noisy_audio = self._mix_at_snr(audio, total_noise, global_snr_db)
        
        # 클리핑 방지
        peak_level = np.max(np.abs(noisy_audio))
        if peak_level > 1.0:
            noisy_audio = noisy_audio / peak_level * 0.99
        
        return noisy_audio
    
    def apply_all_noise(
        self,
        audio: np.ndarray,
        proximity_boost_db: float = None,
        pop_frequency: float = None,
        pop_pattern: str = 'cardioid',
        hum_freq: float = 60.0,
        electrical_snr_db: float = None
    ) -> np.ndarray:
        """
        모든 마이크 잡음을 한번에 적용
        
        물리 기반 노이즈 합성:
        1. 근접 효과 (Proximity Effect)
        2. 팝노이즈 (Pop Noise)
        3. 전기적 잡음 5종 (Thermal, Flicker, Shot, Hum, RFI)
        
        Args:
            audio: 입력 오디오
            proximity_boost_db: 근접효과 부스트 강도 (dB)
            pop_frequency: 초당 팝 발생 횟수
            pop_pattern: 팝노이즈 시뮬레이션용 마이크 패턴 ('cardioid' 또는 'figure8')
            hum_freq: 험 주파수 (50.0 or 60.0 Hz)
            electrical_snr_db: 전기적 노이즈 전체의 목표 SNR (dB)
            
        Returns:
            모든 잡음이 적용된 오디오
        """
        # 1. 근접효과
        audio = self.simulate_proximity_effect(audio, proximity_boost_db)
        
        # 2. 팝노이즈 (물리 기반)
        audio = self.add_pop_noise(audio, pop_frequency, pattern=pop_pattern)
        
        # 3. 전기적 잡음 (5종 통합)
        audio = self.add_electrical_noise(
            audio,
            hum_freq=hum_freq,
            global_snr_db=electrical_snr_db
        )
        
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 출력 디렉토리 생성 (noisy만 생성, clean은 복사하지 않음)
    noisy_dir = output_dir / "noisy"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
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
            # 각 파일마다 다른 랜덤 시드 설정 (파일명 기반)
            file_seed = hash(audio_path.stem) % (2**31)
            np.random.seed(file_seed)
            random.seed(file_seed)
            
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # 정규화 (랜덤 RMS 레벨)
            target_rms_db = np.random.uniform(-35, -15)
            audio = normalize_audio(audio, target_rms_db=target_rms_db)
            
            # 랜덤 파라미터 생성
            proximity_boost_db = np.random.uniform(0, 4)  # 0-4dB 근접효과 부스트
            pop_frequency = np.random.uniform(0, 1.5)  # 0-1.5 pops/second
            pop_pattern = random.choice(['cardioid', 'figure8'])  # 마이크 패턴
            hum_freq = random.choice([50.0, 60.0])  # 험 주파수 (50Hz 또는 60Hz)
            electrical_snr_db = np.random.uniform(35, 50)  # 35-50dB SNR
            
            # 잡음 적용 (랜덤 파라미터)
            noisy_audio = simulator.apply_all_noise(
                audio,
                proximity_boost_db=proximity_boost_db,
                pop_frequency=pop_frequency,
                pop_pattern=pop_pattern,
                hum_freq=hum_freq,
                electrical_snr_db=electrical_snr_db
            )
            
            # 저장 (원본 파일명 유지, _noisy 접미사 제거)
            noisy_path = noisy_dir / audio_path.name
            
            sf.write(noisy_path, noisy_audio, sample_rate)
            
        except Exception as e:
            print(f"오류 발생 ({audio_path.name}): {e}")
            continue
    
    print("데이터셋 합성 완료!")
    print(f"   잡음 음성: {noisy_dir}")


def synthesize_for_splits(
    dataset_root: Path,
    splits: Tuple[str, ...] = ("train", "test"),
    num_samples: int = None,
    sample_rate: int = 16000
):
    """
    데이터셋 루트에서 지정된 분할(split)의 clean → noisy 합성 수행

    Args:
        dataset_root: train/test 등의 분할 디렉토리를 포함한 루트 경로
        splits: 합성을 수행할 분할 이름 목록
        num_samples: 각 분할에서 처리할 파일 수 (None이면 전체)
        sample_rate: 샘플링 레이트
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"데이터셋 루트를 찾을 수 없습니다: {dataset_root}")

    for split in splits:
        split_clean_dir = dataset_root / split / "clean"
        split_output_dir = dataset_root / split

        if not split_clean_dir.exists():
            print(f"[건너뜀] clean 디렉토리를 찾을 수 없습니다: {split_clean_dir}")
            continue

        print(f"\n=== '{split}' 분할 합성 시작 ===")
        synthesize_dataset(
            clean_dir=split_clean_dir,
            output_dir=split_output_dir,
            num_samples=num_samples,
            sample_rate=sample_rate
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="마이크 잡음 데이터셋 합성")
    parser.add_argument("--dataset_root", type=str, help="train/test 분할을 포함한 데이터셋 루트")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"], help="합성을 수행할 분할 이름 목록")
    parser.add_argument("--clean_dir", type=str, help="단일 합성을 위한 깨끗한 음성 디렉토리")
    parser.add_argument("--output_dir", type=str, help="단일 합성을 위한 출력 디렉토리")
    parser.add_argument("--num_samples", type=int, default=None, help="생성할 샘플 수")
    parser.add_argument("--sample_rate", type=int, default=16000, help="샘플링 레이트")
    
    args = parser.parse_args()
    
    if args.dataset_root:
        synthesize_for_splits(
            dataset_root=Path(args.dataset_root),
            splits=tuple(args.splits),
            num_samples=args.num_samples,
            sample_rate=args.sample_rate
        )
    elif args.clean_dir and args.output_dir:
        synthesize_dataset(
            clean_dir=Path(args.clean_dir),
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples,
            sample_rate=args.sample_rate
        )
    else:
        parser.error("--dataset_root 또는 --clean_dir/--output_dir 조합 중 하나를 제공해야 합니다.")

