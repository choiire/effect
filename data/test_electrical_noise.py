"""
전기적 노이즈 합성 테스트 스크립트

5가지 유형의 전기적 노이즈를 개별적으로 생성하고 평가합니다.
"""

import numpy as np
import soundfile as sf
from synthesizer import MicrophoneNoiseSimulator
import matplotlib.pyplot as plt
from pathlib import Path


def create_test_tone(duration=3.0, sr=16000):
    """테스트용 톤 신호 생성 (음성 스펙트럼 유사)"""
    t = np.arange(int(duration * sr)) / sr
    
    # 복합 톤 (음성 포먼트 유사)
    signal = 0.3 * np.sin(2 * np.pi * 150 * t)    # F1
    signal += 0.2 * np.sin(2 * np.pi * 850 * t)   # F2
    signal += 0.15 * np.sin(2 * np.pi * 2500 * t) # F3
    
    # 엔벨로프
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    signal *= envelope
    
    # 정규화
    signal = signal / np.max(np.abs(signal)) * 0.5
    
    return signal


def test_individual_noises():
    """개별 노이즈 타입 테스트"""
    
    sr = 16000
    simulator = MicrophoneNoiseSimulator(sample_rate=sr)
    
    # 테스트 신호
    duration = 5.0
    n_samples = int(duration * sr)
    clean_signal = create_test_tone(duration=duration, sr=sr)
    
    # 출력 디렉토리
    output_dir = Path("test_outputs/electrical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 원본 저장
    sf.write(output_dir / "00_clean.wav", clean_signal, sr)
    print(f"원본 신호 저장: {output_dir / '00_clean.wav'}")
    
    print("\n" + "=" * 70)
    print("개별 노이즈 타입 테스트")
    print("=" * 70)
    
    # 1. 열 노이즈 (Thermal - White Noise)
    print("\n[1/5] 열 노이즈 (Thermal/White Noise)")
    noise_thermal = simulator._generate_thermal_noise(n_samples, amplitude=0.01)
    noisy_thermal = simulator._mix_at_snr(clean_signal, noise_thermal, snr_db=20)
    sf.write(output_dir / "01_thermal_noise.wav", noisy_thermal, sr)
    print(f"  저장: 01_thermal_noise.wav (SNR=20dB)")
    
    # 2. 플리커 노이즈 (Flicker - Pink Noise)
    print("\n[2/5] 플리커 노이즈 (Flicker/Pink Noise)")
    noise_flicker = simulator._generate_flicker_noise(n_samples, amplitude=0.01)
    noisy_flicker = simulator._mix_at_snr(clean_signal, noise_flicker, snr_db=20)
    sf.write(output_dir / "02_flicker_noise.wav", noisy_flicker, sr)
    print(f"  저장: 02_flicker_noise.wav (SNR=20dB)")
    
    # 3. 샷 노이즈 (Shot - Crackle)
    print("\n[3/5] 샷 노이즈 (Shot/Crackle)")
    noise_shot = simulator._generate_shot_noise(n_samples, amplitude=0.05, rate=150)
    noisy_shot = clean_signal + noise_shot
    peak = np.max(np.abs(noisy_shot))
    if peak > 1.0:
        noisy_shot /= peak
    sf.write(output_dir / "03_shot_noise.wav", noisy_shot, sr)
    print(f"  저장: 03_shot_noise.wav (Rate=150Hz)")
    
    # 4. 전원 험 (Mains Hum)
    print("\n[4/5] 전원 험 (Mains Hum)")
    
    # 4-1. 부드러운 Hum (무작위 위상)
    noise_hum_smooth = simulator._generate_mains_hum(
        n_samples, 
        base_freq=60.0,
        n_harmonics=10,
        amplitude=0.02,
        phase_mode='random'
    )
    noisy_hum_smooth = simulator._mix_at_snr(clean_signal, noise_hum_smooth, snr_db=15)
    sf.write(output_dir / "04a_mains_hum_smooth.wav", noisy_hum_smooth, sr)
    print(f"  저장: 04a_mains_hum_smooth.wav (60Hz, 10 harmonics, random phase)")
    
    # 4-2. 날카로운 Buzz (고정 위상)
    noise_hum_buzz = simulator._generate_mains_hum(
        n_samples,
        base_freq=60.0,
        n_harmonics=10,
        amplitude=0.02,
        phase_mode='fixed'
    )
    noisy_hum_buzz = simulator._mix_at_snr(clean_signal, noise_hum_buzz, snr_db=15)
    sf.write(output_dir / "04b_mains_buzz_sharp.wav", noisy_hum_buzz, sr)
    print(f"  저장: 04b_mains_buzz_sharp.wav (60Hz, 10 harmonics, fixed phase)")
    
    # 5. RFI/EMI 노이즈
    print("\n[5/5] RFI/EMI 노이즈")
    
    # 5-1. 사각파 변조 (주기적 버즈)
    noise_rfi_periodic = simulator._generate_rfi_noise(
        n_samples,
        carrier_freq=8000,
        modulator_freq=100,
        amplitude=0.01,
        use_noise_modulator=False
    )
    noisy_rfi_periodic = simulator._mix_at_snr(clean_signal, noise_rfi_periodic, snr_db=18)
    sf.write(output_dir / "05a_rfi_periodic_buzz.wav", noisy_rfi_periodic, sr)
    print(f"  저장: 05a_rfi_periodic_buzz.wav (8kHz carrier, 100Hz modulator)")
    
    # 5-2. 노이즈 변조 (데이터 버즈)
    noise_rfi_data = simulator._generate_rfi_noise(
        n_samples,
        carrier_freq=10000,
        modulator_freq=100,
        amplitude=0.01,
        use_noise_modulator=True
    )
    noisy_rfi_data = simulator._mix_at_snr(clean_signal, noise_rfi_data, snr_db=18)
    sf.write(output_dir / "05b_rfi_data_buzz.wav", noisy_rfi_data, sr)
    print(f"  저장: 05b_rfi_data_buzz.wav (10kHz carrier, noise modulator)")
    
    # 6. 통합 "노이즈 칵테일"
    print("\n[6] 통합 노이즈 칵테일")
    noisy_combined = simulator.add_electrical_noise(
        clean_signal.copy(),
        global_snr_db=20
    )
    sf.write(output_dir / "06_combined_electrical_noise.wav", noisy_combined, sr)
    print(f"  저장: 06_combined_electrical_noise.wav (5종 노이즈 통합, SNR=20dB)")
    
    print("\n" + "=" * 70)
    print(f"✅ 모든 테스트 완료! 결과는 {output_dir} 폴더에 저장되었습니다.")
    print("=" * 70)
    
    return output_dir


def plot_spectrogram_comparison(output_dir):
    """스펙트로그램 비교 플롯"""
    
    sr = 16000
    
    # 파일 로드
    files_to_plot = [
        ("00_clean.wav", "Clean Signal"),
        ("01_thermal_noise.wav", "Thermal (White)"),
        ("02_flicker_noise.wav", "Flicker (Pink)"),
        ("04a_mains_hum_smooth.wav", "Mains Hum (Smooth)"),
        ("05a_rfi_periodic_buzz.wav", "RFI (Periodic)"),
        ("06_combined_electrical_noise.wav", "Combined (All 5)")
    ]
    
    fig, axes = plt.subplots(len(files_to_plot), 1, figsize=(12, 14))
    
    from scipy import signal as sig
    
    for idx, (filename, title) in enumerate(files_to_plot):
        filepath = output_dir / filename
        
        if not filepath.exists():
            continue
        
        audio, _ = sf.read(filepath)
        
        # STFT 계산
        f, t, Sxx = sig.spectrogram(audio, sr, nperseg=512, noverlap=256)
        
        # dB 스케일
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        im = axes[idx].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
        axes[idx].set_ylabel('Frequency (Hz)')
        axes[idx].set_title(title)
        axes[idx].set_ylim([0, 8000])
        plt.colorbar(im, ax=axes[idx], label='Power (dB)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrogram_comparison.png', dpi=150)
    print(f"\n스펙트로그램 저장: {output_dir / 'spectrogram_comparison.png'}")
    plt.close()


def analyze_noise_characteristics():
    """노이즈 특성 분석"""
    
    sr = 16000
    duration = 10.0
    n_samples = int(duration * sr)
    
    simulator = MicrophoneNoiseSimulator(sample_rate=sr)
    
    # 각 노이즈 생성
    thermal = simulator._generate_thermal_noise(n_samples, amplitude=1.0)
    flicker = simulator._generate_flicker_noise(n_samples, amplitude=1.0)
    
    # PSD 계산
    from scipy import signal as sig
    
    f_thermal, psd_thermal = sig.welch(thermal, sr, nperseg=2048)
    f_flicker, psd_flicker = sig.welch(flicker, sr, nperseg=2048)
    
    # 플롯
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Thermal (White) - 평탄한 스펙트럼
    axes[0].loglog(f_thermal, psd_thermal, label='Thermal (White)', alpha=0.7)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD')
    axes[0].set_title('Thermal Noise - Flat Spectrum')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Flicker (Pink) - 1/f 스펙트럼
    axes[1].loglog(f_flicker, psd_flicker, label='Flicker (Pink)', color='tab:orange', alpha=0.7)
    
    # 이론적 1/f 기울기 표시
    f_ref = f_flicker[f_flicker > 20]
    psd_ref = psd_flicker[len(f_flicker) - len(f_ref)]
    theoretical_1f = psd_ref * (f_ref[0] / f_ref)
    axes[1].loglog(f_ref, theoretical_1f, '--', label='Theoretical 1/f', color='red', alpha=0.5)
    
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD')
    axes[1].set_title('Flicker Noise - 1/f Spectrum')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    output_dir = Path("test_outputs/electrical")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'psd_analysis.png', dpi=150)
    print(f"PSD 분석 그래프 저장: {output_dir / 'psd_analysis.png'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("전기적 노이즈 합성 테스트")
    print("=" * 70)
    
    # 무작위 시드 고정 (재현성)
    np.random.seed(42)
    
    # 개별 노이즈 테스트
    print("\n[1] 개별 노이즈 타입 테스트")
    output_dir = test_individual_noises()
    
    # 스펙트로그램 비교
    print("\n[2] 스펙트로그램 생성")
    plot_spectrogram_comparison(output_dir)
    
    # PSD 분석
    print("\n[3] 파워 스펙트럼 밀도 분석")
    analyze_noise_characteristics()
    
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)

