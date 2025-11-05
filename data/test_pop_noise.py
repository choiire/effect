"""
팝노이즈 합성 테스트 스크립트

물리 기반 팝노이즈 합성이 제대로 작동하는지 확인하기 위한 테스트
"""

import numpy as np
import soundfile as sf
from synthesizer import MicrophoneNoiseSimulator
import matplotlib.pyplot as plt
from pathlib import Path


def create_test_signal(duration=3.0, sr=16000):
    """테스트용 신호 생성 (음성 스펙트럼 유사)"""
    t = np.arange(int(duration * sr)) / sr
    
    # 기본파 (100Hz)
    signal = 0.3 * np.sin(2 * np.pi * 100 * t)
    
    # 고조파 추가
    signal += 0.2 * np.sin(2 * np.pi * 200 * t)
    signal += 0.15 * np.sin(2 * np.pi * 300 * t)
    signal += 0.1 * np.sin(2 * np.pi * 400 * t)
    
    # 엔벨로프 적용 (시간에 따라 변화)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    signal *= envelope
    
    # 정규화
    signal = signal / np.max(np.abs(signal)) * 0.5
    
    return signal


def test_pop_noise_variations():
    """다양한 설정으로 팝노이즈 테스트"""
    
    sr = 16000
    simulator = MicrophoneNoiseSimulator(sample_rate=sr)
    
    # 테스트 신호 생성
    clean_signal = create_test_signal(duration=4.0, sr=sr)
    
    # 출력 디렉토리
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 원본 저장
    sf.write(output_dir / "00_clean.wav", clean_signal, sr)
    print(f"원본 신호 저장: {output_dir / '00_clean.wav'}")
    
    # 테스트 1: Cardioid 패턴 (약한 효과)
    print("\n테스트 1: Cardioid 패턴")
    pop_cardioid = simulator.add_pop_noise(
        clean_signal.copy(),
        pop_frequency=2.0,  # 2회/초
        intensity=0.7,
        pattern='cardioid'
    )
    sf.write(output_dir / "01_pop_cardioid.wav", pop_cardioid, sr)
    print(f"  저장: {output_dir / '01_pop_cardioid.wav'}")
    
    # 테스트 2: Figure-8 패턴 (강한 효과)
    print("\n테스트 2: Figure-8 패턴")
    pop_figure8 = simulator.add_pop_noise(
        clean_signal.copy(),
        pop_frequency=2.0,
        intensity=0.7,
        pattern='figure8'
    )
    sf.write(output_dir / "02_pop_figure8.wav", pop_figure8, sr)
    print(f"  저장: {output_dir / '02_pop_figure8.wav'}")
    
    # 테스트 3: 약한 강도
    print("\n테스트 3: 약한 강도")
    pop_weak = simulator.add_pop_noise(
        clean_signal.copy(),
        pop_frequency=2.0,
        intensity=0.3,
        pattern='cardioid'
    )
    sf.write(output_dir / "03_pop_weak.wav", pop_weak, sr)
    print(f"  저장: {output_dir / '03_pop_weak.wav'}")
    
    # 테스트 4: 강한 강도
    print("\n테스트 4: 강한 강도")
    pop_strong = simulator.add_pop_noise(
        clean_signal.copy(),
        pop_frequency=2.0,
        intensity=0.9,
        pattern='cardioid'
    )
    sf.write(output_dir / "04_pop_strong.wav", pop_strong, sr)
    print(f"  저장: {output_dir / '04_pop_strong.wav'}")
    
    # 테스트 5: 빈번한 팝
    print("\n테스트 5: 빈번한 팝 (5회/초)")
    pop_frequent = simulator.add_pop_noise(
        clean_signal.copy(),
        pop_frequency=5.0,
        intensity=0.6,
        pattern='cardioid'
    )
    sf.write(output_dir / "05_pop_frequent.wav", pop_frequent, sr)
    print(f"  저장: {output_dir / '05_pop_frequent.wav'}")
    
    # 스펙트로그램 비교
    plot_comparison(clean_signal, pop_cardioid, pop_figure8, sr, output_dir)
    
    print(f"\n✅ 모든 테스트 완료! 결과는 {output_dir} 폴더에 저장되었습니다.")


def plot_comparison(clean, pop_cardioid, pop_figure8, sr, output_dir):
    """스펙트로그램 비교 플롯"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # STFT 계산
    from scipy import signal as sig
    
    signals = [clean, pop_cardioid, pop_figure8]
    titles = ['Original Clean Signal', 'Pop Noise (Cardioid)', 'Pop Noise (Figure-8)']
    
    for idx, (audio, title) in enumerate(zip(signals, titles)):
        f, t, Sxx = sig.spectrogram(audio, sr, nperseg=512, noverlap=256)
        
        # dB 스케일
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        im = axes[idx].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        axes[idx].set_ylabel('Frequency (Hz)')
        axes[idx].set_title(title)
        axes[idx].set_ylim([0, 1000])  # 저주파 영역에 집중
        plt.colorbar(im, ax=axes[idx], label='Power (dB)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrogram_comparison.png', dpi=150)
    print(f"\n스펙트로그램 저장: {output_dir / 'spectrogram_comparison.png'}")
    plt.close()


def analyze_frequency_response():
    """주파수 응답 분석"""
    
    sr = 16000
    simulator = MicrophoneNoiseSimulator(sample_rate=sr)
    
    # 임펄스 생성
    impulse = np.zeros(sr)
    impulse[sr // 2] = 1.0
    
    # 다양한 거리에서의 로우 쉘프 필터 응답 계산
    distances = [2, 5, 10, 20, 50]
    
    plt.figure(figsize=(10, 6))
    
    for distance in distances:
        shelf_gain_db, _ = simulator._calculate_proximity_gain(
            distance, pattern='cardioid'
        )
        
        sos = simulator._design_low_shelf_biquad(shelf_gain_db, fc_hz=150.0, q=0.707)
        
        # 주파수 응답 계산
        from scipy import signal as sig
        w, h = sig.sosfreqz(sos, worN=2048, fs=sr)
        
        # dB로 변환
        h_db = 20 * np.log10(np.abs(h) + 1e-10)
        
        plt.semilogx(w, h_db, label=f'{distance}cm')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Proximity Effect: Low-Shelf Filter Response at Various Distances')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([20, 1000])
    plt.axvline(150, color='red', linestyle='--', alpha=0.5, label='Corner Freq (150Hz)')
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'frequency_response.png', dpi=150)
    print(f"주파수 응답 그래프 저장: {output_dir / 'frequency_response.png'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("물리 기반 팝노이즈 합성 테스트")
    print("=" * 60)
    
    # 주파수 응답 분석
    print("\n[1] 주파수 응답 분석")
    analyze_frequency_response()
    
    # 팝노이즈 변형 테스트
    print("\n[2] 팝노이즈 변형 테스트")
    test_pop_noise_variations()
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

