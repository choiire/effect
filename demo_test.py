"""
프로젝트 설정 및 모듈 테스트

각 컴포넌트가 정상적으로 작동하는지 확인
"""

import sys
from pathlib import Path
import numpy as np
import torch

print("="*60)
print("🧪 마이크 잡음 제거 시스템 - 모듈 테스트")
print("="*60)

# 1. 기본 라이브러리 확인
print("\n1️⃣ 기본 라이브러리 확인...")
try:
    import librosa
    import soundfile as sf
    import scipy
    print("   ✅ librosa, soundfile, scipy")
except ImportError as e:
    print(f"   ❌ 오류: {e}")
    sys.exit(1)

try:
    from pesq import pesq
    from pystoi import stoi
    print("   ✅ pesq, pystoi")
except ImportError as e:
    print(f"   ⚠️ 평가 지표 라이브러리 미설치: {e}")
    print("   → 설치: pip install pesq pystoi")

# 2. PyTorch 확인
print("\n2️⃣ PyTorch 확인...")
print(f"   버전: {torch.__version__}")
print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# 3. 잡음 합성기 테스트
print("\n3️⃣ 잡음 합성기 테스트...")
try:
    from data.synthesizer import MicrophoneNoiseSimulator
    
    simulator = MicrophoneNoiseSimulator(sample_rate=16000)
    
    # 테스트 신호
    test_audio = np.random.randn(16000) * 0.1  # 1초
    
    # 근접효과
    proximity = simulator.simulate_proximity_effect(test_audio, boost_db=6)
    print(f"   ✅ 근접효과 시뮬레이션 (출력 shape: {proximity.shape})")
    
    # 팝노이즈
    pop = simulator.add_pop_noise(test_audio, pop_frequency=2.0)
    print(f"   ✅ 팝노이즈 추가 (출력 shape: {pop.shape})")
    
    # 전기 잡음
    electrical = simulator.add_electrical_noise(test_audio, hum_freq=60)
    print(f"   ✅ 전기 잡음 추가 (출력 shape: {electrical.shape})")
    
    # 통합
    all_noise = simulator.apply_all_noise(test_audio)
    print(f"   ✅ 모든 잡음 적용 (출력 shape: {all_noise.shape})")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# 4. 전처리 필터 테스트
print("\n4️⃣ 전처리 필터 테스트...")
try:
    from models.preprocessing import MicrophoneNoisePreprocessor
    
    preprocessor = MicrophoneNoisePreprocessor(sample_rate=16000)
    
    test_tensor = torch.randn(2, 16000)  # [batch, samples]
    filtered = preprocessor(test_tensor)
    
    print(f"   ✅ 전처리 필터 (입력: {test_tensor.shape}, 출력: {filtered.shape})")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# 5. U-Net 모델 테스트
print("\n5️⃣ U-Net 모델 테스트...")
try:
    from models.unet import WaveformUNet, SpectrogramUNet
    
    # Waveform U-Net
    model_wave = WaveformUNet(n_channels=32)
    test_input = torch.randn(2, 32000)  # 2초
    
    with torch.no_grad():
        output = model_wave(test_input)
    
    params = sum(p.numel() for p in model_wave.parameters()) / 1e6
    print(f"   ✅ WaveformUNet (파라미터: {params:.2f}M, 출력: {output.shape})")
    
    # Spectrogram U-Net
    model_spec = SpectrogramUNet(n_fft=512, hop_length=256)
    
    with torch.no_grad():
        output = model_spec(test_input)
    
    params = sum(p.numel() for p in model_spec.parameters()) / 1e6
    print(f"   ✅ SpectrogramUNet (파라미터: {params:.2f}M, 출력: {output.shape})")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# 6. 손실 함수 테스트
print("\n6️⃣ 손실 함수 테스트...")
try:
    from training.losses import CombinedLoss
    
    criterion = CombinedLoss()
    
    pred = torch.randn(4, 16000)
    target = torch.randn(4, 16000)
    
    loss, loss_dict = criterion(pred, target, return_components=True)
    
    print(f"   ✅ CombinedLoss")
    print(f"      Total: {loss_dict['total']:.4f}")
    print(f"      SI-SDR: {loss_dict['si_sdr']:.4f}")
    print(f"      STFT: {loss_dict['stft']:.4f}")
    print(f"      Time: {loss_dict['time']:.4f}")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# 7. 평가 지표 테스트
print("\n7️⃣ 평가 지표 테스트...")
try:
    from evaluation.metrics import MetricsCalculator
    
    calculator = MetricsCalculator(sample_rate=16000)
    
    target = np.random.randn(16000) * 0.1
    pred = target + np.random.randn(16000) * 0.02
    
    metrics = calculator.calculate_all(pred, target)
    
    print(f"   ✅ 평가 지표 계산 완료")
    if metrics['si_sdr'] is not None:
        print(f"      SI-SDR: {metrics['si_sdr']:.2f} dB")
    if metrics['pesq'] is not None:
        print(f"      PESQ: {metrics['pesq']:.3f}")
    if metrics['stoi'] is not None:
        print(f"      STOI: {metrics['stoi']:.3f}")
    
except Exception as e:
    print(f"   ⚠️ 평가 지표 오류: {e}")
    print("   → pesq, pystoi 설치 필요")

# 8. 설정 파일 테스트
print("\n8️⃣ 설정 시스템 테스트...")
try:
    from training.config import get_default_config, save_config
    
    config = get_default_config()
    print(f"   ✅ 기본 설정 로드")
    print(f"      모델: {config.model.model_type}")
    print(f"      배치 크기: {config.data.batch_size}")
    print(f"      에폭: {config.training.num_epochs}")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")

# 9. 디렉토리 구조 확인
print("\n9️⃣ 디렉토리 구조 확인...")
required_dirs = [
    "data/clean",
    "data/train",
    "data/val",
    "data/test",
    "checkpoints",
    "logs",
    "evaluation/outputs"
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ⚠️ {dir_path} (미생성)")

# 최종 요약
print("\n" + "="*60)
print("📋 테스트 요약")
print("="*60)
print("""
✅ 기본 설정 완료

다음 단계:
1. 깨끗한 음성 데이터를 data/clean/ 에 준비
2. 데이터 합성: python data/synthesizer.py --clean_dir data/clean --output_dir data/train
3. 학습 시작: python training/train.py --config config.yaml
4. 평가: python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth
5. 추론: python inference/denoise.py --input noisy.wav --output clean.wav

자세한 사용법은 USAGE_GUIDE.md 참조
""")
print("="*60)

