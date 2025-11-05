# 마이크 잡음 제거 시스템 (Microphone Noise Removal System)

물리 기반 노이즈 합성과 딥러닝을 결합한 음성 향상(Speech Enhancement) 시스템

## 🎯 주요 기능

- **근접효과 (Proximity Effect)** - 물리 기반 로우 쉘프 필터 시뮬레이션
- **팝노이즈 (Pop Noise)** - 압력 경도 변환기 원리 기반 합성
- **전기적 잡음 (Electrical Noise)** - 5종 노이즈 통합 (Thermal, Flicker, Shot, Hum, RFI)
- **대규모 학습** - 23,000개 샘플 지원
- **실시간 처리** - 경량 U-Net 기반 모델

## ⚡ 빠른 시작 (Quick Start)

### 1. 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# Train 데이터 (모든 23,000개 파일)
python prepare_training_data.py

# Validation 데이터
python prepare_validation_data.py

# (선택) Test 데이터
python prepare_test_data.py
```

### 3. 데이터 검증

```bash
python verify_data_setup.py
```

### 4. 학습 시작

```bash
python training/train.py
```

### 5. 모니터링

```bash
tensorboard --logdir runs/
```

브라우저에서 http://localhost:6006 접속

**자세한 내용**: [QUICKSTART.md](QUICKSTART.md) 참조

## 📁 데이터 구조

```
data/
├── train/              ← 학습 데이터 (~23,000개)
│   ├── clean/         ← 원본 음성 파일 배치
│   └── noisy/         ← 자동 생성됨
│
├── val/               ← 검증 데이터
│   ├── clean/         ← 원본 음성 파일 배치
│   └── noisy/         ← 자동 생성됨
│
└── test/              ← 평가 전용 (학습 X)
    ├── clean/         ← 원본 음성 파일 배치
    └── noisy/         ← 자동 생성됨
```

**자세한 내용**: [DATA_PREPARATION_GUIDE.md](DATA_PREPARATION_GUIDE.md) 참조

## 📊 데이터 사용 원칙

| 데이터셋 | 파일 수 | 학습 사용 | 용도 |
|---------|---------|-----------|------|
| **Train** | ~23,000 | ✅ Yes | 모델 학습 |
| **Val** | ~500-1000 | ⚠️ 검증만 | 손실 계산, 조기 중단 |
| **Test** | 사용자 지정 | ❌ No | 최종 평가 전용 |

**⚠️ 중요**: Test 데이터는 학습에 절대 사용되지 않습니다!

## 🚀 전체 워크플로우

```bash
# 1. 데이터 준비 (모든 23,000개 파일)
python prepare_training_data.py
python prepare_validation_data.py

# 2. 데이터 검증
python verify_data_setup.py

# 3. 학습 시작
python training/train.py

# 4. 모니터링
tensorboard --logdir runs/

# 5. 추론 (학습 후)
python inference/denoise.py \
  --input noisy.wav \
  --output clean.wav \
  --checkpoint checkpoints/best_model.pt

# 6. 평가
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --test_noisy_dir data/test/noisy \
  --test_clean_dir data/test/clean
```

## 🔬 기술적 특징

### 물리 기반 노이즈 합성

#### 1. 근접 효과 (Proximity Effect)

- **로우 쉘프 바이쿼드 필터** (Audio EQ Cookbook 공식)
- **6dB/octave 기울기** (공진 없음, Q=0.707)
- **거리 기반 게인**: `gain_dB = C_pattern / distance_cm`
- **패턴별 상수**: Cardioid (C=60), Figure-8 (C=120)

#### 2. 팝노이즈 (Pop Noise)

- **시간 변화 거리 프로파일** (가우시안 기반)
- **동적 필터링** (5ms 윈도우)
- **역제곱 법칙** 레벨 증가
- **압력 경도 변환기 원리** 정확히 모델링

#### 3. 전기적 잡음 (5종 통합)

##### 내재적 노이즈 (Stochastic)
- **열 노이즈 (Thermal)**: 가우시안 백색 노이즈
- **플리커 노이즈 (Flicker)**: 1/f 핑크 노이즈 (IIR 필터)
- **샷 노이즈 (Shot)**: 푸아송 임펄스 트레인 (크래클)

##### 외재적 노이즈 (Interference)
- **전원 험 (Mains Hum)**: 고조파 가산 합성 (8-15개), 2차 고조파 강조, 위상 제어
- **RFI/EMI**: 진폭 변조 (AM) - 와인 + 데이터 버즈

**자세한 내용**: 
- [data/POP_NOISE_UPDATE.md](data/POP_NOISE_UPDATE.md) - 팝노이즈 알고리즘
- [data/ELECTRICAL_NOISE_UPDATE.md](data/ELECTRICAL_NOISE_UPDATE.md) - 전기적 노이즈 알고리즘

## 📂 프로젝트 구조

```
effect/
├── data/                           # 데이터 처리
│   ├── synthesizer.py             # 노이즈 합성 (물리 기반)
│   ├── dataset.py                 # PyTorch Dataset
│   ├── POP_NOISE_UPDATE.md        # 팝노이즈 알고리즘 문서
│   └── test_pop_noise.py          # 팝노이즈 테스트
│
├── models/                         # 모델 정의
│   ├── unet.py                    # U-Net 아키텍처
│   ├── fullsubnet.py              # Fast FullSubNet (고급)
│   └── preprocessing.py           # 전처리 모듈
│
├── training/                       # 학습 코드
│   ├── train.py                   # 메인 학습 스크립트
│   ├── config.py                  # 설정 및 하이퍼파라미터
│   └── losses.py                  # 손실 함수
│
├── inference/                      # 추론 코드
│   └── denoise.py                 # 실시간 노이즈 제거
│
├── evaluation/                     # 평가 도구
│   ├── evaluate.py                # 메트릭 계산
│   ├── plot_results.py            # 시각화
│   ├── metrics.py                 # 평가 지표
│   └── summarize_results.py       # 통계 요약
│
├── prepare_training_data.py        # Train 데이터 준비
├── prepare_validation_data.py      # Val 데이터 준비
├── prepare_test_data.py            # Test 데이터 준비
├── verify_data_setup.py            # 데이터 검증
│
├── DATA_PREPARATION_GUIDE.md       # 데이터 준비 가이드
├── QUICKSTART.md                   # 빠른 시작 가이드
└── README.md                       # 이 파일
```

## 💡 고급 사용법

### 노이즈 합성 (맞춤 설정)

```python
from data.synthesizer import MicrophoneNoiseSimulator

simulator = MicrophoneNoiseSimulator(sample_rate=16000)

# 물리 기반 노이즈 합성 (통합)
noisy_audio = simulator.apply_all_noise(
    audio=clean_audio,
    proximity_boost_db=6.0,      # 근접 효과
    pop_frequency=2.0,            # 팝노이즈 빈도
    pop_pattern='cardioid',       # 마이크 패턴
    hum_freq=60.0,                # 험 주파수
    electrical_snr_db=30          # 전기적 노이즈 SNR (5종 통합)
)

# 전기적 노이즈만 세밀하게 제어
noisy_audio = simulator.add_electrical_noise(
    audio=clean_audio,
    thermal_amplitude=0.001,      # 열 노이즈 (White)
    flicker_amplitude=0.002,      # 플리커 노이즈 (Pink)
    shot_rate=150,                # 샷 노이즈 (Crackle)
    hum_amplitude=0.015,          # 험 (Hum/Buzz)
    rfi_amplitude=0.005,          # RFI/EMI (Whine)
    global_snr_db=25              # 전체 SNR
)
```

### 학습 파라미터 조정

`training/config.py` 파일 수정:

```python
@dataclass
class DataConfig:
    train_noisy_dir: str = "data/train/noisy"
    train_clean_dir: str = "data/train/clean"
    batch_size: int = 16
    segment_length: int = 64000  # 4초 @ 16kHz

@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
```

## 📈 평가 지표

- **SI-SDR** (Scale-Invariant SDR) - 신호 분리 성능
- **PESQ** (Perceptual Evaluation of Speech Quality) - 음질 (1.0~4.5)
- **STOI** (Short-Time Objective Intelligibility) - 명료도 (0~1)
- **DNSMOS P.835** - 주관적 품질 (SIG, BAK, OVRL)

## 📖 상세 문서

- [빠른 시작 가이드](QUICKSTART.md) - 3단계로 시작하기
- [데이터 준비 가이드](DATA_PREPARATION_GUIDE.md) - 데이터 구조 및 준비 방법
- [팝노이즈 알고리즘](data/POP_NOISE_UPDATE.md) - 물리 기반 합성 알고리즘
- [전기적 노이즈 알고리즘](data/ELECTRICAL_NOISE_UPDATE.md) - 5종 노이즈 통합 시스템

## 🐛 문제 해결

### "잡음 음성 파일을 찾을 수 없습니다"

```bash
# 해당 데이터셋 준비 스크립트 실행
python prepare_training_data.py      # train용
python prepare_validation_data.py    # val용
```

### GPU 메모리 부족

```python
# training/config.py에서 배치 크기 또는 세그먼트 길이 줄이기
batch_size: 8              # 기본 16에서 줄임
segment_length: 32000      # 2초 @ 16kHz
```

### 데이터 검증 실패

```bash
# 데이터 설정 검증 도구 실행
python verify_data_setup.py
```

## ✅ 학습 전 체크리스트

- [ ] `data/train/clean`에 23,000개 파일이 있는가?
- [ ] `prepare_training_data.py` 실행 완료?
- [ ] `prepare_validation_data.py` 실행 완료?
- [ ] `data/train/noisy`에 23,000개 파일이 생성되었는가?
- [ ] Test 데이터가 train/val과 분리되어 있는가?
- [ ] `verify_data_setup.py`가 모든 검증을 통과했는가?

## 🤝 기여

이슈 및 Pull Request를 환영합니다!

## 📚 참고 문헌

- Beranek, L. L. (1954). *Acoustics*. McGraw-Hill.
- Bristow-Johnson, R. (1994). *Cookbook formulae for audio EQ biquad filter coefficients*.
- Microsoft DNS Challenge (2020-2024)
- Fast FullSubNet (ICASSP 2022)

## 📄 라이선스

MIT License

---

**최종 업데이트**: 2025-11-05  
**문의**: 이슈를 생성하거나 PR을 제출해주세요
