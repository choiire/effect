# 마이크 잡음 제거 시스템 (Microphone Noise Reduction)

딥러닝 기반 음성 향상(Speech Enhancement) 시스템으로, 다음 마이크 특화 잡음을 제거합니다:

- **근접효과 (Proximity Effect)**: 저주파수 과도 증폭 (80-250Hz)
- **팝노이즈 (Pop Noise)**: 순간적 고에너지 저주파 펄스
- **전기적 잡음 (Electrical Noise)**: 험(Hum, 50/60Hz) 및 화이트 노이즈

## 프로젝트 구조

```
effect/
├── data/                   # 데이터 관련
│   ├── clean/             # 깨끗한 음성 데이터
│   ├── noisy/             # 합성된 잡음 음성
│   ├── synthesizer.py     # 마이크 잡음 합성기
│   └── dataset.py         # PyTorch Dataset 클래스
├── models/                 # 모델 아키텍처
│   ├── unet.py            # 베이스라인 U-Net
│   ├── fullsubnet.py      # Fast FullSubNet (고급)
│   └── preprocessing.py   # 신호 처리 전처리 필터
├── training/               # 학습 관련
│   ├── train.py           # 메인 학습 스크립트
│   ├── losses.py          # 손실 함수
│   └── config.py          # 학습 설정
├── evaluation/             # 평가
│   ├── metrics.py         # PESQ, STOI, SI-SDR
│   └── evaluate.py        # 평가 스크립트
├── inference/              # 추론
│   └── denoise.py         # 실시간 잡음 제거
├── checkpoints/            # 학습된 모델 저장
└── logs/                   # 학습 로그 및 텐서보드
```

## 설치 방법

```bash
# 1. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 합성
```bash
python data/synthesizer.py --clean_dir data/clean --output_dir data/noisy
```

### 2. 모델 학습
```bash
python training/train.py --config training/config.yaml
```

### 3. 추론 (잡음 제거)
```bash
python inference/denoise.py --input noisy_audio.wav --output clean_audio.wav --checkpoint checkpoints/best_model.pth
```

### 4. 성능 평가
```bash
python evaluation/evaluate.py --test_dir data/test --checkpoint checkpoints/best_model.pth
```

## 평가 지표

- **SI-SDR** (Scale-Invariant SDR): 신호 분리 성능
- **PESQ** (Perceptual Evaluation of Speech Quality): 음질 (1.0~4.5)
- **STOI** (Short-Time Objective Intelligibility): 명료도 (0~1)
- **DNSMOS P.835**: 주관적 품질 (SIG, BAK, OVRL)

## 개발 로드맵

- [x] Phase 1: 프로젝트 구조 및 기본 환경
- [ ] Phase 2: 데이터 합성 파이프라인
- [ ] Phase 3: 베이스라인 모델 구현
- [ ] Phase 4: 학습 및 평가
- [ ] Phase 5: 성능 최적화

## 참고 문헌

- Microsoft DNS Challenge
- Fast FullSubNet (ICASSP 2022)
- Flow Matching for Speech Enhancement

