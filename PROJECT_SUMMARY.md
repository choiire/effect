# 프로젝트 요약 📊

## 마이크 잡음 제거 시스템 개발 완료

딥러닝 기반 음성 향상(Speech Enhancement) 시스템으로, 마이크 특화 잡음을 제거합니다.

---

## 🎯 개발 목표

다음 3가지 마이크 특화 잡음을 제거하는 알고리즘 개발:

1. **근접효과 (Proximity Effect)**: 80-250Hz 저주파 과도 증폭
2. **팝노이즈 (Pop Noise)**: 순간적 고에너지 저주파 펄스  
3. **전기적 잡음 (Electrical Noise)**: 험(50/60Hz) + 화이트 노이즈

---

## 📦 구현된 주요 컴포넌트

### 1. 데이터 합성 파이프라인
**파일**: `data/synthesizer.py`

- ✅ 근접효과 시뮬레이터 (0-12dB 부스트)
- ✅ 팝노이즈 생성기 (랜덤 저주파 펄스)
- ✅ 전기 잡음 추가 (험 + 화이트 노이즈)
- ✅ DNS Challenge 스타일 SNR 분포 (0-40dB)

### 2. 신호 처리 전처리 필터
**파일**: `models/preprocessing.py`

- ✅ 근접효과 보정 필터 (High-pass filter, 80Hz cutoff)
- ✅ 팝노이즈 감지 및 억제 (에너지 임계값 기반)
- ✅ 전기 잡음 제거 (Notch filter, 60Hz + 고조파)

### 3. 딥러닝 모델
**파일**: `models/unet.py`

**SpectrogramUNet (T-F Domain)**
- STFT 기반 스펙트로그램 처리
- U-Net 아키텍처 (인코더-디코더)
- 마스크 예측 또는 직접 스펙트로그램 복원
- 실시간 처리 가능 (저지연)

**WaveformUNet (Time Domain)**
- End-to-End 파형 처리
- 1D Convolution 기반
- 위상 보존 우수
- 고품질 복원

### 4. 손실 함수
**파일**: `training/losses.py`

보고서 권장 복합 손실 함수:
```
Loss = α·SI-SDR + β·STFT + γ·Time-domain
```

- ✅ SI-SDR Loss: 신호 분리 성능
- ✅ Multi-resolution STFT Loss: 주파수 영역 충실도
- ✅ Time-domain Loss: 시간 영역 정확도

### 5. 평가 지표
**파일**: `evaluation/metrics.py`

- ✅ **SI-SDR**: Scale-Invariant SDR (신호 대 왜곡 비율)
- ✅ **PESQ**: 지각적 음질 평가 (1.0~4.5)
- ✅ **STOI**: 음성 명료도 (0~1)
- ✅ **SNR**: Signal-to-Noise Ratio

### 6. 학습 파이프라인
**파일**: `training/train.py`, `training/config.py`

- ✅ 자동 체크포인트 저장
- ✅ TensorBoard 통합
- ✅ Mixed Precision Training (AMP)
- ✅ Learning Rate Scheduler
- ✅ Early Stopping
- ✅ 설정 파일 기반 (YAML)

### 7. 추론 시스템
**파일**: `inference/denoise.py`

- ✅ 단일 파일 처리
- ✅ 디렉토리 일괄 처리
- ✅ 긴 오디오 청크 처리 (메모리 효율적)
- ✅ 크로스페이드 블렌딩

---

## 📁 프로젝트 구조

```
effect/
├── data/
│   ├── synthesizer.py      # 마이크 잡음 합성기 ⭐
│   ├── dataset.py          # PyTorch Dataset
│   └── clean/              # 깨끗한 음성 데이터
├── models/
│   ├── unet.py             # U-Net 모델 (2종) ⭐
│   └── preprocessing.py    # 신호 처리 필터 ⭐
├── training/
│   ├── train.py            # 메인 학습 스크립트 ⭐
│   ├── losses.py           # 복합 손실 함수 ⭐
│   └── config.py           # 설정 관리
├── evaluation/
│   ├── metrics.py          # 평가 지표 ⭐
│   └── evaluate.py         # 평가 스크립트 ⭐
├── inference/
│   └── denoise.py          # 추론 스크립트 ⭐
├── checkpoints/            # 학습된 모델
├── logs/                   # TensorBoard 로그
├── requirements.txt        # 의존성
├── README.md               # 프로젝트 소개
├── USAGE_GUIDE.md          # 상세 사용 가이드 📖
└── demo_test.py            # 모듈 테스트
```

---

## 🚀 빠른 시작

### 1. 설치

```bash
# Windows
quick_start.bat

# Linux/Mac
bash quick_start.sh
```

또는 수동 설치:

```bash
pip install -r requirements.txt
python demo_test.py  # 모듈 테스트
```

### 2. 데이터 합성

```bash
python data/synthesizer.py \
  --clean_dir data/clean \
  --output_dir data/train \
  --num_samples 1000
```

### 3. 모델 학습

```bash
python training/train.py --config config.yaml
```

### 4. 평가

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --test_noisy_dir data/test/noisy \
  --test_clean_dir data/test/clean
```

### 5. 실전 사용

```bash
python inference/denoise.py \
  --input noisy_audio.wav \
  --output clean_audio.wav \
  --checkpoint checkpoints/best_model.pth
```

---

## 📊 기대 성능

보고서 기반 목표 지표:

| 지표 | 목표 | 의미 |
|------|------|------|
| **SI-SDR** | > 10 dB | 잡음 대비 신호 개선 |
| **PESQ** | > 3.5 | 음질 (주관적) |
| **STOI** | > 0.9 | 명료도 |

**실제 성능은 데이터 품질과 학습 규모에 따라 달라집니다.*

---

## 🎓 기술적 특징

### 보고서 기반 설계

1. **생성 모델 vs 판별 모델 선택**
   - 실시간: SpectrogramUNet (T-F Domain)
   - 고품질: WaveformUNet (Time Domain, End-to-End)

2. **복합 손실 함수**
   - SI-SDR (신호 분리)
   - Multi-resolution STFT (주파수 충실도)
   - Time-domain (시간 정확도)

3. **표준화된 평가**
   - DNS Challenge 방식 SNR 분포
   - ITU-T P.862 PESQ
   - STOI, SI-SDR

4. **마이크 특화 설계**
   - 근접효과, 팝노이즈, 전기 잡음 전용 시뮬레이터
   - 신호 처리 기반 전처리 + 딥러닝 결합

### 확장 가능성

- ✅ 다양한 모델 아키텍처 교체 가능
- ✅ 커스텀 잡음 유형 추가 용이
- ✅ 설정 파일 기반 하이퍼파라미터 관리
- ✅ TensorBoard 실시간 모니터링

---

## 📈 다음 단계 (사용자 작업)

### Phase 1: 데이터 준비 ⏳
1. 깨끗한 음성 데이터 확보
   - LibriSpeech (영어)
   - AI-Hub (한국어)
   - 자체 녹음
2. `data/clean/`에 배치
3. Train/Val/Test 분할

### Phase 2: 학습 ⏳
1. 데이터 합성 실행
2. 설정 파일 조정 (config.yaml)
3. 학습 시작 및 모니터링
4. 최고 성능 모델 선택

### Phase 3: 평가 및 최적화 ⏳
1. 테스트 데이터로 평가
2. PESQ, STOI, SI-SDR 확인
3. 하이퍼파라미터 튜닝
4. 실제 환경 테스트

### Phase 4: 배포
1. 모델 최적화 (양자화, 프루닝)
2. 실시간 추론 파이프라인 구축
3. API 서버 구축 (선택)

---

## 📚 참고 자료

### 구현 기반
- Microsoft DNS Challenge
- Fast FullSubNet (ICASSP 2022)
- U-Net for Audio (다양한 논문)
- ITU-T P.862 (PESQ)

### 제공된 보고서
딥러닝 기반 음성 향상(SE) 필터 구현 및 성능 검증을 위한 종합 기술 보고서의 권장사항을 충실히 따름:
- 생성 모델 vs 판별 모델 비교
- DNS Challenge 데이터 합성 전략
- SI-SDR, PESQ, STOI, DNSMOS 평가 체계
- 복합 손실 함수 설계

---

## ✅ 완료 항목

- [x] 프로젝트 구조 설정
- [x] 마이크 잡음 합성 파이프라인
- [x] U-Net 모델 구현 (2종)
- [x] 신호 처리 전처리 필터
- [x] 복합 손실 함수
- [x] 평가 지표 (PESQ, STOI, SI-SDR)
- [x] 학습 파이프라인
- [x] 추론 시스템
- [x] 문서화 (README, USAGE_GUIDE)
- [x] 테스트 스크립트

## ⏳ 남은 작업 (사용자)

- [ ] 깨끗한 음성 데이터 준비
- [ ] 데이터 합성 실행
- [ ] 모델 학습
- [ ] 성능 평가 및 최적화
- [ ] 실제 환경 테스트

---

## 💡 핵심 인사이트

### 보고서로부터 배운 점

1. **품질 vs 속도 트레이드오프**
   - 실시간 필요: T-F Domain (Fast FullSubNet 스타일)
   - 최고 품질: Time Domain (Waveform End-to-End)

2. **평가는 다차원적**
   - SI-SDR: 신호 처리 성능
   - PESQ/STOI: 인간 지각 품질
   - 둘 다 중요!

3. **데이터 다양성이 핵심**
   - 광범위한 SNR 분포 (0-40dB)
   - 다양한 잡음 유형
   - 현실적인 시뮬레이션

4. **전처리 + 딥러닝 결합**
   - 신호 처리로 1차 정제
   - 딥러닝으로 고도화
   - 상호 보완적

---

## 🎉 결론

마이크 근접효과, 팝노이즈, 전기적 잡음을 제거하는 **완전한 End-to-End 시스템**이 구축되었습니다.

**이제 필요한 것:**
1. 데이터 준비
2. 학습 실행
3. 평가 및 튜닝

**모든 도구가 준비되었습니다. 성공적인 학습을 기원합니다! 🚀**

---

*자세한 사용법은 `USAGE_GUIDE.md`를 참조하세요.*

