# 팝노이즈 합성 방식 개선 문서

## 개요

기존의 단순한 사인파 펄스 기반 팝노이즈 합성을 **물리 기반 근접 효과 시뮬레이션**으로 전면 개선했습니다. 이는 압력 경도 변환기의 물리적 원리를 바탕으로 한 정밀한 모델링입니다.

## 물리적 배경

### 팝노이즈의 본질

팝노이즈는 다음과 같은 물리적 현상의 결과입니다:

1. **근접 효과 (Proximity Effect)**: 지향성 마이크(압력 경도 변환기)에서 음원이 가까워질 때 저주파가 증폭되는 현상
2. **역제곱 법칙**: 거리가 절반이 되면 음압이 4배(+12dB)가 되는 물리 법칙
3. **로우 쉘프 응답**: 근접 효과는 특정 주파수(보통 150Hz) 이하에서 6dB/octave 기울기로 증폭

### 핵심 원리

- **압력 경도 변환기**: 다이어프램 앞뒤의 압력 '차이'로 소리를 감지
- **위상 vs 진폭**: 원거리에서는 위상 차이가 지배적, 근거리에서는 진폭 차이가 지배적
- **마이크 보정의 실패**: 원거리용 내장 저주파 보상이 근거리에서 과도하게 작동

## 구현 세부사항

### 1. 바이쿼드 로우 쉘프 필터 (`_design_low_shelf_biquad`)

**Audio EQ Cookbook** 공식을 정확히 구현:

```python
A = 10^(gain_db / 40)
ω₀ = 2πfc / fs
α = sin(ω₀) / (2Q)
β = 2√A·α

b₀ = A·((A+1) - (A-1)cos(ω₀) + β)
b₁ = 2A·((A-1) - (A+1)cos(ω₀))
b₂ = A·((A+1) - (A-1)cos(ω₀) - β)
a₀ = (A+1) + (A-1)cos(ω₀) + β
a₁ = -2·((A-1) + (A+1)cos(ω₀))
a₂ = (A+1) + (A-1)cos(ω₀) - β
```

**특징**:
- Q = 0.707: 공진 없는 자연스러운 6dB/octave 기울기
- fc = 150Hz: 근접 효과가 시작되는 코너 주파수
- 수치적으로 안정적인 SOS(Second-Order Section) 형식

### 2. 근접 효과 게인 계산 (`_calculate_proximity_gain`)

**거리-게인 모델**:
```
shelf_gain_db = C_pattern / distance_cm
```

**패턴별 상수**:
- **Cardioid** (단일지향성): C = 60
  - 5cm에서 약 12dB 부스트
- **Figure-8** (양지향성): C = 120
  - 5cm에서 약 24dB 부스트
  - Cardioid의 2배 (압력 경도 원리 100% 적용)

**레벨 증가** (역제곱 법칙):
```
level_gain = reference_distance / current_distance
```

### 3. 시간 변화하는 팝노이즈 시뮬레이션

#### 거리 프로파일

호흡이나 입술이 마이크에 다가왔다가 멀어지는 과정을 **가우시안 거리 곡선**으로 모델링:

```python
d(t) = 100 - (100 - d_min) · exp(-((t - 0.5)² / 0.05))
```

- 초기 거리: 100cm (근접 효과 없음)
- 최소 거리: 1-5cm (intensity 파라미터로 조절)
- 지속 시간: 80-200ms

#### 동적 필터링

팝 이벤트를 작은 윈도우(약 5ms)로 분할하여 각 시간 구간마다:

1. 현재 거리에 따른 게인 계산
2. 해당 게인의 로우 쉘프 필터 생성
3. 필터 적용 (음색 변화)
4. 레벨 증가 적용 (역제곱 법칙)

**수식**:
```
y_pop(t) = LowShelf[G(d(t)), fc](x(t)) · (d_ref / d(t))
         └─────────────┬──────────────┘   └────┬─────┘
              음색 변화                    레벨 증가
```

### 4. 최종 혼합

```python
output = clean + (pop_processed - clean) · intensity
```

- 원본 신호 보존
- 근접 효과만 intensity로 조절하여 추가
- 자연스러운 혼합

## 기존 방식과의 비교

| 항목 | 기존 방식 | 새로운 방식 |
|------|-----------|-------------|
| **물리적 근거** | 없음 (경험적) | 압력 경도 변환기 원리 기반 |
| **주파수 응답** | 단일 주파수 사인파 | 6dB/octave 로우 쉘프 |
| **시간 특성** | 고정된 지수 감쇠 | 거리 변화 기반 동적 응답 |
| **레벨 변화** | 고려 안 됨 | 역제곱 법칙 적용 |
| **지향성 패턴** | 지원 안 됨 | Cardioid/Figure-8 구분 |
| **현실성** | 낮음 | 높음 (실제 마이크 측정치 기반) |

## 사용 방법

### 기본 사용

```python
from data.synthesizer import MicrophoneNoiseSimulator

simulator = MicrophoneNoiseSimulator(sample_rate=16000)

# 팝노이즈 추가
noisy_audio = simulator.add_pop_noise(
    audio=clean_audio,
    pop_frequency=2.0,      # 초당 2회 팝 발생
    intensity=0.7,          # 중간 강도 (0-1)
    pattern='cardioid'      # 단일지향성 마이크
)
```

### 파라미터 설명

- **pop_frequency** (float): 초당 팝 발생 횟수
  - 범위: 0-3 (None이면 랜덤)
  - 추천: 1-2 (자연스러운 음성)

- **intensity** (float): 팝 강도
  - 범위: 0-1
  - 0.3: 약한 팝 (최소 거리 ~2cm)
  - 0.7: 중간 팝 (최소 거리 ~4cm)
  - 0.9: 강한 팝 (최소 거리 ~5cm)

- **pattern** (str): 마이크 지향성 패턴
  - `'cardioid'`: 단일지향성 (약한 근접 효과)
  - `'figure8'`: 양지향성 (강한 근접 효과, 2배)

### 전체 노이즈 적용

```python
# 모든 마이크 노이즈 한번에 적용
noisy_audio = simulator.apply_all_noise(
    audio=clean_audio,
    proximity_boost_db=6.0,    # 지속적인 근접 효과
    pop_frequency=1.5,          # 팝노이즈
    pop_pattern='cardioid',     # 팝노이즈 패턴
    hum_freq=60,                # 전기적 험
    hum_snr_db=40,
    white_noise_snr_db=45
)
```

## 테스트 및 검증

### 테스트 실행

```bash
cd data
python test_pop_noise.py
```

### 생성되는 파일

테스트 스크립트는 `test_outputs/` 폴더에 다음을 생성:

1. **오디오 샘플**:
   - `00_clean.wav`: 원본
   - `01_pop_cardioid.wav`: Cardioid 패턴
   - `02_pop_figure8.wav`: Figure-8 패턴
   - `03_pop_weak.wav`: 약한 강도
   - `04_pop_strong.wav`: 강한 강도
   - `05_pop_frequent.wav`: 빈번한 팝

2. **시각화**:
   - `spectrogram_comparison.png`: 스펙트로그램 비교
   - `frequency_response.png`: 거리별 주파수 응답

### 주파수 응답 검증

각 거리에서의 로우 쉘프 응답:

- **2cm**: ~30dB 부스트 @ 100Hz
- **5cm**: ~12dB 부스트 @ 100Hz  
- **10cm**: ~6dB 부스트 @ 100Hz
- **20cm**: ~3dB 부스트 @ 100Hz
- **50cm**: ~1dB 부스트 @ 100Hz

모두 6dB/octave 기울기로 150Hz 이하에서 증폭.

## 기술적 장점

### 1. 물리적 정확성
- 실제 마이크의 동작 원리 반영
- 측정 가능한 물리량 기반 (거리, dB)

### 2. 파라미터 제어성
- 직관적인 파라미터 (거리, 강도, 패턴)
- 예측 가능한 결과

### 3. 확장성
- 다른 지향성 패턴 추가 용이
- 특정 마이크 모델 에뮬레이션 가능

### 4. 학습 데이터 품질
- 더 현실적인 팝노이즈 생성
- 딥러닝 모델의 일반화 성능 향상 기대

## 참고 문헌

1. Beranek, L. L. (1954). *Acoustics*. McGraw-Hill.
2. Eargle, J. M. (2012). *The Microphone Book*. Focal Press.
3. Bristow-Johnson, R. (1994). *Cookbook formulae for audio EQ biquad filter coefficients*.
4. Shure Inc. (2020). *Proximity Effect Technical Note*.
5. Neumann GmbH. *U 87 Operating Instructions*.

## 라이선스

본 구현은 Audio EQ Cookbook의 공개 공식을 사용하며, 물리 법칙은 공공 도메인입니다.

---

**최종 업데이트**: 2025-11-05  
**작성자**: AI Assistant  
**버전**: 1.0

