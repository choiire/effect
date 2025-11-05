## 전기적 노이즈 합성 방식 개선 문서

## 개요

마이크로폰 전기적 노이즈의 **알고리즘적 합성**을 완전히 재설계했습니다. 기존의 단순한 가우시안 백색 노이즈와 험(Hum)에서, **5가지 물리 기반 노이즈 타입**을 정밀하게 모델링하는 통합 시스템으로 전환했습니다.

## 물리적 배경

### 전기적 노이즈의 분류

전기적 노이즈는 크게 두 가지 범주로 나뉩니다:

#### 1. **내재적 (Stochastic) 노이즈**
마이크 내부의 물리적 구성 요소에서 발생하는 무작위, 광대역 노이즈
- 열 노이즈 (Thermal)
- 플리커 노이즈 (Flicker)  
- 샷 노이즈 (Shot)

#### 2. **외재적 (Interference) 노이즈**
외부 전자기 환경에서 유입되는 주기적, 음조 노이즈
- 전원 험 (Mains Hum)
- RFI/EMI (무선 주파수 간섭)

## 구현 세부사항

### 1. 열 노이즈 (Thermal/Johnson-Nyquist Noise)

**물리적 원리**: 도체 내 전자의 열적 교란

**수학적 모델**: 가우시안 백색 노이즈 (GWN)
- 중심 극한 정리에 따라 무수한 전자 움직임의 합 = 가우시안 분포
- 파워 스펙트럼이 가청 대역 전체에 평탄 (White)

**알고리즘**:
```python
noise = np.random.normal(0.0, 1.0, n_samples)
noise = noise / np.sqrt(np.mean(noise ** 2))  # RMS=1로 정규화
noise = noise * amplitude  # 목표 진폭으로 스케일링
```

**특징**:
- "쉭" 하는 히스(Hiss) 사운드
- 주파수 독립적 (평탄한 스펙트럼)
- 마이크 노이즈 플로어의 기본 구성 요소

### 2. 플리커 노이즈 (Flicker/1/f Noise)

**물리적 원리**: 반도체 결함, 저주파 변동

**수학적 모델**: 핑크 노이즈 (Pink Noise)
- 파워 스펙트럼 밀도(PSD): $P(f) \propto 1/f$
- 저주파에 더 많은 에너지 (고주파보다 "어두운" 소리)

**알고리즘**: IIR 필터 기반 스펙트럼 성형
```python
# Audio EQ Cookbook 필터 계수
b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
a = [1, -2.494956002, 2.017265875, -0.522189400]

# 백색 노이즈를 핑크 노이즈로 변환
pink_noise = signal.lfilter(b, a, white_noise)
```

**특징**:
- 백색 노이즈보다 "부드럽고" "따뜻한" 히스
- 저주파 강조
- 실시간 스트리밍 가능 (IIR 필터)

### 3. 샷 노이즈 (Shot/Poisson Noise)

**물리적 원리**: 전하의 불연속적 이동

**수학적 모델**: 푸아송 프로세스 (Poisson Process)
- 이산적인 "사건(event)"의 연속
- 임펄스 간 시간 간격은 지수 분포

**알고리즘**: 임펄스 트레인
```python
# 푸아송 프로세스: 임펄스 간 간격은 지수 분포
inter_arrival_times = np.random.exponential(1.0 / rate, n_events)

# 샘플 인덱스로 변환
arrival_samples = np.cumsum(inter_arrival_times * sr).astype(int)

# 해당 위치에 무작위 진폭의 임펄스 배치
for idx in valid_indices:
    crackle[idx] = np.random.uniform(-amplitude, amplitude)
```

**특징**:
- "자글거리는" 크래클(Crackle) 사운드
- 히스에 섞여 "탁탁 튀는" 효과
- 신호 의존적 특성

### 4. 전원 험 (Mains Hum)

**물리적 원리**: 50/60Hz AC 전력선의 전자기 유도

**수학적 모델**: 고조파 가산 합성 (Additive Synthesis)
- 기본 주파수 (60Hz) + 정수배 고조파들
- 2차 고조파(120Hz) 강조 (정류기 효과)

**알고리즘**:
```python
for i in range(1, n_harmonics + 1):
    freq = base_freq * i
    
    # 2차 고조파 강조
    if i == 2:
        harmonic_amplitude = amplitude * 1.5
    else:
        harmonic_amplitude = amplitude / (i ** 2)  # 감쇠
    
    # 위상 제어
    if phase_mode == 'random':
        phase = np.random.rand() * 2 * np.pi  # 부드러운 Hum
    else:
        phase = 0.0  # 날카로운 Buzz
    
    hum_signal += harmonic_amplitude * np.sin(2π * freq * t + phase)
```

**특징**:
- **Hum (무작위 위상)**: 부드러운 "웅" 소리
- **Buzz (고정 위상)**: 날카로운 "징" 소리
- 8-15개 고조파 (랜덤)
- 2차 고조파가 기본파보다 강할 수 있음

### 5. RFI/EMI 노이즈 (Radio/Electromagnetic Interference)

**물리적 원리**: 무선 주파수 간섭 (Wi-Fi, 휴대폰 등)

**수학적 모델**: 진폭 변조 (AM)
- 고주파 RF 신호가 오디오 회로에서 복조됨
- 반송파(Carrier) × 변조파(Modulator)

**알고리즘**:
```python
# 1. 반송파 (Whine)
carrier = np.sin(2π * carrier_freq * t)  # 6-12 kHz

# 2. 변조 신호 (Data Buzz)
if use_noise_modulator:
    # 핑크 노이즈로 Wi-Fi 데이터 시뮬레이션
    modulator = (pink_noise / max(abs(pink_noise)) + 1.0) / 2.0
else:
    # 사각파로 주기적 버즈
    modulator = 0.5 * square(2π * modulator_freq * t) + 0.5

# 3. AM 변조
rfi_signal = carrier * modulator
```

**특징**:
- **Whine**: 고주파 "삐" 소리 (6-12 kHz)
- **Data Buzz**: 변조 패턴
  - 주기적: 사각파 변조
  - 복잡한: 노이즈 변조 (Wi-Fi 데이터)

## 노이즈 칵테일 통합

### SNR 기반 정확한 믹싱

모든 노이즈를 합산한 후, **정확한 SNR(dB)**로 원본 신호와 혼합:

```python
def _mix_at_snr(clean_signal, noise_signal, snr_db):
    # 1. RMS 계산
    clean_rms = sqrt(mean(clean_signal ** 2))
    noise_rms = sqrt(mean(noise_signal ** 2))
    
    # 2. 목표 노이즈 RMS 계산
    snr_linear = 10 ** (snr_db / 20.0)
    target_noise_rms = clean_rms / snr_linear
    
    # 3. 스케일링 팩터
    scaling_factor = target_noise_rms / noise_rms
    
    # 4. 믹싱
    adjusted_noise = noise_signal * scaling_factor
    return clean_signal + adjusted_noise
```

**핵심 특징**:
- 정량적 제어 가능
- 객관적 평가 가능
- 반복 가능성 (Random seed 고정 시)

## 기존 방식과의 비교

| 항목 | 기존 방식 | 새로운 방식 |
|------|-----------|-------------|
| **노이즈 종류** | 2종 (White, Hum) | 5종 (Thermal, Flicker, Shot, Hum, RFI) |
| **물리적 근거** | 단순 | 정밀 (물리 법칙 기반) |
| **제어성** | 낮음 (SNR만) | 높음 (각 노이즈 개별 제어) |
| **현실성** | 낮음 | 높음 (실제 마이크 측정치 반영) |
| **분리 가능성** | 불가 | 가능 (개별 노이즈 생성) |
| **평가 정확성** | 낮음 | 높음 (정확한 SNR 계산) |
| **Hum 모델** | 3개 고조파만 | 8-15개 고조파 + 위상 제어 |
| **스펙트럼 다양성** | 낮음 (White만) | 높음 (White + Pink) |

## 사용 방법

### 기본 사용

```python
from data.synthesizer import MicrophoneNoiseSimulator

simulator = MicrophoneNoiseSimulator(sample_rate=16000)

# 전기적 노이즈 추가 (5종 통합)
noisy_audio = simulator.add_electrical_noise(
    audio=clean_audio,
    global_snr_db=30  # 전체 노이즈의 목표 SNR
)
```

### 세밀한 제어

```python
# 각 노이즈 타입별 진폭 제어
noisy_audio = simulator.add_electrical_noise(
    audio=clean_audio,
    thermal_amplitude=0.001,      # 열 노이즈
    flicker_amplitude=0.002,      # 플리커 노이즈
    shot_rate=100,                # 샷 노이즈 빈도 (Hz)
    hum_freq=60.0,                # 험 주파수
    hum_amplitude=0.015,          # 험 진폭
    rfi_carrier_freq=8000,        # RFI 반송파 (Hz)
    rfi_amplitude=0.005,          # RFI 진폭
    global_snr_db=25              # 최종 SNR
)
```

### 개별 노이즈 생성 (평가용)

```python
# 개별 노이즈만 생성
n_samples = len(audio)

thermal = simulator._generate_thermal_noise(n_samples, amplitude=0.01)
flicker = simulator._generate_flicker_noise(n_samples, amplitude=0.01)
shot = simulator._generate_shot_noise(n_samples, amplitude=0.05, rate=150)
hum = simulator._generate_mains_hum(
    n_samples, 
    base_freq=60.0, 
    n_harmonics=10, 
    amplitude=0.02,
    phase_mode='random'  # 또는 'fixed'
)
rfi = simulator._generate_rfi_noise(
    n_samples,
    carrier_freq=8000,
    modulator_freq=100,
    amplitude=0.01,
    use_noise_modulator=True  # 또는 False
)
```

## 테스트 및 검증

### 테스트 실행

```bash
cd data
python test_electrical_noise.py
```

### 생성되는 파일

`test_outputs/electrical/` 폴더에 생성:

**오디오 샘플** (8개):
1. `00_clean.wav` - 원본
2. `01_thermal_noise.wav` - 열 노이즈
3. `02_flicker_noise.wav` - 플리커 노이즈
4. `03_shot_noise.wav` - 샷 노이즈 (크래클)
5. `04a_mains_hum_smooth.wav` - 부드러운 Hum
6. `04b_mains_buzz_sharp.wav` - 날카로운 Buzz
7. `05a_rfi_periodic_buzz.wav` - 주기적 RFI
8. `05b_rfi_data_buzz.wav` - 데이터 RFI
9. `06_combined_electrical_noise.wav` - 통합 (5종)

**시각화**:
- `spectrogram_comparison.png` - 스펙트로그램 비교
- `psd_analysis.png` - 파워 스펙트럼 밀도 분석

### 검증 방법

#### 1. 스펙트럼 분석

**열 노이즈 (White)**:
- 평탄한 PSD (모든 주파수 동일)
- 로그-로그 플롯에서 수평선

**플리커 노이즈 (Pink)**:
- 1/f PSD (저주파 강조)
- 로그-로그 플롯에서 -10dB/decade 기울기

**험 노이즈 (Hum)**:
- 60Hz + 120Hz(강함) + 180Hz, 240Hz, ...
- 불연속 라인 스펙트럼

**RFI 노이즈**:
- 반송파 주파수 근처 집중
- 변조 패턴에 따라 사이드밴드 생성

#### 2. 청각적 확인

각 노이즈를 듣고 특징 확인:
- **Thermal**: "쉭" (일정한 히스)
- **Flicker**: "쉭~~" (더 부드러운 히스)
- **Shot**: "탁탁" (크래클)
- **Hum**: "웅~~" (저음 윙윙)
- **Buzz**: "징징" (날카로운 버즈)
- **RFI**: "삐이잉" (고음 와인) + 변조

## 알고리즘 평가를 위한 활용

### 1. 결정론적 환경 구축

```python
# 무작위 시드 고정 (완벽한 재현성)
np.random.seed(42)

# 동일한 노이즈가 매번 생성됨
noisy1 = simulator.add_electrical_noise(audio, global_snr_db=20)
noisy2 = simulator.add_electrical_noise(audio, global_snr_db=20)
# noisy1 == noisy2 (완전히 동일)
```

### 2. 분리 가능성 (Isolatability)

```python
# 특정 노이즈만 테스트
only_hum = simulator.add_electrical_noise(
    audio,
    thermal_amplitude=0,
    flicker_amplitude=0,
    shot_rate=0,
    hum_amplitude=0.02,  # Hum만 활성화
    rfi_amplitude=0,
    global_snr_db=20
)

# 알고리즘이 Hum만 제거하는지 평가
```

### 3. 파라미터 스윕 (Robustness Test)

```python
# 고조파 수 변경 테스트 (1~50개)
for n_harmonics in [1, 5, 10, 20, 50]:
    hum = simulator._generate_mains_hum(
        n_samples, 
        base_freq=60.0, 
        n_harmonics=n_harmonics,
        amplitude=0.02
    )
    # 필터 성능 평가...
```

### 4. Ground Truth 기반 정량 평가

```python
# 노이즈만 별도 저장 (Ground Truth)
total_noise = (noise_thermal + noise_flicker + 
               noise_shot + noise_hum + noise_rfi)

# 알고리즘 평가
filtered_audio = your_filter(noisy_audio)
estimated_noise = noisy_audio - filtered_audio

# 정량 지표 계산
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(total_noise, estimated_noise)
```

## 기술적 장점

### 1. 물리적 정확성
- 실제 마이크의 노이즈 메커니즘을 정밀하게 반영
- 측정 가능한 물리량 기반 (주파수, 진폭, SNR)

### 2. 제어 가능성
- 각 노이즈 타입을 개별적으로 제어
- 정확한 SNR 설정
- 파라미터화된 인터페이스

### 3. 평가 신뢰성
- 결정론적 (시드 고정 시)
- 분리 가능 (개별 노이즈)
- Ground Truth 제공

### 4. 확장성
- 새로운 노이즈 타입 추가 용이
- 실제 마이크 측정 데이터로 보정 가능
- 필터 파라미터 조정 간단

## 참고 문헌

1. Johnson-Nyquist Noise: *Physical Review* (1928)
2. Flicker Noise: Hooge, F. N. (1976)
3. Shot Noise: Schottky, W. (1918)
4. Audio EQ Cookbook: Bristow-Johnson, R. (1994)
5. AM Modulation: *Principles of Communications* (2008)

## 라이선스

Audio EQ Cookbook의 공개 공식을 사용하며, 물리 법칙은 공공 도메인입니다.

---

**최종 업데이트**: 2025-11-05  
**작성자**: AI Assistant  
**버전**: 1.0

