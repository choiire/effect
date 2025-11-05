# 빠른 시작 가이드 (Quick Start)

23,000개의 학습 데이터로 음성 향상 모델을 학습하는 전체 과정입니다.

## 📋 전제 조건

1. Python 3.8 이상 설치
2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. 데이터 배치:
   ```
   data/train/clean/    ← 23,000개 음성 파일 (.wav)
   data/val/clean/      ← 검증용 음성 파일
   data/test/clean/     ← 평가 전용 음성 파일
   ```

## 🚀 3단계로 시작하기

### 1단계: 학습 데이터 준비 (모든 23,000개 파일)

```bash
python prepare_training_data.py
```

**실행 결과:**
- ✅ `data/train/noisy/*.wav` (23,000개 생성)
- ✅ `data/train/clean/*.wav` (23,000개)
- ⏱️ 예상 시간: 20-40분

### 2단계: 검증 데이터 준비

```bash
python prepare_validation_data.py
```

**실행 결과:**
- ✅ `data/val/noisy/*.wav` 생성
- ✅ `data/val/clean/*.wav`

### 3단계: 학습 시작

```bash
python training/train.py
```

**학습 설정:**
- 학습 데이터: 23,000개 모두 사용 ✅
- 검증 데이터: val 폴더 사용
- Test 데이터: 학습에 사용 안 함 ✅

**모니터링:**
```bash
tensorboard --logdir runs/
```

브라우저에서 http://localhost:6006 접속

## 📊 진행 상황 확인

### 데이터 준비 확인

```bash
# Windows PowerShell
(Get-ChildItem "data\train\clean\*.wav").Count   # 23000 확인
(Get-ChildItem "data\train\noisy\*.wav").Count   # 23000 확인

# Linux/Mac
ls data/train/clean/*.wav | wc -l   # 23000 확인
ls data/train/noisy/*.wav | wc -l   # 23000 확인
```

### 학습 상태 확인

```bash
python check_training.py
```

## 🧪 (선택) 테스트 데이터 준비

⚠️ **주의**: 이 데이터는 학습에 사용되지 않습니다!

```bash
python prepare_test_data.py
```

## 📈 학습 후 평가

### 모델 평가

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --test_noisy_dir data/test/noisy \
  --test_clean_dir data/test/clean \
  --output_dir evaluation/results
```

### 결과 시각화

```bash
# 비교 플롯 생성
python evaluation/plot_results.py --results_dir evaluation/results

# 통계 요약
python evaluation/summarize_results.py --results_file evaluation/results/metrics.json

# 비교 테이블
python evaluation/show_comparison_table.py --results_file evaluation/results/metrics.json
```

## 🎯 데이터 사용 원칙 요약

| 폴더 | 파일 수 | 학습 사용 | 용도 |
|------|---------|-----------|------|
| **data/train/** | 23,000 | ✅ Yes | 모델 학습 |
| **data/val/** | ~500-1000 | ⚠️ 검증만 | 손실 계산, 조기 중단 |
| **data/test/** | 사용자 지정 | ❌ No | 최종 평가 전용 |

## 🔧 고급 설정

### 학습 파라미터 조정

`training/config.py` 파일 수정 또는 `config.yaml` 생성:

```yaml
data:
  train_noisy_dir: "data/train/noisy"
  train_clean_dir: "data/train/clean"
  val_noisy_dir: "data/val/noisy"
  val_clean_dir: "data/val/clean"
  batch_size: 16
  segment_length: 64000  # 4초 @ 16kHz

model:
  in_channels: 1
  hidden_channels: 64
  num_layers: 12
  kernel_size: 3

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
```

### 맞춤형 노이즈 설정

노이즈 파라미터를 직접 제어하려면 `data/synthesizer.py` 수정:

```python
noisy_audio = simulator.apply_all_noise(
    audio,
    proximity_boost_db=6.0,      # 근접 효과 강도
    pop_frequency=1.5,            # 팝 발생 빈도
    pop_pattern='cardioid',       # 마이크 패턴
    hum_snr_db=35,                # 험 노이즈 레벨
    white_noise_snr_db=40         # 화이트 노이즈 레벨
)
```

## 🐛 문제 해결

### "잡음 음성 파일을 찾을 수 없습니다"

➡️ 데이터 준비 스크립트를 다시 실행하세요:
```bash
python prepare_training_data.py
```

### 학습이 너무 느림

➡️ 배치 크기 줄이기:
```python
# training/config.py에서
batch_size: 8  # 기본 16에서 줄임
```

### GPU 메모리 부족

➡️ 세그먼트 길이 줄이기:
```python
# training/config.py에서
segment_length: 32000  # 2초 @ 16kHz
```

### 검증 손실이 개선되지 않음

➡️ 학습률 조정:
```python
learning_rate: 0.0005  # 기본 0.001에서 줄임
```

## 📚 다음 단계

1. **TensorBoard 모니터링**: 학습 곡선 관찰
2. **체크포인트 저장**: 최고 성능 모델 자동 저장
3. **테스트 평가**: 학습 후 test 데이터로 최종 평가
4. **결과 분석**: 시각화 및 통계 분석

## 📖 상세 문서

- **데이터 준비**: `DATA_PREPARATION_GUIDE.md`
- **팝노이즈 합성**: `data/POP_NOISE_UPDATE.md`
- **학습 가이드**: `TRAINING_GUIDE.md`
- **평가 방법**: `EVALUATION_GUIDE.md`

## ✅ 체크리스트

학습 시작 전 확인:

- [ ] data/train/clean에 23,000개 파일이 있는가?
- [ ] prepare_training_data.py 실행 완료?
- [ ] prepare_validation_data.py 실행 완료?
- [ ] data/train/noisy에 23,000개 파일이 생성되었는가?
- [ ] Test 데이터가 train/val과 분리되어 있는가?
- [ ] requirements.txt의 모든 패키지가 설치되었는가?
- [ ] GPU가 사용 가능한가? (선택, CUDA 확인)

모두 체크되었다면:

```bash
python training/train.py
```

행운을 빕니다! 🚀

---

**작성일**: 2025-11-05  
**질문**: 문제가 있으면 이슈를 생성하세요

