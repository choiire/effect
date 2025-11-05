# 사용 가이드 📖

마이크 잡음 제거 시스템 사용 방법을 단계별로 안내합니다.

## 목차
1. [설치](#1-설치)
2. [데이터 준비](#2-데이터-준비)
3. [데이터 합성](#3-데이터-합성)
4. [모델 학습](#4-모델-학습)
5. [모델 평가](#5-모델-평가)
6. [실전 사용 (추론)](#6-실전-사용-추론)

---

## 1. 설치

### 1.1 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 1.2 의존성 설치

```bash
pip install -r requirements.txt
```

**필수 라이브러리:**
- PyTorch (GPU 사용 시 CUDA 버전 확인)
- librosa, soundfile (오디오 처리)
- pesq, pystoi (평가 지표)

---

## 2. 데이터 준비

### 2.1 깨끗한 음성 데이터 확보

다음 중 하나를 선택:

**옵션 A: LibriSpeech (영어)**
```bash
# LibriSpeech 다운로드 예시
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
```

**옵션 B: AI-Hub 한국어 데이터**
1. [AI-Hub](https://aihub.or.kr) 접속
2. "소음 환경 음성인식 데이터" 검색
3. 다운로드 승인 후 `*_SD.wav` (깨끗한 음성) 파일 사용

**옵션 C: 자체 데이터**
- 녹음 환경이 조용한 깨끗한 음성 데이터
- `.wav` 또는 `.flac` 형식
- 16kHz 샘플링 레이트 권장

### 2.2 디렉토리 구조

```
data/
├── clean/          # 깨끗한 음성 파일들
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── (합성 후 생성됨)
    ├── train/
    │   ├── noisy/  # 잡음 음성
    │   └── clean/  # 깨끗한 음성
    └── val/
        ├── noisy/
        └── clean/
```

---

## 3. 데이터 합성

### 3.1 기본 합성

깨끗한 음성에 마이크 잡음(근접효과, 팝노이즈, 전기 잡음)을 추가합니다.

```bash
python data/synthesizer.py \
  --clean_dir data/clean \
  --output_dir data/train \
  --num_samples 1000
```

**파라미터:**
- `--clean_dir`: 깨끗한 음성 디렉토리
- `--output_dir`: 출력 디렉토리
- `--num_samples`: 생성할 샘플 수 (생략 시 모든 파일)
- `--sample_rate`: 샘플링 레이트 (기본 16000Hz)

### 3.2 검증 데이터 생성

학습 데이터와 별도로 검증 데이터도 생성:

```bash
python data/synthesizer.py \
  --clean_dir data/clean_val \
  --output_dir data/val \
  --num_samples 200
```

### 3.3 합성 결과 확인

```
data/train/
├── noisy/          # 잡음이 추가된 음성
│   ├── audio1_noisy.wav
│   └── ...
└── clean/          # 정규화된 깨끗한 음성
    ├── audio1_clean.wav
    └── ...
```

---

## 4. 모델 학습

### 4.1 설정 파일 생성

```bash
python training/config.py
```

생성된 `config.yaml`을 편집:

```yaml
data:
  train_noisy_dir: "data/train/noisy"
  train_clean_dir: "data/train/clean"
  val_noisy_dir: "data/val/noisy"
  val_clean_dir: "data/val/clean"
  batch_size: 16
  segment_length: 64000  # 4초

model:
  model_type: "waveform_unet"  # 또는 "spectrogram_unet"
  n_channels: 32
  use_preprocessing: true

training:
  num_epochs: 100
  learning_rate: 0.001
  device: "cuda"
```

### 4.2 학습 시작

```bash
python training/train.py --config config.yaml
```

**주요 옵션:**
- `--config`: 설정 파일 경로
- `--resume`: 체크포인트에서 재개

### 4.3 학습 모니터링

TensorBoard로 실시간 모니터링:

```bash
tensorboard --logdir logs
```

브라우저에서 `http://localhost:6006` 접속

**확인 항목:**
- Train/Val Loss 그래프
- SI-SDR 개선 추이
- Learning Rate 변화

### 4.4 체크포인트

학습 중 저장되는 파일:
- `checkpoints/best_model.pth`: 최고 성능 모델
- `checkpoints/checkpoint_epoch_X.pth`: 주기적 저장

---

## 5. 모델 평가

### 5.1 테스트 데이터셋 평가

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --test_noisy_dir data/test/noisy \
  --test_clean_dir data/test/clean \
  --save_output \
  --output_dir evaluation/outputs
```

**출력:**
```
📊 평가 결과 요약
========================================
🎯 향상된 음성 (Enhanced):
  SI-SDR:  12.45 dB
  PESQ:     3.821
  STOI:     0.945

📉 원본 잡음 신호 (Noisy):
  SI-SDR:   3.21 dB
  PESQ:     2.134
  STOI:     0.756

✨ SI-SDR 개선량: 9.24 dB
```

### 5.2 단일 파일 평가

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --noisy_file test_noisy.wav \
  --clean_file test_clean.wav \
  --output_file test_enhanced.wav
```

### 5.3 평가 지표 해석

| 지표 | 범위 | 목표 | 의미 |
|------|------|------|------|
| **SI-SDR** | -∞ ~ +∞ dB | > 10 dB | 신호 분리 품질 |
| **PESQ** | 1.0 ~ 4.5 | > 3.5 | 지각적 음질 |
| **STOI** | 0 ~ 1 | > 0.9 | 명료도 (알아듣기 쉬움) |

---

## 6. 실전 사용 (추론)

### 6.1 단일 파일 잡음 제거

```bash
python inference/denoise.py \
  --input noisy_recording.wav \
  --output clean_output.wav \
  --checkpoint checkpoints/best_model.pth
```

### 6.2 여러 파일 일괄 처리

```bash
python inference/denoise.py \
  --input recordings/ \
  --output enhanced/ \
  --checkpoint checkpoints/best_model.pth
```

모든 `.wav`, `.flac`, `.mp3` 파일을 자동으로 처리합니다.

### 6.3 긴 오디오 처리

메모리가 부족한 경우 청크 단위로 처리:

```bash
python inference/denoise.py \
  --input long_audio.wav \
  --output enhanced_long.wav \
  --checkpoint checkpoints/best_model.pth \
  --chunk_size 160000  # 10초씩 처리
```

### 6.4 CPU에서 추론

GPU가 없는 환경:

```bash
python inference/denoise.py \
  --input noisy.wav \
  --output clean.wav \
  --checkpoint checkpoints/best_model.pth \
  --device cpu
```

---

## 7. 성능 최적화 팁

### 7.1 더 나은 결과를 위해

1. **더 많은 데이터**: 최소 1,000개 이상의 훈련 샘플
2. **다양한 잡음**: 실제 사용 환경과 유사한 잡음 포함
3. **하이퍼파라미터 튜닝**:
   ```yaml
   loss:
     si_sdr_weight: 1.0
     stft_weight: 0.5  # 조정
     time_weight: 0.1
   ```

### 7.2 학습 시간 단축

- Mixed Precision 활성화: `mixed_precision: true`
- 배치 크기 증가: GPU 메모리 허용 범위 내에서
- 워커 수 증가: `num_workers: 8`

### 7.3 실시간 처리를 위해

- `model_type: "waveform_unet"` 사용
- `n_channels` 값 감소 (32 → 16)
- 전처리 비활성화: `use_preprocessing: false`

---

## 8. 문제 해결

### 8.1 Out of Memory (OOM) 오류

```yaml
# config.yaml
data:
  batch_size: 8  # 줄이기
  segment_length: 32000  # 짧게
```

### 8.2 PESQ 계산 오류

PESQ는 8kHz 또는 16kHz만 지원:
```python
sample_rate: 16000  # 필수
```

### 8.3 학습이 수렴하지 않음

- Learning rate 감소: `0.001 → 0.0001`
- Loss weight 조정
- 데이터 품질 확인

---

## 9. 고급 사용법

### 9.1 자체 잡음 추가

`data/synthesizer.py`의 `MicrophoneNoiseSimulator` 클래스 확장:

```python
def add_custom_noise(self, audio, params):
    # 커스텀 잡음 로직
    return noisy_audio
```

### 9.2 모델 아키텍처 변경

`models/` 디렉토리에 새 모델 추가 후 `config.yaml`에서 선택

### 9.3 전이 학습

사전 학습된 모델에서 시작:

```bash
python training/train.py \
  --config config.yaml \
  --resume checkpoints/pretrained.pth
```

---

## 10. 추가 리소스

- **보고서 참고**: 프로젝트 루트의 기술 보고서
- **코드 문서**: 각 모듈의 docstring 참조
- **예제 노트북**: (향후 추가 예정)

---

## 문의 및 기여

문제가 발생하거나 개선 사항이 있다면 이슈를 등록해주세요!

