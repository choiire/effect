"""
학습 설정 및 하이퍼파라미터
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    train_noisy_dir: str = "data/train/noisy"
    train_clean_dir: str = "data/train/clean"
    val_noisy_dir: str = "data/val/noisy"
    val_clean_dir: str = "data/val/clean"
    
    sample_rate: int = 16000
    segment_length: int = 64000  # 4초 @ 16kHz
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_type: str = "waveform_unet"  # "spectrogram_unet" or "waveform_unet"
    
    # SpectrogramUNet 설정
    n_fft: int = 512
    hop_length: int = 256
    output_mode: str = "mask"  # "mask" or "spectrogram"
    
    # U-Net 공통 설정
    n_channels: int = 32
    
    # 전처리 필터 사용 여부
    use_preprocessing: bool = True
    apply_proximity_correction: bool = True
    apply_pop_suppression: bool = True
    apply_hum_removal: bool = True
    hum_freq: int = 60  # 50 (유럽) or 60 (북미/한국)


@dataclass
class LossConfig:
    """손실 함수 설정"""
    si_sdr_weight: float = 1.0
    stft_weight: float = 0.5
    time_weight: float = 0.1


@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 학습률 스케줄러
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau" or "cosine"
    patience: int = 10
    factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # 체크포인트
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    
    # 로깅
    log_dir: str = "logs"
    log_every_n_steps: int = 50
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    mixed_precision: bool = True  # AMP (Automatic Mixed Precision)


@dataclass
class Config:
    """전체 설정"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 프로젝트 정보
    project_name: str = "microphone_noise_reduction"
    experiment_name: str = "baseline"
    seed: int = 42


def get_default_config() -> Config:
    """기본 설정 반환"""
    return Config()


# 설정 저장/로드 (YAML)
def save_config(config: Config, path: str):
    """설정을 YAML 파일로 저장"""
    import yaml
    from dataclasses import asdict
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(asdict(config), f, allow_unicode=True, default_flow_style=False)
    
    print(f"설정 저장 완료: {path}")


def load_config(path: str) -> Config:
    """YAML 파일에서 설정 로드"""
    import yaml
    
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Config 객체로 변환
    config = Config(
        data=DataConfig(**config_dict.get('data', {})),
        model=ModelConfig(**config_dict.get('model', {})),
        loss=LossConfig(**config_dict.get('loss', {})),
        training=TrainingConfig(**config_dict.get('training', {}))
    )
    
    # 프로젝트 정보
    config.project_name = config_dict.get('project_name', 'microphone_noise_reduction')
    config.experiment_name = config_dict.get('experiment_name', 'baseline')
    config.seed = config_dict.get('seed', 42)
    
    print(f"설정 로드 완료: {path}")
    return config


if __name__ == "__main__":
    # 기본 설정 생성 및 저장
    config = get_default_config()
    save_config(config, "config.yaml")
    
    print("\n기본 설정:")
    print(f"  - 모델: {config.model.model_type}")
    print(f"  - 배치 크기: {config.data.batch_size}")
    print(f"  - 학습률: {config.training.learning_rate}")
    print(f"  - 에폭: {config.training.num_epochs}")

