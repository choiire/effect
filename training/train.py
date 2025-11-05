"""
메인 학습 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from models.unet import SpectrogramUNet, WaveformUNet
from models.preprocessing import MicrophoneNoisePreprocessor
from training.losses import CombinedLoss
from training.config import Config, load_config, save_config, get_default_config


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """음성 향상 모델 학습기"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Device 설정
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {self.device}")
        
        # 모델 생성
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 전처리기 (선택적)
        if config.model.use_preprocessing:
            self.preprocessor = MicrophoneNoisePreprocessor(
                sample_rate=config.data.sample_rate,
                apply_proximity_correction=config.model.apply_proximity_correction,
                apply_pop_suppression=config.model.apply_pop_suppression,
                apply_hum_removal=config.model.apply_hum_removal,
                hum_freq=config.model.hum_freq
            ).to(self.device)
        else:
            self.preprocessor = None
        
        # 손실 함수
        self.criterion = CombinedLoss(
            si_sdr_weight=config.loss.si_sdr_weight,
            stft_weight=config.loss.stft_weight,
            time_weight=config.loss.time_weight
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        if config.training.scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.training.factor,
                patience=config.training.patience
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs
            )
        
        # AMP (Mixed Precision)
        if config.training.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # 로깅
        self.writer = SummaryWriter(log_dir=config.training.log_dir)
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.no_improve_count = 0
        
        print(f"모델 생성 완료 ({config.model.model_type})")
        print(f"   파라미터 수: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def _create_model(self) -> nn.Module:
        """설정에 따라 모델 생성"""
        config = self.config.model
        
        if config.model_type == "spectrogram_unet":
            return SpectrogramUNet(
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_channels=config.n_channels,
                output_mode=config.output_mode
            )
        elif config.model_type == "waveform_unet":
            return WaveformUNet(n_channels=config.n_channels)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {'si_sdr': 0.0, 'stft': 0.0, 'time': 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 전처리 (선택적)
            if self.preprocessor is not None:
                with torch.no_grad():
                    noisy = self.preprocessor(noisy)
            
            # Forward pass (with AMP)
            with torch.cuda.amp.autocast(enabled=self.config.training.mixed_precision):
                enhanced = self.model(noisy)
                loss, loss_dict = self.criterion(enhanced, clean, return_components=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            # 로깅
            if batch_idx % self.config.training.log_every_n_steps == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                
                for key, value in loss_dict.items():
                    if key != 'total':
                        self.writer.add_scalar(f'train/{key}', value, global_step)
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'si_sdr': f"{loss_dict['si_sdr']:.4f}"
            })
        
        # 에폭 평균
        avg_loss = total_loss / len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        return avg_loss, loss_components
    
    @torch.no_grad()
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {'si_sdr': 0.0, 'stft': 0.0, 'time': 0.0}
        
        for noisy, clean in tqdm(val_loader, desc="Validation"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 전처리
            if self.preprocessor is not None:
                noisy = self.preprocessor(noisy)
            
            # Forward pass
            enhanced = self.model(noisy)
            loss, loss_dict = self.criterion(enhanced, clean, return_components=True)
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
        
        # 평균
        avg_loss = total_loss / len(val_loader)
        for key in loss_components:
            loss_components[key] /= len(val_loader)
        
        return avg_loss, loss_components
    
    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"체크포인트 로드 완료: {path}")
        print(f"   에폭: {self.current_epoch}, 최고 Val Loss: {self.best_val_loss:.4f}")
    
    def train(self, train_loader, val_loader):
        """전체 학습 루프"""
        print(f"\n학습 시작!")
        print(f"   에폭: {self.config.training.num_epochs}")
        print(f"   배치 크기: {self.config.data.batch_size}")
        print(f"   학습 샘플: {len(train_loader.dataset)}")
        print(f"   검증 샘플: {len(val_loader.dataset)}\n")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss, train_components = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_components = self.validate(val_loader)
            
            # 스케줄러 업데이트
            if self.config.training.scheduler == "reduce_on_plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 로깅
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  SI-SDR: {train_components['si_sdr']:.4f} -> {val_components['si_sdr']:.4f}")
            
            # 체크포인트 저장
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Best model 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
                self.no_improve_count = 0
                print(f"  최고 성능 모델 저장! (Val Loss: {val_loss:.4f})")
            else:
                self.no_improve_count += 1
            
            # Early stopping
            if self.no_improve_count >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered (no improvement for {self.no_improve_count} epochs)")
                break
        
        print(f"\n학습 완료!")
        print(f"   최고 Val Loss: {self.best_val_loss:.4f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="음성 향상 모델 학습")
    parser.add_argument("--config", type=str, default=None, help="설정 파일 경로 (YAML)")
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로 (재개)")
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
        # 기본 설정 저장
        save_config(config, "config.yaml")
    
    # 시드 설정
    set_seed(config.seed)
    
    # 데이터로더 생성
    print("데이터셋 로딩...")
    train_loader, val_loader = create_dataloaders(
        train_noisy_dir=config.data.train_noisy_dir,
        train_clean_dir=config.data.train_clean_dir,
        val_noisy_dir=config.data.val_noisy_dir,
        val_clean_dir=config.data.val_clean_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        segment_length=config.data.segment_length,
        sample_rate=config.data.sample_rate
    )
    
    # Trainer 생성
    trainer = Trainer(config)
    
    # 체크포인트에서 재개
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 학습
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

