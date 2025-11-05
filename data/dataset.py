"""
PyTorch Dataset for Speech Enhancement
"""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random


class SpeechEnhancementDataset(Dataset):
    """
    음성 향상 데이터셋
    
    잡음이 섞인 음성(noisy)과 깨끗한 음성(clean) 쌍을 로드
    """
    
    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = None,
        augment: bool = False
    ):
        """
        Args:
            noisy_dir: 잡음 음성 디렉토리
            clean_dir: 깨끗한 음성 디렉토리
            sample_rate: 샘플링 레이트
            segment_length: 고정 길이로 자를 샘플 수 (None이면 전체)
            augment: 데이터 증강 여부
        """
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.augment = augment
        
        # 파일 목록
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))
        
        if len(self.noisy_files) == 0:
            raise ValueError(f"잡음 음성 파일을 찾을 수 없습니다: {noisy_dir}")
        
        print(f"데이터셋 로드 완료: {len(self.noisy_files)}개 샘플")
    
    def __len__(self) -> int:
        return len(self.noisy_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            noisy: 잡음 음성 텐서 [samples] 또는 [segment_length]
            clean: 깨끗한 음성 텐서 [samples] 또는 [segment_length]
        """
        # 파일 경로
        noisy_path = self.noisy_files[idx]
        
        # 대응하는 clean 파일 찾기
        # noisy: xxx_noisy.wav -> clean: xxx_clean.wav
        clean_filename = noisy_path.stem.replace("_noisy", "_clean") + ".wav"
        clean_path = self.clean_dir / clean_filename
        
        if not clean_path.exists():
            raise FileNotFoundError(f"Clean 파일을 찾을 수 없습니다: {clean_path}")
        
        # 오디오 로드
        noisy, _ = librosa.load(noisy_path, sr=self.sample_rate)
        clean, _ = librosa.load(clean_path, sr=self.sample_rate)
        
        # 길이 맞추기
        min_length = min(len(noisy), len(clean))
        noisy = noisy[:min_length]
        clean = clean[:min_length]
        
        # 세그먼트 자르기
        if self.segment_length is not None and len(noisy) > self.segment_length:
            if self.augment:
                # 랜덤 위치에서 자르기
                start = random.randint(0, len(noisy) - self.segment_length)
            else:
                # 중앙에서 자르기
                start = (len(noisy) - self.segment_length) // 2
            
            noisy = noisy[start:start + self.segment_length]
            clean = clean[start:start + self.segment_length]
        
        # 제로 패딩 (필요한 경우)
        if self.segment_length is not None and len(noisy) < self.segment_length:
            pad_length = self.segment_length - len(noisy)
            noisy = np.pad(noisy, (0, pad_length), mode='constant')
            clean = np.pad(clean, (0, pad_length), mode='constant')
        
        # 데이터 증강 (선택적)
        if self.augment:
            # 볼륨 조정 (±3dB)
            gain_db = random.uniform(-3, 3)
            gain = 10 ** (gain_db / 20)
            noisy = noisy * gain
            clean = clean * gain
        
        # Tensor로 변환
        noisy = torch.FloatTensor(noisy)
        clean = torch.FloatTensor(clean)
        
        return noisy, clean


def create_dataloaders(
    train_noisy_dir: str,
    train_clean_dir: str,
    val_noisy_dir: str,
    val_clean_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    segment_length: int = 64000,  # 4초 @ 16kHz
    **kwargs
):
    """
    학습 및 검증 데이터로더 생성
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = SpeechEnhancementDataset(
        noisy_dir=train_noisy_dir,
        clean_dir=train_clean_dir,
        segment_length=segment_length,
        augment=True,
        **kwargs
    )
    
    val_dataset = SpeechEnhancementDataset(
        noisy_dir=val_noisy_dir,
        clean_dir=val_clean_dir,
        segment_length=segment_length,
        augment=False,
        **kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

