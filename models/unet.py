"""
U-Net 기반 음성 향상 모델 (베이스라인)

시간-주파수(T-F) 도메인에서 스펙트로그램을 처리하여
잡음을 제거하는 U-Net 아키텍처
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """U-Net의 기본 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 
                     stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """U-Net의 인코더 블록 (다운샘플링)"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class UpBlock(nn.Module):
    """U-Net의 디코더 블록 (업샘플링)"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Skip connection과 크기 맞추기
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SpectrogramUNet(nn.Module):
    """
    스펙트로그램 기반 U-Net 음성 향상 모델
    
    입력: 잡음이 섞인 스펙트로그램
    출력: 깨끗한 스펙트로그램 또는 마스크
    """
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        n_channels: int = 32,
        output_mode: str = "mask"  # "mask" or "spectrogram"
    ):
        """
        Args:
            n_fft: FFT 크기
            hop_length: STFT hop length
            n_channels: 첫 번째 레이어의 채널 수
            output_mode: "mask"면 마스크 예측, "spectrogram"이면 직접 예측
        """
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_mode = output_mode
        
        # 인코더 (다운샘플링)
        self.down1 = DownBlock(1, n_channels)          # -> 32
        self.down2 = DownBlock(n_channels, n_channels*2)     # -> 64
        self.down3 = DownBlock(n_channels*2, n_channels*4)   # -> 128
        self.down4 = DownBlock(n_channels*4, n_channels*8)   # -> 256
        
        # 보틀넥
        self.bottleneck = ConvBlock(n_channels*8, n_channels*16)  # 512
        
        # 디코더 (업샘플링)
        self.up1 = UpBlock(n_channels*16, n_channels*8)    # 256
        self.up2 = UpBlock(n_channels*8, n_channels*4)     # 128
        self.up3 = UpBlock(n_channels*4, n_channels*2)     # 64
        self.up4 = UpBlock(n_channels*2, n_channels)       # 32
        
        # 출력 레이어
        if output_mode == "mask":
            # 마스크 예측 (0~1 사이 값)
            self.output = nn.Sequential(
                nn.Conv2d(n_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            # 스펙트로그램 직접 예측
            self.output = nn.Conv2d(n_channels, 1, kernel_size=1)
    
    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_audio: [batch, samples] - 잡음이 섞인 오디오
            
        Returns:
            enhanced_audio: [batch, samples] - 향상된 오디오
        """
        # 1. STFT 변환
        stft = torch.stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft).to(noisy_audio.device)
        )
        
        # Magnitude와 Phase 분리
        magnitude = torch.abs(stft)  # [batch, freq, time]
        phase = torch.angle(stft)
        
        # 2. U-Net 처리 (magnitude만)
        x = magnitude.unsqueeze(1)  # [batch, 1, freq, time]
        
        # 인코더
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        
        # 보틀넥
        x = self.bottleneck(x)
        
        # 디코더 (skip connections)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # 출력
        output = self.output(x).squeeze(1)  # [batch, freq, time]
        
        # 3. 마스크 적용 또는 직접 사용
        if self.output_mode == "mask":
            # 예측된 마스크를 magnitude에 곱함
            enhanced_magnitude = magnitude * output
        else:
            # 직접 예측된 magnitude 사용
            enhanced_magnitude = output
        
        # 크기 맞추기
        if enhanced_magnitude.shape != magnitude.shape:
            enhanced_magnitude = F.interpolate(
                enhanced_magnitude.unsqueeze(1), 
                size=magnitude.shape[1:], 
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # 4. 원본 위상 사용하여 iSTFT
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        # iSTFT
        enhanced_audio = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(noisy_audio.device),
            length=noisy_audio.shape[-1]
        )
        
        return enhanced_audio


class WaveformUNet(nn.Module):
    """
    시간 도메인 직접 처리 U-Net (1D Convolution)
    
    STFT 없이 waveform을 직접 처리 - End-to-End
    """
    
    def __init__(self, n_channels: int = 32):
        super().__init__()
        
        # 인코더 (1D Conv)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(n_channels),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels*2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(n_channels*2),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(n_channels*4),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(n_channels*4, n_channels*8, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(n_channels*8),
            nn.LeakyReLU(0.2)
        )
        
        # 보틀넥
        self.bottleneck = nn.Sequential(
            nn.Conv1d(n_channels*8, n_channels*16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(n_channels*16),
            nn.LeakyReLU(0.2)
        )
        
        # 디코더 (Transposed Conv)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(n_channels*16, n_channels*8, kernel_size=15, 
                              stride=2, padding=7, output_padding=0),
            nn.BatchNorm1d(n_channels*8),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(n_channels*16, n_channels*4, kernel_size=15,
                              stride=2, padding=7, output_padding=0),
            nn.BatchNorm1d(n_channels*4),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(n_channels*8, n_channels*2, kernel_size=15,
                              stride=2, padding=7, output_padding=0),
            nn.BatchNorm1d(n_channels*2),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(n_channels*4, n_channels, kernel_size=15,
                              stride=2, padding=7, output_padding=0),
            nn.BatchNorm1d(n_channels),
            nn.ReLU()
        )
        
        # 출력
        self.output = nn.Sequential(
            nn.Conv1d(n_channels*2, 1, kernel_size=15, padding=7),
            nn.Tanh()
        )
    
    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_audio: [batch, samples]
            
        Returns:
            enhanced_audio: [batch, samples]
        """
        x = noisy_audio.unsqueeze(1)  # [batch, 1, samples]
        
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # 보틀넥
        b = self.bottleneck(e4)
        
        # 디코더 (skip connections)
        d1 = self.dec1(b)
        # 크기 맞추기
        if d1.shape[2] != e4.shape[2]:
            d1 = F.interpolate(d1, size=e4.shape[2], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e4], dim=1)
        
        d2 = self.dec2(d1)
        if d2.shape[2] != e3.shape[2]:
            d2 = F.interpolate(d2, size=e3.shape[2], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e3], dim=1)
        
        d3 = self.dec3(d2)
        if d3.shape[2] != e2.shape[2]:
            d3 = F.interpolate(d3, size=e2.shape[2], mode='linear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        
        d4 = self.dec4(d3)
        if d4.shape[2] != e1.shape[2]:
            d4 = F.interpolate(d4, size=e1.shape[2], mode='linear', align_corners=False)
        d4 = torch.cat([d4, e1], dim=1)
        
        # 출력
        output = self.output(d4).squeeze(1)  # [batch, samples]
        
        # 원본 길이 맞추기
        if output.shape[-1] != noisy_audio.shape[-1]:
            output = F.interpolate(
                output.unsqueeze(1), 
                size=noisy_audio.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return output


# 테스트 코드
if __name__ == "__main__":
    print("🧪 U-Net 모델 테스트...")
    
    # 테스트 데이터
    batch_size = 4
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)
    
    noisy_audio = torch.randn(batch_size, num_samples)
    
    # 1. SpectrogramUNet 테스트
    print("\n1️⃣ SpectrogramUNet (T-F Domain)")
    model_spec = SpectrogramUNet(n_fft=512, hop_length=256, output_mode="mask")
    
    with torch.no_grad():
        enhanced = model_spec(noisy_audio)
    
    print(f"   입력 shape: {noisy_audio.shape}")
    print(f"   출력 shape: {enhanced.shape}")
    print(f"   파라미터 수: {sum(p.numel() for p in model_spec.parameters()) / 1e6:.2f}M")
    
    # 2. WaveformUNet 테스트
    print("\n2️⃣ WaveformUNet (Time Domain)")
    model_wave = WaveformUNet(n_channels=32)
    
    with torch.no_grad():
        enhanced = model_wave(noisy_audio)
    
    print(f"   입력 shape: {noisy_audio.shape}")
    print(f"   출력 shape: {enhanced.shape}")
    print(f"   파라미터 수: {sum(p.numel() for p in model_wave.parameters()) / 1e6:.2f}M")
    
    print("\n✅ 모델 테스트 완료")

