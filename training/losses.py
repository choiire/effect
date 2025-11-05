"""
음성 향상을 위한 손실 함수들

SI-SDR, Multi-resolution STFT, Perceptual Loss 등
보고서에서 권장하는 복합 손실 함수 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss
    
    음성 분리 성능의 핵심 지표 - 높을수록 좋음
    Loss로 사용하기 위해 음수로 변환
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, samples] - 예측된 신호
            target: [batch, samples] - 타깃 신호
            
        Returns:
            -SI-SDR (낮을수록 좋음)
        """
        # 평균 제거
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # <target, pred>
        dot_product = (target * pred).sum(dim=-1, keepdim=True)
        
        # ||target||^2
        target_energy = (target ** 2).sum(dim=-1, keepdim=True) + self.eps
        
        # 스케일 팩터: s = <target, pred> / ||target||^2
        scale = dot_product / target_energy
        
        # 투영: s * target
        projection = scale * target
        
        # 잔여 (잡음): pred - projection
        residual = pred - projection
        
        # SI-SDR = 10 * log10(||projection||^2 / ||residual||^2)
        projection_energy = (projection ** 2).sum(dim=-1) + self.eps
        residual_energy = (residual ** 2).sum(dim=-1) + self.eps
        
        si_sdr = 10 * torch.log10(projection_energy / residual_energy)
        
        # Loss로 사용하기 위해 음수 반환 (최대화 -> 최소화)
        return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT Loss
    
    여러 해상도의 STFT에서 magnitude와 spectral convergence를 측정
    주파수 영역에서의 충실도를 보장
    """
    
    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [50, 120, 240],
        win_lengths: list = [240, 600, 1200]
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    
    def stft(self, x: torch.Tensor, n_fft: int, hop_length: int, 
             win_length: int) -> torch.Tensor:
        """STFT 계산"""
        window = torch.hann_window(win_length).to(x.device)
        
        stft_result = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        
        return torch.abs(stft_result)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, samples]
            target: [batch, samples]
            
        Returns:
            Multi-resolution STFT loss
        """
        total_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # STFT magnitude 계산
            pred_mag = self.stft(pred, fft_size, hop_size, win_length)
            target_mag = self.stft(target, fft_size, hop_size, win_length)
            
            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / \
                     (torch.norm(target_mag, p='fro') + 1e-8)
            
            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-5),
                torch.log(target_mag + 1e-5)
            )
            
            total_loss += sc_loss + log_mag_loss
        
        return total_loss / len(self.fft_sizes)


class TimeDomainLoss(nn.Module):
    """
    시간 도메인 손실 (L1 + L2 결합)
    """
    
    def __init__(self, l1_weight: float = 0.5, l2_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss


class CombinedLoss(nn.Module):
    """
    복합 손실 함수
    
    보고서에서 권장하는 방식:
    Loss = α * SI-SDR + β * STFT + γ * Time-domain
    """
    
    def __init__(
        self,
        si_sdr_weight: float = 1.0,
        stft_weight: float = 0.5,
        time_weight: float = 0.1
    ):
        """
        Args:
            si_sdr_weight: SI-SDR 손실 가중치 (α)
            stft_weight: STFT 손실 가중치 (β)
            time_weight: Time-domain 손실 가중치 (γ)
        """
        super().__init__()
        
        self.si_sdr_weight = si_sdr_weight
        self.stft_weight = stft_weight
        self.time_weight = time_weight
        
        self.si_sdr_loss = SISDRLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.time_loss = TimeDomainLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ):
        """
        Args:
            pred: [batch, samples] - 예측 신호
            target: [batch, samples] - 타깃 신호
            return_components: True면 각 손실 성분도 반환
            
        Returns:
            total_loss 또는 (total_loss, loss_dict)
        """
        # 각 손실 계산
        si_sdr_loss = self.si_sdr_loss(pred, target)
        stft_loss = self.stft_loss(pred, target)
        time_loss = self.time_loss(pred, target)
        
        # 가중합
        total_loss = (
            self.si_sdr_weight * si_sdr_loss +
            self.stft_weight * stft_loss +
            self.time_weight * time_loss
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'si_sdr': si_sdr_loss.item(),
                'stft': stft_loss.item(),
                'time': time_loss.item()
            }
            return total_loss, loss_dict
        
        return total_loss


# 테스트 코드
if __name__ == "__main__":
    print("🧪 손실 함수 테스트...\n")
    
    # 테스트 데이터
    batch_size = 4
    num_samples = 16000  # 1초
    
    target = torch.randn(batch_size, num_samples)
    pred = target + 0.1 * torch.randn(batch_size, num_samples)  # 약간의 노이즈
    
    # 1. SI-SDR Loss
    print("1️⃣ SI-SDR Loss")
    si_sdr_loss = SISDRLoss()
    loss = si_sdr_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # 2. Multi-resolution STFT Loss
    print("\n2️⃣ Multi-resolution STFT Loss")
    stft_loss = MultiResolutionSTFTLoss()
    loss = stft_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # 3. Combined Loss
    print("\n3️⃣ Combined Loss")
    combined_loss = CombinedLoss(si_sdr_weight=1.0, stft_weight=0.5, time_weight=0.1)
    loss, loss_dict = combined_loss(pred, target, return_components=True)
    print(f"   Total: {loss_dict['total']:.4f}")
    print(f"   - SI-SDR: {loss_dict['si_sdr']:.4f}")
    print(f"   - STFT: {loss_dict['stft']:.4f}")
    print(f"   - Time: {loss_dict['time']:.4f}")
    
    print("\n✅ 손실 함수 테스트 완료")

