"""
음성 향상 평가 지표

보고서에서 권장하는 PESQ, STOI, SI-SDR 지표 구현
"""

import torch
import numpy as np
from typing import Union, List

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("경고: pesq 모듈이 설치되지 않았습니다. PESQ 점수는 계산되지 않습니다.")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("경고: pystoi 모듈이 설치되지 않았습니다. STOI 점수는 계산되지 않습니다.")


def calculate_si_sdr(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) 계산
    
    Args:
        pred: 예측 신호
        target: 타깃 신호
        eps: 수치 안정성을 위한 작은 값
        
    Returns:
        SI-SDR 값 (dB) - 높을수록 좋음
    """
    # Numpy로 변환
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 1차원 확인
    pred = pred.flatten()
    target = target.flatten()
    
    # 평균 제거
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    
    # 스케일 팩터
    alpha = np.dot(target, pred) / (np.dot(target, target) + eps)
    
    # 투영
    projection = alpha * target
    
    # 잔여
    residual = pred - projection
    
    # SI-SDR
    si_sdr = 10 * np.log10(
        (np.sum(projection ** 2) + eps) / (np.sum(residual ** 2) + eps)
    )
    
    return float(si_sdr)


def calculate_pesq(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000
) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality) 계산
    
    Args:
        pred: 예측 신호
        target: 타깃 신호
        sample_rate: 샘플링 레이트 (8000 or 16000만 지원)
        
    Returns:
        PESQ 점수 (1.0~4.5, 높을수록 좋음)
    """
    # Numpy로 변환
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 1차원 확인
    pred = pred.flatten()
    target = target.flatten()
    
    # 길이 맞추기
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    
    # PESQ mode 결정
    if sample_rate == 16000:
        mode = 'wb'  # wideband
    elif sample_rate == 8000:
        mode = 'nb'  # narrowband
    else:
        raise ValueError(f"PESQ는 8kHz 또는 16kHz만 지원합니다. (현재: {sample_rate}Hz)")
    
    if not PESQ_AVAILABLE:
        return None
    
    try:
        score = pesq(sample_rate, target, pred, mode)
        return float(score)
    except Exception as e:
        print(f"PESQ 계산 오류: {e}")
        return None


def calculate_stoi(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000
) -> float:
    """
    STOI (Short-Time Objective Intelligibility) 계산
    
    Args:
        pred: 예측 신호
        target: 타깃 신호
        sample_rate: 샘플링 레이트
        
    Returns:
        STOI 점수 (0~1, 높을수록 좋음)
    """
    # Numpy로 변환
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 1차원 확인
    pred = pred.flatten()
    target = target.flatten()
    
    # 길이 맞추기
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    
    if not STOI_AVAILABLE:
        return None
    
    try:
        score = stoi(target, pred, sample_rate, extended=False)
        return float(score)
    except Exception as e:
        print(f"STOI 계산 오류: {e}")
        return None


def calculate_snr(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> float:
    """
    Signal-to-Noise Ratio (SNR) 계산
    
    Args:
        pred: 예측 신호
        target: 타깃 신호 (깨끗한 신호)
        
    Returns:
        SNR 값 (dB) - 높을수록 좋음
    """
    # Numpy로 변환
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    # 신호 파워
    signal_power = np.sum(target ** 2)
    
    # 노이즈 (잔여)
    noise = pred - target
    noise_power = np.sum(noise ** 2) + eps
    
    snr = 10 * np.log10(signal_power / noise_power)
    
    return float(snr)


class MetricsCalculator:
    """
    여러 지표를 한번에 계산하는 클래스
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def calculate_all(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ) -> dict:
        """
        모든 지표 계산
        
        Returns:
            metrics dict: {'si_sdr', 'pesq', 'stoi', 'snr'}
        """
        metrics = {}
        
        # SI-SDR (필수)
        try:
            metrics['si_sdr'] = calculate_si_sdr(pred, target)
        except Exception as e:
            print(f"⚠️ SI-SDR 계산 실패: {e}")
            metrics['si_sdr'] = None
        
        # PESQ
        try:
            metrics['pesq'] = calculate_pesq(pred, target, self.sample_rate)
        except Exception as e:
            print(f"⚠️ PESQ 계산 실패: {e}")
            metrics['pesq'] = None
        
        # STOI
        try:
            metrics['stoi'] = calculate_stoi(pred, target, self.sample_rate)
        except Exception as e:
            print(f"⚠️ STOI 계산 실패: {e}")
            metrics['stoi'] = None
        
        # SNR
        try:
            metrics['snr'] = calculate_snr(pred, target)
        except Exception as e:
            print(f"⚠️ SNR 계산 실패: {e}")
            metrics['snr'] = None
        
        return metrics
    
    def calculate_batch(
        self,
        pred_batch: Union[np.ndarray, torch.Tensor],
        target_batch: Union[np.ndarray, torch.Tensor]
    ) -> dict:
        """
        배치 데이터에 대해 평균 지표 계산
        
        Args:
            pred_batch: [batch, samples]
            target_batch: [batch, samples]
            
        Returns:
            평균 metrics dict
        """
        # Numpy로 변환
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.cpu().numpy()
        if isinstance(target_batch, torch.Tensor):
            target_batch = target_batch.cpu().numpy()
        
        batch_size = pred_batch.shape[0]
        
        # 각 샘플에 대해 계산
        all_metrics = []
        for i in range(batch_size):
            metrics = self.calculate_all(pred_batch[i], target_batch[i])
            all_metrics.append(metrics)
        
        # 평균 계산
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if m[key] is not None]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = None
        
        return avg_metrics


def format_metrics(metrics: dict) -> str:
    """
    지표를 보기 좋게 포맷팅
    
    Args:
        metrics: 지표 딕셔너리
        
    Returns:
        포맷된 문자열
    """
    lines = []
    
    if 'si_sdr' in metrics and metrics['si_sdr'] is not None:
        lines.append(f"  SI-SDR: {metrics['si_sdr']:>7.2f} dB")
    
    if 'pesq' in metrics and metrics['pesq'] is not None:
        lines.append(f"  PESQ:   {metrics['pesq']:>7.3f}")
    
    if 'stoi' in metrics and metrics['stoi'] is not None:
        lines.append(f"  STOI:   {metrics['stoi']:>7.3f}")
    
    if 'snr' in metrics and metrics['snr'] is not None:
        lines.append(f"  SNR:    {metrics['snr']:>7.2f} dB")
    
    return "\n".join(lines)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 평가 지표 테스트...\n")
    
    # 테스트 신호 생성
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)
    
    # 타깃 신호 (깨끗한 음성)
    target = np.random.randn(num_samples) * 0.1
    
    # 예측 신호 (약간의 노이즈 추가)
    pred = target + np.random.randn(num_samples) * 0.02
    
    # 지표 계산
    calculator = MetricsCalculator(sample_rate=sample_rate)
    metrics = calculator.calculate_all(pred, target)
    
    print("📊 평가 결과:")
    print(format_metrics(metrics))
    
    print("\n✅ 평가 지표 테스트 완료")

