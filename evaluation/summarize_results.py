"""평가 결과 요약 통계 계산"""
import pandas as pd

df = pd.read_csv('results.csv')

print('='*60)
print('평가 결과 요약 통계')
print('='*60)
print(f'\n총 샘플 수: {len(df)}개')

print('\n향상된 음성 (Enhanced):')
print(f'  SI-SDR: {df["si_sdr"].mean():.2f} ± {df["si_sdr"].std():.2f} dB (최소: {df["si_sdr"].min():.2f}, 최대: {df["si_sdr"].max():.2f})')
print(f'  STOI:   {df["stoi"].mean():.3f} ± {df["stoi"].std():.3f} (최소: {df["stoi"].min():.3f}, 최대: {df["stoi"].max():.3f})')
print(f'  SNR:    {df["snr"].mean():.2f} ± {df["snr"].std():.2f} dB (최소: {df["snr"].min():.2f}, 최대: {df["snr"].max():.2f})')

print('\n원본 잡음 신호 (Noisy):')
print(f'  SI-SDR: {df["noisy_si_sdr"].mean():.2f} ± {df["noisy_si_sdr"].std():.2f} dB (최소: {df["noisy_si_sdr"].min():.2f}, 최대: {df["noisy_si_sdr"].max():.2f})')
print(f'  STOI:   {df["noisy_stoi"].mean():.3f} ± {df["noisy_stoi"].std():.3f} (최소: {df["noisy_stoi"].min():.3f}, 최대: {df["noisy_stoi"].max():.3f})')

print('\n개선량:')
print(f'  SI-SDR 개선량: {df["si_sdr_improvement"].mean():.2f} ± {df["si_sdr_improvement"].std():.2f} dB')
print(f'  (최소: {df["si_sdr_improvement"].min():.2f}, 최대: {df["si_sdr_improvement"].max():.2f})')

print('='*60)

