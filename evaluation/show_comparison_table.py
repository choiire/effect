"""평가 결과를 비교 표로 출력"""
import pandas as pd
from pathlib import Path

# 현재 스크립트 위치 기준으로 results.csv 찾기
script_dir = Path(__file__).parent
csv_path = script_dir / 'results.csv'
df = pd.read_csv(csv_path)

print('\n' + '='*80)
print('평가 결과 비교 표')
print('='*80)
print(f'\n총 평가 샘플 수: {len(df)}개\n')

# 향상된 음성 통계
enhanced_si_sdr_mean = df['si_sdr'].mean()
enhanced_si_sdr_std = df['si_sdr'].std()
enhanced_stoi_mean = df['stoi'].mean()
enhanced_stoi_std = df['stoi'].std()
enhanced_snr_mean = df['snr'].mean()
enhanced_snr_std = df['snr'].std()

# 원본 잡음 신호 통계
noisy_si_sdr_mean = df['noisy_si_sdr'].mean()
noisy_si_sdr_std = df['noisy_si_sdr'].std()
noisy_stoi_mean = df['noisy_stoi'].mean()
noisy_stoi_std = df['noisy_stoi'].std()

# 개선량
improvement_mean = df['si_sdr_improvement'].mean()
improvement_std = df['si_sdr_improvement'].std()

# 표 출력
print(f"{'지표':<15} {'원본 잡음 신호':<25} {'향상된 음성':<25} {'개선량':<15}")
print('-'*80)
print(f"{'SI-SDR (dB)':<15} {f'{noisy_si_sdr_mean:>7.2f} ± {noisy_si_sdr_std:>5.2f}':<25} {f'{enhanced_si_sdr_mean:>7.2f} ± {enhanced_si_sdr_std:>5.2f}':<25} {f'{improvement_mean:>+7.2f} ± {improvement_std:>5.2f}':<15}")
print(f"{'STOI':<15} {f'{noisy_stoi_mean:>7.3f} ± {noisy_stoi_std:>5.3f}':<25} {f'{enhanced_stoi_mean:>7.3f} ± {enhanced_stoi_std:>5.3f}':<25} {f'{enhanced_stoi_mean - noisy_stoi_mean:>+7.3f}':<15}")
print(f"{'SNR (dB)':<15} {'N/A':<25} {f'{enhanced_snr_mean:>7.2f} ± {enhanced_snr_std:>5.2f}':<25} {'N/A':<15}")

print('\n' + '-'*80)
print('\n상세 통계:')
print('-'*80)

# 최소/최대값
print(f"\n{'지표':<15} {'최소값':<15} {'최대값':<15} {'평균':<15} {'표준편차':<15}")
print('-'*80)
print(f"{'향상 SI-SDR':<15} {df['si_sdr'].min():>15.2f} {df['si_sdr'].max():>15.2f} {df['si_sdr'].mean():>15.2f} {df['si_sdr'].std():>15.2f}")
print(f"{'원본 SI-SDR':<15} {df['noisy_si_sdr'].min():>15.2f} {df['noisy_si_sdr'].max():>15.2f} {df['noisy_si_sdr'].mean():>15.2f} {df['noisy_si_sdr'].std():>15.2f}")
print(f"{'개선량':<15} {df['si_sdr_improvement'].min():>15.2f} {df['si_sdr_improvement'].max():>15.2f} {df['si_sdr_improvement'].mean():>15.2f} {df['si_sdr_improvement'].std():>15.2f}")
print(f"{'향상 STOI':<15} {df['stoi'].min():>15.3f} {df['stoi'].max():>15.3f} {df['stoi'].mean():>15.3f} {df['stoi'].std():>15.3f}")
print(f"{'원본 STOI':<15} {df['noisy_stoi'].min():>15.3f} {df['noisy_stoi'].max():>15.3f} {df['noisy_stoi'].mean():>15.3f} {df['noisy_stoi'].std():>15.3f}")

print('\n' + '='*80)
print('\n주요 성과:')
print(f'  - 평균 SI-SDR 개선: {improvement_mean:.2f} dB')
print(f'  - 최대 개선: {df["si_sdr_improvement"].max():.2f} dB')
print(f'  - 개선된 샘플 비율: {(df["si_sdr_improvement"] > 0).sum()}/{len(df)} ({(df["si_sdr_improvement"] > 0).sum()/len(df)*100:.1f}%)')
print('='*80 + '\n')

