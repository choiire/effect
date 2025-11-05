"""평가 결과를 그래프로 시각화"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
script_dir = Path(__file__).parent
csv_path = script_dir / 'results.csv'
df = pd.read_csv(csv_path)

# 출력 디렉토리
output_dir = script_dir / 'plots'
output_dir.mkdir(exist_ok=True)

print("그래프 생성 중...")

# 1. SI-SDR 비교 박스플롯
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1-1. SI-SDR 박스플롯
ax1 = axes[0, 0]
data_si_sdr = [df['noisy_si_sdr'].values, df['si_sdr'].values]
bp1 = ax1.boxplot(data_si_sdr, tick_labels=['원본 잡음', '향상된 음성'], patch_artist=True)
bp1['boxes'][0].set_facecolor('lightcoral')
bp1['boxes'][1].set_facecolor('lightblue')
ax1.set_ylabel('SI-SDR (dB)', fontsize=12)
ax1.set_title('SI-SDR 비교', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 평균값 표시
mean_noisy = df['noisy_si_sdr'].mean()
mean_enhanced = df['si_sdr'].mean()
ax1.axhline(y=mean_noisy, color='red', linestyle='--', alpha=0.5, label=f'원본 평균: {mean_noisy:.2f} dB')
ax1.axhline(y=mean_enhanced, color='blue', linestyle='--', alpha=0.5, label=f'향상 평균: {mean_enhanced:.2f} dB')
ax1.legend()

# 1-2. STOI 비교 박스플롯
ax2 = axes[0, 1]
data_stoi = [df['noisy_stoi'].values, df['stoi'].values]
bp2 = ax2.boxplot(data_stoi, tick_labels=['원본 잡음', '향상된 음성'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightcoral')
bp2['boxes'][1].set_facecolor('lightblue')
ax2.set_ylabel('STOI', fontsize=12)
ax2.set_title('STOI 비교', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 평균값 표시
mean_noisy_stoi = df['noisy_stoi'].mean()
mean_enhanced_stoi = df['stoi'].mean()
ax2.axhline(y=mean_noisy_stoi, color='red', linestyle='--', alpha=0.5, label=f'원본 평균: {mean_noisy_stoi:.3f}')
ax2.axhline(y=mean_enhanced_stoi, color='blue', linestyle='--', alpha=0.5, label=f'향상 평균: {mean_enhanced_stoi:.3f}')
ax2.legend()

# 1-3. SI-SDR 개선량 분포
ax3 = axes[1, 0]
improvement = df['si_sdr_improvement'].values
ax3.hist(improvement, bins=30, edgecolor='black', alpha=0.7, color='green')
ax3.axvline(x=improvement.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'평균: {improvement.mean():.2f} dB')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_xlabel('SI-SDR 개선량 (dB)', fontsize=12)
ax3.set_ylabel('빈도', fontsize=12)
ax3.set_title('SI-SDR 개선량 분포', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 1-4. 산점도: 원본 vs 향상된 SI-SDR
ax4 = axes[1, 1]
ax4.scatter(df['noisy_si_sdr'], df['si_sdr'], alpha=0.6, s=30, color='blue')
# 대각선 (y=x)
min_val = min(df['noisy_si_sdr'].min(), df['si_sdr'].min())
max_val = max(df['noisy_si_sdr'].max(), df['si_sdr'].max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x (개선 없음)')
ax4.set_xlabel('원본 SI-SDR (dB)', fontsize=12)
ax4.set_ylabel('향상된 SI-SDR (dB)', fontsize=12)
ax4.set_title('원본 vs 향상된 SI-SDR', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig(output_dir / 'comparison_overview.png', dpi=300, bbox_inches='tight')
print(f"  [저장] {output_dir / 'comparison_overview.png'}")

# 2. 개별 샘플 비교 (SI-SDR)
fig, ax = plt.subplots(figsize=(16, 6))
n_samples = len(df)
x = np.arange(n_samples)
width = 0.35

# 샘플 인덱스로 정렬 (개선량 순으로)
df_sorted = df.sort_values('si_sdr_improvement', ascending=True).reset_index(drop=True)

ax.bar(x - width/2, df_sorted['noisy_si_sdr'], width, label='원본 잡음', color='lightcoral', alpha=0.8)
ax.bar(x + width/2, df_sorted['si_sdr'], width, label='향상된 음성', color='lightblue', alpha=0.8)

ax.set_xlabel('샘플 번호 (개선량 순)', fontsize=12)
ax.set_ylabel('SI-SDR (dB)', fontsize=12)
ax.set_title('샘플별 SI-SDR 비교 (개선량 순 정렬)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'sample_comparison.png', dpi=300, bbox_inches='tight')
print(f"  [저장] {output_dir / 'sample_comparison.png'}")

# 3. 통계 요약 바 차트
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3-1. 평균값 비교
ax1 = axes[0]
categories = ['SI-SDR (dB)', 'STOI', 'SNR (dB)']
noisy_means = [
    df['noisy_si_sdr'].mean(),
    df['noisy_stoi'].mean(),
    np.nan  # SNR은 원본에 대해 계산 안됨
]
enhanced_means = [
    df['si_sdr'].mean(),
    df['stoi'].mean(),
    df['snr'].mean()
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, [noisy_means[0], noisy_means[1], 0], width, 
                label='원본 잡음', color='lightcoral', alpha=0.8)
bars2 = ax1.bar(x + width/2, enhanced_means, width, 
                label='향상된 음성', color='lightblue', alpha=0.8)

ax1.set_ylabel('값', fontsize=12)
ax1.set_title('평균값 비교', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i < 2:  # SI-SDR, STOI
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if not np.isnan(height1):
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                    f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=9)

# 3-2. 개선량
ax2 = axes[1]
improvement_si_sdr = df['si_sdr_improvement'].mean()
improvement_stoi = (df['stoi'].mean() - df['noisy_stoi'].mean()) * 100  # 퍼센트로 변환

categories_imp = ['SI-SDR\n개선량 (dB)', 'STOI\n개선량 (%)']
improvements = [improvement_si_sdr, improvement_stoi]

bars = ax2.bar(categories_imp, improvements, color=['green', 'green'], alpha=0.7)
ax2.set_ylabel('개선량', fontsize=12)
ax2.set_title('평균 개선량', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 값 표시
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3-3. 개선 성공률
ax3 = axes[2]
improved_count = (df['si_sdr_improvement'] > 0).sum()
total_count = len(df)
not_improved_count = total_count - improved_count

categories_success = ['개선됨', '개선 안됨']
counts = [improved_count, not_improved_count]
colors = ['green', 'red']

bars = ax3.bar(categories_success, counts, color=colors, alpha=0.7)
ax3.set_ylabel('샘플 수', fontsize=12)
ax3.set_title(f'개선 성공률 ({improved_count}/{total_count})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 값과 퍼센트 표시
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_count) * 100
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'statistics_summary.png', dpi=300, bbox_inches='tight')
print(f"  [저장] {output_dir / 'statistics_summary.png'}")

# 4. 개선량 vs 원본 SI-SDR 관계
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['noisy_si_sdr'], df['si_sdr_improvement'], 
                     c=df['stoi'], cmap='viridis', alpha=0.6, s=50)
ax.set_xlabel('원본 SI-SDR (dB)', fontsize=12)
ax.set_ylabel('SI-SDR 개선량 (dB)', fontsize=12)
ax.set_title('원본 품질에 따른 개선량 (색상: STOI)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.colorbar(scatter, label='STOI')
plt.tight_layout()
plt.savefig(output_dir / 'improvement_vs_original.png', dpi=300, bbox_inches='tight')
print(f"  [저장] {output_dir / 'improvement_vs_original.png'}")

print("\n모든 그래프 생성 완료!")
print(f"저장 위치: {output_dir}")

