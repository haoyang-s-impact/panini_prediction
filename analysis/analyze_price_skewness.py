"""Analyze and visualize price skewness in Panini card data."""
# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# %%
# Load the extracted card data
df = pd.read_csv(str(PROJECT_ROOT / 'output/panini_cards_extracted.csv'))
prices = df['price_cny'].dropna()

print(f"Total cards with price data: {len(prices)}")
print(f"\n--- Price (CNY) Summary Statistics ---")
print(f"  Mean:      {prices.mean():>10,.2f}")
print(f"  Median:    {prices.median():>10,.2f}")
print(f"  Std Dev:   {prices.std():>10,.2f}")
print(f"  Min:       {prices.min():>10,.2f}")
print(f"  Max:       {prices.max():>10,.2f}")
print(f"  Skewness:  {prices.skew():>9.2f}")
print(f"  Kurtosis:  {prices.kurtosis():>9.2f}")

print(f"\n--- Percentiles ---")
for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%:  {prices.quantile(q):>10,.2f}")

# %%
# ========== VISUALIZATION ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Panini Card Price Distribution: Skewness Analysis', fontsize=16, fontweight='bold')

# --- 1. Raw price histogram ---
ax1 = axes[0, 0]
ax1.hist(prices, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
ax1.axvline(prices.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ¥{prices.mean():,.0f}')
ax1.axvline(prices.median(), color='orange', linestyle='-', linewidth=2, label=f'Median: ¥{prices.median():,.0f}')
ax1.set_title(f'Raw Price Distribution (Skew = {prices.skew():.2f})', fontsize=12)
ax1.set_xlabel('Price (CNY)')
ax1.set_ylabel('Count')
ax1.legend(fontsize=10)

# --- 2. Log-transformed price histogram ---
log_prices = np.log1p(prices)  # log(1 + x) to handle any zeros safely

ax2 = axes[0, 1]
ax2.hist(log_prices, bins=40, color='seagreen', edgecolor='white', alpha=0.85)
ax2.axvline(log_prices.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {log_prices.mean():.2f}')
ax2.axvline(log_prices.median(), color='orange', linestyle='-', linewidth=2, label=f'Median: {log_prices.median():.2f}')
ax2.set_title(f'Log-Transformed Price (Skew = {log_prices.skew():.2f})', fontsize=12)
ax2.set_xlabel('log(1 + Price)')
ax2.set_ylabel('Count')
ax2.legend(fontsize=10)

# --- 3. Box plot comparison ---
ax3 = axes[1, 0]
bp = ax3.boxplot([prices.values, log_prices.values],
                 labels=['Raw Price (CNY)', 'log(1 + Price)'],
                 patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('seagreen')
for box in bp['boxes']:
    box.set_alpha(0.7)
ax3.set_title('Box Plot: Raw vs Log-Transformed', fontsize=12)
ax3.set_ylabel('Value')

# --- 4. Q-Q plots side by side ---
ax4 = axes[1, 1]
# Q-Q for raw prices
sorted_prices = np.sort(prices.values)
n = len(sorted_prices)
theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, n))
ax4.scatter(theoretical_q, (sorted_prices - sorted_prices.mean()) / sorted_prices.std(),
            alpha=0.5, s=15, color='steelblue', label='Raw prices')
# Q-Q for log prices
sorted_log = np.sort(log_prices.values)
ax4.scatter(theoretical_q, (sorted_log - sorted_log.mean()) / sorted_log.std(),
            alpha=0.5, s=15, color='seagreen', label='Log prices')
# Reference line
lims = [-3, 3]
ax4.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Normal reference')
ax4.set_xlim(lims)
ax4.set_ylim(lims)
ax4.set_title('Q-Q Plot: Normality Check', fontsize=12)
ax4.set_xlabel('Theoretical Quantiles')
ax4.set_ylabel('Sample Quantiles (standardized)')
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'results/price_skewness_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to: results/price_skewness_analysis.png")

# %%
# ========== SUMMARY ==========
print("\n" + "="*60)
print("WHY LOG TRANSFORMATION HELPS")
print("="*60)
print(f"""
Raw price skewness:  {prices.skew():.2f}  (positive = right-skewed)
Log price skewness:  {log_prices.skew():.2f}  (closer to 0 = more symmetric)

The mean (¥{prices.mean():,.0f}) is much higher than the median (¥{prices.median():,.0f}),
confirming a long right tail — a few expensive cards pull the
average up while most cards cluster at lower prices.

Log transformation compresses large values and spreads small ones,
making the distribution more symmetric and closer to normal.
This helps XGBoost (and most ML models) because:

  1. Reduced outlier influence — extreme prices don't dominate
     the loss function (MAE/RMSE) as much
  2. Better splits — tree splits become more balanced when the
     target range is compressed
  3. Relative error focus — predicting log(price) means the model
     optimizes for *proportional* accuracy (e.g., being off by
     2x matters equally whether the card is ¥100 or ¥10,000)
  4. Stabilized variance — raw prices have higher variance at
     higher price points (heteroscedasticity); log fixes this
""")
