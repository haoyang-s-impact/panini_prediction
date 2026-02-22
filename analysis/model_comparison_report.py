# %% [markdown]
# # Model Version Comparison Report
# Trains V1→V4 and compares performance with visualizations.

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.makedirs(str(PROJECT_ROOT / 'results'), exist_ok=True)

CSV_PATH = str(PROJECT_ROOT / "output/panini_cards_extracted.csv")
N_TRIALS = 10

# %% [markdown]
# ## 1. Price Distribution Visualization

# %%
df_raw = pd.read_csv(CSV_PATH)
prices = df_raw['price_cny'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw distribution
axes[0].hist(prices, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(prices.mean(), color='red', linestyle='--', label=f'Mean: {prices.mean():,.0f}')
axes[0].axvline(prices.median(), color='orange', linestyle='--', label=f'Median: {prices.median():,.0f}')
axes[0].set_title('Raw price_cny Distribution')
axes[0].set_xlabel('Price (CNY)')
axes[0].set_ylabel('Count')
axes[0].legend()

# Right: log-transformed
log_prices = np.log1p(prices)
axes[1].hist(log_prices, bins=40, color='seagreen', edgecolor='white', alpha=0.85)
axes[1].axvline(log_prices.mean(), color='red', linestyle='--', label=f'Mean: {log_prices.mean():.2f}')
axes[1].axvline(log_prices.median(), color='orange', linestyle='--', label=f'Median: {log_prices.median():.2f}')
axes[1].set_title('log1p(price_cny) Distribution')
axes[1].set_xlabel('log1p(Price)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'results/price_distribution_log_transform.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/price_distribution_log_transform.png")

# %% [markdown]
# ## 2. Train All 4 Versions

# %%
# ---- Shared helpers ----
def _run_trials(prepare_fn, csv_path, n_trials, use_log=False, tune=False, param_dist=None):
    """Generic trial runner. Returns list of per-trial metric dicts and best model."""
    X, y = prepare_fn(csv_path)[:2]

    if use_log:
        y_real = y.copy()
        y_log = np.log1p(y)
        target = y_log
    else:
        y_real = y
        target = y

    # Optional hyperparameter tuning
    params = {}
    if tune and param_dist:
        search = RandomizedSearchCV(
            XGBRegressor(random_state=42, enable_categorical=True),
            param_distributions=param_dist,
            n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X, target)
        params = search.best_params_

    results = []
    best_model = None
    best_r2 = -np.inf

    for i in range(n_trials):
        if use_log:
            X_tr, X_te, ylog_tr, ylog_te, yr_tr, yr_te = train_test_split(
                X, y_log, y_real, test_size=0.2, random_state=i
            )
        else:
            X_tr, X_te, yr_tr, yr_te = train_test_split(
                X, y_real, test_size=0.2, random_state=i
            )

        model = XGBRegressor(**params, random_state=i, enable_categorical=True)
        model.fit(X_tr, ylog_tr if use_log else yr_tr)

        pred_raw = model.predict(X_te)
        if use_log:
            pred_real = np.maximum(np.expm1(pred_raw), 0)
        else:
            pred_real = pred_raw

        mae = mean_absolute_error(yr_te, pred_real)
        rmse = np.sqrt(mean_squared_error(yr_te, pred_real))
        r2 = r2_score(yr_te, pred_real)
        results.append({'mae': mae, 'rmse': rmse, 'r2': r2})
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return results, best_model, X.columns.tolist()


PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

# ---- V1 loader ----
def _load_v1(csv_path):
    from models.train_price_regressor import load_and_prepare_data
    return load_and_prepare_data(csv_path)

# ---- V2 loader ----
def _load_v2(csv_path):
    from models.train_price_regressor_v2 import load_and_prepare_data
    return load_and_prepare_data(csv_path)

# ---- V3 loader ----
def _load_v3(csv_path):
    from models.train_price_regressor_v3 import load_and_prepare_data
    return load_and_prepare_data(csv_path)

# ---- V4 loader (returns X, y_real, y_log) ----
def _load_v4(csv_path):
    from models.train_price_regressor_v4 import load_and_prepare_data
    return load_and_prepare_data(csv_path)

# %%
print("Training V1 (6 features, raw target)...")
v1_results, v1_model, v1_feats = _run_trials(_load_v1, CSV_PATH, N_TRIALS, use_log=False, tune=False)

print("Training V2 (13 features, raw target)...")
v2_results, v2_model, v2_feats = _run_trials(_load_v2, CSV_PATH, N_TRIALS, use_log=False, tune=False)

print("Training V3 (13 features, raw target, tuned)...")
v3_results, v3_model, v3_feats = _run_trials(_load_v3, CSV_PATH, N_TRIALS, use_log=False, tune=True, param_dist=PARAM_DIST)

print("Training V3+log (13 features, log target, tuned) — ablation...")
v3log_results, v3log_model, v3log_feats = _run_trials(_load_v3, CSV_PATH, N_TRIALS, use_log=True, tune=True, param_dist=PARAM_DIST)

print("Training V4 (24 features, log target, tuned)...")
v4_results, v4_model, v4_feats = _run_trials(_load_v4, CSV_PATH, N_TRIALS, use_log=True, tune=True, param_dist=PARAM_DIST)

print("Training V4-raw (24 features, raw target, tuned) — ablation...")
v4raw_results, v4raw_model, v4raw_feats = _run_trials(_load_v4, CSV_PATH, N_TRIALS, use_log=False, tune=True, param_dist=PARAM_DIST)

print("All versions trained.")

# %% [markdown]
# ## 3. Version Comparison Table

# %%
def _avg(results):
    return {k: np.mean([r[k] for r in results]) for k in results[0]}

metrics = {
    'V1': _avg(v1_results),
    'V2': _avg(v2_results),
    'V3': _avg(v3_results),
    'V3+log': _avg(v3log_results),
    'V4': _avg(v4_results),
    'V4-raw': _avg(v4raw_results),
}

feat_counts = {
    'V1': len(v1_feats),
    'V2': len(v2_feats),
    'V3': len(v3_feats),
    'V3+log': len(v3log_feats),
    'V4': len(v4_feats),
    'V4-raw': len(v4raw_feats),
}

rows = []
v3_r2 = metrics['V3']['r2']
for ver, m in metrics.items():
    rows.append({
        'Version': ver,
        'Features': feat_counts[ver],
        'Avg MAE': f"{m['mae']:,.0f}",
        'Avg RMSE': f"{m['rmse']:,.0f}",
        'Avg R²': f"{m['r2']:.4f}",
        'ΔR² vs V3': f"{(m['r2'] - v3_r2)*100:+.2f}pp",
    })

comparison_df = pd.DataFrame(rows)
print("\n" + "="*70)
print("MODEL VERSION COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# %% [markdown]
# ## 4. Version Comparison Bar Chart

# %%
versions = list(metrics.keys())
r2_values = [metrics[v]['r2'] for v in versions]
colors = ['#7f8c8d', '#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(versions, r2_values, color=colors, edgecolor='white', width=0.6)
ax.axhline(v3_r2, color='gray', linestyle='--', alpha=0.7, label=f'V3 baseline ({v3_r2:.4f})')

for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Average R² (real-space)')
ax.set_title('Model Version Comparison: R² Progression')
ax.legend()
ax.set_ylim(min(r2_values) - 0.05, max(max(r2_values) + 0.05, 1.0))

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'results/model_version_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/model_version_comparison.png")

# %% [markdown]
# ## 5. Log-Transform Ablation

# %%
ablation_versions = ['V3', 'V3+log', 'V4-raw', 'V4']
ablation_r2 = [metrics[v]['r2'] for v in ablation_versions]
ablation_colors = ['#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ablation_versions, ablation_r2, color=ablation_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, ablation_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Average R² (real-space)')
ax.set_title('Log-Transform Ablation: Isolating Target Transform vs New Features')
ax.set_ylim(min(ablation_r2) - 0.05, max(max(ablation_r2) + 0.05, 1.0))

# Compute deltas
log_gain = metrics['V3+log']['r2'] - metrics['V3']['r2']
feat_gain_raw = metrics['V4-raw']['r2'] - metrics['V3']['r2']
feat_gain_log = metrics['V4']['r2'] - metrics['V3+log']['r2']
total_gain = metrics['V4']['r2'] - metrics['V3']['r2']

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'results/log_transform_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/log_transform_ablation.png")

print(f"\nAblation breakdown:")
print(f"  V3 → V3+log (log transform only):     {log_gain*100:+.2f}pp R²")
print(f"  V3 → V4-raw (new features only):       {feat_gain_raw*100:+.2f}pp R²")
print(f"  V3+log → V4 (new features + log):      {feat_gain_log*100:+.2f}pp R²")
print(f"  V3 → V4 (total log+features):          {total_gain*100:+.2f}pp R²")

# %% [markdown]
# ## 6. Feature Importance Comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# V3 feature importance
v3_imp = v3_model.feature_importances_
v3_sorted = np.argsort(v3_imp)
axes[0].barh([v3_feats[i] for i in v3_sorted], v3_imp[v3_sorted], color='#2ecc71')
axes[0].set_title('V3 Feature Importance')
axes[0].set_xlabel('Importance')

# V4-raw feature importance (use raw-target variant for fair comparison)
v4_imp = v4raw_model.feature_importances_
v4_sorted = np.argsort(v4_imp)
# Highlight new V4 features
new_feats = set(v4raw_feats) - set(v3_feats)
v4_colors = ['#e74c3c' if v4raw_feats[i] in new_feats else '#3498db' for i in v4_sorted]
axes[1].barh([v4raw_feats[i] for i in v4_sorted], v4_imp[v4_sorted], color=v4_colors)
axes[1].set_title('V4 Feature Importance (red = new features)')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'results/feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/feature_importance_comparison.png")

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "="*70)
print("SESSION 1 SUMMARY")
print("="*70)

best_ver = max(metrics, key=lambda v: metrics[v]['r2'])
best_r2 = metrics[best_ver]['r2']
# Check +5pp target against best of V4 or V4-raw
v4_best = max(metrics['V4']['r2'], metrics['V4-raw']['r2'])
target_met = (v4_best - v3_r2) >= 0.05

print(f"\nBest version: {best_ver} (R² = {best_r2:.4f})")
print(f"V3 baseline R²: {v3_r2:.4f}")
print(f"V4 (log target) R²: {metrics['V4']['r2']:.4f}")
print(f"V4-raw (raw target) R²: {metrics['V4-raw']['r2']:.4f}")
print(f"Best V4 improvement over V3: {(v4_best - v3_r2)*100:+.2f} percentage points")
print(f"+5pp R² target met: {'YES' if target_met else 'NO'}")

print(f"\nLog-transform effect: {log_gain*100:+.2f}pp (negative = hurts real-space R²)")
print(f"New features effect (raw target): {feat_gain_raw*100:+.2f}pp")

print(f"\nTop 5 V4 features by importance:")
top5_idx = np.argsort(v4_imp)[::-1][:5]
for idx in top5_idx:
    marker = " (NEW)" if v4raw_feats[idx] in new_feats else ""
    print(f"  {v4raw_feats[idx]:20s} {v4_imp[idx]:.4f}{marker}")

print(f"\nGenerated plots:")
print(f"  results/price_distribution_log_transform.png")
print(f"  results/model_version_comparison.png")
print(f"  results/log_transform_ablation.png")
print(f"  results/feature_importance_comparison.png")
print("="*70)
