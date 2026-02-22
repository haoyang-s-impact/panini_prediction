# %%
"""
Session 3 Comparison Report: Tree Models with Image Embeddings.

Runs a 12-configuration ablation matrix (3 frameworks × 4 feature sets)
and generates comparison plots to results/.

Configurations:
- 3 frameworks: XGBoost, LightGBM, CatBoost
- 4 feature sets: tabular-only, tabular+PCA-30, tabular+PCA-50, tabular+PCA-64
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from data.data_utils import merge_features, load_tabular_data, load_embeddings
from models.model_utils import (
    tune_hyperparameters, run_trials, prepare_for_catboost,
    show_feature_importance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PCA_VARIANTS = [30, 50, 64]
FRAMEWORKS = ['XGBoost', 'LightGBM', 'CatBoost']

# %%
# === Embedding-Price Correlation Diagnostic ===

def run_correlation_diagnostic():
    """Check if any embedding dimensions correlate suspiciously with price."""
    print("=" * 60)
    print("EMBEDDING-PRICE CORRELATION DIAGNOSTIC")
    print("=" * 60)

    _, y, images, _ = load_tabular_data()
    emb_df = load_embeddings(pca_n=64)

    # Merge to align
    merged = pd.DataFrame({'image': images, 'price_cny': y}).merge(
        emb_df, on='image', how='left'
    )
    emb_cols = [c for c in merged.columns if c.startswith('emb_')]

    correlations = {}
    for col in emb_cols:
        r = merged[col].corr(merged['price_cny'])
        correlations[col] = r

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 most price-correlated embedding dimensions (PCA-64):")
    print(f"{'Dimension':<12} {'Pearson r':>10}")
    print("-" * 24)
    max_abs_r = 0
    for col, r in sorted_corr[:10]:
        flag = " ⚠️" if abs(r) > 0.5 else ""
        print(f"{col:<12} {r:>10.4f}{flag}")
        max_abs_r = max(max_abs_r, abs(r))

    if max_abs_r > 0.5:
        print(f"\n⚠️  WARNING: max |r| = {max_abs_r:.4f} > 0.5 — possible price leakage")
    else:
        print(f"\n✓ max |r| = {max_abs_r:.4f} — no strong price leakage detected")

    return sorted_corr


# %%
# === Hyperparameter Grids ===

XGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

LGBM_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'num_leaves': [15, 31, 63, 127],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
}

CAT_PARAM_DIST = {
    'iterations': [100, 200, 300, 500],
    'depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'l2_leaf_reg': [1, 3, 5, 7],
    'min_data_in_leaf': [1, 3, 5],
}


# %%
# === Run Single Configuration ===

def run_config(framework, pca_n, include_embeddings, n_iter=10):
    """Run a single configuration: tune + 10-trial evaluation.

    Args:
        framework: 'XGBoost', 'LightGBM', or 'CatBoost'
        pca_n: PCA variant (30, 50, 64) - ignored if include_embeddings=False
        include_embeddings: Whether to include image embeddings
        n_iter: RandomizedSearchCV iterations (default 10 for speed)

    Returns:
        config_name, avg_metrics, all_results, best_model, X
    """
    feat_label = "tab" if not include_embeddings else f"pca{pca_n}"
    config_name = f"{framework.lower()[:3]}-{feat_label}"

    X, y, cat_cols = merge_features(
        pca_n=pca_n,
        include_tabular=True,
        include_embeddings=include_embeddings,
    )

    n_feat = X.shape[1]
    print(f"\n{'='*50}")
    print(f"Config: {config_name} | {framework} | {n_feat} features")
    print(f"{'='*50}", flush=True)

    if framework == 'XGBoost':
        base = XGBRegressor(random_state=42, enable_categorical=True)
        params = tune_hyperparameters(base, XGB_PARAM_DIST, X, y,
                                       n_iter=n_iter)

        def factory(random_state):
            return XGBRegressor(**params, random_state=random_state,
                                enable_categorical=True)

    elif framework == 'LightGBM':
        base = LGBMRegressor(random_state=42, verbose=-1)
        params = tune_hyperparameters(base, LGBM_PARAM_DIST, X, y,
                                       n_iter=n_iter)

        def factory(random_state):
            return LGBMRegressor(**params, random_state=random_state,
                                 verbose=-1)

    elif framework == 'CatBoost':
        X, cat_indices = prepare_for_catboost(X, cat_cols)
        base = CatBoostRegressor(random_seed=42, verbose=0,
                                  bootstrap_type='Bernoulli')
        params = tune_hyperparameters(base, CAT_PARAM_DIST, X, y,
                                       n_iter=n_iter,
                                       cat_features=cat_indices)

        def factory(random_state):
            return CatBoostRegressor(**params, random_seed=random_state,
                                     verbose=0, bootstrap_type='Bernoulli',
                                     cat_features=cat_indices)

    avg, results, best_model = run_trials(factory, X, y)
    return config_name, avg, results, best_model, X


# %%
# === Visualization ===

def plot_framework_comparison(summary_df):
    """Plot 1: Grouped bars — 3 frameworks × (tabular vs best PCA)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each framework, get tabular-only and best PCA config
    data = []
    for fw in FRAMEWORKS:
        fw_prefix = fw.lower()[:3]
        tab_row = summary_df[summary_df['config'] == f"{fw_prefix}-tab"]
        pca_rows = summary_df[
            (summary_df['framework'] == fw) &
            (summary_df['config'] != f"{fw_prefix}-tab")
        ]
        if not tab_row.empty:
            data.append({'framework': fw, 'type': 'Tabular Only',
                         'r2': tab_row.iloc[0]['r2']})
        if not pca_rows.empty:
            best_pca = pca_rows.loc[pca_rows['r2'].idxmax()]
            data.append({'framework': fw, 'type': f'Best PCA ({best_pca["config"]})',
                         'r2': best_pca['r2']})

    plot_df = pd.DataFrame(data)
    x = np.arange(len(FRAMEWORKS))
    width = 0.35

    tab_vals = [plot_df[(plot_df['framework'] == fw) & (plot_df['type'] == 'Tabular Only')]['r2'].values[0]
                for fw in FRAMEWORKS]
    pca_vals = [plot_df[(plot_df['framework'] == fw) & (plot_df['type'] != 'Tabular Only')]['r2'].values[0]
                for fw in FRAMEWORKS]

    bars1 = ax.bar(x - width/2, tab_vals, width, label='Tabular Only', color='#4C72B0')
    bars2 = ax.bar(x + width/2, pca_vals, width, label='+ Best Embeddings', color='#DD8452')

    ax.set_ylabel('Average R²')
    ax.set_title('Framework Comparison: Tabular vs Tabular + Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(FRAMEWORKS)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "session3_framework_comparison.png", dpi=150)
    print(f"Saved: results/session3_framework_comparison.png")
    plt.close()


def plot_pca_effect(summary_df):
    """Plot 2: Line plot — R² vs PCA components for each framework."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for fw in FRAMEWORKS:
        fw_prefix = fw.lower()[:3]
        # Get tabular baseline
        tab_row = summary_df[summary_df['config'] == f"{fw_prefix}-tab"]
        tab_r2 = tab_row.iloc[0]['r2'] if not tab_row.empty else None

        pca_r2s = []
        for pca_n in PCA_VARIANTS:
            row = summary_df[summary_df['config'] == f"{fw_prefix}-pca{pca_n}"]
            pca_r2s.append(row.iloc[0]['r2'] if not row.empty else None)

        # Plot PCA line
        ax.plot(PCA_VARIANTS, pca_r2s, 'o-', label=fw, linewidth=2, markersize=8)

        # Plot tabular baseline as dashed horizontal
        if tab_r2 is not None:
            ax.axhline(y=tab_r2, linestyle='--', alpha=0.4)
            ax.annotate(f'{fw} tab: {tab_r2:.3f}', xy=(PCA_VARIANTS[-1], tab_r2),
                        fontsize=8, alpha=0.6, ha='right')

    ax.set_xlabel('PCA Components')
    ax.set_ylabel('Average R²')
    ax.set_title('Effect of Embedding Dimensionality on Model Performance')
    ax.set_xticks(PCA_VARIANTS)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "session3_pca_effect.png", dpi=150)
    print(f"Saved: results/session3_pca_effect.png")
    plt.close()


def plot_full_results_heatmap(summary_df):
    """Plot 3: Heatmap of all 12 configurations."""
    # Build pivot: frameworks as rows, feature sets as columns
    feat_order = ['tab', 'pca30', 'pca50', 'pca64']
    feat_labels = ['Tabular\nOnly', 'Tab +\nPCA-30', 'Tab +\nPCA-50', 'Tab +\nPCA-64']

    matrix = np.zeros((len(FRAMEWORKS), len(feat_order)))
    for i, fw in enumerate(FRAMEWORKS):
        fw_prefix = fw.lower()[:3]
        for j, feat in enumerate(feat_order):
            row = summary_df[summary_df['config'] == f"{fw_prefix}-{feat}"]
            if not row.empty:
                matrix[i, j] = row.iloc[0]['r2']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=feat_labels, yticklabels=FRAMEWORKS,
                center=0, ax=ax, linewidths=1)
    ax.set_title('R² by Framework and Feature Set')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "session3_full_results.png", dpi=150)
    print(f"Saved: results/session3_full_results.png")
    plt.close()


def plot_feature_importance(best_model, feature_names):
    """Plot 4: Top-20 feature importance for best model."""
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]

    names = [feature_names[i] for i in sorted_idx]
    values = [importance[i] for i in sorted_idx]
    colors = ['#DD8452' if n.startswith('emb_') else '#4C72B0' for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Features (Best Model)\nOrange = embedding, Blue = tabular')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "session3_feature_importance.png", dpi=150)
    print(f"Saved: results/session3_feature_importance.png")
    plt.close()


# %%
# === Main Report ===

def main():
    print("=" * 60)
    print("SESSION 3: TREE MODELS WITH IMAGE EMBEDDINGS")
    print("=" * 60)

    # Step 1: Correlation diagnostic
    corr_results = run_correlation_diagnostic()

    # Step 2: Run all 12 configurations
    all_configs = []

    for fw in FRAMEWORKS:
        # Tabular only
        name, avg, results, model, X = run_config(fw, pca_n=30,
                                                    include_embeddings=False)
        all_configs.append({
            'config': name, 'framework': fw, 'features': 'tabular',
            'n_features': X.shape[1], 'pca_n': 0, **avg,
            '_model': model, '_feature_names': X.columns.tolist(),
        })

        # With embeddings: PCA-30, 50, 64
        for pca_n in PCA_VARIANTS:
            name, avg, results, model, X = run_config(fw, pca_n=pca_n,
                                                        include_embeddings=True)
            all_configs.append({
                'config': name, 'framework': fw, 'features': f'pca{pca_n}',
                'n_features': X.shape[1], 'pca_n': pca_n, **avg,
                '_model': model, '_feature_names': X.columns.tolist(),
            })

    # Step 3: Build summary
    summary_df = pd.DataFrame(all_configs)

    # Find V4-raw baseline for delta computation
    v4_baseline = summary_df[summary_df['config'] == 'xgb-tab'].iloc[0]['r2']
    summary_df['delta_r2'] = summary_df['r2'] - v4_baseline

    # Step 4: Print summary table
    print("\n" + "=" * 90)
    print("FULL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<14} {'Framework':<10} {'Features':>5} {'Avg MAE':>10} "
          f"{'Avg RMSE':>10} {'Avg R2':>8} {'ΔR2 vs V4':>10}")
    print("-" * 90)

    for _, row in summary_df.sort_values('r2', ascending=False).iterrows():
        print(f"{row['config']:<14} {row['framework']:<10} {row['n_features']:>5} "
              f"{row['mae']:>10,.0f} {row['rmse']:>10,.0f} {row['r2']:>8.4f} "
              f"{row['delta_r2']:>+10.4f}")

    # Best overall
    best = summary_df.loc[summary_df['r2'].idxmax()]
    print(f"\n🏆 Best: {best['config']} (R²={best['r2']:.4f}, "
          f"ΔR²={best['delta_r2']:+.4f} vs V4-raw)")

    # Step 5: Generate plots
    print("\nGenerating plots...")
    plot_framework_comparison(summary_df)
    plot_pca_effect(summary_df)
    plot_full_results_heatmap(summary_df)

    # Feature importance for best model
    best_idx = summary_df['r2'].idxmax()
    best_model = all_configs[best_idx]['_model']
    best_features = all_configs[best_idx]['_feature_names']
    show_feature_importance(best_model, best_features)
    plot_feature_importance(best_model, best_features)

    # Step 6: Embedding contribution summary
    print("\n" + "=" * 60)
    print("EMBEDDING CONTRIBUTION ANALYSIS")
    print("=" * 60)
    for fw in FRAMEWORKS:
        fw_prefix = fw.lower()[:3]
        tab_r2 = summary_df[summary_df['config'] == f"{fw_prefix}-tab"].iloc[0]['r2']
        for pca_n in PCA_VARIANTS:
            pca_r2 = summary_df[summary_df['config'] == f"{fw_prefix}-pca{pca_n}"].iloc[0]['r2']
            delta = pca_r2 - tab_r2
            direction = "+" if delta > 0 else ""
            print(f"  {fw:<10} PCA-{pca_n}: {pca_r2:.4f} ({direction}{delta:.4f} vs tabular)")

    return summary_df


# %%
if __name__ == "__main__":
    summary = main()
