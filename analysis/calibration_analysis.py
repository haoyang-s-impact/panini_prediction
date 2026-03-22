# %%
"""
Calibration Analysis: CatBoost regression-to-the-mean diagnostic.

Generates 5-fold cross-validated out-of-sample predictions for the V5 CatBoost
model, then produces diagnostic plots quantifying the overestimation of cheap
cards and underestimation of expensive cards.

Outputs to results/:
  - calibration_pred_vs_actual.png    (Plot 1: scatter with 45° line)
  - calibration_residuals.png         (Plot 2: residuals vs predicted & actual)
  - calibration_error_by_bin.png      (Plot 3: mean signed error by price quartile)
  - calibration_curve.png             (Plot 4: mean pred vs mean actual per bin)
  - calibration_before_after.png      (Plot 5: raw vs calibrated pred-vs-actual)

Usage:
    python -m analysis.calibration_analysis
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from data.data_utils import merge_features
from models.calibration import apply_calibration, fit_calibration, save_calibration
from models.model_utils import prepare_for_catboost
from models.train_price_regressor_v5_catboost import PARAM_DIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
COLOR_RAW = '#4C72B0'
COLOR_CAL = '#55A868'
COLOR_OVER = '#55A868'   # overestimation (positive error)
COLOR_UNDER = '#C44E52'  # underestimation (negative error)


# %%
def get_cv_predictions(n_folds=N_FOLDS):
    """Generate 5-fold CV out-of-sample predictions for V5 CatBoost.

    Returns:
        y_actual: np.ndarray (96,) — true prices in CNY
        y_pred:   np.ndarray (96,) — OOS predicted prices in CNY
    """
    X, y, cat_cols = merge_features(include_embeddings=False)
    X_cb, cat_indices = prepare_for_catboost(X, cat_cols)

    print(f"Samples: {len(X_cb)}, Features: {X_cb.shape[1]}")
    print(f"Price range: {y.min():,.0f} – {y.max():,.0f} CNY\n")

    # Tune hyperparams on full data (consistent with existing training scripts)
    print("Tuning hyperparameters (20 iter, 3-fold CV)...")
    base = CatBoostRegressor(random_seed=42, verbose=0, bootstrap_type='Bernoulli')
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=20, cv=3,
        scoring='r2', random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X_cb, y, cat_features=cat_indices)
    best_params = search.best_params_
    print(f"Best CV R²: {search.best_score_:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 5-fold CV to get OOS predictions for every sample
    print(f"\nGenerating {n_folds}-fold OOS predictions...")
    y_actual = np.empty(len(y))
    y_pred = np.empty(len(y))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cb)):
        X_train = X_cb.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X_cb.iloc[test_idx]

        model = CatBoostRegressor(
            **best_params,
            random_seed=42,
            verbose=0,
            bootstrap_type='Bernoulli',
            cat_features=cat_indices,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)

        y_actual[test_idx] = y.iloc[test_idx].values
        y_pred[test_idx] = preds
        print(f"  Fold {fold+1}: {len(test_idx)} samples")

    return y_actual, y_pred


# %%
def _metrics_text(y_actual, y_pred):
    """Return formatted R², MAE, RMSE string."""
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    return f"R²={r2:.3f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}"


# %%
def plot_pred_vs_actual(y_actual, y_pred):
    """Plot 1: Predicted vs Actual scatter with 45° perfect-calibration line."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_actual, y_pred, alpha=0.6, edgecolors='white', linewidth=0.5,
               s=50, color=COLOR_RAW)

    lo = 0
    hi = max(y_actual.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1, label='Perfect calibration')

    ax.set_xlabel('Actual Price (CNY)')
    ax.set_ylabel('Predicted Price (CNY)')
    ax.set_title('CatBoost V5: Predicted vs Actual (5-fold CV)')
    ax.legend(loc='upper left')
    ax.text(0.95, 0.05, _metrics_text(y_actual, y_pred),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    path = RESULTS_DIR / "calibration_pred_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


# %%
def plot_residuals(y_actual, y_pred):
    """Plot 2: Residual analysis — residuals vs predicted (left), vs actual (right)."""
    residuals = y_pred - y_actual

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, x_vals, xlabel, title in [
        (axes[0], y_pred, 'Predicted Price (CNY)', 'Residuals vs Predicted'),
        (axes[1], y_actual, 'Actual Price (CNY)', 'Residuals vs Actual'),
    ]:
        colors = [COLOR_OVER if r >= 0 else COLOR_UNDER for r in residuals]
        ax.scatter(x_vals, residuals, c=colors, alpha=0.6, edgecolors='white',
                   linewidth=0.5, s=50)
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')

        # Linear trend line
        coeffs = np.polyfit(x_vals, residuals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color='red', linewidth=2,
                label=f'Trend (slope={coeffs[0]:.3f})')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Residual (Pred − Actual)')
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle('CatBoost V5: Residual Analysis (5-fold CV)', fontsize=13, y=1.02)
    fig.tight_layout()
    path = RESULTS_DIR / "calibration_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path.name}")


# %%
def plot_error_by_bin(y_actual, y_pred, n_bins=4):
    """Plot 3: Mean signed error by actual price quartile."""
    residuals = y_pred - y_actual
    bins = pd.qcut(y_actual, n_bins, duplicates='drop')

    df = pd.DataFrame({'actual': y_actual, 'residual': residuals, 'bin': bins})
    grouped = df.groupby('bin', observed=True).agg(
        mean_error=('residual', 'mean'),
        mae=('residual', lambda x: np.mean(np.abs(x))),
        count=('residual', 'count'),
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [f"{iv.left:,.0f}–{iv.right:,.0f}\n(n={int(grouped.loc[iv, 'count'])})"
              for iv in grouped.index]
    colors = [COLOR_OVER if e >= 0 else COLOR_UNDER for e in grouped['mean_error']]

    bars = ax.bar(range(len(grouped)), grouped['mean_error'], color=colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='gray', linewidth=1, linestyle='--')

    for bar, val in zip(bars, grouped['mean_error']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:+,.0f}', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

    ax.set_xlabel('Actual Price Range (CNY)')
    ax.set_ylabel('Mean Signed Error (CNY)')
    ax.set_title('CatBoost V5: Prediction Error by Price Quartile (5-fold CV)\n'
                 'Green = overestimation, Red = underestimation')

    fig.tight_layout()
    path = RESULTS_DIR / "calibration_error_by_bin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


# %%
def plot_calibration_curve(y_actual, y_pred, n_bins=5):
    """Plot 4: Regression calibration curve — mean predicted vs mean actual per bin."""
    try:
        bins = pd.qcut(y_pred, n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(y_pred, n_bins)

    df = pd.DataFrame({'actual': y_actual, 'pred': y_pred, 'bin': bins})
    grouped = df.groupby('bin', observed=True).agg(
        mean_pred=('pred', 'mean'),
        mean_actual=('actual', 'mean'),
        count=('actual', 'count'),
    ).sort_values('mean_pred')

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(grouped['mean_pred'], grouped['mean_actual'],
            'o-', color=COLOR_RAW, markersize=8, linewidth=2, label='CatBoost V5')

    lo = 0
    hi = max(grouped['mean_pred'].max(), grouped['mean_actual'].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1, label='Perfect calibration')

    for _, row in grouped.iterrows():
        ax.annotate(f'n={int(row["count"])}',
                    (row['mean_pred'], row['mean_actual']),
                    textcoords='offset points', xytext=(5, 8), fontsize=8)

    ax.set_xlabel('Mean Predicted Price (CNY)')
    ax.set_ylabel('Mean Actual Price (CNY)')
    ax.set_title('CatBoost V5: Calibration Curve (5-fold CV)\n'
                 'Points above diagonal = model predicts too low')
    ax.legend(loc='upper left')

    fig.tight_layout()
    path = RESULTS_DIR / "calibration_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


# %%
def plot_before_after(y_actual, y_pred):
    """Plot 5: Before vs after linear recalibration — pred-vs-actual comparison."""
    cal_params = fit_calibration(y_pred, y_actual)
    y_cal = apply_calibration(y_pred, cal_params)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    lo = 0
    hi = max(y_actual.max(), y_pred.max(), y_cal.max()) * 1.05

    for ax, preds, color, title in [
        (axes[0], y_pred, COLOR_RAW, 'Before Calibration (Raw)'),
        (axes[1], y_cal, COLOR_CAL, 'After Linear Calibration'),
    ]:
        ax.scatter(y_actual, preds, alpha=0.6, edgecolors='white',
                   linewidth=0.5, s=50, color=color)
        ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1)
        ax.set_xlabel('Actual Price (CNY)')
        ax.set_ylabel('Predicted Price (CNY)')
        ax.set_title(title)
        ax.text(0.95, 0.05, _metrics_text(y_actual, preds),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    fig.suptitle('CatBoost V5: Linear Recalibration Effect (5-fold CV)', fontsize=13, y=1.02)
    fig.tight_layout()
    path = RESULTS_DIR / "calibration_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path.name}")

    return cal_params


# %%
def print_summary(y_actual, y_pred, cal_params):
    """Print text summary of metrics and calibration parameters."""
    y_cal = apply_calibration(y_pred, cal_params)

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print(f"\nRaw model:        {_metrics_text(y_actual, y_pred)}")
    print(f"After calibration: {_metrics_text(y_actual, y_cal)}")

    r2_raw = r2_score(y_actual, y_pred)
    r2_cal = r2_score(y_actual, y_cal)
    mae_raw = mean_absolute_error(y_actual, y_pred)
    mae_cal = mean_absolute_error(y_actual, y_cal)
    print(f"\nImprovement:  R² {r2_raw:.3f} → {r2_cal:.3f} ({r2_cal - r2_raw:+.3f})")
    print(f"              MAE {mae_raw:,.0f} → {mae_cal:,.0f} ({mae_cal - mae_raw:+,.0f} CNY)")

    print(f"\nCalibration params:")
    print(f"  slope     = {cal_params['slope']:.4f}")
    print(f"  intercept = {cal_params['intercept']:,.2f}")
    if cal_params['slope'] > 1:
        print("  → slope > 1 confirms regression-to-the-mean (model under-disperses)")
    print("=" * 60)


# %%
def main():
    print("=" * 60)
    print("CALIBRATION ANALYSIS: CatBoost V5 Regression-to-Mean Diagnostic")
    print("=" * 60)

    y_actual, y_pred = get_cv_predictions()

    print("\nGenerating diagnostic plots...")
    plot_pred_vs_actual(y_actual, y_pred)
    plot_residuals(y_actual, y_pred)
    plot_error_by_bin(y_actual, y_pred)
    plot_calibration_curve(y_actual, y_pred)
    cal_params = plot_before_after(y_actual, y_pred)

    print_summary(y_actual, y_pred, cal_params)

    # Save calibration for the active CatBoost model
    save_calibration(cal_params, "v5_catboost_tab")


if __name__ == "__main__":
    main()
