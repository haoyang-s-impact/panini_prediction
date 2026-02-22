"""
XGBoost regressor V4: Derived features + log-transformed target.

Changes over V3:
- 24 features (V3's 13 + 11 derived from Session 1)
- Log-transformed target: y = log1p(price_cny), inverse via expm1
- Fixed card_year parsing: "2022-23" → 2022 instead of NaN → 0
- NaN-aware: uses XGBoost native missing value handling for numerics
- Dual reporting: log-space R² + real-space MAE/RMSE/R²
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_card_year(val):
    """Parse card_year like '2022-23' → 2022.0, or '2022' → 2022.0."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Take first 4 digits (handles "2022-23", "2022/23", "2022")
    match = __import__('re').match(r'(\d{4})', s)
    if match:
        return float(match.group(1))
    return np.nan


def load_and_prepare_data(csv_path: str):
    """Load data and prepare V4 feature set (24 features, log target)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['price_cny'])

    # --- Numeric features (NaN left as NaN for XGBoost native handling) ---
    numeric_cols = ['serial_max', 'grade', 'card_year', 'bid_times',
                    'rarity_ratio', 'day_of_week', 'hour_of_day']

    # --- Boolean features ---
    bool_cols = ['is_rookie', 'is_autograph', 'has_patch', 'is_refractor',
                 'rookie_auto', 'rookie_patch', 'is_rpa', 'is_numbered',
                 'is_1of1', 'is_base', 'is_weekend']

    # --- Categorical features ---
    cat_cols = ['player_name', 'team', 'card_series', 'parallel_type',
                'grading_company', 'player_tier']

    X = df[numeric_cols + bool_cols + cat_cols].copy()
    y = df['price_cny'].copy()

    # Fix card_year: "2022-23" → 2022.0 (V3 bug: pd.to_numeric("2022-23") → NaN → 0)
    X['card_year'] = df['card_year'].apply(_parse_card_year)

    # Numeric cols: coerce to float, leave NaN (XGBoost handles missing natively)
    for col in numeric_cols:
        if col != 'card_year':  # card_year already handled
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Boolean cols: fill NaN with False, convert to int
    for col in bool_cols:
        X[col] = X[col].fillna(False).astype(int)

    # Categorical cols: fill NaN with 'unknown', convert to category dtype
    for col in cat_cols:
        X[col] = X[col].fillna('unknown').astype('category')

    # Log-transform target
    y_log = np.log1p(y)

    return X, y, y_log


# Hyperparameter search space (same as V3)
PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}


def tune_hyperparameters(X, y_log, n_iter=20, cv=3):
    """Find best hyperparameters using RandomizedSearchCV on log-space target."""
    model = XGBRegressor(random_state=42, enable_categorical=True)

    search = RandomizedSearchCV(
        model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X, y_log)

    print("\n=== Best Hyperparameters ===")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV R2 (log-space): {search.best_score_:.4f}")

    return search.best_params_


def train_and_evaluate(X, y, y_log, params, random_state):
    """Train on log-space, evaluate in both log-space and real-space."""
    X_train, X_test, y_log_train, y_log_test, y_train, y_test = train_test_split(
        X, y_log, y, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(
        **params,
        random_state=random_state,
        enable_categorical=True
    )
    model.fit(X_train, y_log_train)

    # Predict in log-space
    y_log_pred = model.predict(X_test)

    # Log-space metrics
    r2_log = r2_score(y_log_test, y_log_pred)

    # Inverse transform to real-space
    y_pred_real = np.expm1(y_log_pred)
    y_pred_real = np.maximum(y_pred_real, 0)  # clamp negatives

    # Real-space metrics
    mae = mean_absolute_error(y_test, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_real))
    r2 = r2_score(y_test, y_pred_real)

    return model, {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'r2_log': r2_log,
    }


def run_multiple_trials(X, y, y_log, params, n_trials=10):
    """Run model n_trials times and return average metrics."""
    results = []
    best_model = None
    best_r2 = -np.inf

    for i in range(n_trials):
        model, metrics = train_and_evaluate(X, y, y_log, params, random_state=i)
        results.append(metrics)
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model
        print(f"Trial {i+1:2d}: MAE={metrics['mae']:,.2f}, RMSE={metrics['rmse']:,.2f}, "
              f"R2={metrics['r2']:.4f}, R2_log={metrics['r2_log']:.4f}")

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}

    print("\n=== Average over 10 trials ===")
    print(f"MAE:     {avg['mae']:,.2f} CNY")
    print(f"RMSE:    {avg['rmse']:,.2f} CNY")
    print(f"R2:      {avg['r2']:.4f}  (real-space)")
    print(f"R2_log:  {avg['r2_log']:.4f}  (log-space)")

    return best_model, avg


def show_feature_importance(model, feature_names):
    """Print feature importance."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("\n=== Feature Importance ===")
    for idx in sorted_idx:
        print(f"{feature_names[idx]:20} {importance[idx]:.4f}")


def main(register=False):
    csv_path = str(PROJECT_ROOT / "output/panini_cards_extracted.csv")

    print("Loading data...")
    X, y, y_log = load_and_prepare_data(csv_path)
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Price range: {y.min():,.0f} - {y.max():,.0f} CNY")
    print(f"Log price range: {y_log.min():.2f} - {y_log.max():.2f}")

    print("\nTuning hyperparameters (20 iterations, 3-fold CV)...")
    best_params = tune_hyperparameters(X, y_log, n_iter=20, cv=3)

    print("\nEvaluating tuned model over 10 trials...")
    best_model, avg_metrics = run_multiple_trials(X, y, y_log, best_params, n_trials=10)

    show_feature_importance(best_model, X.columns.tolist())

    if register:
        from models.registry import register_model, build_metadata

        metadata = build_metadata(
            X, target_transform="log1p", best_params=best_params,
        )
        register_model(
            model=best_model,
            model_id="v4_xgb_ocr_tabular_research",
            version="v4",
            framework="xgboost",
            pipeline_type="ocr_tabular",
            description="V4 XGBoost, 24 features, log1p target, 10-trial best (80/20 split)",
            metrics=avg_metrics,
            metadata=metadata,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XGBoost V4 trainer")
    parser.add_argument("--register", action="store_true",
                        help="Register best model in the model registry")
    args = parser.parse_args()
    main(register=args.register)
