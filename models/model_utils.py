"""
Model training utilities for V5 tree models.

Provides generic evaluation loop, hyperparameter tuning, and
framework-specific helpers (e.g. CatBoost categorical prep).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def tune_hyperparameters(base_model, param_dist, X, y,
                         n_iter=20, cv=3, cat_features=None):
    """Find best hyperparameters using RandomizedSearchCV.

    Args:
        base_model: Unfitted estimator instance
        param_dist: Dict of parameter distributions
        X, y: Training data (raw target)
        n_iter: Number of random search iterations
        cv: Number of cross-validation folds
        cat_features: For CatBoost, list of categorical feature indices

    Returns:
        best_params: Dict of best parameters
    """
    fit_params = {}
    if cat_features is not None:
        fit_params['cat_features'] = cat_features

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y, **fit_params)

    print("\n=== Best Hyperparameters ===")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  Best CV R2: {search.best_score_:.4f}")

    return search.best_params_


def run_trials(model_factory, X, y, n_trials=10):
    """Run model over multiple random seeds and report average metrics.

    Args:
        model_factory: callable(random_state) -> fitted model
            The factory should handle fitting internally or return an
            unfitted model that will be fit here.
        X: Feature DataFrame
        y: Target Series (raw price_cny)
        n_trials: Number of random seed trials

    Returns:
        avg_metrics: Dict with averaged MAE, RMSE, R2
        all_results: List of per-trial metric dicts
        best_model: Model with highest R2
    """
    results = []
    best_model = None
    best_r2 = -np.inf

    for i in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        model = model_factory(random_state=i)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'mae': mae, 'rmse': rmse, 'r2': r2})

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

        print(f"  Trial {i+1:2d}: MAE={mae:,.0f}, RMSE={rmse:,.0f}, R2={r2:.4f}")

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}

    print(f"\n=== Average over {n_trials} trials ===")
    print(f"  MAE:  {avg['mae']:,.0f} CNY")
    print(f"  RMSE: {avg['rmse']:,.0f} CNY")
    print(f"  R2:   {avg['r2']:.4f}")

    return avg, results, best_model


def prepare_for_catboost(X, cat_cols):
    """Convert categorical columns from pandas category to string for CatBoost.

    CatBoost requires categorical columns as string type, not pandas category.

    Args:
        X: Feature DataFrame
        cat_cols: List of categorical column names

    Returns:
        X_cb: Converted DataFrame
        cat_indices: List of column indices for CatBoost's cat_features param
    """
    X_cb = X.copy()
    cat_indices = []
    for col in cat_cols:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype(str)
            cat_indices.append(X_cb.columns.get_loc(col))
    return X_cb, cat_indices


def show_feature_importance(model, feature_names, top_n=20):
    """Print sorted feature importance.

    Works with XGBoost, LightGBM, and CatBoost (all have feature_importances_).
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print(f"\n=== Top {min(top_n, len(feature_names))} Feature Importance ===")
    for rank, idx in enumerate(sorted_idx[:top_n]):
        name = feature_names[idx]
        marker = " *EMB*" if name.startswith('emb_') else ""
        print(f"  {rank+1:2d}. {name:25s} {importance[idx]:.4f}{marker}")
