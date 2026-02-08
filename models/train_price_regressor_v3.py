"""
XGBoost regressor V3: Hyperparameter tuning with RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_and_prepare_data(csv_path: str):
    """Load data and prepare extended feature set."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['price_cny'])

    numeric_cols = ['serial_max', 'grade', 'card_year', 'bid_times']
    bool_cols = ['is_rookie', 'is_autograph', 'has_patch', 'is_refractor']
    cat_cols = ['player_name', 'team', 'card_series', 'parallel_type', 'grading_company']

    X = df[numeric_cols + bool_cols + cat_cols].copy()
    y = df['price_cny'].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X[bool_cols] = X[bool_cols].fillna(False).astype(int)

    for col in cat_cols:
        X[col] = X[col].fillna('unknown').astype('category')

    return X, y


# Hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}


def tune_hyperparameters(X, y, n_iter=20, cv=3):
    """Find best hyperparameters using RandomizedSearchCV."""
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
    search.fit(X, y)

    print("\n=== Best Hyperparameters ===")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV R2: {search.best_score_:.4f}")

    return search.best_params_


def train_and_evaluate(X, y, params, random_state):
    """Train and evaluate model with tuned params."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(
        **params,
        random_state=random_state,
        enable_categorical=True
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def run_multiple_trials(X, y, params, n_trials=10):
    """Run model n_trials times and return average metrics."""
    results = []
    best_model = None
    best_r2 = -np.inf

    for i in range(n_trials):
        model, metrics = train_and_evaluate(X, y, params, random_state=i)
        results.append(metrics)
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model
        print(f"Trial {i+1:2d}: MAE={metrics['mae']:,.2f}, RMSE={metrics['rmse']:,.2f}, R2={metrics['r2']:.4f}")

    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_r2 = np.mean([r['r2'] for r in results])

    print("\n=== Average over 10 trials ===")
    print(f"MAE:  {avg_mae:,.2f} CNY")
    print(f"RMSE: {avg_rmse:,.2f} CNY")
    print(f"R2:   {avg_r2:.4f}")

    return best_model, {'mae': avg_mae, 'rmse': avg_rmse, 'r2': avg_r2}


def show_feature_importance(model, feature_names):
    """Print feature importance."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("\n=== Feature Importance ===")
    for idx in sorted_idx:
        print(f"{feature_names[idx]:15} {importance[idx]:.4f}")


def main():
    csv_path = "output/panini_cards_extracted.csv"

    print("Loading data...")
    X, y = load_and_prepare_data(csv_path)
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Price range: {y.min():,.0f} - {y.max():,.0f} CNY")

    print("\nTuning hyperparameters (20 iterations, 3-fold CV)...")
    best_params = tune_hyperparameters(X, y, n_iter=20, cv=3)

    print("\nEvaluating tuned model over 10 trials...")
    best_model, avg_metrics = run_multiple_trials(X, y, best_params, n_trials=10)

    show_feature_importance(best_model, X.columns.tolist())


if __name__ == "__main__":
    main()
