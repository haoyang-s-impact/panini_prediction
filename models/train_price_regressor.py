"""
Minimal XGBoost regressor to predict price_cny from Panini card features.
V1: Simple baseline using only boolean + numeric features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_and_prepare_data(csv_path: str):
    """Load data and prepare minimal feature set."""
    df = pd.read_csv(csv_path)

    # Drop rows with missing target
    df = df.dropna(subset=['price_cny'])

    # Select only simple features (no encoding needed)
    feature_cols = [
        'is_rookie', 'is_autograph', 'has_patch', 'is_refractor',
        'serial_max', 'grade'
    ]

    X = df[feature_cols].copy()
    y = df['price_cny'].copy()

    # Fill missing numeric values with 0
    X = X.fillna(0)

    # Convert boolean columns to int
    bool_cols = ['is_rookie', 'is_autograph', 'has_patch', 'is_refractor']
    for col in bool_cols:
        X[col] = X[col].astype(int)

    return X, y


def train_and_evaluate(X, y, random_state):
    """Train and evaluate model with given random state."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def run_multiple_trials(X, y, n_trials=10):
    """Run model n_trials times and return average metrics."""
    results = []
    best_model = None
    best_r2 = -np.inf

    for i in range(n_trials):
        model, metrics = train_and_evaluate(X, y, random_state=i)
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

    print("\nTraining 10 trials...")
    best_model, avg_metrics = run_multiple_trials(X, y, n_trials=10)

    show_feature_importance(best_model, X.columns.tolist())

    # best_model.save_model("output/price_cny_model.json")
    # print("\nBest model saved to output/price_cny_model.json")


if __name__ == "__main__":
    main()
