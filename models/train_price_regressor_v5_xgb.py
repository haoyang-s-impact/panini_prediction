# %%
"""
XGBoost V5: Tabular features + image embeddings.

Uses V4's 24 tabular features plus PCA-reduced image embeddings.
Trains on raw price_cny target (log-transform hurts per Session 1).
"""

import argparse
from xgboost import XGBRegressor
from data.data_utils import merge_features
from models.model_utils import tune_hyperparameters, run_trials, show_feature_importance

# %%
# Hyperparameter search space (same as V4)
PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}


# %%
def main(pca_n=30, no_emb=False):
    include_emb = not no_emb
    mode = "tabular only" if no_emb else f"tabular + PCA-{pca_n}"
    print(f"=== XGBoost V5 ({mode}) ===\n")

    X, y, cat_cols = merge_features(
        pca_n=pca_n,
        include_tabular=True,
        include_embeddings=include_emb,
    )
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Price range: {y.min():,.0f} - {y.max():,.0f} CNY\n")

    # Tune
    print("Tuning hyperparameters (20 iterations, 3-fold CV)...")
    base_model = XGBRegressor(random_state=42, enable_categorical=True)
    best_params = tune_hyperparameters(base_model, PARAM_DIST, X, y)

    # Evaluate
    print("\nEvaluating over 10 trials...")

    def model_factory(random_state):
        return XGBRegressor(
            **best_params,
            random_state=random_state,
            enable_categorical=True,
        )

    avg, results, best_model = run_trials(model_factory, X, y)
    show_feature_importance(best_model, X.columns.tolist())

    return avg, results, best_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost V5 trainer")
    parser.add_argument("--pca", type=int, default=30, choices=[30, 50, 64],
                        help="PCA embedding dimensions (default: 30)")
    parser.add_argument("--no-emb", action="store_true",
                        help="Tabular features only (no embeddings)")
    args = parser.parse_args()
    main(pca_n=args.pca, no_emb=args.no_emb)
