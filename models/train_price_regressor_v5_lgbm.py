# %%
"""
LightGBM V5: Tabular features + image embeddings.

Uses V4's 24 tabular features plus PCA-reduced image embeddings.
Trains on raw price_cny target (log-transform hurts per Session 1).
LightGBM auto-detects pandas category dtype for categorical features.
"""

import argparse
from lightgbm import LGBMRegressor
from data.data_utils import merge_features
from models.model_utils import tune_hyperparameters, run_trials, show_feature_importance

# %%
# LightGBM-specific hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, -1],        # -1 = no limit (leaf-wise)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'num_leaves': [15, 31, 63, 127],       # LightGBM leaf-wise growth
    'reg_alpha': [0, 0.1, 1.0],            # L1 regularization
    'reg_lambda': [0, 0.1, 1.0],           # L2 regularization
}


# %%
def main(pca_n=30, no_emb=False):
    include_emb = not no_emb
    mode = "tabular only" if no_emb else f"tabular + PCA-{pca_n}"
    print(f"=== LightGBM V5 ({mode}) ===\n")

    X, y, cat_cols = merge_features(
        pca_n=pca_n,
        include_tabular=True,
        include_embeddings=include_emb,
    )
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Price range: {y.min():,.0f} - {y.max():,.0f} CNY\n")

    # Tune
    print("Tuning hyperparameters (20 iterations, 3-fold CV)...")
    base_model = LGBMRegressor(random_state=42, verbose=-1)
    best_params = tune_hyperparameters(base_model, PARAM_DIST, X, y)

    # Evaluate
    print("\nEvaluating over 10 trials...")

    def model_factory(random_state):
        return LGBMRegressor(
            **best_params,
            random_state=random_state,
            verbose=-1,
        )

    avg, results, best_model = run_trials(model_factory, X, y)
    show_feature_importance(best_model, X.columns.tolist())

    return avg, results, best_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM V5 trainer")
    parser.add_argument("--pca", type=int, default=30, choices=[30, 50, 64],
                        help="PCA embedding dimensions (default: 30)")
    parser.add_argument("--no-emb", action="store_true",
                        help="Tabular features only (no embeddings)")
    args = parser.parse_args()
    main(pca_n=args.pca, no_emb=args.no_emb)
