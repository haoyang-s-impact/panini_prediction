# %%
"""
CatBoost V5: Tabular features + image embeddings.

Uses V4's 24 tabular features plus PCA-reduced image embeddings.
Trains on raw price_cny target (log-transform hurts per Session 1).
CatBoost requires categorical columns as string type + cat_features indices.
"""

import argparse
from catboost import CatBoostRegressor
from data.data_utils import merge_features
from models.model_utils import (
    tune_hyperparameters, run_trials, show_feature_importance,
    prepare_for_catboost,
)

# %%
# CatBoost-specific hyperparameter search space
PARAM_DIST = {
    'iterations': [100, 200, 300, 500],
    'depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'l2_leaf_reg': [1, 3, 5, 7],           # CatBoost L2 on leaves
    'min_data_in_leaf': [1, 3, 5],
}


# %%
def main(pca_n=30, no_emb=False, register=False):
    include_emb = not no_emb
    mode = "tabular only" if no_emb else f"tabular + PCA-{pca_n}"
    print(f"=== CatBoost V5 ({mode}) ===\n")

    X, y, cat_cols = merge_features(
        pca_n=pca_n,
        include_tabular=True,
        include_embeddings=include_emb,
    )
    # Convert categoricals for CatBoost
    X, cat_indices = prepare_for_catboost(X, cat_cols)
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Categorical indices: {cat_indices}")
    print(f"Price range: {y.min():,.0f} - {y.max():,.0f} CNY\n")

    # Tune
    print("Tuning hyperparameters (20 iterations, 3-fold CV)...")
    base_model = CatBoostRegressor(
        random_seed=42, verbose=0, bootstrap_type='Bernoulli',
    )
    best_params = tune_hyperparameters(
        base_model, PARAM_DIST, X, y, cat_features=cat_indices,
    )

    # Evaluate
    print("\nEvaluating over 10 trials...")

    def model_factory(random_state):
        return CatBoostRegressor(
            **best_params,
            random_seed=random_state,
            verbose=0,
            bootstrap_type='Bernoulli',
            cat_features=cat_indices,
        )

    avg, results, best_model = run_trials(model_factory, X, y)
    show_feature_importance(best_model, X.columns.tolist())

    if register:
        from models.registry import register_model

        suffix = "tab" if no_emb else f"pca{pca_n}"
        model_id = f"v5_catboost_{suffix}"

        metadata = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": X.columns.tolist(),
            "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
            "cat_indices": cat_indices,
            "target": "price_cny",
            "target_transform": "none",
        }
        if best_params:
            metadata["best_params"] = best_params

        register_model(
            model=best_model,
            model_id=model_id,
            version="v5",
            framework="catboost",
            pipeline_type=f"ocr_tabular{'_' + suffix if not no_emb else ''}",
            description=f"V5 CatBoost ({mode}), 10-trial best",
            metrics=avg,
            metadata=metadata,
        )

    return avg, results, best_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatBoost V5 trainer")
    parser.add_argument("--pca", type=int, default=30, choices=[30, 50, 64],
                        help="PCA embedding dimensions (default: 30)")
    parser.add_argument("--no-emb", action="store_true",
                        help="Tabular features only (no embeddings)")
    parser.add_argument("--register", action="store_true",
                        help="Register best model in the model registry")
    args = parser.parse_args()
    main(pca_n=args.pca, no_emb=args.no_emb, register=args.register)
