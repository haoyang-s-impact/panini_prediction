"""Train and save the production V4 XGBoost model.

Trains on ALL data (no hold-out) and registers as the active model in the registry.

Usage:
    python -m models.train_production_model          # Train and register
    python -m models.train_production_model --info   # Show all registered models
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from models.train_price_regressor_v4 import (
    PARAM_DIST,
    _parse_card_year,
    load_and_prepare_data,
)
from models.registry import register_model, print_models


def train_and_save() -> str:
    """Train V4 XGBoost on all data and register as active model.

    Returns the model_id.
    """
    csv_path = str(PROJECT_ROOT / "output" / "panini_cards_extracted.csv")

    # Load and prepare data (V4: 24 features, log target)
    print("Loading data...")
    X, y, y_log = load_and_prepare_data(csv_path)
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Price range: {y.min():,.0f} - {y.max():,.0f} CNY")

    # Tune hyperparameters via cross-validation
    print("\nTuning hyperparameters (20 iterations, 3-fold CV)...")
    base_model = XGBRegressor(random_state=42, enable_categorical=True)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DIST,
        n_iter=20,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y_log)
    best_params = search.best_params_
    cv_r2 = search.best_score_

    print(f"\nBest CV R2 (log-space): {cv_r2:.4f}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Train final model on ALL data with best params
    print("\nTraining final model on all data...")
    model = XGBRegressor(**best_params, random_state=42, enable_categorical=True)
    model.fit(X, y_log)

    # Build metadata
    cat_cols = [col for col in X.columns if X[col].dtype.name == "category"]
    category_mappings = {col: list(X[col].cat.categories) for col in cat_cols}

    metadata = {
        "version": "v4",
        "pipeline_type": "ocr_tabular",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
        "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        "category_mappings": category_mappings,
        "best_params": best_params,
        "cv_r2_log": float(cv_r2),
        "target": "price_cny",
        "target_transform": "log1p",
    }

    # Register as active model
    model_id = register_model(
        model=model,
        model_id="v4_xgb_ocr_tabular",
        version="v4",
        framework="xgboost",
        pipeline_type="ocr_tabular",
        description="V4 XGBoost, 24 tabular features, log1p target, trained on ALL data",
        metrics={"cv_r2_log": float(cv_r2)},
        metadata=metadata,
        set_active=True,
    )

    return model_id


if __name__ == "__main__":
    if "--info" in sys.argv:
        print_models()
    else:
        train_and_save()
