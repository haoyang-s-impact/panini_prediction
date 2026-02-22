"""Train and save the production V4 XGBoost model.

Trains on ALL data (no hold-out) and persists to models/saved/ for serving.

Usage:
    python -m models.train_production_model          # Train and save
    python -m models.train_production_model --info   # Show saved model metadata
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
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

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "saved"


def train_and_save(output_dir: Path | str | None = None) -> Path:
    """Train V4 XGBoost on all data and save model + metadata to disk.

    Returns the output directory path.
    """
    output_dir = Path(output_dir or DEFAULT_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")

    # Build metadata
    cat_cols = [col for col in X.columns if X[col].dtype.name == "category"]
    category_mappings = {}
    for col in cat_cols:
        category_mappings[col] = list(X[col].cat.categories)

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

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return output_dir


if __name__ == "__main__":
    if "--info" in sys.argv:
        model_path = DEFAULT_MODEL_DIR / "metadata.json"
        if model_path.exists():
            with open(model_path) as f:
                print("Saved model metadata:")
                print(json.dumps(json.load(f), indent=2))
        else:
            print(f"No saved model found at {DEFAULT_MODEL_DIR}.")
            print("Run `python -m models.train_production_model` to train and save.")
    else:
        train_and_save()
