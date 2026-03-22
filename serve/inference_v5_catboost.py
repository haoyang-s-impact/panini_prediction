"""Single-image price prediction pipeline for V5 CatBoost.

Takes image bytes → runs OCR → extracts features → predicts price.
Uses raw target (no log transform) and CatBoost categorical handling.

Usage:
    from serve.inference_v5_catboost import predict_from_image
    result = predict_from_image(open("pics/test.jpg", "rb").read())
    print(result["predicted_price_cny"])
"""

import numpy as np
import pandas as pd

from data.data_utils import _parse_card_year
from data.panini_card_ocr_etl import compute_derived_features
from models.calibration import apply_calibration, load_calibration
from serve.inference import _run_ocr, _extract_features_from_ocr
from serve.model_registry import load_model

# Default model ID — can be overridden by callers (e.g., app.py --model)
MODEL_ID = "v5_catboost_tab"


def _features_to_dataframe_catboost(features: dict, metadata: dict) -> pd.DataFrame:
    """Convert feature dict to a single-row DataFrame for CatBoost.

    CatBoost requires categorical columns as string type (not pandas category).
    Uses cat_indices from metadata to identify which columns are categorical.
    """
    # Build wide-format DataFrame and compute derived features
    raw_df = pd.DataFrame([features])
    raw_df = compute_derived_features(raw_df)

    feature_names = metadata["feature_names"]
    feature_dtypes = metadata["feature_dtypes"]

    X = pd.DataFrame()
    for col in feature_names:
        if col in raw_df.columns:
            X[col] = raw_df[col]
        else:
            X[col] = pd.Series([np.nan])

    # Determine which columns are categorical from metadata
    cat_indices = metadata.get("cat_indices", [])
    cat_col_names = [feature_names[i] for i in cat_indices]

    for col in feature_names:
        dtype = feature_dtypes.get(col, "")

        if col == "card_year":
            X[col] = X[col].apply(_parse_card_year)
        elif col in cat_col_names:
            # CatBoost needs string type for categoricals
            val = X[col].iloc[0]
            if pd.isna(val) or val is None:
                val = "unknown"
            X[col] = str(val)
        elif "float" in dtype.lower() or "int" in dtype.lower():
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Boolean columns: fill NaN with False, convert to int
    bool_cols = [
        "is_rookie", "is_autograph", "has_patch", "is_refractor",
        "rookie_auto", "rookie_patch", "is_rpa", "is_numbered",
        "is_1of1", "is_base", "is_weekend",
    ]
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].fillna(False).astype(int)

    return X


def predict_from_image(image_bytes: bytes) -> dict:
    """Full prediction pipeline: image bytes → price prediction.

    Uses V5 CatBoost with raw target (no log transform).

    Returns dict with:
        - predicted_price_cny: float
        - extracted_features: dict (human-readable feature values)
        - ocr_lines: list of {text, confidence} dicts
    """
    # 1. Run OCR (shared with V4)
    ocr_lines = _run_ocr(image_bytes)

    if not ocr_lines:
        return {
            "predicted_price_cny": None,
            "extracted_features": {},
            "ocr_lines": [],
            "error": "No text detected in image.",
        }

    # 2. Extract features (shared with V4)
    features = _extract_features_from_ocr(ocr_lines)

    # 3. Load CatBoost model and prepare features
    model, metadata = load_model(MODEL_ID)
    X = _features_to_dataframe_catboost(features, metadata)

    # 4. Predict (raw target — no inverse transform needed)
    price_pred = float(model.predict(X)[0])
    price_pred = max(price_pred, 0.0)

    # 5. Apply linear calibration if available
    cal_params = load_calibration(MODEL_ID)
    if cal_params is not None:
        price_pred = float(apply_calibration(price_pred, cal_params))


    # 6. Build human-readable features for display
    display_features = {k: v for k, v in features.items() if v is not None}

    return {
        "predicted_price_cny": round(price_pred, 2),
        "extracted_features": display_features,
        "ocr_lines": [
            {"text": line["text"], "confidence": round(line["confidence"], 3)}
            for line in ocr_lines
        ],
    }
