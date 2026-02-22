"""Single-image price prediction pipeline.

Takes image bytes → runs OCR → extracts features → predicts price.

Usage:
    from serve.inference import predict_from_image
    result = predict_from_image(open("pics/test.jpg", "rb").read())
    print(result["predicted_price_cny"])
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_utils import _parse_card_year
from data.panini_card_ocr_etl import (
    extract_bid_times,
    extract_card_series,
    extract_descriptors,
    extract_grading,
    extract_player_name,
    extract_price,
    extract_serial_numbers,
    extract_team,
    extract_temporal,
    compute_derived_features,
)
from serve.model_registry import load_model

# Default model ID — None means use registry active model.
# Can be overridden by callers (e.g., app.py --model).
MODEL_ID = None

# Lazy-loaded singleton for EasyOCR reader
_ocr_reader = None


def _get_ocr_reader():
    """Get or initialize the EasyOCR reader (singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=use_gpu, verbose=False)
    return _ocr_reader


def _run_ocr(image_bytes: bytes) -> list[dict]:
    """Run EasyOCR on image bytes, return list of {text, confidence, bbox}."""
    reader = _get_ocr_reader()

    # EasyOCR needs a file path or numpy array
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        results = reader.readtext(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # EasyOCR returns: (bbox, text, confidence)
    return [
        {"text": text, "confidence": float(conf), "bbox": bbox}
        for bbox, text, conf in results
    ]


def _extract_features_from_ocr(ocr_lines: list[dict]) -> dict:
    """Convert raw OCR output to a flat feature dict (V4 schema).

    Reuses extraction functions from panini_card_ocr_etl.py.
    """
    # Convert to the tuple format the ETL functions expect: (text, confidence, bbox_str)
    ocr_entries = [
        (line["text"], line["confidence"], str(line["bbox"]))
        for line in ocr_lines
    ]

    # Run each extraction function
    player_name, player_conf = extract_player_name(ocr_entries)
    card_year, card_series = extract_card_series(ocr_entries)
    serial_data = extract_serial_numbers(ocr_entries)
    grading_data = extract_grading(ocr_entries)
    end_time, product_id = extract_temporal(ocr_entries)
    descriptor_data = extract_descriptors(ocr_entries)
    team = extract_team(ocr_entries)

    # Extract prices and bids from raw text
    price_cny = None
    price_usd = None
    shipping_price = None
    bid_times = None

    for line in ocr_lines:
        text = line["text"]
        if price_cny is None and price_usd is None and shipping_price is None:
            prices = extract_price(text)
            if prices[0] or prices[1] or prices[2]:
                price_cny, price_usd, shipping_price = prices
        if bid_times is None:
            bid_times = extract_bid_times(text)

    return {
        "player_name": player_name,
        "team": team,
        "card_year": card_year,
        "card_series": card_series,
        "serial_max": serial_data["serial_max"],
        "grade": grading_data["grade"],
        "grading_company": grading_data["grading_company"],
        "is_rookie": descriptor_data["is_rookie"],
        "is_autograph": descriptor_data["is_autograph"],
        "has_patch": descriptor_data["has_patch"],
        "is_refractor": descriptor_data["is_refractor"],
        "parallel_type": descriptor_data["parallel_type"],
        "card_features": descriptor_data["card_features"],
        "bid_times": bid_times,
        "end_time": end_time,
        "price_cny": price_cny,
        "price_usd": price_usd,
        "shipping_price": shipping_price,
    }


def _features_to_dataframe(features: dict, metadata: dict) -> pd.DataFrame:
    """Convert feature dict to a single-row DataFrame matching V4 training schema.

    Applies the same preprocessing as load_and_prepare_data() in V4.
    """
    # First build a wide-format DataFrame (like panini_cards_extracted.csv)
    raw_df = pd.DataFrame([features])

    # Compute derived features (adds player_tier, rarity_ratio, etc.)
    raw_df = compute_derived_features(raw_df)

    # Now select and preprocess exactly the V4 features
    feature_names = metadata["feature_names"]
    feature_dtypes = metadata["feature_dtypes"]

    X = pd.DataFrame()
    for col in feature_names:
        if col in raw_df.columns:
            X[col] = raw_df[col]
        else:
            X[col] = pd.Series([np.nan])

    # Apply V4 preprocessing per dtype
    for col in feature_names:
        dtype = feature_dtypes.get(col, "")

        if col == "card_year":
            X[col] = X[col].apply(_parse_card_year)
        elif "float" in dtype.lower() or "int" in dtype.lower():
            if dtype == "category":
                continue
            X[col] = pd.to_numeric(X[col], errors="coerce")
        elif dtype == "category":
            # Restore category dtype with the same categories as training
            categories = metadata.get("category_mappings", {}).get(col, [])
            val = X[col].iloc[0]
            if pd.isna(val) or val is None:
                val = "unknown"
            X[col] = pd.Categorical([val], categories=categories + (["unknown"] if "unknown" not in categories else []))

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

    Returns dict with:
        - predicted_price_cny: float
        - extracted_features: dict (human-readable feature values)
        - ocr_lines: list of {text, confidence} dicts
    """
    # 1. Run OCR
    ocr_lines = _run_ocr(image_bytes)

    if not ocr_lines:
        return {
            "predicted_price_cny": None,
            "extracted_features": {},
            "ocr_lines": [],
            "error": "No text detected in image.",
        }

    # 2. Extract features
    features = _extract_features_from_ocr(ocr_lines)

    # 3. Load model and prepare features
    model, metadata = load_model(MODEL_ID)
    X = _features_to_dataframe(features, metadata)

    # 4. Predict — apply inverse transform based on metadata
    raw_pred = model.predict(X)
    if metadata.get("target_transform") == "log1p":
        price_pred = float(np.expm1(raw_pred[0]))
    else:
        price_pred = float(raw_pred[0])
    price_pred = max(price_pred, 0.0)

    # 5. Build human-readable features for display
    display_features = {k: v for k, v in features.items() if v is not None}

    return {
        "predicted_price_cny": round(price_pred, 2),
        "extracted_features": display_features,
        "ocr_lines": [
            {"text": line["text"], "confidence": round(line["confidence"], 3)}
            for line in ocr_lines
        ],
    }
