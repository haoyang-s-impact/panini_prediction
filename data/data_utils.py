"""
Data engineering utilities for V5 models.

Handles loading tabular features, image embeddings, and merging them
for tree model training. Reuses V4's exact feature preparation logic.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Feature column definitions (same as V4) ---
NUMERIC_COLS = ['serial_max', 'grade', 'card_year', 'bid_times',
                'rarity_ratio', 'day_of_week', 'hour_of_day']

BOOL_COLS = ['is_rookie', 'is_autograph', 'has_patch', 'is_refractor',
             'rookie_auto', 'rookie_patch', 'is_rpa', 'is_numbered',
             'is_1of1', 'is_base', 'is_weekend']

CAT_COLS = ['player_name', 'team', 'card_series', 'parallel_type',
            'grading_company', 'player_tier']


def _parse_card_year(val):
    """Parse card_year like '2022-23' -> 2022.0, or '2022' -> 2022.0."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    match = re.match(r'(\d{4})', s)
    if match:
        return float(match.group(1))
    return np.nan


def load_tabular_data(csv_path=None):
    """Load and prepare V4's 24 tabular features + raw target.

    Returns:
        X: DataFrame with 24 features (numeric, bool, categorical)
        y: Series of raw price_cny
        images: Series of image filenames (for embedding join)
        feature_info: dict with 'numeric_cols', 'bool_cols', 'cat_cols'
    """
    csv_path = csv_path or str(PROJECT_ROOT / "output/panini_cards_extracted.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['price_cny'])

    # Reset index after dropna so all downstream DataFrames align
    df = df.reset_index(drop=True)

    X = df[NUMERIC_COLS + BOOL_COLS + CAT_COLS].copy()
    y = df['price_cny'].copy()
    images = df['image'].copy()

    # Fix card_year: "2022-23" -> 2022.0
    X['card_year'] = df['card_year'].apply(_parse_card_year)

    # Numeric cols: coerce to float, leave NaN for native tree handling
    for col in NUMERIC_COLS:
        if col != 'card_year':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Boolean cols: fill NaN with False, convert to int
    pd.set_option('future.no_silent_downcasting', True)
    for col in BOOL_COLS:
        X[col] = X[col].fillna(False).astype(int)

    # Categorical cols: fill NaN with 'unknown', convert to category dtype
    for col in CAT_COLS:
        X[col] = X[col].fillna('unknown').astype('category')

    feature_info = {
        'numeric_cols': NUMERIC_COLS,
        'bool_cols': BOOL_COLS,
        'cat_cols': CAT_COLS,
    }

    return X, y, images, feature_info


def load_embeddings(pca_n=30):
    """Load image embedding CSV for given PCA variant.

    Args:
        pca_n: Number of PCA components (30, 50, or 64)

    Returns:
        DataFrame with 'image' column + emb_0..emb_{pca_n-1}
    """
    path = PROJECT_ROOT / f"output/image_embeddings_pca{pca_n}.csv"
    return pd.read_csv(path)


def merge_features(csv_path=None, pca_n=30,
                   include_tabular=True, include_embeddings=True):
    """Load tabular data, optionally merge with image embeddings.

    Args:
        csv_path: Path to panini_cards_extracted.csv (default: output/)
        pca_n: PCA variant to use (30, 50, or 64)
        include_tabular: Include 24 tabular features
        include_embeddings: Include PCA embedding features

    Returns:
        X: Feature DataFrame
        y: Target Series (raw price_cny)
        cat_cols: List of categorical column names in X
    """
    X_tab, y, images, feature_info = load_tabular_data(csv_path)

    parts = []
    cat_cols = []

    if include_tabular:
        parts.append(X_tab)
        cat_cols = feature_info['cat_cols']

    if include_embeddings:
        emb_df = load_embeddings(pca_n)
        # Build a mapping from image filename to embedding row
        emb_merged = pd.DataFrame({'image': images}).merge(
            emb_df, on='image', how='left'
        )
        emb_cols = [c for c in emb_merged.columns if c.startswith('emb_')]
        # Verify 1:1 match
        n_missing = emb_merged[emb_cols[0]].isna().sum()
        if n_missing > 0:
            print(f"WARNING: {n_missing} images have no matching embeddings")
        parts.append(emb_merged[emb_cols].reset_index(drop=True))

    X = pd.concat(parts, axis=1)

    assert len(X) == len(y), f"Row count mismatch: {len(X)} vs {len(y)}"

    return X, y, cat_cols
