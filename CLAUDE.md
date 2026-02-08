# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panini NBA trading card price prediction pipeline. Processes Chinese auction screenshots (from WeChat/闲鱼) through OCR, extracts structured card features, and trains XGBoost regressors to predict sale prices (CNY).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run OCR extraction (EasyOCR, processes pics/ → output/)
python panini_card_extractor_interactive.py

# Run feature extraction ETL (output/raw_ocr_output/ → output/panini_cards_extracted.csv)
python panini_card_ocr_etl.py

# Test feature extractors on sample OCR data
python test_extractors.py

# Train price prediction models (reads output/panini_cards_extracted.csv)
python models/train_price_regressor.py      # V1: boolean + numeric features only
python models/train_price_regressor_v2.py   # V2: adds categorical features (native XGBoost)
python models/train_price_regressor_v3.py   # V3: adds RandomizedSearchCV hyperparameter tuning
python models/train_price_regressor_v4.py   # V4: derived features + log-transformed target

# Run model comparison report (trains all versions, generates plots)
python models/model_comparison_report.py    # Outputs PNGs to output/

# Run price skewness analysis
python models/analyze_price_skewness.py     # Outputs PNG to output/
```

## Architecture

The pipeline has three stages:

**Stage 1: OCR** (`panini_card_extractor_interactive.py` or `extract_panini_info.py`)
- Reads auction screenshot images from `pics/`
- Uses EasyOCR (interactive version) or PaddleOCR (original version) to extract Chinese+English text
- Saves per-image raw OCR text files to `output/raw_ocr_output/` with text, confidence scores, and bounding box coordinates
- Also does basic regex-based categorization into card_identity, grading_info, auction_data

**Stage 2: Feature Extraction ETL** (`panini_card_ocr_etl.py`)
- Loads raw OCR text files from `output/raw_ocr_output/` into a long-format DataFrame (multiple rows per image)
- Aggregates into wide-format (one row per card) via `aggregate_card_features()`
- Uses spatial zones (Y-coordinate ranges) and confidence thresholds for position-based extraction
- Fuzzy-matches player names against `data/nba_players.py` using `rapidfuzz` (threshold: 85)
- Maps Chinese text to English (player names, team names, parallel colors, descriptors)
- Outputs: `output/panini_cards_extracted.csv` with ~37 columns (player, team, card series, grading, prices, boolean flags, derived features)
- `compute_derived_features()` adds 11 derived columns: player_tier, rarity_ratio, rookie_auto, rookie_patch, is_rpa, is_numbered, is_1of1, is_base, day_of_week, hour_of_day, is_weekend

**Stage 3: ML** (`models/train_price_regressor.py`, `_v2.py`, `_v3.py`, `_v4.py`)
- Target: `price_cny` (Chinese Yuan sale price)
- V1: 6 features (boolean flags + serial_max + grade), baseline XGBoost
- V2: 13 features (adds player_name, team, card_series, etc. as native categoricals)
- V3: Same features as V2 + RandomizedSearchCV over 20 iterations with 3-fold CV
- V4: 24 features (V3 + 11 derived), log-transformed target, fixed card_year parsing, NaN-aware numerics
- All versions run 10 random-seed trials and report average MAE/RMSE/R2

**Model Comparison** (`models/model_comparison_report.py`)
- VS Code interactive report (`# %%` cells) that trains all 4 versions + ablation variants
- Generates: `output/price_distribution_log_transform.png`, `output/model_version_comparison.png`, `output/log_transform_ablation.png`, `output/feature_importance_comparison.png`

**Price Analysis** (`models/analyze_price_skewness.py`)
- Visualizes price distribution skewness with histograms, box plots, and Q-Q plots
- Generates: `output/price_skewness_analysis.png`

## Key Data Module

`data/nba_players.py` contains all domain knowledge:
- `NBA_PLAYERS`: ~300 player names for fuzzy matching
- `CHINESE_NAME_MAP`: Chinese→English player name translations (e.g., "库里" → "Stephen Curry")
- `TEAM_MAPPINGS`: Chinese/English/abbreviation → full team names
- `PARALLEL_MAPPINGS`: Chinese color terms → English (e.g., "银折" → "Silver")
- `DESCRIPTOR_KEYWORDS`: Regex patterns for card attributes (autograph, rookie, patch, refractor, RPA)
- `PLAYER_TIERS`: ~80 player names → 4 tiers (superstar/star/starter/rotation) for price modeling

## OCR Text Conventions

The raw OCR data uses `半` (misread of `¥`) for CNY prices and contains mixed Chinese/English text. The `extract_price()` function in `panini_card_ocr_etl.py` handles this mapping. Shipping prices are distinguished by the presence of `运费` in the same text line.

## Documentation

`docs/` contains project documentation:
- `docs/ROADMAP.md` — feature roadmap and next steps
- `docs/FEATURE_EXTRACTION_README.md` — detailed feature extraction documentation
- `docs/session1_retrospective.md` — Session 1 retrospective (model iteration learnings)
- `docs/vectorized-floating-aurora.md` — additional notes

## File Format Notes

- `panini_card_ocr_etl.py` uses `# %%` cell markers for VS Code interactive execution (Python Interactive Window)
- CSV exports use `utf-8-sig` encoding for Excel compatibility with Chinese characters
