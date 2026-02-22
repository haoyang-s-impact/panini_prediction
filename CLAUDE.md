# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panini NBA trading card price prediction pipeline with web-based serving. Processes Chinese auction screenshots (from WeChat/闲鱼) through OCR, extracts structured card features, trains XGBoost regressors to predict sale prices (CNY), and serves predictions via a Streamlit web app with Claude-powered natural language analysis.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- Web App (Session 3) ---
# Train and save production model (must run before app)
python -m models.train_production_model

# Show all registered models
python -m models.train_production_model --info

# Launch web app
streamlit run serve/app.py

# --- Data Pipeline ---
# Run OCR extraction (EasyOCR, processes pics/ → output/)
python -m data.panini_card_extractor_interactive

# Run feature extraction ETL (output/raw_ocr_output/ → output/panini_cards_extracted.csv)
python -m data.panini_card_ocr_etl

# Test feature extractors on sample OCR data
python -m tests.test_extractors

# --- Model Registry ---
# All training scripts support --register to save the best model to the registry.
# The registry lives at models/saved/registry.json.
# Production model is auto-registered as active; research models are not active by default.

# Set active model (for serving):
python -c "from models.registry import set_active_model; set_active_model('v5_catboost_tab')"

# List all registered models:
python -c "from models.registry import print_models; print_models()"

# --- Model Training (research) ---
# Train price prediction models (reads output/panini_cards_extracted.csv)
# Add --register to save the best trial model to the registry.
python models/train_price_regressor.py      # V1: boolean + numeric features only
python models/train_price_regressor_v2.py   # V2: adds categorical features (native XGBoost)
python models/train_price_regressor_v3.py   # V3: adds RandomizedSearchCV hyperparameter tuning
python models/train_price_regressor_v4.py   # V4: derived features + log-transformed target

# V5: Tabular + image embeddings (Session 3 — XGBoost, LightGBM, CatBoost)
# Add --register to save the best trial model to the registry.
python -m models.train_price_regressor_v5_xgb --pca 30       # XGBoost + PCA-30 embeddings
python -m models.train_price_regressor_v5_xgb --no-emb       # XGBoost tabular only (V4 baseline)
python -m models.train_price_regressor_v5_lgbm --pca 30      # LightGBM + PCA-30 embeddings
python -m models.train_price_regressor_v5_catboost --pca 30   # CatBoost + PCA-30 embeddings
python -m models.train_price_regressor_v5_catboost --no-emb --register  # Train + register

# --- Analysis ---
# Run model comparison report (trains all versions, generates plots)
python analysis/model_comparison_report.py  # Session 1: outputs PNGs to results/

# Session 3: Tree models + embeddings comparison (12 configs, outputs to results/)
python -m analysis.session3_comparison_report

# Run price skewness analysis
python analysis/analyze_price_skewness.py   # Outputs PNG to results/

# Extract image embeddings (pics/ → output/image_embeddings_pca{30,50,64}.csv)
python data/extract_image_embeddings.py

# Verify embedding quality (generates PCA, t-SNE, similarity plots)
python analysis/verify_embeddings.py        # Outputs PNGs to results/
```

## Architecture

The project has four stages:

**Stage 1: OCR** (`data/panini_card_extractor_interactive.py` or `data/extract_panini_info.py`)
- Reads auction screenshot images from `pics/`
- Uses EasyOCR (interactive version) or PaddleOCR (original version) to extract Chinese+English text
- Saves per-image raw OCR text files to `output/raw_ocr_output/` with text, confidence scores, and bounding box coordinates
- Also does basic regex-based categorization into card_identity, grading_info, auction_data

**Stage 2: Feature Extraction ETL** (`data/panini_card_ocr_etl.py`)
- Loads raw OCR text files from `output/raw_ocr_output/` into a long-format DataFrame (multiple rows per image)
- Aggregates into wide-format (one row per card) via `aggregate_card_features()`
- Uses spatial zones (Y-coordinate ranges) and confidence thresholds for position-based extraction
- Fuzzy-matches player names against `data/nba_players.py` using `rapidfuzz` (threshold: 85)
- Maps Chinese text to English (player names, team names, parallel colors, descriptors)
- Outputs: `output/panini_cards_extracted.csv` with ~37 columns (player, team, card series, grading, prices, boolean flags, derived features)
- `compute_derived_features()` adds 11 derived columns: player_tier, rarity_ratio, rookie_auto, rookie_patch, is_rpa, is_numbered, is_1of1, is_base, day_of_week, hour_of_day, is_weekend
- Top-level execution guarded behind `if __name__ == "__main__"` — functions are importable without side effects

**Stage 2.5: Image Embeddings** (`data/extract_image_embeddings.py`)
- Loads card auction screenshots from `pics/`
- Preprocesses to 224x224 with ImageNet normalization (Resize(256) + CenterCrop(224))
- Passes through frozen pretrained ResNet50 (avgpool → 2048-d feature vectors)
- Applies StandardScaler + PCA to reduce dimensions
- Outputs three variants for overfitting comparison: `output/image_embeddings_pca{30,50,64}.csv`
- Verification: `analysis/verify_embeddings.py` generates PCA variance, t-SNE, and cosine similarity plots

**Stage 3: ML Training** (`models/train_price_regressor.py`, `_v2.py`, `_v3.py`, `_v4.py`, `_v5_*.py`)
- Target: `price_cny` (Chinese Yuan sale price)
- V1: 6 features (boolean flags + serial_max + grade), baseline XGBoost
- V2: 13 features (adds player_name, team, card_series, etc. as native categoricals)
- V3: Same features as V2 + RandomizedSearchCV over 20 iterations with 3-fold CV
- V4: 24 features (V3 + 11 derived), log-transformed target, fixed card_year parsing, NaN-aware numerics
- V5: 24 tabular + PCA image embeddings (30/50/64), compares XGBoost/LightGBM/CatBoost. Uses raw target (Session 1 finding). Shared utilities in `data/data_utils.py` and `models/model_utils.py`.
- All versions run 10 random-seed trials and report average MAE/RMSE/R2

**Model Registry** (`models/registry.py`, `models/saved/registry.json`)
- Central registry for all trained models — tracks artifacts, metrics, and metadata
- `register_model()` saves `model.joblib` + `metadata.json` to `models/saved/{model_id}/` and updates `registry.json`
- `load_model(model_id)` loads specific model; `load_model()` loads the active model
- `set_active_model(model_id)` sets which model the serving layer uses
- `list_models()` / `print_models()` shows all registered models with metrics
- `build_metadata(X)` helper builds standard metadata dict from training DataFrame
- Model ID convention: `{version}_{framework}_{pipeline}` (e.g., `v5_catboost_tab`, `v4_xgb_ocr_tabular`)
- Production model (`train_production_model.py`) auto-sets `set_active=True`; research scripts register with `set_active=False`

**Stage 4: Serving** (`serve/`, `models/train_production_model.py`)
- `models/train_production_model.py` — trains V4 on all data, registers as active model via `models.registry`. Entry point: `python -m models.train_production_model`.
- `serve/model_registry.py` — thin wrapper around `models.registry.load_model()`. Preserves serve-layer API.
- `serve/inference.py` — `predict_from_image(image_bytes)`: single function that chains EasyOCR → feature extraction → XGBoost prediction. Reuses extraction functions from `data/panini_card_ocr_etl.py` directly. Auto-detects GPU for EasyOCR.
- `serve/claude_reasoning.py` — calls Claude API (Haiku 4.5) to generate natural language analysis of the prediction. Graceful fallback if no API key.
- `serve/app.py` — Streamlit frontend: upload image → display predicted price, extracted features, Claude analysis, raw OCR debug. Auto-detects GPU for EasyOCR.
- EasyOCR reader and XGBoost model are lazy-loaded singletons cached via `@st.cache_resource`
- Model metadata includes `pipeline_type` field for future multi-pipeline support (e.g., when V5 with embeddings needs a different feature extraction chain)

## Serving Module (`serve/`)

The `serve/` package is a standalone Python module that can be imported independently of Streamlit:

```python
from serve.inference import predict_from_image
result = predict_from_image(open("pics/test.jpg", "rb").read())
# → {"predicted_price_cny": 9158.87, "extracted_features": {...}, "ocr_lines": [...]}
```

**Key files:**
- `serve/app.py` — Streamlit frontend: upload image → display predicted price, extracted features, Claude analysis, raw OCR debug. Auto-routes to the correct inference module based on active model's framework.
- `serve/model_registry.py` — thin wrapper around `models.registry.load_model()`. Preserves serve-layer API.
- `serve/inference.py` — V4 XGBoost inference: `predict_from_image(image_bytes) → dict`. Uses log1p target transform. Also exports shared OCR/feature extraction functions.
- `serve/inference_v5_catboost.py` — V5 CatBoost inference: `predict_from_image(image_bytes) → dict`. Uses raw target (no log transform), string categoricals with cat_indices.
- `serve/claude_reasoning.py` — `get_analysis(prediction) → str | None`. Requires `ANTHROPIC_API_KEY` env var.

**Inference routing:** `app.py` checks `get_active_model()["framework"]` and imports from `inference_v5_catboost` for CatBoost or `inference` for XGBoost/LightGBM.

**Model artifacts** (`models/saved/`, gitignored):
- `registry.json` — model registry index (all registered models, active model pointer)
- `{model_id}/model.joblib` — serialized model (XGBoost, LightGBM, or CatBoost)
- `{model_id}/metadata.json` — feature names, dtypes, category mappings, hyperparams, `pipeline_type`

## Analysis Scripts (`analysis/`)

- `analysis/verify_embeddings.py` — PCA variance, t-SNE, cosine similarity plots for image embeddings
- `analysis/model_comparison_report.py` — VS Code interactive report: trains all 4 model versions + ablation variants, generates comparison plots to `results/`
- `analysis/analyze_price_skewness.py` — price distribution visualization (histograms, box plots, Q-Q plots)
- `analysis/session3_comparison_report.py` — Session 3: 12-config ablation matrix (3 frameworks × 4 feature sets), embedding-price correlation diagnostic, generates comparison plots to `results/`

## Data Pipeline (`data/`)

- `data/panini_card_extractor_interactive.py` — Stage 1: EasyOCR extraction (processes `pics/` → `output/raw_ocr_output/`)
- `data/extract_panini_info.py` — Stage 1: PaddleOCR extraction (original version)
- `data/panini_card_ocr_etl.py` — Stage 2: Feature extraction ETL (`output/raw_ocr_output/` → `output/panini_cards_extracted.csv`)
- `data/nba_players.py` — domain knowledge (see Key Data Module below)
- `data/extract_image_embeddings.py` — Stage 2.5: ResNet50 → PCA embedding pipeline
- `data/data_utils.py` — shared data loading for V5: `load_tabular_data()`, `load_embeddings()`, `merge_features()`. Reuses V4's exact feature preparation logic.

## Model Utilities (`models/`)

- `models/registry.py` — model registry: `register_model()`, `load_model()`, `set_active_model()`, `list_models()`, `print_models()`, `build_metadata()`
- `models/model_utils.py` — shared training utilities for V5: `run_trials()`, `tune_hyperparameters()`, `prepare_for_catboost()`, `show_feature_importance()`

## Output Directories

- `output/` — data prep artifacts: CSVs, embeddings, OCR data
- `results/` — model comparison outputs: Session 3 plots and reports

## Key Data Module

`data/nba_players.py` contains all domain knowledge:
- `NBA_PLAYERS`: ~300 player names for fuzzy matching
- `CHINESE_NAME_MAP`: Chinese→English player name translations (e.g., "库里" → "Stephen Curry")
- `TEAM_MAPPINGS`: Chinese/English/abbreviation → full team names
- `PARALLEL_MAPPINGS`: Chinese color terms → English (e.g., "银折" → "Silver")
- `DESCRIPTOR_KEYWORDS`: Regex patterns for card attributes (autograph, rookie, patch, refractor, RPA)
- `PLAYER_TIERS`: ~80 player names → 4 tiers (superstar/star/starter/rotation) for price modeling

## OCR Text Conventions

The raw OCR data uses `半` (misread of `¥`) for CNY prices and contains mixed Chinese/English text. The `extract_price()` function in `data/panini_card_ocr_etl.py` handles this mapping. Shipping prices are distinguished by the presence of `运费` in the same text line.

## Documentation

`docs/` contains project documentation:
- `docs/ROADMAP.md` — feature roadmap and next steps
- `docs/FEATURE_EXTRACTION_README.md` — detailed feature extraction documentation
- `docs/session1_retrospective.md` — Session 1 retrospective (model iteration learnings)
- `docs/session2_retrospective.md` — Session 2 retrospective (image embeddings)
- `docs/session3_retrospective.md` — Session 3 retrospective (tree models + embeddings comparison)
- `docs/SERVING_README.md` — Web app serving architecture, usage, and design decisions
- `docs/codebase_knowledge.md` — codebase inventory, dataset profile, model results (read this to skip re-exploration). **If the actual folder structure or file inventory differs from what this file describes, update it before proceeding.**

## Configuration

- `.env` — `ANTHROPIC_API_KEY` for Claude reasoning (gitignored, optional)
- `.env.example` — template with instructions
- `models/saved/` — serialized model artifacts (gitignored, regenerate via `python -m models.train_production_model`)

## File Format Notes

- `data/panini_card_ocr_etl.py` uses `# %%` cell markers for VS Code interactive execution (Python Interactive Window)
- CSV exports use `utf-8-sig` encoding for Excel compatibility with Chinese characters
