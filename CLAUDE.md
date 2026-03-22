# CLAUDE.md

Panini NBA trading card price prediction pipeline. Processes Chinese auction screenshots (WeChat/闲鱼) through OCR → feature extraction → CatBoost prediction → Streamlit web app with optional Claude-powered analysis.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web app
python -m models.train_production_model        # Train + register production model (run first)
streamlit run serve/app.py                     # Launch web app

# Data pipeline
python -m data.panini_card_extractor_interactive  # OCR: pics/ → output/raw_ocr_output/
python -m data.panini_card_ocr_etl                # ETL: raw_ocr_output/ → panini_cards_extracted.csv
python data/extract_image_embeddings.py           # ResNet50 → PCA embeddings

# Model training (research — add --register to save to registry)
python -m models.train_price_regressor_v5_catboost --no-emb --register  # Best: CatBoost tabular
python -m models.train_price_regressor_v5_xgb --pca 30                 # XGBoost + embeddings
python -m models.train_price_regressor_v5_lgbm --pca 30                # LightGBM + embeddings

# Model registry
python -c "from models.registry import print_models; print_models()"
python -c "from models.registry import set_active_model; set_active_model('v5_catboost_tab')"

# Analysis
python -m analysis.calibration_analysis         # Calibration diagnostics → results/
python -m analysis.session3_comparison_report    # 12-config ablation → results/
python -m tests.test_extractors                 # Test feature extractors
```

## Architecture

```
pics/*.jpg → EasyOCR → Feature ETL (37 cols) → CatBoost (24 features) → Calibration → Streamlit
```

- **OCR** (`data/panini_card_extractor_interactive.py`): EasyOCR extracts Chinese+English text with bounding boxes
- **ETL** (`data/panini_card_ocr_etl.py`): Aggregates OCR → structured features. Fuzzy-matches players via `rapidfuzz` (threshold: 85). `compute_derived_features()` adds 11 columns. Guarded by `if __name__ == "__main__"`
- **Embeddings** (`data/extract_image_embeddings.py`): ResNet50 → PCA-30/50/64. Hurts at 96 samples; available for future use
- **Training** (`models/`): V1–V5 across XGBoost/LightGBM/CatBoost. Best: V5 CatBoost tabular R²=0.44. 10-seed trials, RandomizedSearchCV
- **Registry** (`models/registry.py`): JSON-based at `models/saved/registry.json`. Production model auto-sets `set_active=True`
- **Calibration** (`models/calibration.py`): Linear recalibration (`slope * pred + intercept`). Stored as `calibration.json` in model dir. Auto-applied at inference; backward-compatible
- **Serving** (`serve/`): `app.py` routes to `inference.py` (XGBoost) or `inference_v5_catboost.py` (CatBoost) based on active model framework. `claude_reasoning.py` returns `None` if no API key

## Key Conventions

- OCR text uses `半` (misread of `¥`) for CNY prices. `extract_price()` in ETL handles this
- Shipping prices identified by `运费` in same text line
- CSV exports use `utf-8-sig` encoding for Excel compatibility with Chinese characters
- `data/panini_card_ocr_etl.py` uses `# %%` cell markers for VS Code Interactive Window
- Domain knowledge lives in `data/nba_players.py`: ~300 players, Chinese→English maps, team/parallel/descriptor mappings, player tiers
- Model artifacts in `models/saved/` are gitignored — regenerate via `python -m models.train_production_model`

## Configuration

- `.env` — `ANTHROPIC_API_KEY` for Claude reasoning (optional, see `.env.example`)
- Dataset: 97 auction screenshots, 96 valid pricing records. Prices ¥100–¥349K CNY (median ¥6K)
