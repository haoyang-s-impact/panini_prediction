# Panini NBA Card Price Predictor

End-to-end ML pipeline that predicts sale prices of Panini NBA trading cards from Chinese auction screenshots.

## Overview

This project takes screenshots from Chinese auction platforms (WeChat / 闲鱼 Xianyu), extracts structured card features via bilingual OCR, and trains gradient boosting models to predict sale prices in CNY. A Streamlit web app serves predictions with optional Claude-powered natural language analysis.

**Pipeline:** Auction Screenshot → EasyOCR → Feature Extraction → Price Prediction → Claude Analysis

**Dataset:** 97 Panini NBA card auction listings with 37 extracted features. Prices range from ¥100 to ¥349,000 CNY (median ¥6,057).

## Key Results

| Version | Features | Framework | Avg R² | Description |
|---------|----------|-----------|--------|-------------|
| V1 | 6 | XGBoost | -0.358 | Boolean + numeric baseline |
| V2 | 13 | XGBoost | 0.110 | + categorical features |
| V3 | 13 | XGBoost | 0.254 | + hyperparameter tuning |
| V4 | 24 | XGBoost | 0.353 | + derived features, raw target |
| **V5** | **24** | **CatBoost** | **0.440** | **Best model — tabular only** |

Key findings:
- Log-transforming the target hurts performance on this small dataset
- Image embeddings (ResNet50 + PCA) cause overfitting at 97 samples
- CatBoost's native categorical handling outperforms XGBoost and LightGBM

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Enable Claude AI analysis
cp .env.example .env  # edit to add ANTHROPIC_API_KEY

# 3. Train production model
python -m models.train_production_model

# 4. Launch web app
streamlit run serve/app.py
```

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1    │    │  Stage 2     │    │  Stage 3     │    │  Stage 4     │
│  OCR        │───▶│  Feature ETL │───▶│  ML Training │───▶│  Serving     │
│             │    │              │    │              │    │              │
│ pics/*.jpg  │    │ raw_ocr →    │    │ XGBoost /    │    │ Streamlit    │
│ → EasyOCR   │    │ 37-col CSV   │    │ CatBoost     │    │ web app      │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Stage 1 — OCR** (`data/panini_card_extractor_interactive.py`): EasyOCR extracts Chinese + English text from auction screenshots with bounding box coordinates and confidence scores.

**Stage 2 — Feature ETL** (`data/panini_card_ocr_etl.py`): Aggregates OCR lines into structured features per card — fuzzy-matches player names against ~300 NBA players, maps Chinese text to English, computes 11 derived features (player tier, rarity ratio, temporal features).

**Stage 2.5 — Image Embeddings** (`data/extract_image_embeddings.py`): Frozen ResNet50 extracts 2048-d feature vectors, reduced via PCA to 30/50/64 dimensions. Currently hurts performance at this dataset size.

**Stage 3 — Training** (`models/`): Five model versions (V1–V5) across XGBoost, LightGBM, and CatBoost with 10-seed trial averaging and RandomizedSearchCV tuning.

**Stage 4 — Serving** (`serve/app.py`): Streamlit app accepts image uploads and displays predicted price, extracted features, Claude analysis, and raw OCR debug output.

## Web App

```bash
# Use the active model from the registry
streamlit run serve/app.py

# Override with a specific model
streamlit run serve/app.py -- --model v5_catboost_tab
```

**Programmatic usage:**
```python
from serve.inference import predict_from_image

result = predict_from_image(open("pics/test.jpg", "rb").read())
# → {"predicted_price_cny": 22434.12, "extracted_features": {...}, "ocr_lines": [...]}
```

The app displays:
- Predicted price in CNY
- Extracted card features (player, team, year, series, attributes)
- Claude-powered natural language analysis (optional, requires API key)
- Raw OCR output with confidence scores (expandable debug section)

## Project Structure

```
panini_prediction/
├── data/                          # Data pipeline modules
│   ├── panini_card_extractor_interactive.py  # Stage 1: EasyOCR extraction
│   ├── panini_card_ocr_etl.py               # Stage 2: Feature extraction ETL
│   ├── extract_image_embeddings.py          # Stage 2.5: ResNet50 embeddings
│   ├── nba_players.py                       # Domain knowledge (players, teams, mappings)
│   └── data_utils.py                        # Shared data loading utilities
├── models/                        # Training scripts + registry
│   ├── train_production_model.py            # Train + register production model
│   ├── train_price_regressor.py             # V1 baseline
│   ├── train_price_regressor_v2.py          # V2 + categoricals
│   ├── train_price_regressor_v3.py          # V3 + hyperparameter tuning
│   ├── train_price_regressor_v4.py          # V4 + derived features
│   ├── train_price_regressor_v5_*.py        # V5 multi-framework comparison
│   ├── registry.py                          # Model registry
│   ├── model_utils.py                       # Shared training utilities
│   └── saved/                               # Model artifacts (gitignored)
├── serve/                         # Web app + inference
│   ├── app.py                               # Streamlit frontend
│   ├── inference.py                         # XGBoost inference pipeline
│   ├── inference_v5_catboost.py             # CatBoost inference pipeline
│   ├── claude_reasoning.py                  # Claude API analysis
│   └── model_registry.py                    # Serving-layer registry wrapper
├── analysis/                      # Experiment analysis scripts
│   ├── model_comparison_report.py           # V1–V4 comparison plots
│   ├── session3_comparison_report.py        # Framework × embedding ablation
│   ├── analyze_price_skewness.py            # Price distribution analysis
│   └── verify_embeddings.py                 # Embedding quality verification
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── pics/                          # Source auction screenshots (97 images)
├── output/                        # Processed data artifacts
├── results/                       # Analysis plots
└── requirements.txt
```

## Model Registry

The registry tracks trained models, their artifacts, and metrics at `models/saved/registry.json`.

```bash
# Train and register production model (auto-sets as active)
python -m models.train_production_model

# List all registered models
python -c "from models.registry import print_models; print_models()"

# Switch the active model for serving
python -c "from models.registry import set_active_model; set_active_model('v5_catboost_tab')"

# Register a research model (not active by default)
python -m models.train_price_regressor_v5_catboost --no-emb --register
```

Each registered model stores `model.joblib` and `metadata.json` (feature schema, category mappings, hyperparameters) under `models/saved/{model_id}/`.

## Tech Stack

| Category | Libraries |
|----------|-----------|
| OCR | EasyOCR (Chinese + English) |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn |
| Deep Learning | PyTorch, torchvision (ResNet50 embeddings) |
| Data | pandas, NumPy |
| Text Matching | rapidfuzz |
| Visualization | matplotlib, seaborn |
| Serving | Streamlit |
| AI Analysis | Anthropic Claude API (Haiku) |

## Known Limitations

- **Small dataset** — only 97 training samples, the primary constraint on model quality
- **Overfitting risk** — image embeddings (54–88 features) hurt performance at this sample size
- **OCR speed** — EasyOCR is slow on CPU (~2–3s per image); GPU is auto-detected when available
- **Point estimates** — no confidence intervals on predictions
- **Price leakage risk** — auction prices visible in screenshots could leak through image embeddings

## Documentation

- [Serving Architecture](docs/SERVING_README.md) — web app design and usage details
- [Feature Extraction](docs/FEATURE_EXTRACTION_README.md) — detailed ETL documentation
- [Roadmap](docs/ROADMAP.md) — planned features and next steps
- [Session 1 Retrospective](docs/session1_retrospective.md) — model iteration learnings
- [Session 2 Retrospective](docs/session2_retrospective.md) — image embedding experiments
- [Session 3 Retrospective](docs/session3_retrospective.md) — tree model comparison
- [Codebase Knowledge](docs/codebase_knowledge.md) — dataset profile and model results
