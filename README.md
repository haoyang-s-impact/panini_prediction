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
| V5 | 24 | CatBoost | 0.440 | Best model — tabular only |
| **V5 + calibration** | **24** | **CatBoost** | **0.503** | **+ linear recalibration** |

Key findings:
- Log-transforming the target hurts performance on this small dataset
- Image embeddings (ResNet50 + PCA) cause overfitting at 97 samples
- CatBoost's native categorical handling outperforms XGBoost and LightGBM
- Post-hoc linear calibration corrects regression-to-the-mean bias (+0.12 R²)

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
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Stage 1    │    │  Stage 2     │    │  Stage 3     │    │  Stage 3.5  │    │  Stage 4     │
│  OCR        │───▶│  Feature ETL │───▶│  ML Training │───▶│ Calibration │───▶│  Serving     │
│             │    │              │    │              │    │             │    │              │
│ pics/*.jpg  │    │ raw_ocr →    │    │ CatBoost     │    │ linear      │    │ Streamlit    │
│ → EasyOCR   │    │ 37-col CSV   │    │ (24 features)│    │ recalibrate │    │ web app      │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

**Stage 1 — OCR** (`data/panini_card_extractor_interactive.py`): EasyOCR extracts Chinese + English text from auction screenshots with bounding box coordinates and confidence scores.

**Stage 2 — Feature ETL** (`data/panini_card_ocr_etl.py`): Aggregates OCR lines into structured features per card — fuzzy-matches player names against ~300 NBA players, maps Chinese text to English, computes 11 derived features (player tier, rarity ratio, temporal features).

**Stage 2.5 — Image Embeddings** (`data/extract_image_embeddings.py`): Frozen ResNet50 extracts 2048-d feature vectors, reduced via PCA to 30/50/64 dimensions. Currently hurts performance at this dataset size.

**Stage 3 — Training** (`models/`): Five model versions (V1–V5) across XGBoost, LightGBM, and CatBoost with 10-seed trial averaging and RandomizedSearchCV tuning.

**Stage 3.5 — Calibration** (`models/calibration.py`): Linear recalibration corrects regression-to-the-mean bias. A 2-parameter transform (`slope * pred + intercept`) is fitted on 5-fold CV out-of-sample predictions and stored as `calibration.json` alongside the model. Auto-applied at inference time; backward-compatible (absent file = no transform).

**Stage 4 — Serving** (`serve/app.py`): Streamlit app accepts image uploads and displays predicted price (with calibration applied), extracted features, Claude analysis, and raw OCR debug output.

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
│   ├── calibration.py                       # Linear recalibration (fit/apply/save/load)
│   ├── model_utils.py                       # Shared training utilities
│   └── saved/                               # Model artifacts + calibration.json (gitignored)
├── serve/                         # Web app + inference
│   ├── app.py                               # Streamlit frontend
│   ├── inference.py                         # XGBoost inference pipeline
│   ├── inference_v5_catboost.py             # CatBoost inference pipeline
│   ├── claude_reasoning.py                  # Claude API analysis
│   └── model_registry.py                    # Serving-layer registry wrapper
├── analysis/                      # Experiment analysis scripts
│   ├── model_comparison_report.py           # V1–V4 comparison plots
│   ├── session3_comparison_report.py        # Framework × embedding ablation
│   ├── calibration_analysis.py              # Calibration diagnostics + recalibration
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

Each registered model stores `model.joblib`, `metadata.json` (feature schema, category mappings, hyperparameters), and optionally `calibration.json` (linear recalibration parameters) under `models/saved/{model_id}/`.

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
- **Extreme outliers** — 2 SGA rookie cards (200K + 349K CNY) dominate high-end error; calibration helps but can't fully resolve
- **Overfitting risk** — image embeddings (54–88 features) hurt performance at this sample size
- **OCR data gaps** — card_year 56% populated, end_time 71% parsed, serial_max 25% populated
- **Point estimates** — no confidence intervals or prediction uncertainty
- **Global calibration** — same linear correction for all price ranges; per-segment calibration needs more data

## Next Steps

1. **More data** — scrape additional listings (target 500+ samples) to stabilize all model components. Include unsold listings with asking prices for richer market signal.
2. **External features** — add player stats (career points, All-Star appearances, championships) and card population data as features that don't require more screenshots.
3. **Prediction intervals** — add quantile regression or conformal prediction for uncertainty quantification so users know how much to trust each prediction.
4. **Better image features** — crop to card region (top ~40%), use domain-specific embeddings; revisit once dataset exceeds 500 samples where embeddings may finally help.
5. **Log-target CatBoost** — CatBoost handles targets differently than XGBoost; log-transform may reduce regression-to-mean without post-hoc calibration.
6. **Production hardening** — feedback loop (track predictions vs outcomes), batch inference, monitoring for feature drift.

## Documentation

- [Full Retrospective](docs/full_retrospective.md) — comprehensive overview across all 5 sessions
- [Serving Architecture](docs/SERVING_README.md) — web app design and usage details
- [Feature Extraction](docs/FEATURE_EXTRACTION_README.md) — detailed ETL documentation
- [Roadmap](docs/ROADMAP.md) — planned features and next steps
- [Session 1 Retrospective](docs/session1_retrospective.md) — derived features + log-transform
- [Session 2 Retrospective](docs/session2_retrospective.md) — image embedding pipeline
- [Session 3 Retrospective](docs/session3_retrospective.md) — tree model + embedding comparison
- [Session 4 Retrospective](docs/session4_retrospective.md) — infrastructure + environment
- [Session 5 Retrospective](docs/session5_retrospective.md) — calibration analysis + recalibration
- [Codebase Knowledge](docs/codebase_knowledge.md) — dataset profile and model results
