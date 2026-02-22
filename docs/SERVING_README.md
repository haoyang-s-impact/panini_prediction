# Web App Serving — Price Prediction

## Overview

A Streamlit web app where users upload a card auction screenshot and get an AI-powered price prediction. Supports multiple models (XGBoost, CatBoost, LightGBM) via a model registry. The serving layer bridges the research pipeline (OCR → ETL → model prediction) into a user-facing product with Claude-powered natural language reasoning.

## Architecture

```
[Streamlit App]  →  [serve/inference*.py]  →  [EasyOCR + Model]
  app.py                                           ↓
  --model flag   →  Registry lookup         →  Model Prediction → Display
                                                    ↓
                                      [serve/claude_reasoning.py] → AI Analysis
```

### Inference Routing

```
app.py  reads active model's framework from registry
    ├── framework == "catboost"  →  serve/inference_v5_catboost.py
    └── otherwise               →  serve/inference.py  (XGBoost/LightGBM)
```

Both inference modules share OCR and feature extraction code. Only the prediction step differs (categorical handling, target inverse transform).

### Design Principles

- **Registry-based model management.** Models are registered in `models/saved/registry.json` with artifacts in versioned subdirectories. The active model pointer makes switching instant.
- **Training and serving are separated.** Training lives in `models/`. Serving lives in `serve/`. The bridge is the model registry (`models/registry.py`).
- **Metadata-driven inference.** `target_transform` field determines inverse transform (log1p → expm1, or raw). No hardcoded assumptions per model version.
- **Deterministic pipeline, not MCP.** The pipeline is always OCR → ETL → predict. Claude adds value at the output layer (explaining predictions), not orchestrating the pipeline.
- **Graceful degradation.** Claude reasoning returns `None` if no API key — the core prediction always works.

## Files

### Model Registry

| File | Purpose |
|------|---------|
| `models/registry.py` | Core registry: `register_model()`, `load_model()`, `set_active_model()`, `list_models()` |
| `models/saved/registry.json` | JSON index of all registered models + active model pointer |
| `models/saved/{model_id}/model.joblib` | Serialized model artifact (XGBoost, CatBoost, or LightGBM) |
| `models/saved/{model_id}/metadata.json` | Feature names, dtypes, category mappings, hyperparams |
| `models/train_production_model.py` | Train V4 on ALL data, register as active model |

### Serving

| File | Purpose |
|------|---------|
| `serve/app.py` | Streamlit frontend: upload image → display prediction. Accepts `--model` CLI arg. |
| `serve/model_registry.py` | Thin wrapper around `models.registry.load_model()` |
| `serve/inference.py` | XGBoost/LightGBM inference: OCR → features → predict. Shared OCR functions. |
| `serve/inference_v5_catboost.py` | CatBoost inference: imports shared OCR from `inference.py`, CatBoost-specific categorical handling. |
| `serve/claude_reasoning.py` | Claude API call (Haiku 4.5) for natural language card analysis |
| `.env.example` | API key template (`ANTHROPIC_API_KEY`) |

## Usage

### Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Train and register the production model

```bash
python -m models.train_production_model
```

This trains V4 XGBoost on all 96 samples, tunes hyperparameters via 3-fold CV, and registers as the active model.

### Train and register V5 CatBoost (best model, R²=0.44)

```bash
python -m models.train_price_regressor_v5_catboost --no-emb --register
```

### Set the active model

```bash
python -c "from models.registry import set_active_model; set_active_model('v5_catboost_tab')"
```

### Show all registered models

```bash
python -m models.train_production_model --info
```

### Launch the web app

```bash
# Use the registry active model
streamlit run serve/app.py

# Override with a specific model
streamlit run serve/app.py -- --model v5_catboost_tab
streamlit run serve/app.py -- --model v4_xgb_ocr_tabular
```

The `--` separator is required by Streamlit to pass arguments to the script. The UI shows which model is serving and whether it's from CLI override or registry.

### Use programmatically (without Streamlit)

```python
# Uses the active model from registry
from serve.inference import predict_from_image
result = predict_from_image(open("pics/test.jpg", "rb").read())
print(result["predicted_price_cny"])    # 22434.12
print(result["extracted_features"])     # {"player_name": "Shai Gilgeous-Alexander", ...}
print(result["ocr_lines"])             # [{"text": "...", "confidence": 0.95}, ...]
```

## Inference Modules

| Module | Frameworks | Target Transform | Categorical Handling |
|--------|-----------|-----------------|---------------------|
| `serve/inference.py` | XGBoost, LightGBM | Reads `metadata["target_transform"]`: `log1p` → expm1 inverse, `none` → raw | pandas `category` dtype |
| `serve/inference_v5_catboost.py` | CatBoost | Raw prediction (no inverse) | String dtype + `cat_indices` from metadata |

## Key Design Patterns

### Lazy Singleton (EasyOCR Reader)

The EasyOCR reader takes ~10s to initialize (loading language models into memory). A lazy singleton initializes it on first use and keeps it resident. In Streamlit, `@st.cache_resource` achieves this — the object is created once per app instance and reused for all subsequent requests. Auto-detects GPU via `torch.cuda.is_available()`.

### Model Registry

The registry (`models/registry.py`) manages `models/saved/registry.json` and per-model artifact directories. Each model gets a versioned directory (`models/saved/{model_id}/`) containing `model.joblib` + `metadata.json`. The `active_model` pointer determines which model the serving layer loads by default.

### Feature Schema Enforcement

`_features_to_dataframe()` in `serve/inference.py` applies identical preprocessing to training: dtype coercion, category mappings with the exact same categories as training, NaN handling, `_parse_card_year()`. The metadata.json stores `feature_names`, `feature_dtypes`, and `category_mappings` to make this reproducible.

### Graceful Degradation

The Claude reasoning module (`serve/claude_reasoning.py`) is designed to fail silently. If no API key is set, or the API call fails, or the `anthropic` package isn't installed — it returns `None` and the app shows predictions without the analysis section. The core product (price prediction) always works; LLM integration is an enhancement, not a dependency.

## Model Performance Summary

| Model ID | Framework | R² | Notes |
|----------|-----------|---:|-------|
| `v1_xgb_ocr_tabular` | XGBoost | -0.36 | 6 features, baseline |
| `v2_xgb_ocr_tabular` | XGBoost | 0.11 | 13 features |
| `v3_xgb_ocr_tabular` | XGBoost | 0.25 | 13 features + tuning |
| `v4_xgb_ocr_tabular` | XGBoost | 0.35 | 24 features, log1p target (production) |
| **`v5_catboost_tab`** | **CatBoost** | **0.44** | **24 features, raw target (best)** |
| `v5_xgb_tab` | XGBoost | 0.34 | 24 features, raw target |
| `v5_lgbm_tab` | LightGBM | 0.22 | 24 features, raw target |

## Performance

| Step | Latency |
|------|---------|
| First prediction (EasyOCR model loading) | ~10s |
| Subsequent predictions (OCR on image) | ~2-3s |
| Model loading (cached) | <1s |
| Claude reasoning (API call) | ~1-2s |

## Expandability

### Adding a new model framework:

1. Create `serve/inference_{framework}.py` with a `predict_from_image()` function
2. Import shared OCR/feature extraction from `serve/inference.py`
3. Add routing in `app.py`'s `_get_predict_fn()` for the new framework
4. Train and register: `python -m models.train_price_regressor_v5_{framework} --no-emb --register`

### Adding embeddings to serving:

When a model using tabular + image embeddings is served:
1. `serve/inference_{variant}.py` would also run ResNet50 + PCA on the uploaded image
2. Append embedding columns to the feature DataFrame before prediction
3. The `pipeline_type` field in metadata distinguishes `ocr_tabular` from `ocr_tabular_pca30`

### Deploying to the web:

1. **Wrap in FastAPI:** `serve/` is a standalone Python package. Add `api.py` with a `POST /predict` endpoint that calls `predict_from_image()`.
2. **Containerize:** Dockerfile with EasyOCR + model frameworks + model artifacts.

## Gaps & Known Limitations

| Gap | Impact | Mitigation |
|-----|--------|------------|
| 96 training samples | Low model confidence, high variance | Core limitation — need more auction screenshots |
| No confidence interval | User gets point estimate only | Could add quantile regression or bootstrap |
| EasyOCR slow on CPU (~2-3s/image) | User waits during inference | GPU auto-detected; lighter OCR possible |
| Price visible in screenshots | Possible data leakage through embeddings | Monitor embedding-only model performance |
| CatBoost model trained on 80/20 split | Not trained on all data like V4 production | Create `train_production_model_catboost.py` for full-data training |

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| **Model registry over MLflow** | Lightweight, file-based, sufficient for current scale. Add MLflow when needed. |
| **`--model` CLI arg** | Override active model without changing registry state. Useful for A/B testing. |
| **Separate CatBoost inference module** | CatBoost needs string categoricals + cat_indices. Separate module keeps each clean. |
| **Shared OCR/feature extraction** | OCR and feature extraction are framework-agnostic. Only the prediction step differs. |
| **Metadata-driven inference** | `target_transform` field determines inverse transform. No hardcoded assumptions. |
| **Streamlit** | Fastest path to interactive MVP. Swap to FastAPI + React when deploying publicly. |
| **Claude Haiku 4.5** | ~$0.001/prediction. Graceful fallback if no API key. Enhancement, not dependency. |
