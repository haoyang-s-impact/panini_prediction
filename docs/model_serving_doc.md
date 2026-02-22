# Model Registry & Serving

## Overview

The model registry tracks all trained models and their artifacts. Each training script can register its best model, and the serving layer loads whichever model is active (or one specified via CLI).

```
[Training Scripts]  →  [models/registry.py]  →  [models/saved/registry.json]
     V1-V5               register_model()         + {model_id}/model.joblib
                                                   + {model_id}/metadata.json

[Streamlit App]     →  [serve/inference*.py]  →  [models/registry.py]
  app.py --model         predict_from_image()       load_model(model_id)
```

## Quick Start

```bash
# Train and register the production model (V4 XGBoost, set as active)
python -m models.train_production_model

# Train and register V5 CatBoost (not active by default)
python -m models.train_price_regressor_v5_catboost --no-emb --register

# List all registered models
python -m models.train_production_model --info

# Switch active model
python -c "from models.registry import set_active_model; set_active_model('v5_catboost_tab')"

# Launch app with active model
streamlit run serve/app.py

# Launch app with a specific model (overrides active)
streamlit run serve/app.py -- --model v5_catboost_tab
streamlit run serve/app.py -- --model v4_xgb_ocr_tabular
```

## Registry

### Files

| File | Purpose |
|------|---------|
| `models/registry.py` | Core registry module — register, load, list, set active |
| `models/saved/registry.json` | JSON index of all registered models |
| `models/saved/{model_id}/model.joblib` | Serialized model artifact |
| `models/saved/{model_id}/metadata.json` | Feature schema, dtypes, category mappings, hyperparams |

### registry.json Schema

```json
{
  "schema_version": 1,
  "active_model": "v5_catboost_tab",
  "models": {
    "v5_catboost_tab": {
      "model_id": "v5_catboost_tab",
      "version": "v5",
      "framework": "catboost",
      "pipeline_type": "ocr_tabular",
      "description": "V5 CatBoost (tabular only), 10-trial best",
      "registered_at": "2026-02-22T...",
      "n_samples": 96,
      "n_features": 24,
      "target": "price_cny",
      "target_transform": "none",
      "metrics": { "mae": 20039, "rmse": 42724, "r2": 0.4387 },
      "artifact_dir": "v5_catboost_tab"
    }
  }
}
```

### Model ID Convention

Format: `{version}_{framework}_{pipeline}`

| Model ID | Description |
|----------|-------------|
| `v1_xgb_ocr_tabular` | V1 XGBoost, 6 features, baseline |
| `v2_xgb_ocr_tabular` | V2 XGBoost, 13 features, native categoricals |
| `v3_xgb_ocr_tabular` | V3 XGBoost, 13 features, tuned hyperparams |
| `v4_xgb_ocr_tabular` | V4 XGBoost, 24 features, trained on ALL data (production) |
| `v4_xgb_ocr_tabular_research` | V4 XGBoost, 24 features, 80/20 split (research) |
| `v5_catboost_tab` | V5 CatBoost, 24 tabular features |
| `v5_xgb_tab` | V5 XGBoost, 24 tabular features |
| `v5_lgbm_tab` | V5 LightGBM, 24 tabular features |
| `v5_xgb_pca30` | V5 XGBoost, tabular + PCA-30 embeddings |
| `v5_catboost_pca30` | V5 CatBoost, tabular + PCA-30 embeddings |

### Python API

```python
from models.registry import (
    register_model,    # Save model + update registry
    load_model,        # Load by ID or active model
    list_models,       # List all registered models
    print_models,      # Print formatted table
    set_active_model,  # Set active model for serving
    get_active_model,  # Get active model's registry entry
    build_metadata,    # Build standard metadata dict from training DataFrame
)
```

## Training Scripts

All training scripts support `--register` to save the best trial model to the registry.

| Script | Command | Registers As |
|--------|---------|-------------|
| V1 XGBoost | `python models/train_price_regressor.py --register` | `v1_xgb_ocr_tabular` |
| V2 XGBoost | `python models/train_price_regressor_v2.py --register` | `v2_xgb_ocr_tabular` |
| V3 XGBoost | `python models/train_price_regressor_v3.py --register` | `v3_xgb_ocr_tabular` |
| V4 XGBoost | `python models/train_price_regressor_v4.py --register` | `v4_xgb_ocr_tabular_research` |
| V5 XGBoost | `python -m models.train_price_regressor_v5_xgb --no-emb --register` | `v5_xgb_tab` |
| V5 CatBoost | `python -m models.train_price_regressor_v5_catboost --no-emb --register` | `v5_catboost_tab` |
| V5 LightGBM | `python -m models.train_price_regressor_v5_lgbm --no-emb --register` | `v5_lgbm_tab` |
| Production | `python -m models.train_production_model` | `v4_xgb_ocr_tabular` (auto-active) |

- **Production model** (`train_production_model.py`) trains on ALL data and sets `set_active=True` automatically.
- **Research scripts** (V1-V5) register with `set_active=False` — they train on 80/20 splits for fair evaluation.

## Serving

### Architecture

```
serve/app.py  (Streamlit frontend, shared UI)
    ├── --model v4_xgb_ocr_tabular  →  serve/inference.py  (XGBoost/LightGBM)
    └── --model v5_catboost_tab     →  serve/inference_v5_catboost.py  (CatBoost)
```

The UI is shared — only the prediction backend differs based on the model's `framework` field.

### Inference Modules

| Module | Frameworks | Target Transform | Categorical Handling |
|--------|-----------|-----------------|---------------------|
| `serve/inference.py` | XGBoost, LightGBM | Reads `metadata["target_transform"]`: `log1p` → expm1 inverse, `none` → raw | pandas `category` dtype |
| `serve/inference_v5_catboost.py` | CatBoost | Raw prediction (no inverse) | String dtype + `cat_indices` from metadata |

Both modules share OCR and feature extraction code. The shared functions (`_run_ocr`, `_extract_features_from_ocr`) live in `serve/inference.py` and are imported by the CatBoost module.

### Routing Logic

`app.py` reads the active model's `framework` from the registry:
- `framework == "catboost"` → uses `serve/inference_v5_catboost.py`
- Otherwise → uses `serve/inference.py`

### Files

| File | Purpose |
|------|---------|
| `serve/app.py` | Streamlit frontend. Accepts `--model` CLI arg. Routes to correct inference backend. |
| `serve/inference.py` | XGBoost/LightGBM inference. Shared OCR + feature extraction. |
| `serve/inference_v5_catboost.py` | CatBoost inference. Imports shared functions from `inference.py`. |
| `serve/model_registry.py` | Thin wrapper around `models.registry.load_model()`. |
| `serve/claude_reasoning.py` | Claude API (Haiku 4.5) for natural language analysis. Optional. |

### App CLI

```bash
# Use the registry active model (default)
streamlit run serve/app.py

# Override with a specific model
streamlit run serve/app.py -- --model v4_xgb_ocr_tabular
streamlit run serve/app.py -- --model v5_catboost_tab

# The UI shows which model is serving and whether it's from CLI override or registry
```

The `--` separator is required by Streamlit to pass arguments to the script.

### Programmatic Usage

```python
# XGBoost (V4)
from serve.inference import predict_from_image
result = predict_from_image(open("pics/test.jpg", "rb").read())

# CatBoost (V5) — specify model ID
from serve.inference_v5_catboost import predict_from_image
import serve.inference_v5_catboost as mod
mod.MODEL_ID = "v5_catboost_tab"
result = predict_from_image(open("pics/test.jpg", "rb").read())
```

## Model Performance Summary

| Model ID | Framework | R² | MAE (CNY) | Notes |
|----------|-----------|---:|----------:|-------|
| `v1_xgb_ocr_tabular` | XGBoost | -0.36 | — | 6 features, baseline |
| `v2_xgb_ocr_tabular` | XGBoost | 0.11 | — | 13 features |
| `v3_xgb_ocr_tabular` | XGBoost | 0.25 | — | 13 features + tuning |
| `v4_xgb_ocr_tabular` | XGBoost | 0.35 | — | 24 features, log1p target |
| `v5_catboost_tab` | CatBoost | **0.44** | 20,039 | 24 features, raw target, **best** |
| `v5_xgb_tab` | XGBoost | 0.34 | — | 24 features, raw target |
| `v5_lgbm_tab` | LightGBM | 0.22 | — | 24 features, raw target |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Registry-based serving** | Supports multiple models, framework-agnostic. Active model pointer makes switching instant. |
| **`--model` CLI arg** | Override active model without changing registry state. Useful for A/B testing. |
| **Separate CatBoost inference module** | CatBoost needs string categoricals + cat_indices (different from XGBoost pandas category). Separate module keeps each clean. |
| **Shared OCR/feature extraction** | OCR and feature extraction are framework-agnostic. Only the prediction step differs. |
| **metadata-driven inference** | `target_transform` field determines inverse transform. No hardcoded assumptions. |
| **`set_active=False` for research** | Research models (80/20 split) shouldn't accidentally become the serving model. Only `train_production_model.py` auto-activates. |
