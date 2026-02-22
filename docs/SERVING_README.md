# Web App Serving — Price Prediction MVP

## Overview

A Streamlit web app where users upload a card auction screenshot and get an AI-powered price prediction. The serving layer bridges the research pipeline (OCR → ETL → XGBoost V4) into a user-facing product with Claude-powered natural language reasoning.

## Architecture

```
[Streamlit App]  →  [serve/inference.py]  →  [EasyOCR + XGBoost V4]
      ↓                                            ↓
[Upload Image]   →  OCR → Feature Extraction → Model Prediction → Display
                                                    ↓
                                      [serve/claude_reasoning.py] → AI Analysis
```

### Design Principles

- **Training and serving are separated.** Training lives in `models/train_production_model.py`. Serving lives in `serve/`. The bridge is `models/saved/` on disk (model.joblib + metadata.json).
- **Deterministic pipeline, not MCP.** The pipeline is always OCR → ETL → predict. Claude adds value at the output layer (explaining predictions), not orchestrating the pipeline. MCP is for AI agent infrastructure, not web apps.
- **No MLflow (yet).** With one model version and 97 samples, `joblib` + `metadata.json` is sufficient. MLflow earns its place when V5 with a different pipeline (tabular + embeddings) arrives.
- **No abstract base classes.** With one pipeline type, a factory pattern is over-engineering. The `pipeline_type` string in `metadata.json` is the only expandability hook needed.
- **Graceful degradation.** Claude reasoning returns `None` if no API key — the core prediction always works.

## Files

### Training

| File | Purpose |
|------|---------|
| `models/train_production_model.py` | Train V4 on ALL data (no hold-out), save model artifacts to `models/saved/` |
| `models/saved/model.joblib` | Serialized XGBoost model (525KB, gitignored) |
| `models/saved/metadata.json` | Feature names, dtypes, category mappings, hyperparams, `pipeline_type` |

### Serving

| File | Purpose |
|------|---------|
| `serve/__init__.py` | Package init |
| `serve/model_registry.py` | Serving-only: `load_model()` loads pre-trained model + metadata from disk |
| `serve/inference.py` | `predict_from_image(image_bytes)`: OCR → features → price prediction |
| `serve/claude_reasoning.py` | Claude API call (Haiku 4.5) for natural language card analysis |
| `serve/app.py` | Streamlit frontend: upload image → display prediction + features + reasoning |
| `.env.example` | API key template (`ANTHROPIC_API_KEY`) |

### Modified (from research pipeline)

| File | Change |
|------|--------|
| `data/panini_card_ocr_etl.py` | Guarded top-level execution behind `if __name__ == "__main__"` so extraction functions are importable without side effects |
| `requirements.txt` | Added streamlit, anthropic, python-dotenv, joblib |
| `.gitignore` | Added `.env`, `models/saved/` |

## Usage

### Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Train and save the production model

```bash
python -m models.train_production_model
```

This trains V4 XGBoost on all 96 samples (no hold-out — performance already validated in research), tunes hyperparameters via 3-fold CV, and saves to `models/saved/`.

### Show saved model metadata

```bash
python -m models.train_production_model --info
```

### Launch the web app

```bash
streamlit run serve/app.py
```

### Use programmatically (without Streamlit)

```python
from serve.inference import predict_from_image

result = predict_from_image(open("pics/test.jpg", "rb").read())
print(result["predicted_price_cny"])    # 9158.87
print(result["extracted_features"])     # {"player_name": "Stephen Curry", ...}
print(result["ocr_lines"])             # [{"text": "...", "confidence": 0.95}, ...]
```

## Key Design Patterns

### Lazy Singleton (EasyOCR Reader)

The EasyOCR reader takes ~10s to initialize (loading language models into memory). A lazy singleton initializes it on first use and keeps it resident. In Streamlit, `@st.cache_resource` achieves this — the object is created once per app instance and reused for all subsequent requests. Auto-detects GPU via `torch.cuda.is_available()`.

### Model Persistence

Training is offline, batch, exploratory — you try many configurations and pick the best. Serving is online, single-instance, deterministic — you load ONE trained model and run it on new inputs. The bridge is **model persistence**: serializing the trained model + its metadata (feature schema, preprocessing config) so the serving layer can reproduce the exact same transformation pipeline.

### Feature Schema Enforcement

`_features_to_dataframe()` in `serve/inference.py` applies identical preprocessing to training: dtype coercion, category mappings with the exact same categories as training, NaN handling, `_parse_card_year()`. The metadata.json stores `feature_names`, `feature_dtypes`, and `category_mappings` to make this reproducible.

### Graceful Degradation

The Claude reasoning module (`serve/claude_reasoning.py`) is designed to fail silently. If no API key is set, or the API call fails, or the `anthropic` package isn't installed — it returns `None` and the app shows predictions without the analysis section. The core product (price prediction) always works; LLM integration is an enhancement, not a dependency.

## Inference Test Results

Uploaded a Stephen Curry 2022-23 Panini Prizm card:
- **OCR detected:** player name, team, card year, series, bid count (38), actual price (10,750 CNY)
- **Model predicted:** 9,159 CNY (within 15% of actual — reasonable for R²=0.35)
- **OCR lines:** 35 text lines extracted

### Performance

| Step | Latency |
|------|---------|
| First prediction (EasyOCR model loading) | ~10s |
| Subsequent predictions (OCR on image) | ~2-3s |
| Model loading (cached) | <1s |
| Claude reasoning (API call) | ~1-2s |

## Expandability

### When V5 (tabular + embeddings) arrives:

1. **Train V5:** Import from both V4 training and `data/extract_image_embeddings.py`. Concatenate tabular features with PCA-reduced image embeddings. Save with `pipeline_type: "ocr_tabular_embeddings"` in metadata.

2. **Update inference:** `_features_to_dataframe()` checks `metadata["pipeline_type"]`. If `"ocr_tabular_embeddings"`, also run ResNet50 + PCA on the uploaded image and append embedding columns to the feature DataFrame.

3. **Add MLflow:** When there are 2+ models with different pipelines to compare, wrap `model_registry.py` with MLflow experiment tracking. The `load_model()` interface stays the same — MLflow adds tracking, not new API.

4. **Add model selector:** Once MLflow manages multiple versions, add a sidebar dropdown in `app.py` to let users pick which model to use.

### When deploying to the web:

5. **Wrap in FastAPI:** `serve/` is a standalone Python package. Add `api.py` with a `POST /predict` endpoint that calls `predict_from_image()`. Streamlit becomes a thin frontend calling the API.

6. **Containerize:** Dockerfile with EasyOCR + XGBoost + model artifacts. Deploy on Railway/Render/Fly.io.

### Expandability hooks already in place:
- `metadata.json` has `pipeline_type` field (string-based routing, no framework)
- `serve/` is importable independently of Streamlit
- `model_registry.py` interface is stable (`load_model()`)
- Claude reasoning is optional (graceful fallback)

## Gaps & Known Limitations

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Model trained on log target but R² measured in log-space (CV R²=0.12) | Production model may overfit on 96 samples | Known — dataset size is the bottleneck, not the model. More data needed. |
| No price range / confidence interval | User gets point estimate only | Could add quantile regression or bootstrap in future |
| No bid count prediction | Deferred for MVP | Would need separate model or heuristic |
| EasyOCR slow on CPU (~2-3s/image) | User waits during inference | GPU auto-detected; could also pre-process with lighter OCR |
| Price visible in screenshots | Possible data leakage through embeddings (future V5) | Monitor embedding-only model performance |
| 96 training samples | Low model confidence, high variance | Core limitation — need more auction screenshots |

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| **Not MCP** | Pipeline is deterministic (OCR → ETL → predict). MCP is for AI agent tool orchestration, not web apps. |
| **Not MLflow** | One model version, 97 samples. `joblib` + `metadata.json` is sufficient. Add MLflow when V5 introduces a different pipeline. |
| **No pipeline registry** | One pipeline type. A factory pattern is over-engineering. `pipeline_type` string in metadata is enough. |
| **V4 only** | V3 and V4 share the same OCR pipeline. V4 is strictly better. Serve only V4 until V5 arrives with a different pipeline. |
| **Streamlit** | Fastest path to interactive MVP. Swap to FastAPI + React when deploying publicly. |
| **Claude Haiku 4.5** | ~$0.001/prediction. Graceful fallback if no API key. Enhancement, not dependency. |
| **Training in `models/`, serving in `serve/`** | Clean separation for future MLflow integration. Training concerns stay with training scripts. |
