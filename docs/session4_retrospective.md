# Session 4: Infrastructure & Environment — Retrospective

## 1. Context & Goal

Session 3 produced V5 CatBoost (R²=0.44) and three V5 training scripts, but the project was still a collection of research scripts. There was no way to serve predictions, switch between models, or deploy the pipeline as a product. The codebase had flat file structure with scripts at the root level.

Session 4's goal: **build the infrastructure to bridge research into production** — a serving pipeline, model registry, folder organization, and documentation.

**Success criteria:**
1. Serve predictions from a web app (upload image → get price)
2. Support multiple model frameworks with seamless switching
3. Organize the codebase into a maintainable package structure

---

## 2. Plan (What Was Intended)

### Files Created

| # | File | Purpose |
|---|------|---------|
| 1 | `serve/__init__.py` | Package init |
| 2 | `serve/app.py` | Streamlit frontend |
| 3 | `serve/inference.py` | XGBoost/LightGBM inference pipeline (OCR → features → predict) |
| 4 | `serve/inference_v5_catboost.py` | CatBoost-specific inference (string categoricals + cat_indices) |
| 5 | `serve/model_registry.py` | Thin wrapper around `models.registry` for serving layer |
| 6 | `serve/claude_reasoning.py` | Claude Haiku API for natural language card analysis |
| 7 | `models/registry.py` | Central model registry (register, load, switch, list) |
| 8 | `models/train_production_model.py` | Train on all data + register as active model |
| 9 | `data/data_utils.py` | Shared data loading for V5 (tabular, embeddings, merge) |
| 10 | `models/model_utils.py` | Shared training utilities (trials, tuning, CatBoost prep) |
| 11 | `docs/SERVING_README.md` | Serving architecture documentation |
| 12 | `.env.example` | API key template for Claude reasoning |

### Folder Reorganization

| Before | After |
|--------|-------|
| `panini_card_ocr_etl.py` (root) | `data/panini_card_ocr_etl.py` |
| `extract_panini_info.py` (root) | `data/extract_panini_info.py` |
| `panini_card_extractor_interactive.py` (root) | `data/panini_card_extractor_interactive.py` |
| `app.py` (root) | `serve/app.py` |
| `test_extractors.py` (root) | `tests/test_extractors.py` |
| `output/*.png` (analysis plots) | `results/*.png` |
| `models/analyze_*.py` | `analysis/analyze_*.py` |
| `models/model_comparison_report.py` | `analysis/model_comparison_report.py` |

---

## 3. Execution Timeline

### Feb 14 — Initial Serving Setup

**Commits:** `c4632e7`, `ff90154`

- Moved analysis scripts from `models/` to `analysis/`
- Created `serve/` package: Streamlit app, inference module, Claude reasoning, model registry wrapper
- Created `models/train_production_model.py` for full-data training + artifact persistence
- Created `docs/SERVING_README.md` with architecture docs
- Updated `requirements.txt` with Streamlit and anthropic dependencies
- Refactored `panini_card_ocr_etl.py` for importability (functions reusable without side effects)

### Feb 22 — Folder Reorg + Registry System

**Commits:** `2ae9122`, `72edfc9`, `00d3891`

- Reorganized all source files into `data/`, `models/`, `serve/`, `analysis/`, `tests/`
- Moved analysis plots from `output/` to `results/`
- Created `data/data_utils.py` and `models/model_utils.py` (shared utilities for V5)
- Created V5 training scripts (`v5_xgb`, `v5_lgbm`, `v5_catboost`) with CLI args
- **Built full model registry** (`models/registry.py`): JSON-based tracking with per-model artifact directories
- Created `serve/inference_v5_catboost.py` for CatBoost-specific categorical handling
- Added `--register` flag to all V1–V5 training scripts
- Refactored `train_production_model.py` to use registry instead of direct file persistence
- Updated `serve/app.py` with framework-aware inference routing

### Feb 28 — README

**Commits:** `a6e7c3c`, `55aa837`

- Created initial README.md with project overview, architecture, and quick start

---

## 4. Key Engineering Decisions

### Model Registry over MLflow

**Decision:** File-based JSON registry (`models/saved/registry.json`) with per-model artifact directories.

**Why:** MLflow adds significant dependency weight (database, UI server, tracking server) for a 96-sample research project. A lightweight JSON registry with `register_model()` / `load_model()` / `set_active_model()` provides the core functionality — model versioning, artifact storage, active model pointer — without operational overhead. Easy to replace with MLflow later if needed.

### Separate CatBoost Inference Module

**Decision:** `serve/inference_v5_catboost.py` exists alongside `serve/inference.py` rather than conditional logic in one module.

**Why:** CatBoost requires fundamentally different categorical handling — string-typed columns with explicit `cat_indices` metadata, versus XGBoost's pandas `category` dtype. The target inverse transform also differs (V4 XGBoost uses log1p → expm1; V5 CatBoost uses raw). Separate modules keep each clean while sharing OCR and feature extraction code via imports from `serve/inference.py`.

### Metadata-Driven Feature Schema

**Decision:** Each model artifact stores complete schema in `metadata.json` — feature names, dtypes, category mappings, target transform, hyperparameters.

**Why:** Ensures inference applies exactly the same preprocessing as training without hardcoding assumptions per model version. When the active model changes, the inference pipeline adapts automatically based on the metadata. No code changes needed to serve a new model.

### Graceful Degradation for Claude Reasoning

**Decision:** Claude API integration returns `None` if API key missing or call fails. Core predictions always work.

**Why:** The product is price prediction, not LLM analysis. Users should be able to deploy without configuring API keys. Claude analysis is an enhancement layer — it explains predictions in natural language but doesn't affect the prediction itself.

### Lazy Singleton for EasyOCR

**Decision:** EasyOCR reader initialized once via `@st.cache_resource`, cached for the lifetime of the Streamlit session.

**Why:** EasyOCR model loading takes ~10 seconds (downloading/loading language models). Without caching, every prediction would pay this cost. The singleton pattern with Streamlit's resource cache ensures one-time initialization with automatic GPU detection via `torch.cuda.is_available()`.

### Framework-Aware Inference Routing

**Decision:** `app.py` reads `framework` from the active model's registry entry and dynamically imports the correct inference module.

```
app.py
  ├── framework == "catboost"  →  serve/inference_v5_catboost.py
  └── otherwise                →  serve/inference.py
```

**Why:** Each framework has different categorical handling and target transforms. Routing at the app level keeps inference modules focused on one framework each, while the shared OCR/feature extraction layer (`_run_ocr()`, `_extract_features_from_ocr()`) avoids duplication.

### Production Model Auto-Registers as Active

**Decision:** `train_production_model.py` calls `register_model(..., set_active=True)`. Research training scripts use `set_active=False`.

**Why:** Clear separation between production and research workflows. The serving layer always has a default model. Researchers can register experimental models without disrupting production. Users can override via `--model` CLI flag for A/B testing.

---

## 5. Key Results

### Infrastructure Delivered

| Component | What It Does |
|-----------|-------------|
| **Model Registry** | 246 lines. Register, load, switch, list models. JSON-based with artifact directories. |
| **Streamlit App** | 197 lines. Upload image → OCR → features → prediction → Claude analysis. |
| **XGBoost Inference** | 228 lines. Full pipeline: EasyOCR → feature extraction → XGBoost prediction. Shared OCR functions. |
| **CatBoost Inference** | 125 lines. CatBoost-specific categorical handling + calibration auto-loading. |
| **Claude Reasoning** | 57 lines. Natural language analysis via Claude Haiku 4.5. Graceful fallback. |
| **Production Training** | 127 lines. Full-data training + registry registration + calibration fitting. |
| **Shared Utilities** | 280 lines. Data loading, trial runner, hyperparameter tuning, CatBoost prep. |

### Codebase Structure

```
panini_prediction/
├── data/          # Data pipeline (OCR, ETL, embeddings, domain knowledge)
├── models/        # Training scripts, registry, utilities
├── serve/         # Web app, inference, Claude reasoning
├── analysis/      # Experiment reports and diagnostics
├── tests/         # Unit tests
├── docs/          # Documentation and retrospectives
├── pics/          # Source auction screenshots
├── output/        # Processed data artifacts
└── results/       # Analysis plots
```

### Serving Performance

| Step | Latency |
|------|---------|
| First prediction (EasyOCR model load) | ~10s |
| Subsequent OCR | ~2–3s |
| Model prediction (cached) | <1s |
| Claude reasoning (API call) | ~1–2s |

---

## 6. Gaps & Limitations

| Gap | Impact | Status |
|-----|--------|--------|
| No batch inference | One image at a time | Sufficient for current use |
| No A/B testing infrastructure | Can't run shadow models | `--model` CLI flag is a manual workaround |
| No monitoring/alerting | No drift detection or prediction tracking | Not needed at current scale |
| No feedback loop | Predictions aren't compared to actual outcomes | Would need scraping of sale results |
| CatBoost not in production training script | `train_production_model.py` still trains V4 XGBoost | CatBoost training done via research scripts + `--register` |

---

## 7. Success Criteria Assessment

| Criterion | Target | Result | Met? |
|-----------|--------|--------|------|
| Web app serves predictions | Upload image → get price | Streamlit app with OCR, features, prediction, Claude analysis | **YES** |
| Multiple framework support | Switch between XGBoost/CatBoost | Registry + framework-aware routing | **YES** |
| Organized codebase | Clean package structure | `data/`, `models/`, `serve/`, `analysis/`, `tests/` | **YES** |

**All three success criteria met.**

---

## 8. Recommendations for Session 5

1. **Calibration** — the serving model shows regression-to-the-mean bias (overestimates cheap cards, underestimates expensive ones). Diagnose and correct with post-hoc recalibration.
2. **Upgrade production model** — `train_production_model.py` still trains V4 XGBoost. Should train V5 CatBoost (R²=0.44 vs 0.35).
3. **Prediction intervals** — users get point estimates with no confidence bounds.
4. **More data** — 96 samples is the binding constraint for model quality.
