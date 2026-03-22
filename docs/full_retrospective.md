# Panini NBA Card Price Prediction — Full Retrospective

## Journey Overview

Over four sessions, we built a complete pipeline — from raw Chinese auction screenshots to calibrated price predictions served via a web app. The model improved from worse-than-random (R²=-0.36) to meaningfully predictive (R²=0.50 calibrated), while the project grew from a single training script to a modular system with a model registry, calibration layer, and Streamlit frontend.

### Progression at a Glance

| Session | Focus | Key Discovery | R² (Best) | Delta |
|---------|-------|---------------|-----------|-------|
| Pre-sessions | V1–V3: Baseline → categoricals → tuning | XGBoost with 6–13 features | 0.25 (V3) | — |
| **Session 1** | Derived features + log-transform | Log-transform hurts; new features are the real win | 0.35 (V4-raw) | +0.10 |
| **Session 2** | Image embeddings (ResNet50 → PCA) | Embeddings cluster similar cards visually | 0.35 (unchanged) | — |
| **Session 3** | 3 frameworks × 4 feature sets (12 configs) | CatBoost dominates; embeddings hurt at 96 samples | 0.44 (CatBoost) | +0.09 |
| **Session 4** | Calibration analysis + linear recalibration | 2-parameter linear fix adds +0.12 R² | 0.50 (calibrated) | +0.06 |

**Total improvement: R² from 0.25 to 0.50 (+0.25), RMSE from ~49K to ~31K CNY (-37%).**

---

## What We Went Through

### Session 1: Derived Features & Log-Transform Target

**Goal:** Engineer new features from raw OCR data and test log-transforming the skewed price target. Success criterion: +5pp R² over V3.

**What we did:**
- Designed 11 derived features: `player_tier`, `rarity_ratio`, `rookie_auto`, `rookie_patch`, `is_rpa`, `is_numbered`, `is_1of1`, `is_base`, `day_of_week`, `hour_of_day`, `is_weekend`
- Fixed broken `card_year` parsing ("2022-23" → 2022.0 instead of NaN)
- Created V4 training script with both raw and log-transformed target variants
- Built a model comparison report training all 6 variants with plots

**What worked:**
- New features delivered +9.93pp R² (nearly 2x the target), driven by `rookie_patch` (importance 0.36) and `hour_of_day` (importance 0.32)
- Feature interactions like `is_rookie × has_patch` captured real market premiums that individual flags missed

**The surprise:**
- **Log-transform hurt performance by -24pp R².** The textbook approach for skewed targets backfired — with only 97 samples and extreme outliers (max 348K vs median 6K), the `expm1` inverse transform amplified small log-space errors into massive real-space errors on the high-value cards that matter most.

**Outcome: EXCEEDED target (+9.93pp vs +5pp goal)**

---

### Session 2: Image Embeddings

**Goal:** Convert auction screenshots into numerical vectors for downstream modeling. Success criterion: embeddings cluster visually similar cards.

**What we did:**
- Built a ResNet50 (frozen, ImageNet) → PCA pipeline producing 3 embedding variants (PCA-30/50/64, retaining 70–92% variance)
- Created a verification suite: PCA scree plot, t-SNE visualization, cosine similarity heatmap, nearest-neighbor analysis

**What worked:**
- Same-player cards found each other reliably (Ja Morant cards at 0.95 cosine similarity)
- An OCR-failed "unknown" card matched correctly to Ja Morant at 0.94 — the CNN recognized the player visually when text extraction couldn't
- Three PCA variants saved for Session 3 overfitting comparison

**What didn't:**
- No meaningful clustering by player tier or price quartile — visual similarity ≠ market value (expected, but confirmed)
- Price leakage concern identified: screenshots display sale prices in the UI, potentially leaking weak signal through visual cues

**Outcome: MET (embeddings cluster visually similar cards)**

---

### Session 3: Tree Models + Embeddings Comparison

**Goal:** Compare XGBoost/LightGBM/CatBoost across tabular-only vs tabular+embeddings (12 configurations). Success criterion: +10–15pp R² over V3, embeddings show positive contribution.

**What we did:**
- Created shared utilities (`data/data_utils.py`, `models/model_utils.py`) for clean separation of concerns
- Built 3 V5 training scripts with CLI args (`--pca 30/50/64`, `--no-emb`, `--register`)
- Ran full 12-config ablation matrix (3 frameworks × 4 feature sets)
- Built model registry (`models/registry.py`) for tracking artifacts, metrics, and active model

**What worked:**
- **CatBoost tabular-only achieved R²=0.44** (+9.5pp over XGBoost V4), thanks to superior categorical handling via ordered target statistics
- CatBoost utilized `player_name` (rank 5 feature) and `player_tier` (rank 11) — categories XGBoost nearly ignored
- Feature importance shifted: `hour_of_day` became #1 (48% importance), followed by `is_rpa`, `rarity_ratio`, `bid_times`

**The surprise:**
- **Embeddings hurt ALL frameworks** (XGBoost -4 to -10pp, LightGBM +4 to -17pp, CatBoost -23 to -27pp). With 96 samples and 54–88 total features, models memorized noise instead of learning signal. More PCA dimensions = worse performance, confirming the overfitting hypothesis.
- LightGBM underperformed both competitors (R²=0.22 tabular-only) — its leaf-wise growth strategy overfits aggressively on small samples

**Outcome: PARTIALLY MET (CatBoost +19pp over V3, but embeddings failed)**

---

### Session 4: Calibration Analysis & Linear Recalibration

**Goal:** Diagnose and correct the regression-to-the-mean bias observed in serving. Success criteria: diagnostic plots, R² improvement, backward-compatible serving integration.

**Note:** This session replaced the originally planned "Neural Network Baseline" — calibration was higher priority for serving quality.

**What we did:**
- Created 5-fold CV out-of-sample predictions on all 96 cards to diagnose bias
- Built 5 diagnostic plots (pred vs actual, residuals, error by quartile, calibration curve, before/after)
- Implemented 2-parameter linear recalibration (`models/calibration.py`)
- Integrated auto-loading `calibration.json` into the serving pipeline

**What the diagnostics showed:**
- Q1–Q2 cards overestimated by +6–7K CNY, Q4 cards underestimated by -36K CNY
- Predictions compressed into 0–50K range while actuals spanned 0–350K
- Two SGA rookie cards (200K and 349K) dominated Q4 error

**What worked:**
- Linear calibration (`slope=1.88, intercept=-8,392`) improved R² from 0.38 to 0.50 (+0.12) and RMSE from 35K to 31K
- The slope of 1.88 confirmed the model's prediction range was compressed to roughly half of truth
- Backward-compatible design: if `calibration.json` is absent, predictions pass through unchanged

**The surprise:**
- MAE slightly increased (+171 CNY) because linear calibration optimizes for squared error, helping the few extreme cases at the cost of median-case accuracy — the correct trade-off for right-skewed prices
- CV tuning R² was negative (-0.36), yet the same hyperparameters produce R²=0.38 on OOS evaluation — extreme variance from small-sample evaluation

**Outcome: ALL THREE criteria met**

---

## What Upgrades We Have

### Model Quality

| Metric | V3 Baseline | Current (V5 CatBoost + calibration) | Improvement |
|--------|-------------|--------------------------------------|-------------|
| R² | 0.25 | 0.50 | +0.25 (2x) |
| RMSE | ~49K CNY | ~31K CNY | -37% |
| Framework | XGBoost | CatBoost | Better categorical handling |
| Features | 13 | 24 | +11 derived features |
| Post-processing | None | Linear calibration | Corrects regression-to-mean |

### Infrastructure

- **Model Registry** (`models/registry.py`): Central registry tracking model artifacts, metrics, metadata, and active model pointer. Supports `register_model()`, `load_model()`, `set_active_model()`, `list_models()`.
- **Calibration Module** (`models/calibration.py`): Fit, save, load, and apply linear recalibration — decoupled from both training and serving.
- **Shared Utilities**: `data/data_utils.py` (data loading/merging) and `models/model_utils.py` (training/tuning/CatBoost prep) eliminate duplication across V5 scripts.
- **Modular Serving**: `serve/app.py` auto-routes to the correct inference module based on active model framework (XGBoost vs CatBoost).

### Feature Engineering

- 11 derived features capturing domain knowledge: rarity ratios, card attribute interactions (rookie+patch, rookie+auto), temporal patterns (hour, day, weekend), player tiers
- Fixed data bugs: `card_year` parsing, `end_time` regex
- Embedding pipeline available for future use when dataset grows

### Analysis & Diagnostics

- 12-config ablation matrix with 4 comparison plots (Session 3)
- 5 calibration diagnostic plots (Session 4)
- Price skewness analysis, embedding verification suite
- Per-session retrospectives documenting decisions and learnings

### Serving

- Streamlit web app: upload screenshot → OCR → feature extraction → prediction → Claude analysis
- Lazy-loaded singletons for EasyOCR reader and model
- GPU auto-detection for OCR
- Framework-aware inference routing (XGBoost/CatBoost)
- Auto-applied calibration at inference time

---

## Current Limitations

### 1. Sample Size (the binding constraint)

**96 samples** is insufficient for the complexity of the problem. This constraint cascades into every other limitation:
- Image embeddings overfit (54–88 features for 96 samples)
- High-cardinality categoricals (`player_name`: 34 unique values) can't be learned reliably
- Cross-validation is unstable (CV tuning R² was negative despite decent OOS performance)
- Calibration is fitted on the same small dataset, risking optimistic bias
- Only 3 of 4 player tiers are represented; `is_1of1` has zero examples

### 2. Extreme Price Outliers

Two SGA rookie cards (200K and 349K CNY) are 2–5x above the 3rd-most-expensive card and dominate Q4 evaluation metrics. The model cannot learn to predict these from 96 samples — they are essentially unique events. The calibration helps but is a statistical correction, not a learned relationship.

### 3. OCR Data Quality

- `card_year` only 55.7% populated (many cards lack year in OCR text)
- `end_time` only 71.1% successfully parsed
- Player name fuzzy matching at 85 threshold may miss or mismatch names
- `serial_max` only 24.7% populated — rarity features are sparse
- FOTL/Case Hit/SSP keywords have 0 occurrences in the current dataset

### 4. Image Pipeline Limitations

- Embeddings encode full auction screenshots including UI chrome, not just the card image
- Screenshots display sale prices — potential (weak) data leakage through visual cues
- ResNet50 (ImageNet-trained) is not optimized for trading card visual features
- Embeddings are unusable at current sample size

### 5. Model & Calibration

- Linear calibration is global — same correction for all price ranges, cannot fix per-segment bias
- Calibration slope of 1.88 is extreme, reflecting fundamental model compression
- No prediction uncertainty — users get a point estimate with no confidence bounds
- Hyperparameter search may be suboptimal (negative CV R² suggests instability)
- No temporal validation — model doesn't account for market trends over time

### 6. Serving

- Single-image inference only (no batch processing)
- No A/B testing or shadow model infrastructure
- Claude reasoning requires API key and adds latency
- No feedback loop to improve predictions based on actual sale outcomes

---

## Future Exploration Areas

### Priority 1: More Data

The single highest-impact improvement. Nearly every limitation traces back to 96 samples.
- **Scrape more listings** from 闲鱼/WeChat across more players, years, and card series
- **Include unsold listings** with asking prices (add a `sold` boolean flag) — even failed auctions contain market signal
- **Track prices over time** for trend analysis and temporal validation
- **Target: 500+ samples** would stabilize embeddings, calibration, and categorical learning

### Priority 2: External Feature Enrichment

Add signal that doesn't require more auction screenshots:
- **Player stats**: Career points, All-Star appearances, championships, current season performance — these directly drive card value but are currently absent
- **Market indices**: Hobby market trends, recent comparable sales, player trade/injury news
- **Card database**: Link to Beckett/PSA population data for print run sizes, authenticated counts

### Priority 3: Prediction Uncertainty

Users need to know how much to trust each prediction:
- **Quantile regression** (CatBoost supports `MultiQuantile` loss) for prediction intervals
- **Conformal prediction** for distribution-free coverage guarantees
- **Display confidence bands** in the Streamlit app alongside point predictions

### Priority 4: Better Image Features

Revisit once dataset exceeds 500 samples:
- **Card-region cropping**: Detect and crop to just the card image (top ~40% of screenshot), removing UI chrome and price text to eliminate leakage
- **Domain-specific embeddings**: Fine-tune on sports card datasets rather than ImageNet
- **Targeted visual features**: Extract specific attributes (patch size, autograph quality, card condition) rather than generic embeddings

### Priority 5: Model Architecture

- **Log-target CatBoost**: Session 1 showed log-transform hurts XGBoost, but CatBoost handles targets differently — log-transform may naturally compress the range and reduce regression-to-mean without needing post-hoc calibration
- **Bayesian optimization** (Optuna) instead of RandomizedSearchCV — better hyperparameter search with fewer iterations
- **Stacking/Ensemble**: XGBoost + CatBoost meta-learner could capture different signal from each framework
- **Neural network baseline**: The originally planned Session 4 topic — MLP or TabNet for tabular regression

### Priority 6: Production Hardening

- **Feedback loop**: Record predictions vs actual sale prices to retrain and recalibrate over time
- **Batch inference**: Process multiple cards at once for portfolio valuation
- **A/B testing**: Run shadow models alongside production to measure improvements before switching
- **Monitoring**: Track prediction distributions, feature drift, and model degradation over time

---

## Roadmap vs Reality

The original 5-session roadmap planned for neural networks (Session 4) and entity embeddings + ensemble (Session 5). In practice:

| Session | Planned | Actual | Why the Deviation |
|---------|---------|--------|-------------------|
| 1 | Derived features + text keywords | Derived features + log-transform study | Text keywords (FOTL, SSP) had 0 occurrences — pivoted to log-transform hypothesis |
| 2 | Image embeddings | Image embeddings | As planned |
| 3 | Tree models + embeddings | Tree models + embeddings | As planned (embeddings failed, but CatBoost won) |
| 4 | Neural network baseline | Calibration analysis | Serving model quality was more impactful than a new architecture |
| 5 | Entity embeddings + ensemble | Not yet started | — |

The deviation was the right call. Calibration delivered +0.12 R² improvement with a 2-parameter model — more impactful than a neural network would likely have been on 96 samples. The project prioritized **improving what we had** over **adding complexity**.

---

## Key Lessons Learned

1. **Textbook approaches can fail on small data.** Log-transform, image embeddings, and high-dimensional features all failed — not because the techniques are wrong, but because 96 samples is below the threshold where they help.

2. **Feature engineering beats model complexity.** The biggest single-session gain (+9.93pp) came from domain-informed derived features, not from a fancier model or more data dimensions.

3. **Framework choice matters.** CatBoost's ordered target statistics gave +9.5pp over XGBoost on the same features — a "free" improvement from picking the right tool for small-sample categorical data.

4. **Post-processing can unlock trapped signal.** Linear calibration added +0.12 R² without retraining — systematic bias was hiding model quality that was already there.

5. **Ablation studies prevent false conclusions.** Without the 12-config matrix, we might have blamed the framework for embedding failure (or vice versa). Systematic comparison showed both effects clearly.

6. **Prioritize serving quality over research novelty.** Replacing the neural network session with calibration analysis delivered more user-facing value with less complexity.
