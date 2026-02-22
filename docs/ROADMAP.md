# Panini Price Prediction - Future Roadmap

## Current State

| Version | Approach | Features | Best R² |
|---------|----------|----------|---------|
| V1 | Baseline XGBoost | 6 features (bool + numeric) | -0.36 |
| V2 | + More features | 13 features (+ categorical) | 0.11 |
| V3 | + Hyperparameter tuning | RandomizedSearchCV | 0.25 |
| V4-raw | + Derived features | 24 features, raw target | 0.35 |
| **V5 CatBoost** | **+ Framework comparison** | **24 tabular (CatBoost)** | **0.44** |
| V5 + embeddings | + Image embeddings | 24 tab + PCA-30/50/64 | 0.17-0.30 (hurts) |

---

## Areas for Exploration

### 1. More Data

- Scrape more listings (more players, years, series)
- Include unsold listings with asking price (with a "sold" flag)
- Track prices over time for trend analysis

### 2. Feature Engineering

- **Derived features:** `serial_number / serial_max` (rarity ratio), `days_until_end`
- **Interactions:** `is_rookie × is_autograph`, `player × team`
- **Target transform:** Log-transform `price_cny` (prices often skewed)
- **Player stats:** Career points, All-Star appearances, championships

### 3. Different Models

| Model | Why try it |
|-------|-----------|
| LightGBM | Faster, native categorical, often matches XGBoost |
| CatBoost | Best native categorical handling |
| Random Forest | Simpler baseline, less prone to overfitting |
| Neural Network | Can learn complex non-linear patterns |
| Ensemble | Stack XGB + LGBM + CatBoost for robustness |

### 4. Image Data (High Potential)

Images contain signals text features miss:

- **Card condition:** Centering, corners, surface (even graded cards vary)
- **Visual appeal:** Refractor shine, patch quality/size, autograph placement
- **Player pose:** Action shot vs portrait affects desirability

**Approaches:**

- CNN (ResNet/EfficientNet) pretrained → extract embeddings → combine with tabular
- Vision Transformer (ViT) for patch-based analysis
- Multimodal model (image + tabular jointly)

### 5. Text/NLP Features

- Embed `card_features` text with sentence transformers
- Extract keywords: "1/1", "FOTL", "case hit", "SSP"

### 6. Temporal Features

- Day of week, month (holiday demand spikes)
- Player recent performance (injury, trade, playoff run)
- Market trends (hobby boom/bust cycles)

---

## Suggested Priority

1. **Target log-transform** - Quick win, handles price skew
2. **Image embeddings** - High signal, differentiates similar cards
3. **Try LightGBM/CatBoost** - Easy swap, often better
4. **External player data** - If available

---

## Next 5 Sessions

### Session 1: Derived Features & Text Keywords

**Topic**: Feature engineering for tabular data

**Implement**:
- Log-transform `price_cny` target (handle skewness)
- Rarity ratio: `serial_number / serial_max`
- Interaction features: `rookie_auto`, `graded_high`
- Temporal features: `day_of_week`, `hour_of_day` from `end_time`
- Text keyword booleans: `is_fotl`, `is_case_hit`, `is_ssp`, `is_1of1`
- Player tier mapping (star / all-star / regular)

**Deliverables**: `train_price_regressor_v4.py`, updated `data/panini_card_ocr_etl.py`

**Success Criteria**: +5% R² over V3

---

### Session 2: Image Embeddings

**Topic**: Transfer learning for visual features

**Implement**:
- Pretrained ResNet50/EfficientNet (frozen weights)
- Image preprocessing: resize 224×224, ImageNet normalization
- Extract 2048-d vectors → PCA reduce to 64-128 dims
- Save embeddings to `output/image_embeddings.csv`

**Deliverables**: `extract_image_embeddings.py`

**Success Criteria**: Embeddings cluster similar cards (visual inspection)

---

### Session 3: Tree Models with Embeddings ✅

**Topic**: Gradient boosting with visual features

**Implement**:
- V5 feature set: original + derived + image embeddings
- LightGBM implementation (native categorical)
- CatBoost implementation (best categorical handling)
- Ablation study: compare with/without image embeddings

**Deliverables**: `train_price_regressor_v5_xgb.py`, `_v5_lgbm.py`, `_v5_catboost.py`, `data/data_utils.py`, `models/model_utils.py`, `analysis/session3_comparison_report.py`

**Results**: CatBoost tabular-only is best (R²=0.44, +9.5pp over XGBoost V4). Embeddings hurt all frameworks — overfitting on 96 samples with 54-88 features. See `docs/session3_retrospective.md`.

**Success Criteria**: +10-15% R² over V3, image embeddings show positive contribution

---

### Session 4: Neural Network Baseline

**Topic**: PyTorch MLP for tabular regression

**Implement**:
- Data pipeline: StandardScaler, train/val/test split (60/20/20)
- Baseline MLP: Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Output
- Training loop: MSE loss, Adam optimizer, early stopping

**Deliverables**: `nn/data_loader.py`, `nn/models.py`, `train_price_nn_v1.py`

**Success Criteria**: Match or exceed best tree model performance

---

### Session 5: Entity Embeddings & Ensemble

**Topic**: Learnable categorical representations + model combination

**Implement**:
- Entity embeddings: player(16d), team(8d), series(8d), parallel(4d)
- Hyperparameter tuning (learning rate, dropout, embedding dims)
- Ensemble: best tree + best NN (weighted average)
- Final comparison report

**Deliverables**: `train_price_nn_v2.py`, `train_ensemble.py`

**Success Criteria**: +5% over baseline MLP, ensemble achieves best overall R²

---

## Success Metrics Summary

| Session | Target |
|---------|--------|
| 1 - Derived Features | +5% R² over V3 |
| 2 - Image Embeddings | Clusters visually verified |
| 3 - Tree + Embeddings | +10-15% R² over V3 |
| 4 - Baseline MLP | Match best tree |
| 5 - Entity + Ensemble | Best overall R² |
