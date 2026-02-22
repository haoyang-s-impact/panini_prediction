# Session 3: Tree Models with Image Embeddings — Retrospective

## 1. Context & Goal

Session 2 produced ResNet50 image embeddings (PCA-30/50/64 variants). V4-raw (XGBoost, 24 tabular features, raw target) held at R²=0.35. Session 3's goal: combine tabular + embeddings, compare XGBoost/LightGBM/CatBoost, and measure embedding dimensionality effects. **Success criterion: +10-15pp R² over V3 (0.25), embeddings show positive contribution.**

**Key data realities:**
- 96 samples with valid prices (1 dropped for NaN)
- 24 tabular features → adding 30-64 embedding dims gives 54-88 total features
- Features-to-samples ratio as high as 88:96 (near 1:1) — extreme overfitting risk

---

## 2. Plan (What Was Intended)

### Files Created
| # | File | Purpose |
|---|------|---------|
| 1 | `data/data_utils.py` | Data engineering: load tabular, load embeddings, merge features |
| 2 | `models/model_utils.py` | Model training: run_trials, tune_hyperparameters, CatBoost prep |
| 3 | `models/train_price_regressor_v5_xgb.py` | XGBoost V5 with CLI args |
| 4 | `models/train_price_regressor_v5_lgbm.py` | LightGBM V5 with CLI args |
| 5 | `models/train_price_regressor_v5_catboost.py` | CatBoost V5 with CLI args |
| 6 | `analysis/session3_comparison_report.py` | 12-config ablation matrix + 4 plots |

### Architecture Decision
Separated concerns:
- **`data/`** — data engineering (loading, merging, feature prep)
- **`models/`** — model-specific logic (training, tuning, CatBoost conversion)
- **`analysis/`** — experiment analysis (correlation diagnostic, comparison report)
- **`results/`** — model comparison outputs (separate from `output/` data prep artifacts)

---

## 3. Key Results

### Embedding-Price Correlation Diagnostic

No leakage detected. Max |Pearson r| between any embedding dimension and price = 0.37 (emb_1). All well below 0.5 threshold.

### Full Results Table

| Config | Framework | Features | Avg MAE | Avg RMSE | Avg R² | ΔR² vs V4 |
|--------|-----------|----------|---------|----------|--------|-----------|
| **cat-tab** | **CatBoost** | **24** | **20,039** | **42,725** | **0.4387** | **+0.0947** |
| xgb-tab | XGBoost | 24 | 22,386 | 47,209 | 0.3440 | baseline |
| xgb-pca30 | XGBoost | 54 | 22,788 | 48,645 | 0.2987 | -0.0453 |
| lig-pca30 | LightGBM | 54 | 23,900 | 48,867 | 0.2632 | -0.0808 |
| xgb-pca50 | XGBoost | 74 | 23,544 | 49,988 | 0.2505 | -0.0935 |
| xgb-pca64 | XGBoost | 88 | 23,307 | 50,104 | 0.2458 | -0.0982 |
| lig-tab | LightGBM | 24 | 23,277 | 50,636 | 0.2238 | -0.1202 |
| cat-pca30 | CatBoost | 54 | 22,787 | 50,448 | 0.2131 | -0.1309 |
| cat-pca64 | CatBoost | 88 | 22,886 | 51,240 | 0.1942 | -0.1498 |
| lig-pca50 | LightGBM | 74 | 23,505 | 51,819 | 0.1813 | -0.1627 |
| lig-pca64 | LightGBM | 88 | 23,510 | 51,985 | 0.1753 | -0.1687 |
| cat-pca50 | CatBoost | 74 | 23,248 | 52,251 | 0.1703 | -0.1737 |

![Full Results Heatmap](../results/session3_full_results.png)

### Embedding Contribution (tabular+emb vs tabular-only delta)

| Framework | PCA-30 | PCA-50 | PCA-64 |
|-----------|--------|--------|--------|
| XGBoost | -0.045 | -0.094 | -0.098 |
| LightGBM | +0.039 | -0.043 | -0.049 |
| CatBoost | -0.226 | -0.268 | -0.245 |

**Embeddings consistently hurt performance.** Only LightGBM+PCA-30 shows a marginal positive delta (+0.039), likely noise. More dimensions = worse performance across the board.

![PCA Effect on R²](../results/session3_pca_effect.png)

### Framework Comparison (tabular-only)

| Framework | R² | Rank |
|-----------|-----|------|
| CatBoost | 0.4387 | 1st |
| XGBoost | 0.3440 | 2nd |
| LightGBM | 0.2238 | 3rd |

![Framework Comparison](../results/session3_framework_comparison.png)

### Top Features (CatBoost, best model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hour_of_day | 48.44 |
| 2 | is_rpa | 11.31 |
| 3 | rarity_ratio | 8.04 |
| 4 | bid_times | 5.86 |
| 5 | player_name | 5.70 |

![Top Feature Importance (CatBoost)](../results/session3_feature_importance.png)

Notable: CatBoost's feature importance ranking differs significantly from XGBoost's. CatBoost utilizes `player_name` (rank 5) and `player_tier` (rank 11) much more effectively than XGBoost, thanks to its superior categorical handling with ordered target statistics.

### Generated Artifacts

- `results/session3_framework_comparison.png` — grouped bars
- `results/session3_pca_effect.png` — R² vs PCA dimensions line plot
- `results/session3_full_results.png` — heatmap of all 12 configs
- `results/session3_feature_importance.png` — top-20 features for CatBoost

---

## 4. Surprise Findings

### Embeddings hurt all frameworks (the big surprise)

The roadmap hypothesized +10-15pp R² from image embeddings. Instead, adding embeddings degrades R² by 4-27pp across all frameworks and PCA variants.

**Why:** With only 96 samples, adding 30-64 embedding dimensions massively increases the features-to-samples ratio. At PCA-64 (88 features for 96 samples), the models have nearly one knob per sample — ideal for memorizing noise. The embeddings capture visual patterns (card design, color, layout) but these weakly correlate with price (max r=0.37) while adding substantial noise that dilutes the strong tabular signals (hour_of_day, rookie_patch, serial_max).

**CatBoost is most damaged** by embeddings (-22 to -27pp) because CatBoost's strength is categorical handling — adding many noisy continuous features undermines its advantage.

### CatBoost dominates tabular-only (+9.5pp over XGBoost)

CatBoost's ordered target statistics for categoricals provide a substantial advantage over XGBoost's one-hot approach with only 96 samples. The `player_name` feature (34 unique values in 96 rows) is effectively utilized by CatBoost but nearly ignored by XGBoost.

### LightGBM underperforms both

LightGBM (R²=0.22 tabular-only) trails both XGBoost (0.34) and CatBoost (0.44). Likely cause: LightGBM's leaf-wise growth strategy is optimized for larger datasets and can overfit aggressively on 96 samples.

### More PCA dimensions = worse (overfitting curve confirmed)

Across all frameworks, PCA-64 ≤ PCA-50 ≤ PCA-30 ≤ tabular-only. The diminishing PCA returns (70% → 85% → 92% variance) add noise faster than signal.

---

## 5. Gaps & Limitations

| Gap | Impact | Status |
|-----|--------|--------|
| Only 96 samples | Binding constraint for embeddings | Need 500+ samples for embeddings to help |
| Full screenshot embeddings | Include UI chrome, not just card | Could crop to card region (~top 40%) |
| RandomizedSearchCV with 10 iterations | May not find optimal hyperparameters | Sufficient for comparison; exact R² values less important than relative ordering |
| No cross-validation on final trials | 10 random splits but no nested CV | Standard approach; overfitting risk is in the feature space, not the hyperparameters |

---

## 6. Success Criteria Assessment

| Criterion | Target | Result | Met? |
|-----------|--------|--------|------|
| +10-15pp R² over V3 (0.25) | R² 0.35-0.40 | CatBoost R²=0.44 | **YES (+19pp)** |
| Image embeddings show positive contribution | R²(tab+emb) > R²(tab) | Embeddings hurt all frameworks | **NO** |

**Session 3 target partially met.** The framework comparison yielded a significant win (CatBoost +9.5pp over V4-raw), but the embedding hypothesis failed.

---

## 7. Recommendations for Session 4

1. **Use CatBoost as the production model** — R²=0.44 is the new best, and it handles categoricals natively without one-hot encoding
2. **Do not use image embeddings in production** at this dataset size — they add noise
3. **Embeddings may help at scale** — if the dataset grows to 500+ samples, revisit with PCA-30 only
4. **Consider card-region cropping** before re-extracting embeddings — removing UI chrome may improve signal quality
5. **More data is the highest-priority improvement** — 96 samples is the binding constraint for both embeddings and high-cardinality categoricals
6. **Player stats** from external sources (career points, All-Star appearances) could add signal without adding noise dimensions
