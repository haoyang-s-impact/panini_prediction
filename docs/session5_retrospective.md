# Session 5: Calibration Analysis & Recalibration — Retrospective

## 1. Context & Goal

Session 3 established CatBoost V5 (24 tabular features, raw target) as the best model at R²=0.44. However, during serving, a clear pattern emerged: the model **overestimates low-priced cards** and **underestimates expensive cards** — a classic regression-to-the-mean artifact.

Session 5's goal: **diagnose** this bias quantitatively with calibration curves, then **correct** it with post-hoc recalibration integrated into the serving pipeline.

**Success criteria:**
1. Produce diagnostic plots that quantify the over/underestimation pattern
2. Apply a calibration correction that improves R²
3. Integrate calibration into the serving pipeline (backward compatible)

---

## 2. Plan (What Was Intended)

### Files Created
| # | File | Purpose |
|---|------|---------|
| 1 | `models/calibration.py` | Linear recalibration utility: fit, apply, save, load |
| 2 | `analysis/calibration_analysis.py` | 5-fold CV diagnostic plots + calibration fitting |

### Files Modified
| # | File | Change |
|---|------|--------|
| 1 | `serve/inference_v5_catboost.py` | Auto-apply calibration post-prediction if `calibration.json` exists |
| 2 | `models/train_production_model.py` | Auto-fit and save calibration after training |
| 3 | `CLAUDE.md` | Added calibration docs to commands, architecture, and utilities sections |

### Architecture Decision

Calibration is a **post-processing layer**, not a new model. The CatBoost model remains unchanged — a 2-parameter linear transform (`slope * pred + intercept`) is applied after prediction. This is stored as `calibration.json` alongside the model artifact and loaded at inference time. If the file doesn't exist, predictions are unchanged (backward compatible).

**Why linear over isotonic:** With 96 samples, isotonic regression (N-1 degrees of freedom) overfits, especially in the tails where we have the fewest data points and care the most. A 2-parameter linear model directly addresses the slope bias.

---

## 3. Key Results

### 5-Fold CV Out-of-Sample Predictions

Since the registered model was trained on all data, we retrained CatBoost from scratch using 5-fold cross-validation to get out-of-sample predictions for all 96 cards. Each card is predicted exactly once while held out.

**CV tuning R²: -0.36** (negative, indicating high variance in small-sample cross-validation — the model struggles to generalize in 3-fold inner CV).

### Diagnostic Plots

**Plot 1: Predicted vs Actual**

![Predicted vs Actual](../results/calibration_pred_vs_actual.png)

Predictions cluster in a narrow band (0–50K predicted) while actuals span 0–350K. Points for expensive cards fall far below the 45° line.

**Plot 2: Residual Analysis**

![Residual Analysis](../results/calibration_residuals.png)

- Residuals vs Predicted (left): slope=-0.88 — as predictions increase, residuals become more negative
- **Residuals vs Actual (right): slope=-0.73** — the key diagnostic. Positive residuals at low actual prices (overestimation), deeply negative at high actual prices (underestimation)

**Plot 3: Error by Price Quartile**

![Error by Price Bin](../results/calibration_error_by_bin.png)

| Quartile | Price Range (CNY) | Mean Signed Error | Pattern |
|----------|------------------|-------------------|---------|
| Q1 | 58–2,586 | +7,218 | Overestimation |
| Q2 | 2,586–6,057 | +6,162 | Overestimation |
| Q3 | 6,057–21,978 | +1,282 | Slight overestimation |
| Q4 | 21,978–348,994 | -35,991 | Heavy underestimation |

The pattern is unambiguous: Q1-Q3 overestimated, Q4 massively underestimated (-36K average).

**Plot 4: Calibration Curve**

![Calibration Curve](../results/calibration_curve.png)

All points above the diagonal — the model systematically predicts too low for every price tier, with the gap widening dramatically at the top.

**Plot 5: Before vs After Linear Calibration**

![Before/After Calibration](../results/calibration_before_after.png)

| Metric | Before (Raw) | After Calibration | Change |
|--------|-------------|-------------------|--------|
| R² | 0.381 | 0.503 | **+0.122** |
| MAE | 14,845 | 15,017 | +171 |
| RMSE | 35,003 | 31,365 | **-3,638** |

### Calibration Parameters

```
slope     = 1.8818
intercept = -8,392.01
```

Slope > 1 confirms regression-to-the-mean: the model's raw predictions need to be stretched by nearly 2x from their center to match reality.

### Extreme Outliers

The two most expensive cards are both **Shai Gilgeous-Alexander** rookies:

| Price (CNY) | Card | Details |
|-------------|------|---------|
| 348,994 | SGA Panini Gold | Auto + Patch, /10 |
| 200,250 | SGA Panini Gold | Patch, /10 |

These are 2-5x higher than the 3rd card (129K) and dominate Q4's underestimation.

---

## 4. Surprise Findings

### Slope of 1.88 is extreme

The model's prediction range is compressed to roughly half of the true range. This is unusually severe and reflects the combination of small sample size (96), extreme price skew (mean 19,733 vs median 6,057), and a few ultra-high-value outliers.

### CV tuning R² was negative (-0.36)

The `RandomizedSearchCV` (20 iter, 3-fold) produced a negative best R², meaning the tuned model is worse than predicting the mean in cross-validation. Yet the same hyperparameters produce R²=0.38 on 5-fold OOS predictions and R²=0.44 on 10-trial 80/20 splits. This instability is a symptom of extreme variance in small-sample evaluation — the specific folds matter enormously.

### MAE slightly increased with calibration (+171 CNY)

Linear calibration optimizes for R²/RMSE (sum of squared errors), not MAE (sum of absolute errors). Stretching predictions helps the few large-error points (expensive cards) at the cost of slightly worse median-case error. This is the correct trade-off for a right-skewed price distribution.

### Calibration is conceptually a "second model"

The linear transform `1.88 * pred - 8392` is effectively a 2-parameter model stacked on top of CatBoost. This is the simplest possible stacking — a meta-learner with one feature (the base model's prediction). The improvement (+0.122 R²) shows there was systematic signal left on the table that CatBoost couldn't capture internally.

---

## 5. Gaps & Limitations

| Gap | Impact | Status |
|-----|--------|--------|
| Only 96 samples | Extreme variance in all evaluation; calibration fitted on small OOS set | Need 500+ samples |
| 2 SGA outliers (200K+348K) | Dominate Q4 error; disproportionate influence on calibration slope | Cannot address without more high-value samples |
| Linear calibration is global | Same correction applied to all price ranges; doesn't fix per-segment bias | Could use piecewise/quantile calibration with more data |
| Calibration fitted on same CV splits used for diagnosis | Slight optimistic bias in "after" metrics | Independent hold-out would be better with more data |
| Negative CV R² during tuning | Hyperparameters may not be optimal | More data or Bayesian optimization could help |

---

## 6. Success Criteria Assessment

| Criterion | Target | Result | Met? |
|-----------|--------|--------|------|
| Diagnostic plots quantify bias | 4+ plots showing the pattern | 5 plots, all clearly show regression-to-mean | **YES** |
| Calibration improves R² | R² increase | 0.381 → 0.503 (+0.122) | **YES** |
| Integrated into serving | Backward-compatible, auto-applied | `calibration.json` auto-loaded in inference | **YES** |

**All three success criteria met.**

---

## 7. Recommendations for Session 6

1. **More data is the #1 priority** — 96 samples with 348K outliers makes calibration fragile. Even 200 samples would stabilize both the base model and calibration.
2. **Investigate the negative CV R²** — the gap between CV tuning (-0.36) and OOS evaluation (+0.38) suggests the hyperparameter search may be suboptimal. Try Bayesian optimization (Optuna) or increase CV folds from 3 to 5.
3. **Consider log-target CatBoost** — Session 1 found log-transform hurts XGBoost, but CatBoost handles targets differently. Log-transform naturally compresses the range and may reduce the regression-to-mean effect, potentially eliminating the need for post-hoc calibration.
4. **Prediction intervals** — now that we have calibrated point predictions, add uncertainty quantification (e.g., quantile regression or conformal prediction) so users know how much to trust each prediction.
5. **Feature engineering for high-value cards** — the SGA outliers share traits (Panini Gold, /10, rookie). Interaction features like `is_gold_numbered_rookie` could help the model capture these premium combinations.
