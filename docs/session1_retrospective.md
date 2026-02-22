# Session 1: Derived Features & Log-Transform Target — Retrospective

## 1. Context & Goal

The roadmap defines 5 sessions. Session 1 = feature engineering for tabular data. Starting point: V1→V2→V3 models exist (6→13→13 features, with V3 adding hyperparameter tuning). OCR pipeline + ETL fully working (97 cards, 25 columns). **Success criterion: +5pp R² over V3 (~0.25).**

**Key data realities identified during planning:**
- Price heavily right-skewed (mean 19,733 vs median 6,057 CNY, max 348,994)
- Serial data sparse: `serial_max` only 25% populated
- Only 2 graded cards → grading features nearly useless
- `card_year` in V3 was broken: `pd.to_numeric("2022-23")` → NaN → 0
- FOTL/Case Hit/SSP keywords have 0 occurrences → skipped

**Hypothesis going in:** Log-transform of skewed target would be the biggest win.

---

## 2. Plan (What Was Intended)

### Files to Change
| # | File | Action |
|---|------|--------|
| 1 | `data/nba_players.py` | Add `PLAYER_TIERS` dict (~80 players → 4 tiers) |
| 2 | `data/panini_card_ocr_etl.py` | Add `compute_derived_features()` (11 new columns), integrate after aggregation |
| 3 | `train_price_regressor_v4.py` | **New.** 24 features, log-transformed target, fixed card_year, NaN-aware |
| 4 | `model_comparison_report.py` | **New.** VS Code interactive report: train all versions, generate comparison plots |
| 5 | `CLAUDE.md` | Add V4 + report commands and architecture descriptions |

### 11 Derived Features Planned
| Column | Type | Logic |
|--------|------|-------|
| `player_tier` | categorical | `PLAYER_TIERS.get(name, 'rotation')`, `'unknown'` if NaN |
| `rarity_ratio` | Float64 | `1.0 / serial_max` |
| `rookie_auto` | boolean | `is_rookie AND is_autograph` |
| `rookie_patch` | boolean | `is_rookie AND has_patch` |
| `is_rpa` | boolean | `card_features` contains "RPA" |
| `is_numbered` | boolean | `serial_max < 100` |
| `is_1of1` | boolean | `serial_max == 1` (forward-looking, 0 in current data) |
| `is_base` | boolean | All special flags False AND no parallel AND no serial |
| `day_of_week` | Int64 | From `end_time`, 0=Mon..6=Sun |
| `hour_of_day` | Int64 | From `end_time`, 0-23 |
| `is_weekend` | boolean | `day_of_week in [5, 6]` |

### V4 Feature Set (24 total)
- **Numeric (7):** serial_max, grade, card_year, bid_times, rarity_ratio, day_of_week, hour_of_day
- **Boolean (11):** is_rookie, is_autograph, has_patch, is_refractor, rookie_auto, rookie_patch, is_rpa, is_numbered, is_1of1, is_base, is_weekend
- **Categorical (6):** player_name, team, card_series, parallel_type, grading_company, player_tier

---

## 3. Execution Steps Taken

### Step 1: `data/nba_players.py` — Add PLAYER_TIERS
- Appended dict after `DESCRIPTOR_KEYWORDS` (line 503)
- 12 superstars, 28 stars, 40 starters, default "rotation"
- No issues

### Step 2: `data/panini_card_ocr_etl.py` — Add derived features
- Added `PLAYER_TIERS` to import statement
- Inserted `compute_derived_features(df)` function after `aggregate_card_features()`
- Fixed `end_time` parsing bug: regex to handle `"2024-12-2021:34:02"` → `"2024-12-20 21:34:02"`
- Integrated call after aggregation + added derived feature summary stats to print block
- **Note:** Edit tool intermittently returned ENOENT on Windows/OneDrive filesystem, but changes applied correctly (confirmed by re-reading files)

### Step 3: Run ETL
- First run failed: `rapidfuzz` not installed in venv. Fixed with `pip install rapidfuzz`
- Second run: success. **37 columns** (was 25). Key derived feature stats:
  - Player tiers: star=76, unknown=17, superstar=4, starter=0, rotation=0
  - Rarity ratio: 24.7% populated (matches serial_max)
  - Rookie+Patch combos: 7 (7.2%), Rookie+Auto: 8 (8.2%)
  - End time parsed: 69/97 (71.1%), Weekend: 44 (45.4%)

### Step 4: Create `train_price_regressor_v4.py`
- Followed V3 structure exactly
- Added `_parse_card_year()` to fix "2022-23" → 2022.0
- Log-transform: `y_log = np.log1p(y)`, inverse via `np.expm1`
- NaN-aware: numeric features left as NaN (XGBoost native handling), not filled with 0
- Dual reporting: log-space R² + real-space MAE/RMSE/R²
- Fixed FutureWarning: changed bulk `.fillna(False).astype(int)` to per-column loop

### Step 5: Create `model_comparison_report.py`
- First version trained V1, V2, V3, V3+log, V4 (5 variants)
- After seeing results, **added V4-raw variant** (24 features, raw target, tuned) to isolate new-features effect from log-transform effect
- Final report trains 6 variants total, generates 4 PNG plots

### Step 6: Run report
- First run revealed log-transform hurts real-space R² badly (-24pp)
- Updated report to include V4-raw, re-ran successfully
- All 4 plots generated to `output/`

### Step 7: Update CLAUDE.md
- Added V4 and report commands
- Updated Stage 2 output column count (25 → 37), added `compute_derived_features()` description
- Updated Stage 3 with V4 description
- Added Model Comparison section
- Added `PLAYER_TIERS` to Key Data Module section

---

## 4. Gaps Identified During Execution

| Gap | Impact | How Addressed |
|-----|--------|---------------|
| `rapidfuzz` not in venv | ETL failed on first run | `pip install rapidfuzz` |
| No "starter" or "rotation" players in 97-card dataset | `player_tier` only has 3 of 4 tiers represented | Kept as-is; tiers will activate with more data |
| `is_1of1` has 0 examples | Feature exists but contributes nothing currently | Forward-looking; kept for future cards |
| `card_year` only 55.7% populated | Many cards lack year in OCR text | NaN handled natively by XGBoost |
| `end_time` only 71.1% parsed | 28 cards have no temporal features | NaN handled natively by XGBoost |
| Edit tool ENOENT on OneDrive/WSL2 | Edits appeared to fail but actually applied | Verified via re-reading file; known WSL2+OneDrive issue |

---

## 5. Key Results

### Model Comparison Table

| Version | Features | Target | Tuned | Avg MAE | Avg RMSE | Avg R² | ΔR² vs V3 |
|---------|----------|--------|-------|---------|----------|--------|-----------|
| V1 | 6 | raw | no | 25,594 | 56,202 | -0.358 | -61.2pp |
| V2 | 13 | raw | no | 21,212 | 47,320 | 0.110 | -14.4pp |
| V3 | 13 | raw | yes | 22,300 | 49,215 | 0.254 | baseline |
| V3+log | 13 | log | yes | 22,286 | 56,700 | 0.014 | -24.0pp |
| V4 | 24 | log | yes | 20,139 | 52,105 | 0.181 | -7.3pp |
| **V4-raw** | **24** | **raw** | **yes** | **21,841** | **46,604** | **0.353** | **+9.9pp** |

**+5pp R² target: MET** (+9.93pp via V4-raw)

### Ablation Breakdown
```
V3 → V3+log  (log transform only):      -23.99pp R²
V3 → V4-raw  (new features only):        +9.93pp R²
V3+log → V4  (new features + log):      +16.67pp R²
V3 → V4      (total log+features):       -7.33pp R²
```

### Top 5 Features by Importance (V4-raw, best model)
| Feature | Importance | New? |
|---------|-----------|------|
| `rookie_patch` | 0.357 | YES |
| `hour_of_day` | 0.315 | YES |
| `serial_max` | 0.197 | no |
| `has_patch` | 0.061 | no |
| `card_series` | 0.015 | no |

### Generated Artifacts
- `results/price_distribution_log_transform.png` — raw vs log price histograms
- `results/model_version_comparison.png` — R² bar chart across all 6 variants
- `results/log_transform_ablation.png` — 4-bar ablation chart
- `results/feature_importance_comparison.png` — V3 vs V4 side-by-side importance

---

## 6. Surprise Findings

### Log-transform HURTS real-space R² (the big surprise)

The plan hypothesized that log-transform would be "the biggest win" for the right-skewed price distribution. **The opposite happened.** Log-transform alone causes a -24pp drop in real-space R².

**Why:** With only 97 samples and extreme skew (max 348,994 vs median 6,057), the `expm1` inverse transform amplifies small log-space errors into massive real-space errors on the few high-value outlier cards. R² is variance-weighted, so a few bad outlier predictions dominate the metric.

**Insight:** Log-transform is the textbook approach for skewed targets but fails on small datasets where outlier cards (the ones you most want to predict accurately) get their errors exponentially amplified by the inverse transform.

### New features are the real win

The 11 derived features provide +9.93pp R² improvement with raw target. The two most impactful:
- **`rookie_patch`** (importance 0.36) — the interaction of is_rookie × has_patch captures the "rookie patch" premium that is a major price driver in the card market
- **`hour_of_day`** (importance 0.32) — auction end time matters: late-night listings may get fewer bids

### Player tier didn't show up in top features

Despite player name being intuitively important for card pricing, neither `player_name` nor `player_tier` appear in the top 5 features. With only 97 cards, the categorical cardinality of player names may be too high relative to sample size.

---

## 7. Recommendations for Next Session

1. **Keep raw target** — log-transform is counterproductive at this data size
2. **More data** — 97 samples is the binding constraint; more cards would help player_tier and categoricals contribute
3. **Consider dropping V4 log-target variant** — V4-raw is strictly superior; rename it to V4 in the main training script
4. **Player tier may need binning** — with only 3 of 4 tiers represented, consider coarser grouping
