# Codebase Knowledge Reference

Quick-reference for Claude sessions. Eliminates the need to re-explore the codebase from scratch each time.

Last updated: Session 2 (2026-02-14)

---

## 1. Project Structure

```
panini_prediction/
├── CLAUDE.md                              # Project instructions for Claude Code
├── requirements.txt                       # Python dependencies
├── panini_card_extractor_interactive.py   # Stage 1: EasyOCR extraction
├── extract_panini_info.py                 # Stage 1: PaddleOCR extraction (original)
├── panini_card_ocr_etl.py                 # Stage 2: Feature extraction ETL
├── test_extractors.py                     # Unit tests for feature extractors
├── data/
│   ├── nba_players.py                     # Domain knowledge (players, teams, tiers, mappings)
│   └── extract_image_embeddings.py        # Stage 2.5: ResNet50 → PCA embeddings
├── models/
│   ├── train_price_regressor.py           # V1: 6 features, baseline XGBoost
│   ├── train_price_regressor_v2.py        # V2: 13 features + categoricals
│   ├── train_price_regressor_v3.py        # V3: V2 + RandomizedSearchCV
│   ├── train_price_regressor_v4.py        # V4: 24 features + derived + log target
│   ├── model_comparison_report.py         # Interactive report: all versions + ablation
│   └── analyze_price_skewness.py          # Price distribution visualization
├── analysis/
│   └── verify_embeddings.py               # Embedding verification: PCA, t-SNE, similarity
├── docs/
│   ├── ROADMAP.md                         # 5-session feature roadmap
│   ├── FEATURE_EXTRACTION_README.md       # Feature extraction documentation
│   ├── session1_retrospective.md          # Session 1 learnings
│   ├── session2_retrospective.md          # Session 2 learnings
│   └── codebase_knowledge.md              # This file
├── pics/                                  # 97 source images (auction screenshots)
├── output/                                # All generated artifacts
│   ├── raw_ocr_output/                    # 97 .txt files (OCR per image)
│   ├── panini_cards_extracted.csv         # Main dataset (97 rows × 37 cols)
│   ├── image_embeddings_pca30.csv         # Embeddings: 97 × 31 (image + 30 dims)
│   ├── image_embeddings_pca50.csv         # Embeddings: 97 × 51 (image + 50 dims)
│   ├── image_embeddings_pca64.csv         # Embeddings: 97 × 65 (image + 64 dims)
│   ├── pca_explained_variance_ratio.npy   # Saved PCA variance for verification
│   ├── pca_variance_analysis.png          # Scree plot + cumulative variance curve
│   ├── image_embedding_tsne.png           # t-SNE by player tier + price quartile
│   ├── image_similarity_analysis.png      # Cosine similarity heatmap + tier comparison
│   ├── model_version_comparison.png       # R² bar chart across V1-V4
│   ├── feature_importance_comparison.png  # V3 vs V4 feature importance
│   ├── price_distribution_log_transform.png
│   ├── log_transform_ablation.png
│   ├── price_skewness_analysis.png
│   ├── ocr_data.csv                       # Legacy long-format OCR data
│   ├── ocr_quality_report.csv             # Feature completeness stats
│   └── cards_needing_review.csv           # Low-confidence rows flagged for review
└── .venv/                                 # Python 3.12.3 virtual environment
```

---

## 2. Key Functions by File

### `panini_card_ocr_etl.py` (Stage 2 ETL)
| Function | Lines | Purpose |
|----------|-------|---------|
| `load_ocr_data()` | 383-407 | Load `.txt` files from raw_ocr_output/ → long-format DataFrame |
| `aggregate_card_features()` | 418-553 | Group by image → one row per card with all extracted features |
| `extract_player_name()` | 150-177 | Fuzzy-match against NBA_PLAYERS (threshold 85%) |
| `extract_card_series()` | 179-197 | Regex: `(20\d{2}[-/]?\d{2})\s+(Panini\s+[A-Za-z\s]+?)` |
| `extract_serial_numbers()` | 199-236 | Parse `X/Y编` patterns |
| `extract_grading()` | 238-275 | Find PSA/BGS/SGC/CSG grades |
| `extract_temporal()` | 277-299 | Extract end_time and product_id |
| `extract_descriptors()` | 301-359 | Find autograph/rookie/patch/refractor/RPA flags |
| `extract_team()` | 361-370 | Map team keywords → full team names |
| `compute_derived_features()` | 556-613 | Add 11 engineered columns from existing features |
| `extract_price()` | ~372-381 | Parse `半` (misread `¥`) prices, distinguish shipping via `运费` |

### `data/nba_players.py` (Domain Knowledge)
| Export | Purpose |
|--------|---------|
| `NBA_PLAYERS` | ~300 player names for fuzzy matching |
| `CHINESE_NAME_MAP` | Chinese → English player names (e.g., "库里" → "Stephen Curry") |
| `TEAM_MAPPINGS` | Chinese/English/abbreviation → full team names |
| `PARALLEL_MAPPINGS` | Chinese color terms → English (e.g., "银折" → "Silver") |
| `DESCRIPTOR_KEYWORDS` | Regex patterns for card attributes |
| `PLAYER_TIERS` | ~80 players → 4 tiers (superstar/star/starter/rotation) |

### `data/extract_image_embeddings.py` (Stage 2.5)
| Function | Purpose |
|----------|---------|
| `get_preprocessing_transform()` | Resize(256) → CenterCrop(224) → ImageNet normalize |
| `load_resnet50_feature_extractor()` | ResNet50 IMAGENET1K_V2, remove FC head, eval mode |
| `extract_all_embeddings()` | Loop 97 images → 2048-d vectors |
| `reduce_dimensions()` | StandardScaler → PCA(64), slice for 30/50 variants |
| `save_embeddings_csv()` | Save per-variant CSV with `image, emb_0, ..., emb_N` |

### `models/train_price_regressor_v4.py` (Best Model)
| Key Detail | Value |
|------------|-------|
| Features | 24 total (7 numeric + 11 boolean + 6 categorical) |
| Target | `price_cny` (raw, NOT log-transformed — log hurts R²) |
| Tuning | RandomizedSearchCV, 20 iterations, 3-fold CV |
| Trials | 10 random seeds, reports average MAE/RMSE/R² |
| Categoricals | Native XGBoost `enable_categorical=True` |
| NaN handling | Native XGBoost (NaN left as-is, not filled) |

---

## 3. Dataset Profile: `panini_cards_extracted.csv`

**Shape:** 97 rows × 37 columns

### Column Inventory

**Identifier (1):**
- `image` — WeChat filename, links to `pics/` (primary key)

**Card Identity (5):**
- `product_id` — flea market product ID
- `player_name` — fuzzy-matched player (82% coverage)
- `team` — team name (77% coverage)
- `card_year` — e.g., "2022-23" (56% coverage)
- `card_series` — e.g., "Panini Prizm" (56% coverage)

**Serial Data (3):**
- `card_number` — card number in set
- `serial_number` — formatted "5/25"
- `serial_max` — max serial (25% coverage, sparse)

**Boolean Flags (4):**
- `is_rookie`, `is_autograph`, `has_patch`, `is_refractor`

**Descriptors (2):**
- `parallel_type` — e.g., "Panini Silver"
- `card_features` — concatenated features string

**Grading (3):**
- `grading_company`, `grade`, `grade_subtype` — only 2 graded cards total

**Prices & Auction (6):**
- `price_cny` — **TARGET** (mean 19,733, median 6,057, max 348,994 CNY)
- `price_usd`, `shipping_price`, `bid_times`, `end_time`, `sold_status`

**Quality Metadata (2):**
- `ocr_avg_confidence`, `needs_review`

**Derived Features (11):** added by `compute_derived_features()`
- `player_tier` — categorical: superstar(4)/star(76)/unknown(17)/starter(0)/rotation(0)
- `rarity_ratio` — 1/serial_max (25% populated)
- `rookie_auto`, `rookie_patch` — interaction flags
- `is_rpa`, `is_numbered`, `is_1of1`, `is_base` — boolean flags
- `day_of_week`, `hour_of_day`, `is_weekend` — temporal (71% populated)

### Key Data Characteristics
- Price heavily right-skewed: mean 19,733 vs median 6,057 CNY
- Only 2 graded cards → grading features not useful for modeling
- `serial_max` only 25% populated → sparse but important (3rd most important feature)
- Player tier distribution: star dominates (76/97), only 4 superstars
- `card_year` parsing: "2022-23" → 2022.0 via regex (V4 fixed this from V3's broken `pd.to_numeric`)

---

## 4. Image Data Profile

- **97 images** in `pics/`: 57 JPG + 40 PNG
- **Filename pattern:** `微信图片_YYYYMMDDhhmmss_N_1274.{jpg|png}` (WeChat screenshots)
- **Two aspect ratios:** ~0.46 portrait (78 images, phone) and ~1.44 landscape (19 images, tablet)
- **Content:** Full auction screenshots — card photo in upper 30-40%, rest is UI/text/pricing
- **Spatial zones** (from ETL, portrait images):
  - `card_image`: Y=250-700
  - `player_info`: Y=700-1250
  - `description`: Y=1250-1600
  - `pricing`: Y=1600-1900
  - `metadata`: Y=2000-2600
- **1:1 mapping** to CSV via `image` column

---

## 5. Embedding Data Profile

| Variant | File | Columns | Cumulative Variance | Total Features (with 24 tabular) |
|---------|------|---------|--------------------|---------------------------------|
| PCA-30 | `image_embeddings_pca30.csv` | image + 30 emb | 70.4% | 54 |
| PCA-50 | `image_embeddings_pca50.csv` | image + 50 emb | 85.1% | 74 |
| PCA-64 | `image_embeddings_pca64.csv` | image + 64 emb | 91.8% | 88 |

- **Pipeline:** ResNet50 (IMAGENET1K_V2, frozen) → 2048-d avgpool → StandardScaler → PCA
- **Verification results:**
  - Nearest-neighbor strong: Ja Morant cards find each other (cosine sim 0.948)
  - t-SNE shows no strong tier clustering (visual ≠ market — expected)
  - Within-tier vs between-tier similarity differences small (~0.01-0.03)
- **Known risk:** price visible in screenshots → possible data leakage via visual cues

---

## 6. Model Performance Summary

| Version | Features | Target | Tuned | Avg R² | Notes |
|---------|----------|--------|-------|--------|-------|
| V1 | 6 | raw | no | -0.358 | Baseline, boolean + numeric only |
| V2 | 13 | raw | no | 0.110 | + categorical features |
| V3 | 13 | raw | yes | 0.254 | + RandomizedSearchCV |
| V3+log | 13 | log | yes | 0.014 | Log-transform hurts (-24pp) |
| V4-log | 24 | log | yes | 0.181 | + derived features, log target |
| **V4-raw** | **24** | **raw** | **yes** | **0.353** | **Best model (+10pp over V3)** |

### Key Findings
- **Log-transform hurts R²** on this small skewed dataset — inverse expm1 amplifies outlier errors
- **Best approach:** raw target + more features (V4-raw)
- **Top features:** rookie_patch (0.36), hour_of_day (0.32), serial_max (0.20)
- **Player tier not in top features** — categorical cardinality too high for 97 samples

---

## 7. Installed Packages (Key)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.9.1 | Deep learning (embedding extraction) |
| torchvision | 0.24.1 | Pretrained models, image transforms |
| xgboost | 3.1.2 | Gradient boosting models |
| scikit-learn | 1.8.0 | PCA, StandardScaler, metrics, RandomizedSearchCV |
| pandas | 2.3.3 | DataFrames, CSV I/O |
| numpy | 2.2.6 | Numerical arrays |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical plots |
| easyocr | 1.7.2 | OCR extraction (Chinese + English) |
| rapidfuzz | 3.14.3 | Fuzzy string matching |
| pillow | 12.0.0 | Image loading |
| openpyxl | 3.1.5 | Excel export |

Python: 3.12.3 | venv at `.venv/bin/activate`

---

## 8. Conventions & Gotchas

- **`# %%` cell markers** — all scripts use VS Code Interactive Window format
- **CSV encoding** — `utf-8-sig` for Excel compatibility with Chinese characters
- **`matplotlib.use('Agg')`** — non-interactive backend for all plot-generating scripts
- **`PROJECT_ROOT`** — `Path(__file__).resolve().parent.parent` for scripts in subdirs, `.parent` for root-level
- **Edit tool ENOENT** — WSL2 + OneDrive causes intermittent ENOENT on edits, but changes still apply
- **OCR price convention** — `半` = misread of `¥` in Chinese OCR text
- **Confidence thresholds** — player_name(0.50), card_series(0.60), price(0.70), product_id(0.85), grades(0.75)
- **Fuzzy match threshold** — 85% similarity via rapidfuzz token_sort_ratio
