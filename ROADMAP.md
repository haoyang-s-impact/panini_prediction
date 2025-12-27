# Panini Price Prediction - Future Roadmap

## Current State

| Version | Approach | Features |
|---------|----------|----------|
| V1 | Baseline XGBoost | 6 features (bool + numeric) |
| V2 | + More features | 13 features (+ categorical) |
| V3 | + Hyperparameter tuning | RandomizedSearchCV |

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
