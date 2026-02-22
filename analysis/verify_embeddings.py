# %% [markdown]
# # Image Embedding Verification
# Observability and fact-checking for image embeddings extracted by
# `data/extract_image_embeddings.py`. Generates PCA variance analysis,
# t-SNE clustering, cosine similarity heatmap, and nearest-neighbor checks.

# %%
"""
Verify image embedding quality through multiple analyses.

Reads embedding CSVs and card metadata, produces:
- results/pca_variance_analysis.png  (scree plot + cumulative variance)
- results/image_embedding_tsne.png   (t-SNE by player tier + price quartile)
- results/image_similarity_analysis.png (cosine similarity heatmap + tier comparison)
- Console: nearest-neighbor analysis for sample cards

Usage:
    python analysis/verify_embeddings.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = PROJECT_ROOT / 'output'
RESULTS_DIR = PROJECT_ROOT / 'results'
CSV_PATH = DATA_DIR / 'panini_cards_extracted.csv'
PCA_VARIANTS = [30, 50, 64]
PCA_MAX = max(PCA_VARIANTS)
EMBEDDINGS_PATH = DATA_DIR / f'image_embeddings_pca{PCA_MAX}.csv'
PCA_VARIANCE_PATH = DATA_DIR / 'pca_explained_variance_ratio.npy'


# %% [markdown]
# ## Load Data

# %%
def load_data():
    """Load embedding CSV and card metadata, merge on image column."""
    embeddings_df = pd.read_csv(EMBEDDINGS_PATH)
    cards_df = pd.read_csv(CSV_PATH)

    emb_cols = [c for c in embeddings_df.columns if c.startswith('emb_')]
    merged = embeddings_df.merge(
        cards_df[['image', 'player_tier', 'price_cny', 'player_name', 'card_series', 'team']],
        on='image', how='left'
    )

    print(f"Loaded {len(merged)} cards with {len(emb_cols)} embedding dimensions")
    return merged, emb_cols


# %% [markdown]
# ## PCA Variance Analysis

# %%
def plot_pca_variance(output_dir):
    """Generate scree plot and cumulative variance curve with 30/50/64 cutoffs.

    Uses saved variance ratios from the original 2048->64 PCA reduction,
    not a re-computed PCA on already-reduced data.
    """
    explained = np.load(str(PCA_VARIANCE_PATH))
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Scree plot
    components = np.arange(1, len(explained) + 1)
    axes[0].bar(components, explained * 100, color='steelblue', alpha=0.7, width=1.0)
    for cutoff in PCA_VARIANTS:
        if cutoff <= len(explained):
            axes[0].axvline(x=cutoff, color='red', linestyle='--', alpha=0.7,
                          label=f'PCA-{cutoff}')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot: Variance per Component')
    axes[0].legend()

    # Right: Cumulative variance curve
    axes[1].plot(components, cumulative * 100, color='steelblue', linewidth=2)
    for cutoff in PCA_VARIANTS:
        if cutoff <= len(cumulative):
            pct = cumulative[cutoff - 1] * 100
            axes[1].plot(cutoff, pct, 'ro', markersize=8)
            axes[1].annotate(f'PCA-{cutoff}: {pct:.1f}%',
                           xy=(cutoff, pct),
                           xytext=(cutoff + 2, pct - 3),
                           fontsize=10, color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance (%)')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'pca_variance_analysis.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA variance analysis to {output_path}")

    # Print summary table
    print("\nVariance explained summary:")
    for cutoff in PCA_VARIANTS:
        if cutoff <= len(cumulative):
            print(f"  PCA-{cutoff:2d}: {cumulative[cutoff-1]*100:.1f}% cumulative variance")


# %% [markdown]
# ## t-SNE Visualization

# %%
def plot_tsne(merged, emb_cols, output_dir):
    """Generate t-SNE scatter plots colored by player tier and price quartile."""
    X_emb = merged[emb_cols].values

    perplexity = min(30, len(X_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(X_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: color by player_tier
    tier_order = ['superstar', 'star', 'starter', 'rotation', 'unknown']
    tier_colors = {
        'superstar': '#e74c3c', 'star': '#3498db', 'starter': '#2ecc71',
        'rotation': '#95a5a6', 'unknown': '#bdc3c7'
    }
    for tier in tier_order:
        mask = merged['player_tier'] == tier
        if mask.any():
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                          c=tier_colors[tier], label=tier, alpha=0.7,
                          s=50, edgecolors='white', linewidth=0.5)
    axes[0].set_title('t-SNE of Image Embeddings (by Player Tier)')
    axes[0].legend()
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Right: color by price quartile
    prices = merged['price_cny'].fillna(0)
    quartiles = pd.qcut(prices, q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'],
                        duplicates='drop')
    q_colors = {
        'Q1 (low)': '#2ecc71', 'Q2': '#3498db',
        'Q3': '#f39c12', 'Q4 (high)': '#e74c3c'
    }
    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        mask = quartiles == q
        if mask.any():
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                          c=q_colors[q], label=q, alpha=0.7,
                          s=50, edgecolors='white', linewidth=0.5)
    axes[1].set_title('t-SNE of Image Embeddings (by Price Quartile)')
    axes[1].legend()
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    output_path = output_dir / 'image_embedding_tsne.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualization to {output_path}")


# %% [markdown]
# ## Cosine Similarity Analysis

# %%
def plot_similarity_heatmap(merged, emb_cols, output_dir):
    """Generate cosine similarity heatmap and within/between tier comparison."""
    X_emb = merged[emb_cols].values
    tiers = merged['player_tier'].fillna('unknown')

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(X_emb)

    # Sort by player_tier for visual clustering
    tier_order = ['superstar', 'star', 'starter', 'rotation', 'unknown']
    tier_rank = tiers.map({t: i for i, t in enumerate(tier_order)})
    sort_idx = tier_rank.argsort()
    sim_sorted = sim_matrix[sort_idx][:, sort_idx]
    tiers_sorted = tiers.iloc[sort_idx].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Heatmap
    im = axes[0].imshow(sim_sorted, cmap='RdYlBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Add tier boundary lines
    boundaries = []
    for tier in tier_order:
        count = (tiers_sorted == tier).sum()
        if count > 0:
            boundaries.append(count)
    cumsum = np.cumsum(boundaries[:-1])
    for b in cumsum:
        axes[0].axhline(y=b - 0.5, color='black', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=b - 0.5, color='black', linewidth=0.5, alpha=0.5)

    axes[0].set_title('Cosine Similarity (sorted by Player Tier)')
    axes[0].set_xlabel('Card Index')
    axes[0].set_ylabel('Card Index')

    # Right: Within-tier vs between-tier similarity
    within_sims = {}
    between_sims = {}
    for tier in tier_order:
        mask = tiers == tier
        n_tier = mask.sum()
        if n_tier < 2:
            continue
        tier_indices = np.where(mask)[0]
        other_indices = np.where(~mask)[0]

        # Within-tier: upper triangle of sub-matrix
        sub = sim_matrix[np.ix_(tier_indices, tier_indices)]
        triu_idx = np.triu_indices(n_tier, k=1)
        within_sims[tier] = sub[triu_idx].mean()

        # Between-tier
        if len(other_indices) > 0:
            cross = sim_matrix[np.ix_(tier_indices, other_indices)]
            between_sims[tier] = cross.mean()

    tiers_present = [t for t in tier_order if t in within_sims]
    x_pos = np.arange(len(tiers_present))
    width = 0.35

    within_vals = [within_sims[t] for t in tiers_present]
    between_vals = [between_sims.get(t, 0) for t in tiers_present]

    axes[1].bar(x_pos - width/2, within_vals, width, label='Within Tier',
               color='steelblue', alpha=0.8)
    axes[1].bar(x_pos + width/2, between_vals, width, label='Between Tiers',
               color='coral', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(tiers_present, rotation=15)
    axes[1].set_ylabel('Average Cosine Similarity')
    axes[1].set_title('Within-Tier vs Between-Tier Similarity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'image_similarity_analysis.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity analysis to {output_path}")

    # Print summary
    print("\nSimilarity summary:")
    for tier in tiers_present:
        w = within_sims[tier]
        b = between_sims.get(tier, float('nan'))
        diff = w - b
        print(f"  {tier:12s}: within={w:.3f}  between={b:.3f}  delta={diff:+.3f}")


# %% [markdown]
# ## Nearest Neighbor Analysis

# %%
def print_nearest_neighbors(merged, emb_cols, n_neighbors=3):
    """Print nearest neighbors for sample cards (one per player tier)."""
    X_emb = merged[emb_cols].values
    sim_matrix = cosine_similarity(X_emb)

    print("\n" + "=" * 70)
    print("NEAREST NEIGHBOR ANALYSIS")
    print("=" * 70)

    # Pick one sample card per tier
    tier_order = ['superstar', 'star', 'starter', 'rotation', 'unknown']
    for tier in tier_order:
        tier_mask = merged['player_tier'] == tier
        if not tier_mask.any():
            continue

        # Pick first card in this tier
        sample_idx = merged[tier_mask].index[0]
        sample = merged.iloc[sample_idx]

        print(f"\nQuery: [{tier}] {sample.get('player_name', 'N/A')} "
              f"| {sample.get('card_series', 'N/A')} "
              f"| {sample.get('team', 'N/A')} "
              f"| price={sample.get('price_cny', 'N/A')}")

        # Get top neighbors (excluding self)
        sims = sim_matrix[sample_idx].copy()
        sims[sample_idx] = -1  # exclude self
        top_indices = np.argsort(sims)[::-1][:n_neighbors]

        for rank, idx in enumerate(top_indices, 1):
            neighbor = merged.iloc[idx]
            sim_score = sims[idx]
            same_player = (sample.get('player_name', '') == neighbor.get('player_name', '')
                          and pd.notna(sample.get('player_name')))
            same_tier = (sample.get('player_tier', '') == neighbor.get('player_tier', ''))

            markers = []
            if same_player:
                markers.append("SAME PLAYER")
            if same_tier:
                markers.append("same tier")

            marker_str = f" <{'  '.join(markers)}>" if markers else ""
            print(f"  #{rank} (sim={sim_score:.3f}): [{neighbor.get('player_tier', 'N/A')}] "
                  f"{neighbor.get('player_name', 'N/A')} "
                  f"| {neighbor.get('card_series', 'N/A')} "
                  f"| price={neighbor.get('price_cny', 'N/A')}"
                  f"{marker_str}")

    print("\n" + "=" * 70)


# %% [markdown]
# ## Main

# %%
def main():
    print("=" * 60)
    print("IMAGE EMBEDDING VERIFICATION")
    print("=" * 60)

    merged, emb_cols = load_data()

    print("\n--- PCA Variance Analysis ---")
    plot_pca_variance(RESULTS_DIR)

    print("\n--- t-SNE Visualization ---")
    plot_tsne(merged, emb_cols, RESULTS_DIR)

    print("\n--- Cosine Similarity Analysis ---")
    plot_similarity_heatmap(merged, emb_cols, RESULTS_DIR)

    print("\n--- Nearest Neighbor Analysis ---")
    print_nearest_neighbors(merged, emb_cols)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Plots saved to {RESULTS_DIR}/:")
    print(f"  - pca_variance_analysis.png")
    print(f"  - image_embedding_tsne.png")
    print(f"  - image_similarity_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
