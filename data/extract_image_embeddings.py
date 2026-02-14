# %% [markdown]
# # Image Embedding Extraction
# Extract visual features from card auction screenshots using pretrained ResNet50.
# Outputs PCA-reduced embeddings at 30, 50, and 64 dimensions for overfitting comparison.

# %%
"""
Extract image embeddings from card auction screenshots using pretrained ResNet50.

Reads images from pics/, passes through frozen ResNet50 (avgpool layer -> 2048-d),
applies StandardScaler + PCA to reduce dimensions, saves three CSV variants.

Usage:
    python data/extract_image_embeddings.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# %% [markdown]
# ## Configuration

# %%
PICS_DIR = PROJECT_ROOT / 'pics'
OUTPUT_DIR = PROJECT_ROOT / 'output'
PCA_VARIANTS = [30, 50, 64]
PCA_MAX = max(PCA_VARIANTS)


# %% [markdown]
# ## Image Preprocessing

# %%
def get_preprocessing_transform():
    """Standard ImageNet preprocessing for ResNet50."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# %% [markdown]
# ## Model Loading

# %%
def load_resnet50_feature_extractor():
    """Load pretrained ResNet50, remove classification head, set to eval mode."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove final FC layer to get 2048-d avgpool output
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


# %% [markdown]
# ## Embedding Extraction

# %%
def extract_single_embedding(image_path, model, transform):
    """Extract 2048-d embedding for a single image."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        embedding = model(img_tensor)  # [1, 2048, 1, 1]

    return embedding.squeeze().numpy()  # [2048]


def extract_all_embeddings(pics_dir, model, transform):
    """Extract embeddings for all images in directory.

    Returns:
        filenames: list of image filenames
        embeddings: numpy array of shape (n_images, 2048)
    """
    filenames = []
    embeddings = []
    errors = []

    image_files = sorted([
        f for f in os.listdir(pics_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(image_files)} images in {pics_dir}")

    for i, filename in enumerate(image_files):
        filepath = os.path.join(pics_dir, filename)
        try:
            emb = extract_single_embedding(filepath, model, transform)
            filenames.append(filename)
            embeddings.append(emb)
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"  Processed {i + 1}/{len(image_files)}: {filename}")
        except Exception as e:
            errors.append((filename, str(e)))
            print(f"  ERROR processing {filename}: {e}")

    if errors:
        print(f"\n{len(errors)} images failed:")
        for fname, err in errors:
            print(f"  {fname}: {err}")

    embeddings_array = np.stack(embeddings)
    print(f"\nExtracted {embeddings_array.shape[0]} embeddings, shape: {embeddings_array.shape}")

    return filenames, embeddings_array


# %% [markdown]
# ## PCA Dimensionality Reduction

# %%
def reduce_dimensions(embeddings, n_components):
    """Apply StandardScaler + PCA to reduce embedding dimensionality.

    Args:
        embeddings: numpy array (n_images, 2048)
        n_components: target dimensions (must be < n_images)

    Returns:
        reduced: numpy array (n_images, n_components)
        pca: fitted PCA object
        scaler: fitted StandardScaler object
    """
    n_samples = embeddings.shape[0]
    if n_components >= n_samples:
        print(f"WARNING: n_components ({n_components}) >= n_samples ({n_samples}). "
              f"Reducing to {n_samples - 1}.")
        n_components = n_samples - 1

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings_scaled)

    # Report variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA: {n_components} components from {embeddings.shape[1]} features")
    print(f"Cumulative variance explained:")
    for k in PCA_VARIANTS:
        if k <= n_components:
            print(f"  {k:3d} components: {cumvar[k-1]*100:.1f}%")

    return reduced, pca, scaler


# %% [markdown]
# ## Save Embeddings

# %%
def save_embeddings_csv(filenames, reduced_embeddings, n_components, output_dir):
    """Save PCA-reduced embeddings as CSV for a specific component count."""
    sliced = reduced_embeddings[:, :n_components]
    col_names = [f'emb_{i}' for i in range(n_components)]

    df = pd.DataFrame(sliced, columns=col_names)
    df.insert(0, 'image', filenames)

    output_path = output_dir / f'image_embeddings_pca{n_components}.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  Saved {output_path.name}: {df.shape[0]} rows x {df.shape[1]} cols")

    return df


# %% [markdown]
# ## Main

# %%
def main():
    print("=" * 60)
    print("IMAGE EMBEDDING EXTRACTION")
    print("=" * 60)

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Load model
    print("\nLoading pretrained ResNet50...")
    model = load_resnet50_feature_extractor()
    transform = get_preprocessing_transform()
    print("Model loaded (frozen, eval mode).")

    # Extract raw embeddings
    print(f"\nExtracting 2048-d embeddings from {PICS_DIR}...")
    filenames, raw_embeddings = extract_all_embeddings(PICS_DIR, model, transform)

    # PCA reduction (fit once at max, slice for smaller variants)
    print(f"\nReducing dimensions via PCA (max {PCA_MAX} components)...")
    reduced, pca, scaler = reduce_dimensions(raw_embeddings, n_components=PCA_MAX)

    # Save PCA variance data for verification script
    np.save(str(OUTPUT_DIR / 'pca_explained_variance_ratio.npy'),
            pca.explained_variance_ratio_)

    # Save three variants
    print(f"\nSaving embedding CSVs...")
    for n in PCA_VARIANTS:
        n_actual = min(n, reduced.shape[1])
        save_embeddings_csv(filenames, reduced, n_actual, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(filenames)}")
    print(f"Raw embedding dim: {raw_embeddings.shape[1]}")
    print(f"PCA variants saved: {PCA_VARIANTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
