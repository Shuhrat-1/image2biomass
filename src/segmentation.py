"""Unsupervised segmentation via DINOv2 patch tokens (path B).

No masks are required: DINOv2 patch embeddings cluster by semantics, so k-means
on the patch tokens of one image yields regions (green / dead / soil). The class
fractions can be used as features or as a standalone biomass proxy, and the
masks are a strong visual for the report and video.

This module is analysis-only — it does not touch the training loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

try:
    import config
    import dinov2_features
except ModuleNotFoundError:
    from src import config
    from src import dinov2_features


def segment_image(image_path: Path, n_clusters: int = 3,
                  random_state: int = 42) -> dict:
    """Cluster one image's patch tokens into regions.

    Returns:
      - 'labels': (grid_h, grid_w) cluster id per patch.
      - 'fractions': (n_clusters,) fraction of patches per cluster.
      - 'grid': (h, w) patch grid size.
    """
    patches, (gh, gw) = dinov2_features.extract_patch_tokens(image_path)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(patches)              # (n_patches,)
    frac = np.bincount(labels, minlength=n_clusters) / len(labels)
    return {"labels": labels.reshape(gh, gw),
            "fractions": frac, "grid": (gh, gw)}


def order_clusters_by_greenness(image_path: Path, seg: dict) -> np.ndarray:
    """Map cluster ids to a green-ness ranking using mean RGB of each region.

    Returns an array mapping cluster_id -> rank (0 = most green). Helps label
    clusters consistently as green / dead / soil across images.
    """
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB").resize(
        (config.DINOV2_IMG_SIZE, config.DINOV2_IMG_SIZE)))
    gh, gw = seg["grid"]
    patch_px = config.DINOV2_IMG_SIZE // gh
    labels = seg["labels"]
    greenness = []
    for c in range(int(labels.max()) + 1):
        ys, xs = np.where(labels == c)
        if len(ys) == 0:
            greenness.append(-1)
            continue
        # average excess-green (2G - R - B) over the cluster's patches
        vals = []
        for y, x in zip(ys, xs):
            block = img[y*patch_px:(y+1)*patch_px, x*patch_px:(x+1)*patch_px]
            r, g, b = block[..., 0].mean(), block[..., 1].mean(), block[..., 2].mean()
            vals.append(2*g - r - b)
        greenness.append(np.mean(vals))
    return np.argsort(np.argsort(-np.array(greenness)))  # rank, 0 = greenest


def visualize_segmentation(image_path: Path, n_clusters: int = 3):
    """Show the original image next to its cluster mask (for report/video)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    seg = segment_image(image_path, n_clusters=n_clusters)
    img = Image.open(image_path).convert("RGB").resize(
        (config.DINOV2_IMG_SIZE, config.DINOV2_IMG_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(seg["labels"], cmap="viridis")
    axes[1].set_title(f"DINOv2 unsupervised segmentation (k={n_clusters})")
    axes[1].axis("off")
    fig.tight_layout()
    plt.show()
    return seg


def compute_fraction_features(wide, n_clusters: int = 3,
                              img_dir: Path = config.LABELLED_IMG_DIR) -> np.ndarray:
    """Cluster-fraction features for every image (greenness-ordered for consistency).

    Returns (N, n_clusters) array aligned with wide, where columns are ordered
    by greenness (col 0 = greenest region fraction).
    """
    feats = []
    for p in wide["image_path"]:
        path = Path(img_dir) / Path(p).name
        seg = segment_image(path, n_clusters=n_clusters)
        rank = order_clusters_by_greenness(path, seg)
        ordered = np.zeros(n_clusters)
        for cluster_id, r in enumerate(rank):
            ordered[r] = seg["fractions"][cluster_id]
        feats.append(ordered)
    return np.array(feats, dtype=np.float32)
