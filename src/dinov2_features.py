"""DINOv2 feature extraction (lazy-loaded, cached).

Loads a frozen DINOv2 ViT once and exposes two views of each image:
- CLS token  -> a global embedding for regression.
- patch tokens -> per-region embeddings for unsupervised segmentation.

Because the backbone is frozen, embeddings are deterministic and computed once,
then cached to disk (npz) and reused across all CV folds — far faster than
running the ViT every epoch.

Heavy imports (torch, the hub model) happen lazily so the rest of the project
runs even when DINOv2 is disabled or unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import config
except ModuleNotFoundError:
    from src import config

_MODEL = None       # cached backbone
_DEVICE = None


def _load_backbone():
    """Load the frozen DINOv2 backbone once (lazy)."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch  # local import: only needed when DINOv2 is used
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = torch.hub.load("facebookresearch/dinov2", config.DINOV2_NAME)
        _MODEL.eval().to(_DEVICE)
        for p in _MODEL.parameters():
            p.requires_grad = False
    return _MODEL, _DEVICE


def _build_transform():
    """Validation-style transform sized for DINOv2 (multiple of 14)."""
    from torchvision import transforms
    s = config.DINOV2_IMG_SIZE
    return transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_embeddings(wide, img_dir: Path = config.LABELLED_IMG_DIR,
                       batch_size: int = 32, use_cache: bool = True) -> dict:
    """Compute (or load) frozen DINOv2 embeddings for every image.

    Returns a dict with:
      - 'cls':     (N, D) global embeddings, ordered like wide.
      - 'image_id':(N,) ids aligned with the rows.
    Patch tokens are not cached here (large); use extract_patch_tokens for
    segmentation on demand.
    """
    if use_cache and config.DINOV2_CACHE.exists():
        data = np.load(config.DINOV2_CACHE, allow_pickle=True)
        return {"cls": data["cls"], "image_id": data["image_id"]}

    import torch
    from PIL import Image

    model, device = _load_backbone()
    tf = _build_transform()
    ids = wide["image_id"].tolist()
    paths = [Path(img_dir) / Path(p).name for p in wide["image_path"]]

    cls_list = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths])
            imgs = imgs.to(device)
            out = model(imgs)            # (B, D) CLS embedding for dinov2 hub model
            cls_list.append(out.cpu().numpy())
    cls = np.concatenate(cls_list).astype(np.float32)

    config.ensure_dirs()
    np.savez(config.DINOV2_CACHE, cls=cls, image_id=np.array(ids, dtype=object))
    return {"cls": cls, "image_id": np.array(ids, dtype=object)}


def extract_patch_tokens(image_path: Path):
    """Return per-patch embeddings for one image (for segmentation).

    Shape: (n_patches, D) plus the patch grid (h, w) so masks can be reshaped.
    """
    import torch
    from PIL import Image

    model, device = _load_backbone()
    tf = _build_transform()
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.forward_features(img)
        patches = out["x_norm_patchtokens"][0].cpu().numpy()  # (n_patches, D)
    grid = config.DINOV2_IMG_SIZE // 14
    return patches, (grid, grid)
