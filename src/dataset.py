"""PyTorch Dataset and DataLoader factory for the image biomass task.

Reads images from LABELLED_IMG_DIR, returns (image_tensor, target_tensor).
Targets are returned in log1p space (consistent with the tabular baseline);
invert with expm1 at metric time. Uses ImageNet normalization so the same
pipeline feeds both a from-scratch CNN and a pretrained backbone.

Folds come from splits.py, so the CNN evaluates on exactly the same folds as
the tabular baseline (fair modality comparison).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import config
    from splits import get_cv_folds
except ModuleNotFoundError:
    from src import config
    from src.splits import get_cv_folds

# ImageNet statistics (required by pretrained ResNet/EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    """Image transforms. Train adds light augmentation; val is deterministic.

    Augmentation is kept mild: the pasture biomass signal is texture/density,
    so aggressive crops or color jitter could destroy the target relationship.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BiomassDataset(Dataset):
    """Quadrat image -> 5 biomass targets (log1p).

    Parameters
    ----------
    wide : pd.DataFrame
        Wide-format frame (one row per image) with target columns and image_path.
    indices : np.ndarray
        Positional row indices into `wide` for this split (from splits.py).
    img_dir : Path
        Directory holding the .jpg files.
    transform : callable
        Image transform pipeline.
    """

    def __init__(self, wide: pd.DataFrame, indices: np.ndarray,
                 img_dir: Path = config.LABELLED_IMG_DIR,
                 transform: transforms.Compose | None = None) -> None:
        self.df = wide.iloc[indices].reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform or build_transforms(train=False)
        self.targets = config.TARGETS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # Use only the filename; ignore any stale folder prefix in image_path.
        path = self.img_dir / Path(row["image_path"]).name
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        y = np.log1p(row[self.targets].to_numpy(dtype=np.float32))
        return image, torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def make_fold_loaders(wide: pd.DataFrame, fold: tuple[np.ndarray, np.ndarray],
                      img_size: int = 224, batch_size: int = 32,
                      num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) for one CV fold."""
    tr_idx, val_idx = fold
    pin = torch.cuda.is_available()  # pin_memory only helps with a GPU
    train_ds = BiomassDataset(wide, tr_idx, transform=build_transforms(img_size, True))
    val_ds = BiomassDataset(wide, val_idx, transform=build_transforms(img_size, False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


def get_all_fold_loaders(wide: pd.DataFrame, n_splits: int = 5,
                         **kwargs) -> list[tuple[DataLoader, DataLoader]]:
    """Build loaders for every CV fold (same folds as the tabular baseline)."""
    folds = get_cv_folds(wide, n_splits=n_splits)
    return [make_fold_loaders(wide, f, **kwargs) for f in folds]