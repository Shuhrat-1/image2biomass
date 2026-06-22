"""Central project configuration: paths, constants, environment detection.

Terminology (to avoid confusion):
- LABELLED: the full labelled dataset (357 images). We split it via
  cross-validation into fold-train / fold-validation. This is what was
  originally the Kaggle "train" set.
- KAGGLE_TEST: the hidden Kaggle test set (labels unavailable). NOT used for
  local evaluation; kept only for reference / potential submission.

All paths are relative to the project root and work in both local and Colab
environments (auto-detected).
"""
from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
def in_colab() -> bool:
    """True if running inside Google Colab."""
    return "google.colab" in sys.modules


# Project root.
# - Local: one level up from src/.
# - Colab: expects the project under /content/image2biomass (adjust if needed).
if in_colab():
    PROJECT_ROOT: Path = Path("/content/image2biomass")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"

# Labelled dataset (the 357 images we actually train/evaluate on)
LABELLED_CSV: Path = DATA_RAW / "labelled.csv"
LABELLED_IMG_DIR: Path = DATA_RAW / "labelled"

# Hidden Kaggle test (labels unavailable; not used for local evaluation)
KAGGLE_TEST_CSV: Path = DATA_RAW / "kaggle_test.csv"
KAGGLE_TEST_IMG_DIR: Path = DATA_RAW / "kaggle_test"

# EDA outputs
WIDE_CSV: Path = DATA_PROCESSED / "labelled_wide.csv"

# Domain constants
TARGETS: list[str] = [
    "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g",
]
COMP_TARGETS: list[str] = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
TARGET_WEIGHTS: dict[str, float] = {
    "Dry_Green_g": 0.1, "Dry_Dead_g": 0.1, "Dry_Clover_g": 0.1,
    "GDM_g": 0.2, "Dry_Total_g": 0.5,
}

META_NUM: list[str] = ["Pre_GSHH_NDVI", "Height_Ave_cm"]
META_CAT: list[str] = ["State", "Species"]
# month added as a feature based on EDA (correlates with height/green/clover)
TABULAR_NUM: list[str] = ["Pre_GSHH_NDVI", "Height_Ave_cm", "month"]
TABULAR_CAT: list[str] = ["State", "Species_grouped"]

RANDOM_STATE: int = 42


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Method toggles — enable/disable model families without touching code.
# DINOv2 flags default to False so the base submission never loads the heavy
# ViT unless explicitly requested.
# --------------------------------------------------------------------------- #
RUN_NAIVE: bool = True
RUN_TABULAR: bool = True
RUN_SIMPLE_CNN: bool = True
RUN_RESNET_FROZEN: bool = True
RUN_RESNET_FINETUNE: bool = True
RUN_DINOV2_REGRESSION: bool = False
RUN_DINOV2_SEGMENTATION: bool = False

# DINOv2 settings
DINOV2_NAME: str = "dinov2_vits14"   # ViT-S/14, 384-dim embeddings (light, fast)
DINOV2_DIM: int = 384
DINOV2_IMG_SIZE: int = 224           # must be a multiple of 14
# Cache for precomputed frozen embeddings (computed once, reused across folds)
DINOV2_CACHE: Path = DATA_PROCESSED / "dinov2_embeddings.npz"
