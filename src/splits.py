"""Cross-validation splitting strategy, shared across all models.

Single source of truth so the tabular baseline and the CNN evaluate on
identical folds. Folds keep image groups intact and stratify by binned
Dry_Total_g to balance the biomass range across folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:
    import config
except ModuleNotFoundError:
    from src import config


def make_strata(wide: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    """Bin Dry_Total_g into quantile strata for balanced folds.

    Continuous target -> categorical bins so StratifiedGroupKFold can balance
    the biomass range across folds. Falls back to fewer bins if duplicates.
    """
    try:
        strata = pd.qcut(wide["Dry_Total_g"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        strata = pd.qcut(wide["Dry_Total_g"].rank(method="first"),
                         q=n_bins, labels=False)
    return strata.to_numpy()


def get_cv_folds(wide: pd.DataFrame, n_splits: int = 5,
                 n_bins: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) positional index pairs.

    Groups = image_id (an image never spans train and test).
    Stratification = binned Dry_Total_g.
    """
    strata = make_strata(wide, n_bins=n_bins)
    groups = wide["image_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.RANDOM_STATE)
    return list(sgkf.split(wide, strata, groups))


def describe_folds(wide: pd.DataFrame,
                   folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Sanity-check folds: size and mean Total per fold, group disjointness."""
    rows = []
    for i, (tr, te) in enumerate(folds):
        tr_groups = set(wide.iloc[tr]["image_id"])
        te_groups = set(wide.iloc[te]["image_id"])
        rows.append({
            "fold": i,
            "n_train": len(tr), "n_test": len(te),
            "test_mean_Total": round(wide.iloc[te]["Dry_Total_g"].mean(), 2),
            "group_overlap": len(tr_groups & te_groups),  # must be 0
        })
    return pd.DataFrame(rows).set_index("fold")
