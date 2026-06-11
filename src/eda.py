"""EDA functions for CSIRO Image2Biomass.

Organized into blocks: loading/transformation, statistics, visualization.
Imported into the notebook (eda_first.ipynb); no standalone run required.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import config
except ModuleNotFoundError:  # when imported via the src package
    from src import config


# --------------------------------------------------------------------------- #
# Loading and transformation
# --------------------------------------------------------------------------- #
def load_long(csv_path: Path = config.LABELLED_CSV) -> pd.DataFrame:
    """Load the long CSV, extract image_id and parse the date."""
    df = pd.read_csv(csv_path)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"], format="%Y/%m/%d")
    return df


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide: one row = one image, 5 target columns."""
    meta_cols = ["image_id", "image_path", "Sampling_Date",
                 "State", "Species", *config.META_NUM]
    meta = df_long[meta_cols].drop_duplicates("image_id").set_index("image_id")
    wide_t = df_long.pivot(index="image_id", columns="target_name", values="target")
    wide = meta.join(wide_t).reset_index()
    wide["month"] = wide["Sampling_Date"].dt.month
    return wide


def group_rare_species(wide: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Collapse rare species (< min_count) into 'Other' to reduce categorical noise."""
    counts = wide["Species"].value_counts()
    rare = counts[counts < min_count].index
    wide = wide.copy()
    wide["Species_grouped"] = wide["Species"].where(~wide["Species"].isin(rare), "Other")
    return wide


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def target_stats(wide: pd.DataFrame) -> pd.DataFrame:
    """Target summary: moments, skew, kurtosis, zero fraction."""
    rows = []
    for t in config.TARGETS:
        s = wide[t]
        rows.append({
            "target": t, "mean": s.mean(), "std": s.std(),
            "min": s.min(), "q25": s.quantile(0.25), "median": s.median(),
            "q75": s.quantile(0.75), "max": s.max(),
            "skew": s.skew(), "kurtosis": s.kurtosis(),
            "zero_pct": (s == 0).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("target").round(3)


def outlier_summary(wide: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Number of IQR-rule outliers per target."""
    rows = []
    for t in config.TARGETS:
        s = wide[t]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        rows.append({"target": t, "lower": round(lo, 2), "upper": round(hi, 2),
                     "n_outliers": int(((s < lo) | (s > hi)).sum())})
    return pd.DataFrame(rows).set_index("target")


def check_total_constraint(wide: pd.DataFrame, tol: float = 1e-2) -> dict:
    """Check Dry_Total_g == sum(components)."""
    recon = wide[config.COMP_TARGETS].sum(axis=1)
    diff = (recon - wide["Dry_Total_g"]).abs()
    return {"max_abs_diff": float(diff.max()),
            "n_violations": int((diff > tol).sum()),
            "frac_ok": float((diff <= tol).mean())}


def full_corr(wide: pd.DataFrame) -> pd.DataFrame:
    """Full numeric correlation matrix: targets + metadata + month."""
    cols = config.TARGETS + config.META_NUM + ["month"]
    return wide[cols].corr().round(3)


def group_balance(wide: pd.DataFrame) -> dict:
    """Group sizes by State and Species."""
    return {"by_State": wide["State"].value_counts().to_dict(),
            "by_Species": wide["Species"].value_counts().to_dict(),
            "n_images": len(wide), "n_states": wide["State"].nunique(),
            "n_species": wide["Species"].nunique()}


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_target_distributions(wide: pd.DataFrame) -> None:
    """Target histograms in raw and log1p scale (show the skew effect)."""
    n = len(config.TARGETS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    for j, t in enumerate(config.TARGETS):
        s = wide[t]
        axes[0, j].hist(s, bins=30, color="#4c9f70", edgecolor="white")
        axes[0, j].set_title(f"{t}\nskew={s.skew():.2f}")
        axes[1, j].hist(np.log1p(s), bins=30, color="#3a6ea5", edgecolor="white")
        axes[1, j].set_title(f"log1p({t})")
    axes[0, 0].set_ylabel("raw")
    axes[1, 0].set_ylabel("log1p")
    fig.tight_layout()
    plt.show()


def plot_target_boxplots(wide: pd.DataFrame) -> None:
    """Boxplots of all targets on one axis (overall biomass scale)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([wide[t] for t in config.TARGETS], labels=config.TARGETS)
    ax.set_ylabel("grams")
    ax.set_title("Target distribution (IQR + outliers)")
    plt.xticks(rotation=20)
    fig.tight_layout()
    plt.show()


def plot_corr_heatmap(wide: pd.DataFrame) -> None:
    """Heatmap of the full correlation matrix."""
    corr = full_corr(wide)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation: targets + metadata")
    fig.tight_layout()
    plt.show()


def plot_meta_scatter(wide: pd.DataFrame) -> None:
    """Scatter of key metadata against Dry_Total_g with a trend line."""
    fig, axes = plt.subplots(1, len(config.META_NUM),
                             figsize=(6 * len(config.META_NUM), 5))
    for ax, m in zip(np.atleast_1d(axes), config.META_NUM):
        x, y = wide[m].values, wide["Dry_Total_g"].values
        ax.scatter(x, y, alpha=0.5, color="#4c9f70")
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, color="#c0392b", lw=2)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(m)
        ax.set_ylabel("Dry_Total_g")
        ax.set_title(f"{m} vs Total (r={r:.2f})")
    fig.tight_layout()
    plt.show()


def plot_target_by_group(wide: pd.DataFrame, group: str = "State",
                         target: str = "Dry_Total_g") -> None:
    """Boxplot of a target across groups (distribution shift between State/Species)."""
    groups = wide[group].value_counts().index.tolist()
    data = [wide.loc[wide[group] == g, target] for g in groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=groups)
    ax.set_ylabel(target)
    ax.set_title(f"{target} by {group}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    plt.show()


def show_sample_images(wide: pd.DataFrame, img_dir: Path = config.LABELLED_IMG_DIR,
                       n: int = 6) -> None:
    """Show several images annotated with biomass (data understanding + video)."""
    from matplotlib.image import imread

    sample = wide.sample(min(n, len(wide)), random_state=config.RANDOM_STATE)
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, (_, r) in zip(np.array(axes).ravel(), sample.iterrows()):
        path = img_dir / Path(r["image_path"]).name
        try:
            ax.imshow(imread(path))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "file not found", ha="center")
        ax.set_title(f"Total={r['Dry_Total_g']:.0f}g | {r['State']}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
