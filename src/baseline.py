"""Tabular baselines on metadata features (no image).

Establishes the lower bound the CNN must beat and quantifies how much
metadata (NDVI, height, species, state, month) alone explains biomass.

Models: naive median, RandomForest, HistGradientBoosting.
Targets trained in log1p space; metrics reported on the original scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import config
    from splits import get_cv_folds
except ModuleNotFoundError:
    from src import config
    from src.splits import get_cv_folds


# --------------------------------------------------------------------------- #
# Feature pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """One-hot for categoricals, passthrough for numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.TABULAR_CAT),
            ("num", "passthrough", config.TABULAR_NUM),
        ]
    )


def make_model(kind: str) -> Pipeline:
    """Build a preprocessing + regressor pipeline for one target."""
    if kind == "rf":
        reg = RandomForestRegressor(n_estimators=300, max_depth=None,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    elif kind == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=300,
                                            random_state=config.RANDOM_STATE)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return Pipeline([("prep", build_preprocessor()), ("reg", reg)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def weighted_score(per_target_rmse: dict[str, float]) -> float:
    """Competition-style weighted RMSE across targets."""
    return sum(config.TARGET_WEIGHTS[t] * rmse for t, rmse in per_target_rmse.items())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R2 on the original scale."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_naive(wide: pd.DataFrame, folds: list) -> pd.DataFrame:
    """Naive baseline: predict the train-fold median for every test row."""
    records = []
    for i, (tr, te) in enumerate(folds):
        for t in config.TARGETS:
            med = wide.iloc[tr][t].median()
            y_true = wide.iloc[te][t].to_numpy()
            y_pred = np.full_like(y_true, med, dtype=float)
            m = _metrics(y_true, y_pred)
            records.append({"model": "naive_median", "fold": i, "target": t, **m})
    return pd.DataFrame(records)


def evaluate_model(wide: pd.DataFrame, folds: list, kind: str) -> pd.DataFrame:
    """Train one model per target in log1p space; report original-scale metrics."""
    records = []
    for i, (tr, te) in enumerate(folds):
        X_tr, X_te = wide.iloc[tr], wide.iloc[te]
        for t in config.TARGETS:
            model = make_model(kind)
            model.fit(X_tr, np.log1p(X_tr[t]))
            y_pred = np.expm1(model.predict(X_te))
            y_pred = np.clip(y_pred, 0, None)  # biomass is non-negative
            m = _metrics(X_te[t].to_numpy(), y_pred)
            records.append({"model": kind, "fold": i, "target": t, **m})
    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold results: mean +/- std per model and target."""
    agg = (results.groupby(["model", "target"])[["rmse", "mae", "r2"]]
           .agg(["mean", "std"]).round(3))
    return agg


def weighted_scores_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Weighted competition score per model (mean over folds)."""
    rows = []
    for model in results["model"].unique():
        sub = results[results["model"] == model]
        per_target = sub.groupby("target")["rmse"].mean().to_dict()
        rows.append({"model": model, "weighted_rmse": round(weighted_score(per_target), 3)})
    return pd.DataFrame(rows).set_index("model").sort_values("weighted_rmse")


def feature_importance(wide: pd.DataFrame, target: str = "Dry_Total_g",
                       kind: str = "rf") -> pd.DataFrame:
    """Fit on all data and return feature importance for one target (RF only)."""
    if kind != "rf":
        raise ValueError("feature_importance currently supports kind='rf'")
    model = make_model("rf")
    model.fit(wide, np.log1p(wide[target]))
    prep = model.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(config.TABULAR_CAT)
    names = list(cat_names) + config.TABULAR_NUM
    imp = model.named_steps["reg"].feature_importances_
    return (pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def run_all_baselines(wide: pd.DataFrame, n_splits: int = 5) -> dict:
    """Full baseline evaluation. Returns dict of result frames."""
    folds = get_cv_folds(wide, n_splits=n_splits)
    naive = evaluate_naive(wide, folds)
    rf = evaluate_model(wide, folds, "rf")
    hgb = evaluate_model(wide, folds, "hgb")
    all_res = pd.concat([naive, rf, hgb], ignore_index=True)
    return {
        "raw": all_res,
        "summary": summarize(all_res),
        "weighted": weighted_scores_by_model(all_res),
        "folds": folds,
    }


def weighted_per_fold(results: pd.DataFrame) -> dict:
    """Per-fold weighted RMSE for each model (enables mean +/- std and paired tests).

    Unlike weighted_scores_by_model (which averages RMSE across folds first),
    this computes the weighted score within each fold, so we get a distribution
    over folds — needed for std and for paired significance tests vs other models.
    """
    out = {}
    for model in results["model"].unique():
        sub = results[results["model"] == model]
        fold_scores = []
        for f in sorted(sub["fold"].unique()):
            per_target = sub[sub["fold"] == f].set_index("target")["rmse"].to_dict()
            fold_scores.append(weighted_score(per_target))
        out[model] = {
            "fold_weighted": fold_scores,
            "weighted_rmse_mean": float(np.mean(fold_scores)),
            "weighted_rmse_std": float(np.std(fold_scores)),
        }
    return out
