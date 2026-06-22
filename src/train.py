"""Training loop and cross-validated evaluation for image models.

- Loss: MSE in log1p space (model and targets are in log space).
- Model selection / early stopping: by validation weighted-RMSE on the ORIGINAL
  scale (expm1), the same metric used to compare against the tabular baseline.
- Device-agnostic (CUDA if available, else CPU) so it runs locally and on Colab.
- Cross-validated over the same folds as the tabular baseline (splits.py).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import config
    from baseline import _metrics, weighted_score
    from dataset import make_fold_loaders
    from model import build_model
    from splits import get_cv_folds
except ModuleNotFoundError:
    from src import config
    from src.baseline import _metrics, weighted_score
    from src.dataset import make_fold_loaders
    from src.model import build_model
    from src.splits import get_cv_folds


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cap for log-space predictions before expm1. log1p(500) ~ 6.2 is well above
# the largest real target (log1p(185) ~ 5.2) yet keeps expm1 from exploding.
LOG_CLIP_MAX: float = 6.2


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (y_true, y_pred) on ORIGINAL scale."""
    model.eval()
    trues, preds = [], []
    for images, y_log in loader:
        images = images.to(device)
        out = model(images).cpu().numpy()
        trues.append(y_log.numpy())
        preds.append(out)
    y_true = np.expm1(np.concatenate(trues))          # invert log1p
    # Clip predictions in LOG space before expm1: an unstable early-epoch
    # prediction of e.g. log=25 would explode to expm1(25)~7e10 and wreck the
    # metric. Real log1p targets are <= ~5.3 (log1p(185)); cap generously.
    log_pred = np.clip(np.concatenate(preds), 0, LOG_CLIP_MAX)
    y_pred = np.clip(np.expm1(log_pred), 0, None)     # non-negative
    return y_true, y_pred


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target metrics + weighted score, matching the tabular baseline."""
    per_target_rmse = {}
    rows = {}
    for j, t in enumerate(config.TARGETS):
        m = _metrics(y_true[:, j], y_pred[:, j])
        rows[t] = m
        per_target_rmse[t] = m["rmse"]
    return {"per_target": rows, "weighted_rmse": weighted_score(per_target_rmse)}


# --------------------------------------------------------------------------- #
# Training one fold
# --------------------------------------------------------------------------- #
def train_one_fold(model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, device: torch.device,
                   max_epochs: int = 50, lr: float = 1e-3,
                   weight_decay: float = 1e-4, patience: int = 8,
                   verbose: bool = True) -> dict:
    """Train with early stopping on val weighted-RMSE. Returns best-epoch results.

    Restores the weights of the best epoch before returning.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )

    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_eval = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for images, y_log in train_loader:
            images, y_log = images.to(device), y_log.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), y_log)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_loader.dataset)

        y_true, y_pred = predict(model, val_loader, device)
        ev = evaluate_predictions(y_true, y_pred)
        val_score = ev["weighted_rmse"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_weighted_rmse": val_score})
        if verbose:
            print(f"  epoch {epoch:02d} | train_loss {train_loss:.4f} "
                  f"| val_wRMSE {val_score:.3f}")

        if val_score < best_score - 1e-4:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_eval = ev
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best wRMSE {best_score:.3f})")
                break

    model.load_state_dict(best_state)
    return {"best_weighted_rmse": best_score, "best_eval": best_eval,
            "history": pd.DataFrame(history)}


# --------------------------------------------------------------------------- #
# Cross-validated training
# --------------------------------------------------------------------------- #
def cross_validate(wide: pd.DataFrame, kind: str = "resnet18", n_splits: int = 5,
                   img_size: int = 224, batch_size: int = 32,
                   num_workers: int = 2, model_kwargs: dict | None = None,
                   train_kwargs: dict | None = None,
                   verbose: bool = True) -> dict:
    """Train `kind` over all CV folds; aggregate per-target and weighted metrics.

    Returns raw per-fold frame, summary (mean/std), and mean weighted-RMSE.
    """
    device = get_device()
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    folds = get_cv_folds(wide, n_splits=n_splits)

    records = []
    fold_weighted = []
    for i, fold in enumerate(folds):
        if verbose:
            print(f"[fold {i}] {kind} on {device}")
        train_loader, val_loader = make_fold_loaders(
            wide, fold, img_size=img_size, batch_size=batch_size,
            num_workers=num_workers,
        )
        model = build_model(kind, **model_kwargs)
        res = train_one_fold(model, train_loader, val_loader, device,
                             verbose=verbose, **train_kwargs)
        fold_weighted.append(res["best_weighted_rmse"])
        for t, m in res["best_eval"]["per_target"].items():
            records.append({"model": kind, "fold": i, "target": t, **m})

    raw = pd.DataFrame(records)
    summary = (raw.groupby(["model", "target"])[["rmse", "mae", "r2"]]
               .agg(["mean", "std"]).round(3))
    return {
        "raw": raw,
        "summary": summary,
        "weighted_rmse_mean": float(np.mean(fold_weighted)),
        "weighted_rmse_std": float(np.std(fold_weighted)),
        "fold_weighted": fold_weighted,
    }


# --------------------------------------------------------------------------- #
# Final model for deployment (trained on all data) + checkpointing
# --------------------------------------------------------------------------- #
def train_final_model(wide: pd.DataFrame, kind: str = "resnet18",
                      img_size: int = 224, batch_size: int = 32,
                      num_workers: int = 2, val_frac: float = 0.15,
                      model_kwargs: dict | None = None,
                      train_kwargs: dict | None = None,
                      verbose: bool = True) -> tuple[nn.Module, dict]:
    """Train one deployable model on (almost) all data.

    A small internal validation split (val_frac) is held out only for early
    stopping; the model is meant for the Gradio app, not for CV reporting.
    Returns (trained_model, info_dict).
    """
    from sklearn.model_selection import GroupShuffleSplit

    device = get_device()
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}

    # Group-aware holdout for early stopping (keep an image's rows together).
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                            random_state=config.RANDOM_STATE)
    tr_idx, val_idx = next(gss.split(wide, groups=wide["image_id"]))

    train_loader, val_loader = make_fold_loaders(
        wide, (tr_idx, val_idx), img_size=img_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    model = build_model(kind, **model_kwargs)
    res = train_one_fold(model, train_loader, val_loader, device,
                         verbose=verbose, **train_kwargs)
    info = {"kind": kind, "img_size": img_size, "model_kwargs": model_kwargs,
            "val_weighted_rmse": res["best_weighted_rmse"],
            "targets": config.TARGETS}
    return model, info


def save_checkpoint(model: nn.Module, info: dict, path: Path) -> None:
    """Save model weights + metadata needed to rebuild it for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "info": info}, path)
    # Also dump info as JSON for easy inspection.
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(info, f, indent=2)


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint and load weights for inference."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    info = ckpt["info"]
    model = build_model(info["kind"], **info["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, info


# --------------------------------------------------------------------------- #
# DINOv2 regression on precomputed embeddings (fast: trains only the head)
# --------------------------------------------------------------------------- #
def cross_validate_dinov2(wide: pd.DataFrame, n_splits: int = 5,
                          max_epochs: int = 100, lr: float = 1e-3,
                          weight_decay: float = 1e-4, patience: int = 12,
                          batch_size: int = 64, verbose: bool = True) -> dict:
    """Train a DINOv2 regression head over CV folds using cached embeddings.

    Embeddings are precomputed once (frozen backbone); each fold only trains a
    small MLP head on the D-dim vectors, so this is fast even on CPU.
    Uses the same folds as every other model for a fair comparison.
    """
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import dinov2_features
        from model import DINOv2Regressor
    except ModuleNotFoundError:
        from src import dinov2_features
        from src.model import DINOv2Regressor

    device = get_device()
    emb = dinov2_features.compute_embeddings(wide)  # {'cls': (N,D), 'image_id'}
    X = torch.tensor(emb["cls"], dtype=torch.float32)
    Y = torch.tensor(np.log1p(wide[config.TARGETS].to_numpy(dtype=np.float32)))

    folds = get_cv_folds(wide, n_splits=n_splits)
    records, fold_weighted = [], []

    for i, (tr, te) in enumerate(folds):
        if verbose:
            print(f"[fold {i}] dinov2 head on {device}")
        tr_loader = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=batch_size,
                               shuffle=True)
        model = DINOv2Regressor(in_dim=config.DINOV2_DIM).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        best_score, best_state, no_improve = float("inf"), None, 0
        best_eval = None
        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            # validate
            model.eval()
            with torch.no_grad():
                log_pred = model(X[te].to(device)).cpu().numpy()
            log_pred = np.clip(log_pred, 0, LOG_CLIP_MAX)
            y_pred = np.clip(np.expm1(log_pred), 0, None)
            y_true = wide.iloc[te][config.TARGETS].to_numpy()
            ev = evaluate_predictions(y_true, y_pred)
            if ev["weighted_rmse"] < best_score - 1e-4:
                best_score, best_eval = ev["weighted_rmse"], ev
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        fold_weighted.append(best_score)
        for t, m in best_eval["per_target"].items():
            records.append({"model": "dinov2", "fold": i, "target": t, **m})

    raw = pd.DataFrame(records)
    summary = (raw.groupby(["model", "target"])[["rmse", "mae", "r2"]]
               .agg(["mean", "std"]).round(3))
    return {"raw": raw, "summary": summary,
            "weighted_rmse_mean": float(np.mean(fold_weighted)),
            "weighted_rmse_std": float(np.std(fold_weighted)),
            "fold_weighted": fold_weighted}
