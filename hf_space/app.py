"""Gradio app for pasture biomass prediction — Hugging Face Space entry point.

End-user scenario: a farmer photographs a 70x30 cm pasture quadrat and gets an
estimate of biomass components (grams). Image-only — no metadata required.

The Space layout keeps app.py at the root, the model modules under src/, and the
checkpoint at models/best_model.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPACE_ROOT / "src"))

import config  # noqa: E402
from dataset import build_transforms  # noqa: E402
from train import load_checkpoint, LOG_CLIP_MAX  # noqa: E402

import gradio as gr  # noqa: E402

CHECKPOINT_PATH = SPACE_ROOT / "models" / "best_model.pt"

_MODEL = None
_INFO = None
_TRANSFORM = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _lazy_load() -> None:
    """Load the model once on first prediction."""
    global _MODEL, _INFO, _TRANSFORM
    if _MODEL is None:
        _MODEL, _INFO = load_checkpoint(CHECKPOINT_PATH, _DEVICE)
        _TRANSFORM = build_transforms(_INFO.get("img_size", 224), train=False)


def predict_biomass(image) -> dict:
    """Predict the 5 biomass targets (grams) from a PIL image."""
    if image is None:
        return {}
    _lazy_load()
    x = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        log_pred = _MODEL(x).cpu().numpy()[0]
    log_pred = np.clip(log_pred, 0, LOG_CLIP_MAX)
    grams = np.clip(np.expm1(log_pred), 0, None)
    return {t: float(round(g, 1)) for t, g in zip(config.TARGETS, grams)}


def format_summary(predictions: dict) -> str:
    """Human-readable grazing-oriented summary."""
    if not predictions:
        return "Upload a pasture quadrat photo to get an estimate."
    total = predictions.get("Dry_Total_g", 0.0)
    green = predictions.get("Dry_Green_g", 0.0)
    green_share = (green / total * 100) if total > 0 else 0
    quality = "good grazing quality" if green_share > 50 else "more dead matter, lower quality"
    return (f"Estimated total dry biomass: **{total:.0f} g** per quadrat (70x30 cm).\n\n"
            f"Green (live) share: {green_share:.0f}% — {quality}.\n\n"
            f"Note: estimate from image only; field conditions vary.")


def _predict_and_summary(image):
    preds = predict_biomass(image)
    return preds, format_summary(preds)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Pasture Biomass Estimator") as demo:
        gr.Markdown(
            "# Pasture Biomass Estimator\n"
            "Upload a photo of a 70x30 cm pasture quadrat to estimate dry biomass "
            "components. Built on the CSIRO Image2Biomass dataset.\n\n"
            "*Image-only model — the realistic field scenario where NDVI/height "
            "measurements are unavailable.*"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Pasture quadrat photo")
                btn = gr.Button("Estimate biomass", variant="primary")
            with gr.Column():
                out_label = gr.Label(label="Predicted biomass (grams)")
                out_text = gr.Markdown()
        btn.click(_predict_and_summary, inputs=inp, outputs=[out_label, out_text])
    return demo


if __name__ == "__main__":
    build_interface().launch()
