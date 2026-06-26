---
title: Pasture Biomass Estimator
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Pasture Biomass Estimator

Estimate pasture dry-biomass components (grams) from a single photo of a
70x30 cm quadrat. Built on the CSIRO Image2Biomass dataset as part of a
Master's in Green Data Science project (ISA, ULisboa).

The model is an **image-only** fine-tuned ResNet18 — the realistic field
scenario where a farmer has only a phone photo, not NDVI or height measurements.

## How it works
Upload a quadrat photo; the model predicts five targets: green, dead, clover,
GDM, and total dry matter (grams), plus a short grazing-quality summary.

## Note on accuracy
In our study, cheap ground measurements (NDVI + height) actually predict biomass
*better* than imagery. This app demonstrates the image-only case for field use;
the full modality comparison is in the project report.
