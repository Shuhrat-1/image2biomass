# Image2Biomass: Predicting Pasture Biomass from Quadrat Photographs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shuhrat-1/image2biomass/blob/main/notebooks/train_colab.ipynb)

Master's in Green Data Science — Practical Machine Learning (ISA, ULisboa).

Predict five pasture biomass targets (grams) from 70x30 cm quadrat photographs,
using the [CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)
dataset. The central question is a **modality comparison**: how much does the
image add over two cheap ground measurements (NDVI and pasture height)?

## Train on Colab (one click)

Click the **Open in Colab** badge above, then:
1. **Runtime → Change runtime type → GPU (T4)**.
2. **Runtime → Run all**.
3. When the Drive cell prompts, authorize access to your Google Drive
   (the images live in `MyDrive/image2biomass_data/`).

The notebook clones this repo for code and mounts Drive for the image data,
then runs the full comparison and saves a deployable checkpoint.

## Key result

| Model | Weighted RMSE | R2 (Total) | Type |
|-------|--------------:|-----------:|------|
| RandomForest (metadata) | **11.83** | **0.72** | tabular |
| ResNet18 fine-tuned     | 17.60 | 0.33 | image |
| SimpleCNN (from scratch)| 19.68 | –    | image |
| Naive median            | 24.12 | –    | floor |
| ResNet18 frozen         | 24.73 | –    | image |

**Finding:** cheap ground measurements (NDVI + height) outperform the best image
model by 33% on 357 samples. Fine-tuning is essential — frozen ImageNet features
do worse than the naive median. See the report for full analysis.

## Repository structure

```
src/                    Core modules (pure functions; imported by notebooks)
  config.py             Paths, constants, Colab auto-detection
  eda.py                Loading, long->wide, statistics, plots
  splits.py             StratifiedGroupKFold (shared by all models)
  baseline.py           Tabular baselines (naive / RF / HistGBM)
  dataset.py            PyTorch Dataset + DataLoader factory
  model.py              SimpleCNN + pretrained ResNet/EfficientNet
  train.py              Training loop, CV, checkpointing
  app.py                Gradio app (image-only inference)
notebooks/
  eda_first.ipynb       EDA (Run All)
  baseline_tabular.ipynb  Tabular baseline (Run All)
  train_colab.ipynb     CNN training on Colab GPU
project_proposal.md     Project proposal
DEPLOY.md               Hugging Face deployment guide
```

## Data

Images (~1 GB) are **not** committed to Git. Download from Kaggle:
[CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass/data).

Expected layout after download:
```
data/raw/
  labelled.csv          (committed — small)
  labelled/             *.jpg  (from Kaggle "train")
  kaggle_test.csv
  kaggle_test/          *.jpg  (from Kaggle "test")
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

1. **EDA** — open `notebooks/eda_first.ipynb` → Run All.
2. **Tabular baseline** — open `notebooks/baseline_tabular.ipynb` → Run All.
3. **CNN training** — open `notebooks/train_colab.ipynb` in Google Colab
   (Runtime → GPU), follow the Drive-mount cell, Run All.

All paths come from `src/config.py`; no command-line arguments needed.
`config.in_colab()` auto-detects the environment.

## Method

A four-tier comparison aligned with the course:
1. Naive median baseline.
2. Tabular RandomForest / gradient boosting on metadata (sessions 4, 7).
3. CNN from scratch (session 10).
4. Transfer learning: fine-tuned vs frozen ResNet18 (session 12).
5. Gradio + Hugging Face deployment (session 11).

Targets are modelled in log1p space; predictions are inverted (expm1) and
clipped to be non-negative. Evaluation uses StratifiedGroupKFold by image,
stratified by binned total biomass, so all models share identical folds.

## Deployment

A Gradio app (`src/app.py`) demonstrates the image-only model in the realistic
farmer scenario (photo in, biomass out). See `DEPLOY.md`.

## Use of AI

AI-generated code is annotated with `prompt:` comments documenting the prompts
used and any manual modifications, per course guidelines.

## Team

[Shuhrat Maksumov / Student ID 28548], [Xavier Loreto / Student ID 28648]
