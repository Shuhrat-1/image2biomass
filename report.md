# Image2Biomass: Predicting Pasture Biomass from Quadrat Photographs

**Master's in Green Data Science — Practical Machine Learning (ISA, ULisboa)**

**Team:** [Name 1 / Student ID 1], [Name 2 / Student ID 2]

---

## 1. Introduction

Pasture biomass is a key input for grazing management: knowing how much dry
matter is available helps farmers decide when and how to graze livestock without
over- or under-using a paddock. Measuring biomass directly is labour-intensive,
so estimating it from photographs is attractive — a farmer could photograph a
field and get an instant estimate.

We use the CSIRO Image2Biomass dataset (Kaggle), where each 70x30 cm quadrat
photograph is paired with five ground-truth biomass targets in grams: green,
dead, clover, GDM (green dry matter), and total dry matter.

Beyond raw prediction, our central question is a **modality comparison**: how
much does the image add over two cheap ground measurements already available in
the data — NDVI and pasture height? This reframes the project from "train a CNN"
to "quantify whether expensive pixel analysis beats cheap field measurements",
which is both more scientifically interesting and more practically relevant.

## 2. Data

**Source.** CSIRO Image2Biomass (Kaggle): 357 labelled quadrat photographs from
19 sites across four Australian states (Tasmania, NSW, WA, Victoria), collected
2014-2017. Each image carries metadata: state, pasture species, pre-grazing NDVI,
average sward height, and sampling date.

**Format.** The raw CSV is in long format (1785 rows = 357 images x 5 targets).
We pivot to wide format (one row per image, five target columns) and extract
`month` from the date.

**Target characteristics.** All five targets are strongly right-skewed
(skewness 1.4-2.8, kurtosis 3-9). Clover is the hardest: 37.8% zeros and the
highest skew (2.84). We therefore model targets in **log1p** space and invert
predictions with expm1, clipping to non-negative values.

**Cleaning and feature engineering.** No missing values in the metadata. We
collapse rare species (<10 images) into an "Other" category to reduce
categorical noise. We verified the physical constraint
Total = Green + Dead + Clover holds in 99.7% of rows (one rounding violation).

**Key correlations (motivation for the project).** Height correlates with green
mass (r=0.65) and total (r=0.50); NDVI correlates with GDM (r=0.47) and total
(r=0.36). Together these two ground measurements form a strong predictor — but
r≈0.5 leaves most of the variance unexplained, which is the headroom an image
could close. NDVI also shows heteroscedasticity (saturation on dense pasture),
suggesting imagery might help where NDVI plateaus.

## 3. Data Organization

Observations are correlated within an image (5 target rows) and within a site
(shared soil, lighting, species), so a naive random split would leak
information and inflate scores. We use **StratifiedGroupKFold** with:
- **Groups = image_id** — all rows of one image stay in the same fold.
- **Stratification = binned Dry_Total_g** — the biomass range is balanced
  across folds.

We chose 5-fold cross-validation over a single train/test split for a more
reliable estimate. GroupKFold by state was rejected because states are
imbalanced (Tas 138 ... WA 32); the smallest state would yield an unstable fold.
State is instead used for error analysis. The **same folds** are used for every
model (tabular and image) so comparisons are fair.

Note on the Kaggle hidden test: it omits metadata, so a fair modality study must
be done on the locally labelled data via cross-validation, not the leaderboard.

## 4. Methods

We build a four-tier comparison aligned with the course syllabus:

**Tier 0 — Naive baseline.** Predict the training-fold median per target. The
reference floor any useful model must beat.

**Tier 1 — Tabular models (sessions 4, 7).** RandomForest and
HistGradientBoosting on metadata features (NDVI, height, month, species, state),
one model per target in log1p space. One-hot encoding for categoricals.

**Tier 2 — CNN from scratch (session 10).** A small 4-block convnet
(conv-BN-ReLU-pool) with a regression head, trained on 224x224 RGB. Kept small
because ~285 training images would overfit a large network.

**Tier 3 — Transfer learning (session 12).** A pretrained ResNet18 with a
5-output regression head, in two regimes: **frozen** (feature extraction, head
only) and **fine-tuned** (whole backbone). ImageNet normalization; mild
augmentation (flips, light jitter) chosen so as not to destroy the
texture/density signal.

**Training.** Adam optimizer, MSE loss in log1p space, early stopping on
validation weighted-RMSE (original scale). Fine-tuning uses a smaller learning
rate (1e-4) since the whole backbone is updated. All image models share the
dataset pipeline and CV folds.

**Deployment (session 11).** A Gradio app serving the image-only model,
deployed to Hugging Face Spaces (see Section 7).

## 5. Results

**Evaluation metric.** Competition-style weighted RMSE (target weights
0.1/0.1/0.1/0.2/0.5, with Total weighted highest) plus per-target RMSE, MAE, and
R2 on the original gram scale, aggregated over 5 folds (mean ± std).

**Main comparison table** (weighted RMSE, lower is better):

| Model | Weighted RMSE | R2 (Total) | Type |
|-------|--------------:|-----------:|------|
| RandomForest (metadata)   | 11.83 | 0.72 | tabular |
| ResNet18 fine-tuned       | 18.23 | 0.33 | image |
| SimpleCNN (from scratch)  | 18.70 | – | image |
| Naive median              | 24.12 | -0.08 | floor |
| ResNet18 frozen           | 24.74 | – | image |

All weighted-RMSE values are mean over 5 folds: fine-tune 18.23 ± 2.83,
simple_cnn 18.70 ± 2.64, frozen 24.74 ± 2.22. A final deployable model trained on
an enlarged train split reached 15.94 on a single holdout; the table reports the
more conservative cross-validated figure.

**Tabular per-target R2 (RandomForest):** Green 0.79, GDM 0.80, Total 0.72,
Clover 0.65, Dead 0.39. **Feature importance:** Height (0.57) and NDVI (0.24)
dominate (81% combined), consistent with the EDA.

**Image per-target R2 (best image model — fine-tuned ResNet):** Green 0.49,
GDM 0.42, Total 0.33, Dead 0.27, Clover 0.30. The image model is beaten by the
RandomForest on every single target; the largest gaps are Clover (0.30 vs 0.65)
and GDM (0.42 vs 0.80).

## 6. Analysis

**1. Metadata outperforms imagery.** The RandomForest on two cheap ground
measurements (11.83) beats the best image model (18.23) by ~35%. On 357
samples, the direct physical signal in NDVI/height beats what a CNN extracts from
raw pixels. This is the project's central, quantified conclusion.

**2. Fine-tuning is essential; feature extraction fails.** The frozen ResNet (24.74) performs *worse than the naive median* (24.12): unadapted ImageNet
features are useless for pasture. Unfreezing the backbone cuts error sharply.
This contrast directly demonstrates the value of fine-tuning on small,
domain-shifted data.

**3. A small from-scratch CNN beats the frozen ResNet.** A compact network
trained on-task (18.70) is statistically indistinguishable from fine-tuning
(18.23) and clearly beats the frozen ResNet (24.74) — on small data, "small + adapted" beats
"large + borrowed".

**4. Where imagery is weakest.** Dead matter (R2 0.27) and clover (0.30) have the
lowest image R2, matching intuition: dead biomass and scattered clover are hard
to read from a top-down photo. The image model is beaten by the RandomForest on
every target, with the widest gaps on GDM (0.42 vs 0.80) and clover
(0.30 vs 0.65).

**Methodological note.** Fine-tuning showed early-epoch validation instability;
early stopping on validation weighted-RMSE was used to select the best epoch.

**Interpretation.** For this dataset and scale, the image does not add predictive
value beyond cheap ground measurements — it underperforms them. Practically,
where NDVI/height are available they are the better and cheaper predictor. The
image-only model remains useful only in the field scenario where such
measurements are unavailable, which is exactly what the deployed app targets.

## 7. Deployment

A Gradio app (`src/app.py`) serves the image-only fine-tuned ResNet: a user
uploads a quadrat photo and receives the five biomass estimates plus a short
grazing-quality summary (total biomass, green share). It is deployed to Hugging
Face Spaces [link TBD].

The app deliberately uses the image-only model: it represents the realistic
field scenario where a farmer has only a phone photo, not NDVI or height. This is
consistent with the analysis — the demo illustrates the practical use case, while
the scientific comparison lives in the report.

## 8. Future Work

The image model underperforms metadata; promising directions to close the gap:
- **DINOv2 self-supervised features** instead of ImageNet ResNet — features
  learned on vast unlabelled data may transfer better to pasture texture.
- **Segmentation approach** — predict green/dead/soil masks, then derive
  component fractions, closer to how an agronomist assesses a sward.
- **Multimodal fusion** — combine image + metadata; expected to beat either
  modality alone on the labelled set.
- **Cross-validation ensembling** and **test-time augmentation** for cheap,
  low-risk gains.

## 9. References

- CSIRO Image2Biomass dataset (Kaggle competition).
- He et al., "Deep Residual Learning for Image Recognition" (ResNet), 2016.
- Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011.
- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning
  Library", NeurIPS 2019.
- [Add any other sources used.]

## 10. Contributions

- **[Name 1]:** [e.g. EDA, tabular baseline, report.]
- **[Name 2]:** [e.g. CNN pipeline, training, deployment.]

*(Fill in per actual division of work.)*
