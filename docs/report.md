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
r~0.5 leaves most of the variance unexplained, which is the headroom an image
could close. NDVI also shows heteroscedasticity (saturation on dense pasture),
suggesting imagery might help where NDVI plateaus.

## 3. Data Organization

Observations are correlated within an image (5 target rows) and within a site
(shared soil, lighting, species), so a naive random split would leak information
and inflate scores. We use **StratifiedGroupKFold** with groups = image_id (all
rows of one image stay in the same fold) and stratification = binned Dry_Total_g
(the biomass range is balanced across folds).

We chose 5-fold cross-validation over a single train/test split for a more
reliable estimate. GroupKFold by state was rejected because states are imbalanced
(Tas 138 ... WA 32); the smallest state would yield an unstable fold. State is
instead used for error analysis. The **same folds** are used for every model
(tabular and image) so comparisons are fair.

Note on the Kaggle hidden test: it omits metadata, so a fair modality study must
be done on the locally labelled data via cross-validation, not the leaderboard.

## 4. Methods

We build a comparison aligned with the course syllabus:

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

**Tier 4 — Self-supervised features (DINOv2).** A frozen DINOv2 ViT-S/14
(self-supervised, 384-dim embeddings) with the same regression head. Embeddings
are precomputed once and cached, so only the head is trained per fold. This is
compared directly against frozen ResNet18 to isolate the effect of pretraining
type (self-supervised vs ImageNet) at the same frozen regime.

**Training.** Adam optimizer, MSE loss in log1p space, early stopping on
validation weighted-RMSE (original scale). Fine-tuning uses a smaller learning
rate (1e-4) since the whole backbone is updated. All image models share the
dataset pipeline and CV folds.

**Deployment (session 11).** A Gradio app serving the image-only model, deployed
to Hugging Face Spaces (see Section 7).

## 5. Results

**Evaluation metric.** Competition-style weighted RMSE (target weights
0.1/0.1/0.1/0.2/0.5, with Total weighted highest) plus per-target RMSE, MAE, and
R2 on the original gram scale, aggregated over 5 folds (mean +/- std).

**Main comparison table** (weighted RMSE, lower is better):

| Model | Weighted RMSE (mean +/- std) | R2 (Total) | Type |
|-------|-----------------------------:|-----------:|------|
| RandomForest (metadata)   | 11.83 +/- 1.73 | 0.72 | tabular |
| DINOv2 frozen             | 16.50 +/- 2.69 | 0.45 | image |
| ResNet18 fine-tuned       | 16.83 +/- 2.83 | 0.47 | image |
| SimpleCNN (from scratch)  | 18.08 +/- 3.47 | – | image |
| ResNet18 frozen           | 23.05 +/- 2.13 | – | image |
| Naive median              | 24.12 +/- 4.66 | -0.08 | floor |

A final deployable ResNet model trained on an enlarged train split reached 15.94
on a single holdout; the table reports the more conservative cross-validated
figures.

**Tabular per-target R2 (RandomForest):** Green 0.79, GDM 0.80, Total 0.72,
Clover 0.65, Dead 0.39. **Feature importance:** Height (0.57) and NDVI (0.24)
dominate (81% combined), consistent with the EDA.

**Image per-target R2 (DINOv2 frozen, best image model):** Green 0.56, GDM 0.51,
Clover 0.48, Total 0.45, Dead 0.16. DINOv2 is strongest on green/GDM and weakest
on dead matter, which is hard to read from a top-down photo.

## 6. Analysis

This study yields **two key results**, both supported by the 5-fold comparison.

### Key result 1 — Modality: cheap ground measurements beat imagery
The RandomForest on NDVI + height (11.83 +/- 1.73) is the best model overall,
ahead of the best image model, DINOv2 frozen (16.50 +/- 2.69) — a 28.3% lower
weighted RMSE. On 357 samples, the direct physical signal in NDVI/height is more
informative than anything a network extracts from raw pixels.

*Statistical support.* A paired Wilcoxon test across the 5 folds gives
statistic=0, p=0.0625: RandomForest beat DINOv2 in **all five folds** without
exception (RF per-fold [12.0, 8.9, 11.8, 14.4, 12.0] vs DINOv2
[17.6, 11.7, 19.9, 17.0, 16.3]). With only 5 folds, p=0.0625 is the minimum
achievable value, so this is the strongest possible consistency at this sample
size, even though it does not cross the conventional 0.05 threshold. The RF gap
(4.67) is 2.7x its own std (1.73), so the advantage is well outside RF's noise.
Caveat: folds are grouped by image and therefore not fully independent.

### Key result 2 — Within imagery: pretraining quality outweighs fine-tuning
At the same frozen regime, ResNet18 (ImageNet) scores 23.05 — barely better than
the naive median (24.12) — while DINOv2 (self-supervised) reaches 16.50, a 28.4%
improvement. The only difference is the pretraining, so this isolates its effect:
self-supervised features transfer to pasture far better than ImageNet features.
Frozen DINOv2 (16.50) also matches fully fine-tuned ResNet (16.83); the gap is
negligible (0.33, ~0.12 sigma), but the direction is notable — a strong frozen
backbone equals a weaker fine-tuned one, so *what the network was pretrained on
matters as much as whether its backbone is adapted* on this small dataset.

### Supporting observations
**Fine-tuning rescues a weak backbone.** ResNet improves from 23.05 (frozen) to
16.83 (fine-tuned), a 27% gain: on a weak (ImageNet) backbone, adapting it is
essential. DINOv2 needs no fine-tuning to reach the same level.

**A small from-scratch CNN is competitive.** SimpleCNN (18.08) is close to
fine-tuned ResNet (16.83) and beats frozen ResNet — on small data, a compact
on-task network rivals a large adapted one.

**Where imagery is weakest.** Dead matter (R2 0.16) and total (0.45) are the
hardest for the image model; the RandomForest beats DINOv2 on every target. Dead
biomass and scattered clover are difficult to read from a top-down photo.

**Methodological note.** Fine-tuning showed early-epoch validation instability;
early stopping on validation weighted-RMSE selected the best epoch.

**Interpretation.** For this dataset and scale, the image does not add predictive
value beyond cheap ground measurements. Practically, where NDVI/height are
available they are the better and cheaper predictor; the image-only model remains
useful only in the field scenario where such measurements are unavailable, which
is what the deployed app targets.

## 7. Deployment

A Gradio app (src/app.py) serves an image-only fine-tuned ResNet: a user uploads
a quadrat photo and receives the five biomass estimates plus a short
grazing-quality summary (total biomass, green share). It is deployed to Hugging
Face Spaces [link TBD].

The app deliberately uses the image-only model: it represents the realistic field
scenario where a farmer has only a phone photo, not NDVI or height. This is
consistent with the analysis — the demo illustrates the practical use case, while
the scientific comparison lives in the report. (DINOv2 gives the best image
accuracy but needs a heavier backbone; ResNet was chosen for the lightweight
deployment.)

## 8. Future Work

The image model still underperforms metadata; remaining directions:
- **Multimodal fusion** — combine the DINOv2 embedding with metadata; expected
  to beat either modality alone on the labelled set.
- **Cross-validation ensembling** and **test-time augmentation** for cheap,
  low-risk gains.
- **Domain-specific pretraining** — continue pretraining the backbone on
  vegetation/pasture imagery before fine-tuning.

Note: DINOv2 self-supervised features (originally proposed as future work) were
implemented and are reported above — they gave the best image model (16.50),
far above the ImageNet ResNet baseline.

## 9. References

- CSIRO Image2Biomass dataset (Kaggle competition).
- He et al., "Deep Residual Learning for Image Recognition" (ResNet), 2016.
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
  2023.
- Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011.
- Paszke et al., "PyTorch", NeurIPS 2019.
- [Add any other sources used.]

## 10. Contributions

- **[Name 1]:** [e.g. EDA, tabular baseline, report.]
- **[Name 2]:** [e.g. CNN pipeline, DINOv2, deployment.]

*(Fill in per actual division of work.)*
