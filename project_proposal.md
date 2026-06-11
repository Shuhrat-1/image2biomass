# Image2Biomass: Predicting Pasture Biomass from Quadrat Photographs

**Project category:** Image regression (with tabular metadata baseline)

**Team members:** [Shuhrat Maksumov / Student ID 28548], [Xavier Loreto / ID: 28648]

---

## Project plan

**Problem statement.** We predict pasture biomass from photographs of 70x30 cm
field quadrats, using the CSIRO Image2Biomass dataset. Each image has five
regression targets in grams: green, dead, clover, GDM, and total dry matter.
Accurate biomass estimation helps farmers decide when and how to graze
livestock, making this a practically significant agricultural problem. Beyond
prediction, our central scientific question is a modality comparison: how much
does the image add over two cheap ground measurements (NDVI and pasture height)?

**Challenges.** The dataset is small (357 labelled images), which makes deep
networks prone to overfitting. The five targets are strongly right-skewed
(skew 1.4-2.8) with many zeros (clover ~38%), so we train on a log1p transform.
Observations are correlated by site and species, so a naive random split leaks
information; we use grouped cross-validation. Finally, the Kaggle hidden test
omits metadata, so a fair modality study must compare models on the locally
labelled data rather than the leaderboard.

**Dataset.** CSIRO Image2Biomass ([Kaggle](https://www.kaggle.com/competitions/csiro-biomass/overview)): 
357 quadrat photographs with pairedground-truth biomass, plus metadata (state, species, pre-grazing NDVI, average height, date). We split the labelled set with StratifiedGroupKFold by image,
stratified by binned total biomass, so the CNN and the tabular baseline are
evaluated on identical folds.

**Method.** We build a four-tier comparison aligned with the course: (1) a naive
median baseline; (2) a tabular RandomForest / gradient boosting on metadata
(sessions 4, 7); (3) a CNN trained from scratch (session 10); and (4) transfer
learning by fine-tuning a pretrained ResNet18, compared against frozen feature
extraction (session 12). A Gradio app deployed to Hugging Face (session 11)
demonstrates the image-only model in the realistic farmer scenario. All targets
are modelled in log1p space; predictions are inverted and clipped to be
non-negative.

**Evaluation.** We report a competition-style weighted RMSE (target weights
0.1/0.1/0.1/0.2/0.5) plus per-target RMSE, MAE, and R2 on the original scale,
aggregated over 5 folds (mean +/- std). We compare every model on the same folds
and analyse where each modality succeeds or fails (e.g. dead matter and clover
are hardest from imagery). The key deliverable is a quantified answer to whether
the image adds predictive value beyond cheap ground measurements.
