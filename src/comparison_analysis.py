# ═══ FULL TABLE mean ± std + paired Wilcoxon
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

# 1. Table models: per-fold weighted (new function weighted_per_fold)
folds = splits.get_cv_folds(wide, n_splits=5)
naive_df = baseline.evaluate_naive(wide, folds)
rf_df    = baseline.evaluate_model(wide, folds, "rf")
tab_all  = pd.concat([naive_df, rf_df], ignore_index=True)
tab_wpf  = baseline.weighted_per_fold(tab_all)   # {'naive_median':..., 'rf':...}

# 2. Collect all models: tabular + image (results from the notebook)
all_models = {}
all_models["RandomForest"]   = tab_wpf["rf"]
all_models["naive_median"]   = tab_wpf["naive_median"]
for name, r in results.items():           # simple_cnn, resnet_frozen, finetune, dinov2
    all_models[name] = {"weighted_rmse_mean": r["weighted_rmse_mean"],
                        "weighted_rmse_std":  r["weighted_rmse_std"],
                        "fold_weighted":      r["fold_weighted"]}

# 3. Table mean ± std
rows = [{"model": m,
         "weighted_rmse": f"{d['weighted_rmse_mean']:.2f} ± {d['weighted_rmse_std']:.2f}",
         "_sort": d["weighted_rmse_mean"]} for m, d in all_models.items()]
comp = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
print(comp.to_string(index=False))

# 4. Paired Wilcoxon: RandomForest vs DINOv2 by fold
rf_folds   = all_models["RandomForest"]["fold_weighted"]
best_image = "dinov2_frozen" if "dinov2_frozen" in all_models else "resnet18_finetune"
img_folds  = all_models[best_image]["fold_weighted"]
print(f"\nRF per-fold:     {[round(x,2) for x in rf_folds]}")
print(f"{best_image} per-fold: {[round(x,2) for x in img_folds]}")
try:
    stat, p = wilcoxon(rf_folds, img_folds)
    print(f"\nWilcoxon RF vs {best_image}: statistic={stat}, p={p:.4f}")
    print("Note: only 5 folds and folds are grouped by image (not fully independent),")
    print("so treat the p-value as indicative, not definitive.")
except ValueError as e:
    print(f"\nWilcoxon needs >0 differences: {e}")
