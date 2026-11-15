import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

# Load results from phase2_seed42_results.json
import json
with open("results/phase2/phase2_seed42_results.json", "r") as f:
    res = json.load(f)

# Re-run CIC -> UNSW Tier A to get exact probabilities and ROC curve
from scripts.run_phase2_empirical import audit_split_protocols, audit_unsw_partitioning, prepare_shared_matrices

split_info = audit_split_protocols()
cic_df = split_info["merged_cic"]
unsw_part_info = audit_unsw_partitioning()
unsw_df = unsw_part_info["df_te_final"] # target test

data_a = prepare_shared_matrices(
    cic_df, unsw_df,
    split_info["primary_tr_indices"], split_info["primary_te_indices"],
    unsw_part_info["df_tr_dedup"], unsw_part_info["df_te_final"],
    feature_tier="tier_a",
    harmonize_units=True,
    log1p=False,
    subsample_max=200_000,
    seed=42,
)

scaler_cic_a = RobustScaler().fit(data_a["X_cic_tr"])
X_c_tr_sa = scaler_cic_a.transform(data_a["X_cic_tr"])
X_u_cross_sa = scaler_cic_a.transform(data_a["X_unsw_te"])
y_u_te = data_a["y_unsw_te"]

xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
xgb.fit(X_c_tr_sa, data_a["y_cic_tr"])
p_te = xgb.predict_proba(X_u_cross_sa)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_u_te, p_te)
auroc = roc_auc_score(y_u_te, p_te)
print(f"AUROC: {auroc:.4f}")

# Threshold from source val
# Evaluate MCC over all thresholds
best_mcc = -1.0
best_th_mcc = 0.5
mccs = []
for th in np.linspace(0.01, 0.99, 99):
    pred = (p_te >= th).astype(int)
    m = matthews_corrcoef(y_u_te, pred)
    mccs.append((th, m))
    if m > best_mcc:
        best_mcc = m
        best_th_mcc = th

th_src = 0.54 # from source val
pred_src = (p_te >= th_src).astype(int)
cm_src = confusion_matrix(y_u_te, pred_src, labels=[0, 1])
tn, fp, fn, tp = cm_src.ravel()
tpr_src = tp / (tp + fn)
fpr_src = fp / (fp + tn)
mcc_src = matthews_corrcoef(y_u_te, pred_src)

print(f"\nCONFUSION MATRIX AT SOURCE THRESHOLD (th={th_src:.4f}):")
print(f"  TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
print(f"  TPR={tpr_src:.4f}, FPR={fpr_src:.4f}, MCC={mcc_src:.4f}")
print(f"  ORACLE MAX MCC: {best_mcc:.4f} at threshold {best_th_mcc:.4f}")

print("\n10 ROC POINTS (FPR, TPR, Threshold, MCC):")
step = max(1, len(fpr) // 10)
indices = list(range(0, len(fpr), step))[:10]
if (len(fpr) - 1) not in indices:
    indices.append(len(fpr) - 1)

print(f"{'Point':<6} | {'Threshold':<12} | {'FPR':<10} | {'TPR':<10} | {'MCC':<10}")
print("-" * 56)
for i, idx in enumerate(indices):
    th = thresholds[idx]
    pred_i = (p_te >= th).astype(int)
    m_i = matthews_corrcoef(y_u_te, pred_i)
    print(f"{i+1:<6} | {th:12.4f} | {fpr[idx]:10.4f} | {tpr[idx]:10.4f} | {m_i:10.4f}")

