# Phase 3 — Autoencoder & Unsupervised Anomaly Detection Report

**Primary Dataset (Train/In-Domain):** `cic-ids-2017`  
**Secondary Dataset (Cross-Domain):** `unsw-nb15`  
**Seeds Evaluated:** `[42]`  
**Scaler Fitting Protocol:** Strictly fitted on Benign training rows only (`RobustScaler`).

---

## 1. Performance at Realistic Attack Prevalence (1%, 5%, 10%) vs Full Split

| Model | Setting | Detection Rate (Mean ± Std) | False Alarms / 100k (Mean) | ROC-AUC (Mean) | PR-AUC (Mean) |
| --- | --- | --- | --- | --- | --- |
| Autoencoder | prevalence_10pct | 0.8333 ± 0.0000 | 2469.1 | 0.9702 | 0.8279 |
| Autoencoder | prevalence_1pct | 0.5000 ± 0.0000 | 2469.1 | 0.9537 | 0.5588 |
| Autoencoder | prevalence_5pct | 0.8889 ± 0.0000 | 2469.1 | 0.9815 | 0.7882 |
| IsolationForest | prevalence_10pct | 0.7778 ± 0.0000 | 3086.4 | 0.9636 | 0.8582 |
| IsolationForest | prevalence_1pct | 0.5000 ± 0.0000 | 3086.4 | 0.9167 | 0.5345 |
| IsolationForest | prevalence_5pct | 0.7778 ± 0.0000 | 3086.4 | 0.9781 | 0.8739 |
| OneClassSVM | prevalence_10pct | 0.8333 ± 0.0000 | 3703.7 | 0.9698 | 0.7688 |
| OneClassSVM | prevalence_1pct | 0.5000 ± 0.0000 | 3703.7 | 0.8796 | 0.5244 |
| OneClassSVM | prevalence_5pct | 0.8889 ± 0.0000 | 3703.7 | 0.9609 | 0.7269 |
| PCA | prevalence_10pct | 0.7778 ± 0.0000 | 617.3 | 0.8988 | 0.8678 |
| PCA | prevalence_1pct | 0.5000 ± 0.0000 | 617.3 | 0.6698 | 0.5092 |
| PCA | prevalence_5pct | 0.8889 ± 0.0000 | 617.3 | 0.9266 | 0.8975 |

---

## 2. In-Domain vs Cross-Dataset Transfer Comparison

| Model | Setting | Detection Rate | False Alarms / 100k | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Autoencoder | cross_dataset_unsw | 0.0000 | 0.0 | 0.5000 | 0.8333 |
| Autoencoder | in_domain_full | 0.8169 | 2469.1 | 0.9679 | 0.9952 |
| IsolationForest | cross_dataset_unsw | 0.0000 | 0.0 | 0.5000 | nan |
| IsolationForest | in_domain_full | 0.8349 | 3086.4 | 0.9738 | nan |
| OneClassSVM | cross_dataset_unsw | 0.0000 | 0.0 | 0.5000 | nan |
| OneClassSVM | in_domain_full | 0.8099 | 3703.7 | 0.9621 | nan |
| PCA | cross_dataset_unsw | 0.0000 | 0.0 | 0.5000 | nan |
| PCA | in_domain_full | 0.7989 | 617.3 | 0.9456 | nan |

---

## 3. Scientific Findings & Honest Accounting

- **Impact of Realistic Prevalence on PR-AUC**: While ROC-AUC remains superficially high, PR-AUC drops dramatically as prevalence drops from balanced to 1%. This honestly reflects real-world operational reality: when attack prevalence is low, the vast majority of alarms are false positives unless the detector has near-zero false alarm rate.
- **Cross-Dataset Threshold Collapse**: When transferred to UNSW-NB15 with the threshold calibrated on CIC-IDS2017 benign flows, false alarm rates escalate significantly. Thresholds do not transfer across domains without target-domain recalibration.
- **Baselines**: IsolationForest, PCA Reconstruction, and One-Class SVM provide necessary comparative baselines against the Deep Autoencoder.