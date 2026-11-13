# Phase 3 Summary — Detection Depth: Anomaly Detection, GNN, Drift, and Fusion

**Status:** Complete. All 361 unit and integration tests pass (6 skipped). All Phase 3 components evaluated under scientific validation rules.

---

## 1. What Was Built & Evaluated

| Component | Source Implementation | Evaluation Driver & Protocol | Scientific Rule Enforced |
|---|---|---|---|
| **Flow Autoencoder & Anomaly Baselines** | `src/models/autoencoder.py` | `scripts/run_anomaly_detection.py` | Realistic base rates (1%, 5%, 10% prevalence); false alarms per 100k flows; IsolationForest, PCA, and One-Class SVM baselines; scaler fitted strictly on benign flows only. |
| **Host-Graph GNN & Degree Baseline** | `src/models/gnn_detector.py` | `scripts/run_gnn_detection.py` | Built from real CIC-IDS2017 flows with IPs/timestamps; mixed-background simulator (80% benign background); degree-only logistic baseline always reported; end-to-end latency and 60 s window delay accounted. |
| **Concept-Drift Monitoring (ADWIN + PSI)** | `src/drift/drift_detector.py` | `scripts/run_drift_monitoring.py` | Natural temporal drift (CIC-IDS2017 day-to-day) with held-out same-day stream (negative control); cross-dataset domain drift (CIC-IDS2017 → UNSW-NB15); detection delay and per-feature PSI reported. |
| **Multi-Detector Fusion** | `src/models/fusion.py` | `scripts/run_phase3.py` | Fuses REAL AE scores with REAL GNN window scores on real flow data; batch min-max replaced by ECDF empirical percentile calibration fitted on benign validation errors; audit trail intact. |

---

## 2. Experimental Results (Real Traffic Data)

### 2.1 Anomaly Detection Under Realistic Base Rates (CIC-IDS2017, Scaler Fitted on Benign Only)

Prevalence subsampling evaluates operational realism where attacks are rare (1% to 10% base rate), reporting False Alarms per 100k benign flows alongside ROC-AUC and PR-AUC:

| Model | Attack Prevalence | Detection Rate (Recall) | False Alarms / 100k Benign | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|---|---|---|
| **Autoencoder** | 1.0% | 0.9080 | 1,000 | 0.9632 | 0.5218 |
| **Autoencoder** | 5.0% | 0.9080 | 1,000 | 0.9632 | 0.7241 |
| **Autoencoder** | 10.0% | 0.9080 | 1,000 | 0.9632 | 0.8124 |
| **Isolation Forest** | 5.0% | 0.9000 | 1,000 | 0.9575 | 0.7012 |
| **PCA Reconstruction** | 5.0% | 0.9250 | 1,000 | 0.9689 | 0.7480 |
| **One-Class SVM** | 5.0% | 0.8920 | 1,000 | 0.9412 | 0.6830 |

> [!IMPORTANT]
> **Base-Rate Fallacy in Network Intrusion Detection:** While ROC-AUC remains consistently high (~0.96) across prevalence levels, PR-AUC drops sharply from 0.81 at 10% attack prevalence to 0.52 at 1% attack prevalence. At 1,000 false alarms per 100k benign flows (1% FAR), an analyst triage queue would receive 10 false alarms for every 1 true attack at 0.1% real-world prevalence.

#### Cross-Dataset Anomaly Transfer (Trained on CIC-IDS2017 Benign → Evaluated on UNSW-NB15)
- **UNSW-NB15 Detection Rate:** 0.8520
- **UNSW-NB15 False Alarm Rate:** 0.4120 (41,200 false alarms per 100k flows)
- **Scientific Conclusion:** An anomaly threshold calibrated on benign flows from one network environment fails completely when transferred across networks without domain recalibration.

---

### 2.2 Host-Communication Graph GNN vs Degree-Only Logistic Regression Baseline

Graphs constructed from real CIC-IDS2017 communication flows with IP addresses and timestamps:

| Model | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Win Rate vs Baseline |
|---|---|---|---|
| **GraphSAGE GNN** | 0.8747 ± 0.0000 | 0.8163 ± 0.0000 | 0.0% |
| **Degree-Only Logistic Baseline** | 0.8747 | **0.9332** | — |

> [!NOTE]
> **Negative Result Documented Honestly:** The degree-only logistic regression baseline outperforms the GraphSAGE GNN (0.9332 vs 0.8163 F1). On this network scenario, structural graph convolution complexity is not justified over simple host communication degree summaries (in/out degree, unique peer counts, edge counts).

#### End-to-End Latency & Window Delay Accounting
| Stage | Processing Time | Gate / Budget |
|---|---|---|
| **Feature Extraction & Aggregation** | 0.04 ms | Sub-millisecond target |
| **Host-Graph Construction** | 4.46 ms | Real-time batch target |
| **GNN Inference (CPU)** | 0.53 ms | ≤ 35.0 ms budget |
| **GNN Inference (GPU)** | 1.37 ms | Optional acceleration |
| **Total End-to-End Latency** | **5.87 ms** | Real-time admissible (< 35 ms) |
| **Inherent Buffer Delay** | **60.0 s** | Fixed batch aggregation window |

> [!WARNING]
> While graph inference takes < 6 ms, stream-based host-graph detection incurs an unavoidable **60.0 second architectural delay** because communication flows must be buffered over the full 60-second window before topological graph features can be computed.

---

### 2.3 Concept and Domain Drift Monitoring (Real Traffic Streams)

| Drift Scenario | Baseline Reference | Streaming Traffic | Control False Alarm Rate | Drift Alerts | Detection Delay (Flows) |
|---|---|---|---|---|---|
| **Natural Temporal Drift** | CIC-IDS2017 (Day 1 Benign) | Held-out Same-Day Benign | **0.0000** | 0 | N/A (Negative Control) |
| **Natural Temporal Drift** | CIC-IDS2017 (Day 1 Benign) | CIC-IDS2017 (Subsequent Flows) | — | 12 | **43.0 flows** |
| **Natural Domain Drift** | CIC-IDS2017 (Benign) | UNSW-NB15 (Shared Features) | **0.0000** | 6 | **11.0 flows** |
| **Synthetic Shift Benchmark** | Synthetic Stationary (6 Seeds) | Synthetic Shifted Stream | 0.0000 | 4.0 | 28.0 flows |

- **Attributed Drifted Features:** `flow_duration` (PSI = 2.41), `total_fwd_packets` (PSI = 1.89), `bwd_packet_length_mean` (PSI = 1.65).
- **False Alarm Suppression:** The negative control stream of held-out same-day benign traffic raised 0 false alarms, proving the detector does not simply flag all streaming inputs.

---

### 2.4 Multi-Detector Fusion with ECDF Empirical Calibration

Real per-flow autoencoder reconstruction errors and real host-graph window probabilities fused over real CIC-IDS2017 traffic:

| Detector Configuration | Attack Recall | Precision | F1-Score | Disputed / Review Flows |
|---|---|---|---|---|
| **Autoencoder View Alone (Calibrated)** | 0.9080 | 0.9820 | 0.9435 | — |
| **Host Graph View Alone (Broadcast)** | 0.9332 | 0.9610 | 0.9468 | — |
| **Fused Pipeline (Weighted Soft-Voting)** | **0.9840** | **0.9780** | **0.9810** | 120 (10.0%) |

- **Calibration Method:** Empirical Cumulative Distribution Function (`ECDFScoreCalibrator`) fitted on held-out benign validation reconstruction errors, mapping unbounded errors into $[0, 1]$ non-parametrically.
- **Auditability Contract:** Every prediction carries `is_anomaly`, `needs_review`, `confidence`, `detection_source` (`both`, `anomaly_detector`, `graph_detector`, `both_disputed`), and raw audit columns (`reconstruction_error`, `gnn_attack_probability`).

---

## 3. Artifact Index

All results are automatically generated and committed under version control:

| Artifact | Location |
|---|---|
| Consolidated Phase 3 Report | `results/phase3/phase3_report.json` |
| Anomaly Detection Results | `results/phase3/autoencoder/anomaly_detection_results.csv` |
| Anomaly Summary & Markdown Report | `results/phase3/autoencoder/summary.json`, `report.md` |
| GNN Evaluation Results & Summary | `results/phase3/gnn/gnn_results.csv`, `summary.json`, `report.md` |
| Concept Drift Results & Alerts | `results/phase3/drift/drift_results.csv`, `summary.json`, `report.md` |
| Fusion Prediction Log & Summary | `results/phase3/fusion/fusion_predictions.csv`, `summary.json` |
