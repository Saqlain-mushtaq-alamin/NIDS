# Phase 3 — Graph Neural Network (GNN) Detection Report

**Dataset Source:** `real_cic2017`  
**Seeds Evaluated:** `[42]`  
**Class Weighting:** Balanced inverse frequency per window label.  

---

## 1. GNN vs Degree-Only Logistic Regression Baseline

| Model | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | GNN Win Rate |
| --- | --- | --- | --- |
| **GraphSAGE GNN** | 0.8747 ± 0.0000 | 0.8163 ± 0.0000 | 0.0% |
| **Degree-Only Logistic Baseline** | 0.8747 | 0.9332 | — |

> **Scientific Verdict on GNN Value:** Degree-only logistic regression baseline is competitive with or outperforms GNN; GNN structural complexity is not justified on this scenario.

---

## 2. End-to-End Latency & Window Delay Accounting

| Pipeline Stage | Measured Latency | Operational Gate / Budget |
| --- | --- | --- |
| **Feature Extraction & Aggregation** | 0.05 ms | Sub-millisecond target |
| **Host-Graph Construction** | 4.82 ms | Real-time batch target |
| **GNN Inference (CPU)** | 0.54 ms | ≤ 35.0 ms budget |
| **GNN Inference (GPU)** | 2.79 ms | Optional acceleration |
| **Total Processing Latency (End-to-End)** | 7.67 ms | Real-time admissible |
| **Inherent Window Buffer Delay** | **60.0 s** | Fixed batch aggregation window |

> [!NOTE]
> While graph inference runs within the 35 ms budget once the graph is built, stream-based host-graph detection incurs a **60.0 second architectural delay** because communication flows must be buffered over the full 60-second window before topological graph features can be computed.