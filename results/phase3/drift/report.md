# Phase 3 — Concept-Drift Monitoring Report

## 1. Natural Drift Evaluation (Real Traffic)

Evaluated on natural concept and domain drift using real network traffic flows:
- **CIC-IDS2017 Day-to-Day Temporal Drift:** Baseline established on initial benign flows; evaluated against a held-out same-day stream (negative control) and subsequent chronological flows.
- **Cross-Dataset Domain Shift:** CIC-IDS2017 baseline transferred to UNSW-NB15 stream on shared behavioral features.

| Drift Scenario | Baseline Dataset | Stream Dataset | Control False Alarm Rate | Drift Alerts | Detection Delay (Flows) |
| --- | --- | --- | --- | --- | --- |
| **CIC-IDS2017 Temporal** | CIC-IDS2017 (Day 1 Benign) | CIC-IDS2017 (Day 2+ Flows) | 0.0000 | 12.0 | 43.0 flows |
| **Cross-Dataset Transfer** | CIC-IDS2017 (Benign) | UNSW-NB15 | 0.0000 | 6.0 | 11.0 flows |

---

## 2. Synthetic Benchmark (Controlled Sanity Check)

| Metric | Measured Value | Operational Gate |
| --- | --- | --- |
| **Mean Alerts on Shifted Stream** | 4.0 | ≥ 1 alert on genuine shift |
| **Mean Alerts on Control Stream** | 0.0 | Bound by ADWIN delta (0.002) |
| **Monitoring Coverage** | 50.0% | Proportion of drifted features monitored |
| **Recall on Monitored Shifted Features** | 100.0% | Target high recall on watched subspace |

> [!NOTE]
> All natural drift results are evaluated directly on real traffic captures with timestamps. The detector demonstrates sub-50-flow detection delay on natural shifts with negligible false alarm rate on stationary benign streams.
