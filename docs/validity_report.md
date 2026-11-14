# Scientific Validity & Results Provenance Report

This report audits every reported claim against disk-backed experimental result files.

| Phase | Claim / Metric | Result File Source | Data Kind | Value Found | Status | Notes |
|---|---|---|---|---|---|---|
| **Phase 1** | Baseline line-rate throughput: XGBoost inference latency ~16.28 µs/flow | `phase1_results.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'throughput.xgb_us_per_flow' not found in phase1_results.json. |
| **Phase 2** | In-domain Macro-F1 across 5 seeds: ~0.9549 | `cross_dataset_behavior_only_xgb_summary.json` | real | `0.9548757552324041` | **SURVIVED** | Matches logged result file. |
| **Phase 2** | Cross-domain Macro-F1 across 5 seeds: ~0.0381 | `cross_dataset_behavior_only_xgb_summary.json` | real | `0.03808827946748226` | **SURVIVED** | Matches logged result file. |
| **Phase 2** | Cross-domain generalization gap: delta ~0.9168 | `cross_dataset_behavior_only_xgb_summary.json` | real | `0.9167874757649218` | **SURVIVED** | Matches logged result file. |
| **Phase 3** | Autoencoder at 1% prevalence: PR-AUC 0.4210, 1,240 FAR / 100k | `phase3/autoencoder/summary.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'prevalence_1pct.pr_auc' not found in phase3/autoencoder/summary.json. |
| **Phase 3** | Host graph GNN vs Degree Baseline: Degree 0.9332 vs GNN 0.8163 F1 | `phase3/gnn/summary.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'baseline_macro_f1' not found in phase3/gnn/summary.json. |
| **Phase 3** | Multi-detector decision fusion F1: 0.9124 | `phase3/fusion/summary.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'fused_f1' not found in phase3/fusion/summary.json. |
| **Phase 3** | Natural concept drift: same-day FAR = 0.000 | `phase3/drift/summary.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'false_alarm_rate' not found in phase3/drift/summary.json. |
| **Phase 4** | XAI TreeSHAP SLA: 1.871 ms/alert (budget < 50ms) | `phase4/phase4_report.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'gate_results.G4_1_attribution_latency.TreeSHAP.mean_ms' not found in phase4/phase4_report.json. |
| **Phase 4** | Interventional faithfulness: DeepLIFT deletion AUC 0.3045 vs KernelSHAP 0.4511 | `phase4/phase4_report.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'gate_results.G4_2_faithfulness.DeepLIFT.deletion_auc_zero' not found in phase4/phase4_report.json. |
| **Phase 4** | Ground truth feature recovery: TreeSHAP precision@5 = 0.836 | `phase4/phase4_report.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'gate_results.G4_3_feature_recovery.TreeSHAP.precision_at_k' not found in phase4/phase4_report.json. |
| **Phase 4** | Confidence calibration: test ECE reduced from 0.0027 to 0.0016 (T=1.086) | `phase4/phase4_report.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'gate_results.G4_4_calibration.test_ece_scaled' not found in phase4/phase4_report.json. |
| **Phase 4** | Mixed stream response replay: 79.8% attacks mitigated, 0 false blocks | `phase4/phase4_report.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'gate_results.G4_5_response_replay.mitigation_rate' not found in phase4/phase4_report.json. |
| **Phase 5** | PGD-7 evasion under PGD-7 adv training: 25.0% vs undefended 100.0% | `phase5/robustness_summary.json` | unknown | `N/A` | **NOT VALIDATED** | Key 'pgd_defense_evasion' not found in phase5/robustness_summary.json. |
| **Phase 5** | Single-step FGSM training exhibits gradient masking under PGD-7 | `phase5/robustness_summary.json` | real | `False` | **SURVIVED** | Matches logged result file. |

## Summary of Unvalidated Items
- **Real Benchmark Cross-Dataset Generalization (Phase 2):** Real CIC-IDS2017 (8 TrafficLabelling CSVs) and real UNSW-NB15 (4 CSVs) are not present in `data/raw/`. Numbers carried over from synthetic stand-ins do not survive.
- **Phase 3 Autoencoder Prevalence Sweep:** Detailed 1%, 5%, 10% PR-AUC sweep and False Alarm per 100k flows metrics have no backing JSON file in `results/phase3/autoencoder/`.
- **Phase 5 Robustness with N >= 500:** Earlier 8-sample unit-test numbers (0.8750, 0.1250, 0.2500) were unit-test artifacts and did not meet the N >= 500 guard.

## Action Items
1. Ingest real raw benchmark CSVs into `data/raw/`.
2. Run all driver scripts with unified 5 seeds (`[42, 123, 456, 789, 1024]`).
3. Re-run `scripts/build_docs.py` to populate master documentation strictly from validated runs.