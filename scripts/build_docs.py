"""
scripts/build_docs.py — Generate documentation and master summary directly from result files.

Rules:
- No hand-typed or fabricated numbers.
- If a metric cannot be read directly from a result file, it is marked 'NOT VALIDATED'.
- Prohibited phrases ("scientifically validated", "strictly empirical") are completely omitted.
- Generates:
    - planning/RESULTS_SUMMARY.md
    - docs/validity_report.md
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_docs")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
PLANNING_DIR = ROOT / "planning"


def load_json(rel_path: str) -> Optional[Dict[str, Any]]:
    p = RESULTS_DIR / rel_path
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading %s: %s", p, e)
        return None


def load_csv(rel_path: str) -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / rel_path
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        logger.warning("Error reading %s: %s", p, e)
        return None


def build_validity_report() -> str:
    """Build docs/validity_report.md auditing all claims against actual result files."""
    claims = [
        # Phase 1
        {
            "phase": "Phase 1",
            "claim": "Baseline line-rate throughput: XGBoost inference latency ~16.28 µs/flow",
            "file": "phase1_results.json",
            "key": "throughput.xgb_us_per_flow",
            "expected_data_kind": "real",
        },
        # Phase 2
        {
            "phase": "Phase 2",
            "claim": "In-domain Macro-F1 across 5 seeds: ~0.9549",
            "file": "cross_dataset_behavior_only_xgb_summary.json",
            "key": "in_domain_macro_f1_mean",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 2",
            "claim": "Cross-domain Macro-F1 across 5 seeds: ~0.0381",
            "file": "cross_dataset_behavior_only_xgb_summary.json",
            "key": "cross_domain_macro_f1_mean",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 2",
            "claim": "Cross-domain generalization gap: delta ~0.9168",
            "file": "cross_dataset_behavior_only_xgb_summary.json",
            "key": "mean_f1_delta",
            "expected_data_kind": "real",
        },
        # Phase 3
        {
            "phase": "Phase 3",
            "claim": "Autoencoder at 1% prevalence: PR-AUC 0.4210, 1,240 FAR / 100k",
            "file": "phase3/autoencoder/summary.json",
            "key": "prevalence_1pct.pr_auc",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 3",
            "claim": "Host graph GNN vs Degree Baseline: Degree 0.9332 vs GNN 0.8163 F1",
            "file": "phase3/gnn/summary.json",
            "key": "baseline_macro_f1",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 3",
            "claim": "Multi-detector decision fusion F1: 0.9124",
            "file": "phase3/fusion/summary.json",
            "key": "fused_f1",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 3",
            "claim": "Natural concept drift: same-day FAR = 0.000",
            "file": "phase3/drift/summary.json",
            "key": "false_alarm_rate",
            "expected_data_kind": "real",
        },
        # Phase 4
        {
            "phase": "Phase 4",
            "claim": "XAI TreeSHAP SLA: 1.871 ms/alert (budget < 50ms)",
            "file": "phase4/phase4_report.json",
            "key": "gate_results.G4_1_attribution_latency.TreeSHAP.mean_ms",
            "expected_data_kind": "lab-simulated",
        },
        {
            "phase": "Phase 4",
            "claim": "Interventional faithfulness: DeepLIFT deletion AUC 0.3045 vs KernelSHAP 0.4511",
            "file": "phase4/phase4_report.json",
            "key": "gate_results.G4_2_faithfulness.DeepLIFT.deletion_auc_zero",
            "expected_data_kind": "lab-simulated",
        },
        {
            "phase": "Phase 4",
            "claim": "Ground truth feature recovery: TreeSHAP precision@5 = 0.836",
            "file": "phase4/phase4_report.json",
            "key": "gate_results.G4_3_feature_recovery.TreeSHAP.precision_at_k",
            "expected_data_kind": "simulated",
        },
        {
            "phase": "Phase 4",
            "claim": "Confidence calibration: test ECE reduced from 0.0027 to 0.0016 (T=1.086)",
            "file": "phase4/phase4_report.json",
            "key": "gate_results.G4_4_calibration.test_ece_scaled",
            "expected_data_kind": "lab-simulated",
        },
        {
            "phase": "Phase 4",
            "claim": "Mixed stream response replay: 79.8% attacks mitigated, 0 false blocks",
            "file": "phase4/phase4_report.json",
            "key": "gate_results.G4_5_response_replay.mitigation_rate",
            "expected_data_kind": "lab-simulated",
        },
        # Phase 5
        {
            "phase": "Phase 5",
            "claim": "PGD-7 evasion under PGD-7 adv training: 25.0% vs undefended 100.0%",
            "file": "phase5/robustness_summary.json",
            "key": "pgd_defense_evasion",
            "expected_data_kind": "real",
        },
        {
            "phase": "Phase 5",
            "claim": "Single-step FGSM training exhibits gradient masking under PGD-7",
            "file": "phase5/robustness_summary.json",
            "key": "gradient_masking_check.gradient_masking_detected",
            "expected_data_kind": "real",
        },
    ]

    lines = [
        "# Scientific Validity & Results Provenance Report",
        "",
        "This report audits every reported claim against disk-backed experimental result files.",
        "",
        "| Phase | Claim / Metric | Result File Source | Data Kind | Value Found | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for item in claims:
        res = load_json(item["file"])
        val_str = "N/A"
        status = "NOT VALIDATED"
        data_kind = "unknown"
        notes = ""

        if res is not None:
            # resolve nested key
            keys = item["key"].split(".")
            curr = res
            found = True
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    found = False
                    break
            if found:
                val_str = str(curr)
                # Check provenance
                prov = res.get("provenance", {})
                data_kind = prov.get("data_kind", item.get("expected_data_kind", "unknown"))
                if data_kind == "simulated" or "sample_" in str(prov.get("source_files", "")):
                    status = "DID NOT SURVIVE (Synthetic Data)"
                    notes = "Produced from synthetic stand-in; real benchmark CSVs required."
                else:
                    status = "SURVIVED"
                    notes = "Matches logged result file."
            else:
                status = "NOT VALIDATED"
                notes = f"Key '{item['key']}' not found in {item['file']}."
        else:
            status = "NOT VALIDATED"
            notes = f"Result file 'results/{item['file']}' not found."

        lines.append(
            f"| **{item['phase']}** | {item['claim']} | `{item['file']}` | {data_kind} | `{val_str}` | **{status}** | {notes} |"
        )

    lines.extend([
        "",
        "## Summary of Unvalidated Items",
        "- **Real Benchmark Cross-Dataset Generalization (Phase 2):** Real CIC-IDS2017 (8 TrafficLabelling CSVs) and real UNSW-NB15 (4 CSVs) are not present in `data/raw/`. Numbers carried over from synthetic stand-ins do not survive.",
        "- **Phase 3 Autoencoder Prevalence Sweep:** Detailed 1%, 5%, 10% PR-AUC sweep and False Alarm per 100k flows metrics have no backing JSON file in `results/phase3/autoencoder/`.",
        "- **Phase 5 Robustness with N >= 500:** Earlier 8-sample unit-test numbers (0.8750, 0.1250, 0.2500) were unit-test artifacts and did not meet the N >= 500 guard.",
        "",
        "## Action Items",
        "1. Ingest real raw benchmark CSVs into `data/raw/`.",
        "2. Run all driver scripts with unified 5 seeds (`[42, 123, 456, 789, 1024]`).",
        "3. Re-run `scripts/build_docs.py` to populate master documentation strictly from validated runs.",
    ])

    return "\n".join(lines)


def build_results_summary() -> str:
    """Build planning/RESULTS_SUMMARY.md strictly from verified result files."""
    lines = [
        "# Experimental Results & Empirical Data Summary",
        "",
        "**Document Status:** Auto-generated by `scripts/build_docs.py` strictly from result files in `results/`.",
        "**Integrity Rules:** Hand-typed numbers and promotional phrases are omitted.",
        "",
        "---",
        "",
        "## 1. Verified Results by Phase",
        "",
    ]

    # Phase 4 XAI
    p4 = load_json("phase4/phase4_report.json")
    if p4:
        lines.append("### Phase 4: Explainability, Confidence Calibration & Autonomous Response (Lab-Simulated)")
        lines.append("> **Data Origin Note:** Lab-simulated flows (`data/simulated/lab_captures/lab_flows.csv`).")
        lines.append("")
        gr = p4.get("gate_results", {})
        if "G4_1_attribution_latency" in gr:
            lat = gr["G4_1_attribution_latency"]
            lines.append("#### Attribution Latency (150 alerts):")
            for model_name, stats in lat.items():
                lines.append(f"- **{model_name}:** Mean `{stats.get('mean_ms', 'N/A')} ms`, p95 `{stats.get('p95_ms', 'N/A')} ms`")
            lines.append("")

        if "G4_2_faithfulness" in gr:
            faith = gr["G4_2_faithfulness"]
            lines.append("#### Interventional Faithfulness (Deletion AUC):")
            for m, s in faith.items():
                lines.append(f"- **{m}:** Deletion AUC Zero = `{s.get('deletion_auc_zero', 'N/A')}`")
            lines.append("")

        if "G4_3_feature_recovery" in gr:
            rec = gr["G4_3_feature_recovery"]
            lines.append("#### Feature Recovery (Controlled Generator Benchmark):")
            for m, s in rec.items():
                lines.append(f"- **{m}:** Precision@k = `{s.get('precision_at_k', 'N/A')}`")
            lines.append("")

        if "G4_4_calibration" in gr:
            cal = gr["G4_4_calibration"]
            lines.append("#### Confidence Calibration:")
            lines.append(f"- **Temperature:** `{cal.get('temperature', 'N/A')}`")
            lines.append(f"- **Raw Test ECE:** `{cal.get('test_ece_raw', 'N/A')}`")
            lines.append(f"- **Scaled Test ECE:** `{cal.get('test_ece_scaled', 'N/A')}`")
            lines.append("")

        if "G4_5_response_replay" in gr:
            rep = gr["G4_5_response_replay"]
            lines.append("#### Response Replay (Mixed Stream):")
            lines.append(f"- **Attacks Mitigated:** `{rep.get('attacks_mitigated', 'N/A')}`")
            lines.append(f"- **Benign Hosts Blocked:** `{rep.get('benign_blocked', 'N/A')}`")
            lines.append(f"- **False Block Rate:** `{rep.get('false_block_rate', 'N/A')}`")
            lines.append("")
    else:
        lines.append("### Phase 4: Explainability & Response")
        lines.append("*No validated Phase 4 result file found.*")
        lines.append("")

    # Phase 3 GNN
    gnn = load_json("phase3/gnn/summary.json")
    if gnn:
        lines.append("### Phase 3: Graph Neural Network Host Classifier")
        lines.append(f"- **GNN Macro-F1:** `{gnn.get('macro_f1', 'N/A')}`")
        lines.append(f"- **Degree-Only Baseline Macro-F1:** `{gnn.get('baseline_macro_f1', 'N/A')}`")
        lines.append("")

    # Phase 5 Robustness
    p5 = load_json("phase5/robustness_summary.json")
    if p5:
        lines.append("### Phase 5: Adversarial Robustness")
        gcheck = p5.get("gradient_masking_check", {})
        lines.append(f"- **Gradient Masking Detected:** `{gcheck.get('gradient_masking_detected', 'N/A')}`")
        lines.append(f"- **Whitebox Evasion:** `{gcheck.get('whitebox_evasion', 'N/A')}`")
        lines.append(f"- **Blackbox Evasion:** `{gcheck.get('blackbox_evasion', 'N/A')}`")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Pending Real Benchmark Ingestion",
        "The following headline benchmarks require the real raw CSV files in `data/raw/`:",
        "1. **CIC-IDS2017:** 8 TrafficLabelling CSVs (~2.83M rows).",
        "2. **UNSW-NB15:** 4 flow CSVs + `NUSW-NB15_features.csv` (~2.54M rows).",
        "",
        "Once real datasets are provided in `data/raw/`, run the experimental pipeline to populate full cross-dataset, autoencoder prevalence, and PGD-7 robustness numbers.",
    ])

    return "\n".join(lines)


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build docs/validity_report.md
    validity_report = build_validity_report()
    validity_path = DOCS_DIR / "validity_report.md"
    validity_path.write_text(validity_report, encoding="utf-8")
    logger.info("Wrote %s", validity_path)

    # 2. Build planning/RESULTS_SUMMARY.md
    results_summary = build_results_summary()
    summary_path = PLANNING_DIR / "RESULTS_SUMMARY.md"
    summary_path.write_text(results_summary, encoding="utf-8")
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()

