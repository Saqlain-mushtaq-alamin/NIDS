"""
scripts/run_phase5.py — Phase 5 Adversarial Robustness Testing Experiment Driver.

Executes:
1. Attack implementations: FGSM and PGD with feature-space physical constraints (Section 5.1).
2. Epsilon sweeps (ε ∈ {0.01, 0.05, 0.1, 0.2, 0.5}) across undefended baseline models (Section 5.2.1).
3. Adversarial training (50% clean + 50% FGSM-perturbed examples) (Section 5.2.2).
4. Defended model evaluation under attack (Section 5.2.3).
5. Defense improvement computation vs literature benchmark (+35% ACC, -12.5% FPR, arXiv:2502.15561) (Section 5.2.4).
6. Cross-dataset adversarial evaluation to measure out-of-domain transfer impact (Section 5.2.5).
7. Honest framing report generation per §3.4 and §6 of the research proposal (Section 5.3).

Usage:
    python scripts/run_phase5.py
    python scripts/run_phase5.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loader import load_dataset
from src.data.preprocessor import Preprocessor
from src.evaluation.robustness import (
    FeatureSpaceAttacker,
    compute_defense_improvement,
    evaluate_cross_dataset_robustness,
    format_results_markdown_table,
    run_epsilon_sweep,
)
from src.models.baseline_dnn import DNNBaseline
from src.models.baseline_xgb import XGBBaseline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logger
from src.utils.reproducibility import set_global_seed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 Adversarial Robustness Testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robustness.yaml",
        help="Path to Phase 5 config file.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution for smoke testing with fewer epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase5",
        help="Directory where results and reports will be saved.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger("NIDS-Phase5-Robustness")

    logger.info("=" * 72)
    logger.info("Phase 5 — Adversarial Robustness Testing")
    logger.info("=" * 72)

    # 1. Config & Seeding
    cfg = load_config(args.config)
    seed = getattr(cfg.reproducibility, "global_seed", 42)
    set_global_seed(seed)
    logger.info(f"Reproducibility global seed set to: {seed}")

    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    if args.quick:
        epsilons = [0.01, 0.1, 0.5]
        dnn_epochs = 10
    else:
        dnn_epochs = getattr(cfg.defense, "epochs", 20)

    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading & Preprocessing
    logger.info("\n[Step 1/6] Loading and Preprocessing Datasets...")
    df_cic = load_dataset(cfg.evaluation.primary_dataset)
    df_unsw = load_dataset(cfg.evaluation.secondary_dataset)
    logger.info(f"  Primary ({cfg.evaluation.primary_dataset}): {len(df_cic):,} rows")
    logger.info(f"  Secondary ({cfg.evaluation.secondary_dataset}): {len(df_unsw):,} rows")

    # Use behavior_only feature set for cross-dataset compatibility
    prep = Preprocessor(feature_mode="behavior_only", normalization="robust")
    splits = prep.fit_and_split(df_cic, test_ratio=0.25, val_ratio=0.1, seed=seed)

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    # Transform secondary dataset with fitted preprocessor
    X_unsw, y_unsw = prep.transform(df_unsw)

    # Align labels to common binary taxonomy ("Benign" vs "Attack") for cross-dataset evaluation
    y_train = np.where(y_train == "Benign", "Benign", "Attack")
    y_test = np.where(y_test == "Benign", "Benign", "Attack")
    y_unsw = np.where(np.isin(y_unsw, ["Benign", "Normal"]), "Benign", "Attack")

    logger.info(f"  Train: X={X_train.shape}, y={len(y_train)}")
    logger.info(f"  Test:  X={X_test.shape}, y={len(y_test)}")
    logger.info(f"  Target (UNSW): X={X_unsw.shape}, y={len(y_unsw)}")

    # 3. Model Training
    logger.info("\n[Step 2/6] Training Baseline Undefended Models...")
    # XGBoost Baseline
    xgb_model = XGBBaseline(n_estimators=100, max_depth=6, random_state=seed)
    xgb_model.fit(X_train, y_train)
    xgb_clean_acc = float(np.mean(y_test == xgb_model.predict(X_test)))
    logger.info(f"  XGBoost Baseline fitted -- Clean Test ACC: {xgb_clean_acc:.4f}")

    # Undefended DNN Baseline
    dnn_undefended = DNNBaseline(
        hidden_dims=(64, 32),
        epochs=dnn_epochs,
        batch_size=64,
        learning_rate=0.001,
        random_state=seed,
    )
    dnn_undefended.fit(X_train, y_train)
    dnn_clean_acc = float(np.mean(y_test == dnn_undefended.predict(X_test)))
    logger.info(f"  DNN (Undefended) fitted -- Clean Test ACC: {dnn_clean_acc:.4f}")

    # 4. Adversarial Training (Defended Model)
    logger.info("\n[Step 3/6] Implementing Adversarial Training (Section 5.2.2)...")
    attacker = FeatureSpaceAttacker(enforce_constraints=True, min_val=0.0)
    adv_epsilon = getattr(cfg.defense, "epsilon", 0.1)
    adv_ratio = getattr(cfg.defense, "adv_ratio", 0.5)

    dnn_defended = DNNBaseline(
        hidden_dims=(64, 32),
        epochs=dnn_epochs,
        batch_size=64,
        learning_rate=0.001,
        random_state=seed,
    )
    dnn_defended.fit(
        X_train,
        y_train,
        adversarial_attacker=attacker,
        adv_epsilon=adv_epsilon,
        adv_ratio=adv_ratio,
    )
    dnn_def_clean_acc = float(np.mean(y_test == dnn_defended.predict(X_test)))
    logger.info(
        f"  DNN (Adversarially Trained) fitted -- Clean Test ACC: {dnn_def_clean_acc:.4f} "
        f"(ratio={adv_ratio * 100:.0f}% adv, eps={adv_epsilon})"
    )

    # 5. Epsilon Sweeps (Section 5.1.5, 5.2.1, 5.2.3)
    logger.info("\n[Step 4/6] Running Epsilon Sweeps (FGSM & PGD-10)...")
    logger.info(f"  Sweeping eps in {epsilons}")

    # XGBoost sweep via DNN surrogate transfer attack
    logger.info("  Evaluating XGBoost (no defense) via surrogate transfer attack...")
    sweep_xgb = run_epsilon_sweep(
        model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        attacker=attacker,
        epsilons=epsilons,
        attack_types=["fgsm", "pgd"],
        surrogate_model=dnn_undefended.torch_model_,
        benign_label="Benign",
        model_name="XGBoost",
        defense_name="no defense",
        pgd_steps=10,
    )

    # Undefended DNN sweep
    logger.info("  Evaluating DNN (no defense) white-box attack...")
    sweep_dnn_base = run_epsilon_sweep(
        model=dnn_undefended,
        X_test=X_test,
        y_test=y_test,
        attacker=attacker,
        epsilons=epsilons,
        attack_types=["fgsm", "pgd"],
        benign_label="Benign",
        model_name="DNN",
        defense_name="no defense",
        pgd_steps=10,
    )

    # Defended DNN sweep
    logger.info("  Evaluating DNN (adv-trained) white-box attack...")
    sweep_dnn_def = run_epsilon_sweep(
        model=dnn_defended,
        X_test=X_test,
        y_test=y_test,
        attacker=attacker,
        epsilons=epsilons,
        attack_types=["fgsm", "pgd"],
        benign_label="Benign",
        model_name="DNN",
        defense_name="adv-trained",
        pgd_steps=10,
    )

    # Combine into single Results Table DataFrame
    combined_results_df = pd.concat([sweep_xgb, sweep_dnn_base, sweep_dnn_def], ignore_index=True)
    table_csv_path = output_dir / "robustness_sweep.csv"
    combined_results_df.to_csv(table_csv_path, index=False)
    logger.info(f"  Saved full epsilon sweep to: {table_csv_path}")

    # 6. Defense Improvement Computation (Section 5.2.4)
    logger.info("\n[Step 5/6] Computing Defense Improvements (Section 5.2.4)...")
    improvement_fgsm = compute_defense_improvement(
        sweep_dnn_base, sweep_dnn_def, target_epsilon=adv_epsilon, attack_type="FGSM"
    )
    improvement_pgd = compute_defense_improvement(
        sweep_dnn_base, sweep_dnn_def, target_epsilon=adv_epsilon, attack_type="PGD-10"
    )
    logger.info(
        f"  FGSM eps={adv_epsilon}: Delta Detection ACC = {improvement_fgsm['defense_acc_gain_pct']:+.2f}%, "
        f"FPR Reduction = {improvement_fgsm['defense_fpr_reduction_pct']:+.2f}%"
    )
    logger.info(
        f"  PGD-10 eps={adv_epsilon}: Delta Detection ACC = {improvement_pgd['defense_acc_gain_pct']:+.2f}%, "
        f"FPR Reduction = {improvement_pgd['defense_fpr_reduction_pct']:+.2f}%"
    )

    # 7. Cross-Dataset Adversarial Evaluation (Section 5.2.5)
    logger.info("\n[Step 6/6] Evaluating Cross-Dataset Impact of Adversarial Training...")
    cross_results = evaluate_cross_dataset_robustness(
        baseline_model=dnn_undefended,
        defended_model=dnn_defended,
        X_source=X_test,
        y_source=y_test,
        X_target=X_unsw,
        y_target=y_unsw,
        attacker=attacker,
        epsilon=adv_epsilon,
        benign_label="Benign",
    )
    logger.info(
        f"  In-Domain Clean ACC: Baseline={cross_results['source_in_domain']['baseline_clean_acc']:.4f}, "
        f"Defended={cross_results['source_in_domain']['defended_clean_acc']:.4f}"
    )
    logger.info(
        f"  Cross-Domain Clean ACC: Baseline={cross_results['target_cross_domain']['baseline_clean_acc']:.4f}, "
        f"Defended={cross_results['target_cross_domain']['defended_clean_acc']:.4f} "
        f"(Delta={cross_results['target_cross_domain']['cross_clean_acc_change']:+.4f})"
    )

    # Save summary JSON
    summary_data = {
        "dataset_source": cfg.evaluation.primary_dataset,
        "dataset_target": cfg.evaluation.secondary_dataset,
        "defense_epsilon": adv_epsilon,
        "defense_ratio": adv_ratio,
        "defense_improvements": {
            "fgsm": improvement_fgsm,
            "pgd": improvement_pgd,
        },
        "cross_dataset_impact": cross_results,
    }
    summary_json_path = output_dir / "robustness_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"  Saved summary metrics to: {summary_json_path}")

    # Generate Markdown Report (Section 5.3)
    logger.info("\nGenerating Phase 5 Robustness Report with Honest Framing (Section 5.3)...")
    report_md = _generate_report_markdown(
        combined_results_df, improvement_fgsm, improvement_pgd, cross_results
    )
    report_path = output_dir / "robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"  Saved human-readable report to: {report_path}")

    # Print markdown results table to log
    logger.info("\n" + "=" * 72)
    logger.info("PHASE 5 RESULTS TABLE")
    logger.info("=" * 72)
    try:
        print(format_results_markdown_table(combined_results_df))
    except Exception:
        print(format_results_markdown_table(combined_results_df).encode("ascii", "replace").decode("ascii"))

    logger.info("\nPhase 5 execution finished successfully.")
    return 0


def _generate_report_markdown(
    df: pd.DataFrame,
    imp_fgsm: dict,
    imp_pgd: dict,
    cross_results: dict,
) -> str:
    md_table = format_results_markdown_table(df)

    content = f"""# Phase 5 — Adversarial Robustness Evaluation Report

**Generated by:** `scripts/run_phase5.py`  
**Evaluation Scope:** Feature-space adversarial robustness & adversarial training impact.

---

> [!WARNING]
> ### Critical Scope Framing (per §3.4 and §6 of the Research Proposal)
> **Do NOT claim general adversarial robustness based on feature-space FGSM/PGD testing.**  
> The 2026 literature survey on ML-NIDS explicitly warns that feature-space perturbations do not translate directly to realizable network traffic without violating protocol syntax or physical link constraints. All results in this report represent **feature-space model sensitivity evaluations**, not problem-space evasion resilience. Problem-space (packet-level) evasion attacks are explicitly out of scope.

---

## 1. Adversarial Evaluation Results

{md_table}

---

## 2. Defense Improvement vs Literature Target

- **Literature Target (arXiv:2502.15561):** +35.0% Detection Accuracy, -12.5% False Positive Rate under attack.
- **Observed FGSM (ε={imp_fgsm['epsilon']}):**
  - Baseline Adv ACC: `{imp_fgsm['baseline_adv_acc']:.4f}` → Defended Adv ACC: `{imp_fgsm['defended_adv_acc']:.4f}` (**{imp_fgsm['defense_acc_gain_pct']:+.2f}%**)
  - Baseline Adv FPR: `{imp_fgsm['baseline_adv_fpr']:.4f}` → Defended Adv FPR: `{imp_fgsm['defended_adv_fpr']:.4f}` (**{imp_fgsm['defense_fpr_reduction_pct']:+.2f}% reduction**)
- **Observed PGD-10 (ε={imp_pgd['epsilon']}):**
  - Baseline Adv ACC: `{imp_pgd['baseline_adv_acc']:.4f}` → Defended Adv ACC: `{imp_pgd['defended_adv_acc']:.4f}` (**{imp_pgd['defense_acc_gain_pct']:+.2f}%**)
  - Baseline Adv FPR: `{imp_pgd['baseline_adv_fpr']:.4f}` → Defended Adv FPR: `{imp_pgd['defended_adv_fpr']:.4f}` (**{imp_pgd['defense_fpr_reduction_pct']:+.2f}% reduction**)

---

## 3. Cross-Dataset Generalization Impact (Section 5.2.5)

- **Source In-Domain Clean ACC:** Baseline = `{cross_results['source_in_domain']['baseline_clean_acc']:.4f}`, Defended = `{cross_results['source_in_domain']['defended_clean_acc']:.4f}`
- **Target Cross-Domain Clean ACC:** Baseline = `{cross_results['target_cross_domain']['baseline_clean_acc']:.4f}`, Defended = `{cross_results['target_cross_domain']['defended_clean_acc']:.4f}`
- **Cross-Domain Clean Accuracy Delta:** `{cross_results['target_cross_domain']['cross_clean_acc_change']:+.4f}`
- **Target Cross-Domain Adversarial ACC:** Baseline = `{cross_results['target_cross_domain']['baseline_adv_acc']:.4f}`, Defended = `{cross_results['target_cross_domain']['defended_adv_acc']:.4f}` (**Δ = {cross_results['target_cross_domain']['defense_gain_cross_domain']:+.4f}**)

---

## 4. Honest Framing & Threat Model Limitations (Section 5.3)

### 4.1 Feature-Space vs Problem-Space Limitations
- Tabular flow features (e.g. `Flow Duration`, `Total Fwd Packets`, `Flow Bytes/s`) are mathematical abstractions computed over completed flows. Perturbing a feature like `Total Fwd Packets` by adding non-integer or arbitrarily small noise cannot be realized on wire without emitting actual discrete IP packets.
- Perturbations that alter duration or byte counters without adding valid payload or padding packets break TCP checksums and flow reconstruction semantics.

### 4.2 Threat Model Assumptions
- **White-box DNN attacks:** Assumes the adversary possesses exact access to network weights, architecture, normalization statistics, and feature names.
- **Black-box Tree transfer attacks:** Assumes the adversary can train a surrogate neural network on a representative data distribution and transfer adversarial directions to the deployed decision tree ensemble (arXiv:2306.05494).

### 4.3 Citing the Field's Criticism
- **arXiv:2306.05494 ("Evasion Adversarial Attacks Remain Impractical Against ML-based NIDS"):** Highlights that black-box evasion often requires access to >50% of the target's training distribution.
- **arXiv:2502.15561:** Demonstrates defensive adversarial training benefits under controlled feature-space conditions.
- **2026 Survey:** Demonstrates the persistent problem-space gap in contemporary NIDS literature.
"""
    return content


if __name__ == "__main__":
    sys.exit(main())
