"""
scripts/run_phase5.py — Phase 5 Adversarial Robustness Testing Experiment Driver.

Executes:
1. Feature-space attacks (FGSM, PGD-10) with physical constraints and per-feature scaled bounds.
2. Threat model enforcement:
   - Perturb ONLY attack flows (benign flows unperturbed).
   - Manipulate only attacker-controllable features (forward packets, delays, duration).
   - Monotonicity / directional bounds (delays/lengths/counts >= 0).
   - Integer rounding for discrete counts in raw space.
   - Goal = evasion into Benign class; metrics = evasion rate and detection rate under attack.
3. Controls:
   - Random noise of equal magnitude.
   - Query-based black-box attack for tree models in addition to surrogate transfer.
4. Multi-step PGD adversarial training (PGD-7 with random start) alongside FGSM training.
5. Empirical gradient masking verification (only claimed if supported by evidence).
6. Guards: aborts if clean model is constant classifier or if test set has < 500 attack or < 500 benign flows.
7. Generates honest scientific report with flagged citations for manual verification.

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
    check_gradient_masking,
    compute_defense_improvement,
    compute_robustness_metrics,
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
    logger.info("Phase 5 — Adversarial Robustness & Threat Model Evaluation")
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
        min_flows = 0
    else:
        dnn_epochs = getattr(cfg.defense, "epochs", 20)
        min_flows = getattr(cfg.evaluation, "min_flows", 500)

    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading & Preprocessing
    logger.info("\n[Step 1/6] Loading and Preprocessing Datasets...")
    df_cic = load_dataset(cfg.evaluation.primary_dataset)
    df_unsw = load_dataset(cfg.evaluation.secondary_dataset)
    logger.info(f"  Primary ({cfg.evaluation.primary_dataset}): {len(df_cic):,} rows")
    logger.info(f"  Secondary ({cfg.evaluation.secondary_dataset}): {len(df_unsw):,} rows")

    # Use behavior_only feature set for cross-dataset compatibility
    primary_dataset = cfg.evaluation.primary_dataset
    secondary_dataset = cfg.evaluation.secondary_dataset
    prep = Preprocessor(
        feature_mode="behavior_only",
        normalization="robust",
        dataset=primary_dataset,
    )
    splits = prep.fit_and_split(df_cic, test_ratio=0.25, val_ratio=0.1, seed=seed)

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    # Transform secondary dataset with fitted preprocessor
    X_unsw, y_unsw = prep.transform(df_unsw, dataset=secondary_dataset)

    # Align labels to common binary taxonomy ("Benign" vs "Attack")
    y_train = np.where(y_train == "Benign", "Benign", "Attack")
    y_test = np.where(y_test == "Benign", "Benign", "Attack")
    y_unsw = np.where(np.isin(y_unsw, ["Benign", "Normal"]), "Benign", "Attack")

    n_test_attacks = int(np.sum(y_test == "Attack"))
    n_test_benign = int(np.sum(y_test == "Benign"))
    logger.info(f"  Train: X={X_train.shape}, y={len(y_train)}")
    logger.info(f"  Test:  X={X_test.shape}, y={len(y_test)} ({n_test_attacks} attacks, {n_test_benign} benign)")
    logger.info(f"  Target (UNSW): X={X_unsw.shape}, y={len(y_unsw)}")

    # Guard: Check minimum sample size for scientific validity
    if min_flows > 0 and (n_test_attacks < min_flows or n_test_benign < min_flows):
        raise ValueError(
            f"Evaluation aborted: test set must have >= {min_flows} attack flows and "
            f">= {min_flows} benign flows (got {n_test_attacks} attacks, {n_test_benign} benign)."
        )

    # Initialize Attacker with fitted scaler and feature names
    attacker = FeatureSpaceAttacker(
        enforce_constraints=True,
        scaler=prep.scaler_,
        feature_names=prep.feature_names_,
        directional=True,
    )

    # 3. Model Training
    logger.info("\n[Step 2/6] Training Baseline Models...")
    xgb_model = XGBBaseline(n_estimators=100, max_depth=6, random_state=seed)
    xgb_model.fit(X_train, y_train)

    dnn_undefended = DNNBaseline(
        hidden_dims=(64, 32),
        epochs=dnn_epochs,
        batch_size=64,
        learning_rate=0.001,
        random_state=seed,
    )
    dnn_undefended.fit(X_train, y_train)

    # Guard: Constant classifier check
    for m_name, m_obj in [("XGBoost", xgb_model), ("DNN (Undefended)", dnn_undefended)]:
        preds = m_obj.predict(X_test)
        if len(np.unique(preds)) <= 1:
            raise ValueError(f"Clean model '{m_name}' is a constant classifier (predicts {np.unique(preds)[0]}).")

    xgb_clean_metrics = compute_robustness_metrics(y_test, xgb_model.predict(X_test))
    dnn_clean_metrics = compute_robustness_metrics(y_test, dnn_undefended.predict(X_test))
    logger.info(f"  XGBoost Baseline fitted -- Clean ACC: {xgb_clean_metrics['accuracy']:.4f}, DetRate: {xgb_clean_metrics['detection_rate']:.4f}")
    logger.info(f"  DNN (Undefended) fitted -- Clean ACC: {dnn_clean_metrics['accuracy']:.4f}, DetRate: {dnn_clean_metrics['detection_rate']:.4f}")

    # 4. Adversarial Training: FGSM and PGD-7 (Section 5.2.2)
    adv_epsilon = getattr(cfg.defense, "epsilon", 0.1)
    adv_ratio = getattr(cfg.defense, "adv_ratio", 0.5)

    logger.info(f"\n[Step 3/6] Implementing Adversarial Training (FGSM and PGD-7, eps={adv_epsilon})...")
    # 4a. FGSM Adversarial Training
    dnn_fgsm = DNNBaseline(
        hidden_dims=(64, 32),
        epochs=dnn_epochs,
        batch_size=64,
        learning_rate=0.001,
        random_state=seed,
    )
    dnn_fgsm.fit(
        X_train,
        y_train,
        adversarial_attacker=attacker,
        adv_epsilon=adv_epsilon,
        adv_ratio=adv_ratio,
        adv_method="fgsm",
    )
    dnn_fgsm_metrics = compute_robustness_metrics(y_test, dnn_fgsm.predict(X_test))
    logger.info(f"  DNN (FGSM-Trained) -- Clean ACC: {dnn_fgsm_metrics['accuracy']:.4f}, DetRate: {dnn_fgsm_metrics['detection_rate']:.4f}")

    # 4b. Multi-step PGD-7 Adversarial Training
    dnn_pgd = DNNBaseline(
        hidden_dims=(64, 32),
        epochs=dnn_epochs,
        batch_size=64,
        learning_rate=0.001,
        random_state=seed,
    )
    dnn_pgd.fit(
        X_train,
        y_train,
        adversarial_attacker=attacker,
        adv_epsilon=adv_epsilon,
        adv_ratio=adv_ratio,
        adv_method="pgd",
        pgd_steps=7,
        random_start=True,
    )
    dnn_pgd_metrics = compute_robustness_metrics(y_test, dnn_pgd.predict(X_test))
    logger.info(f"  DNN (PGD-7-Trained) -- Clean ACC: {dnn_pgd_metrics['accuracy']:.4f}, DetRate: {dnn_pgd_metrics['detection_rate']:.4f}")

    # 5. Epsilon Sweeps with Controls (FGSM, PGD-10, Random Noise, Query-based Black-box)
    logger.info("\n[Step 4/6] Running Epsilon Sweeps across Models and Attack Types...")

    # 5a. XGBoost (Surrogate Transfer and Query-Based Black-Box)
    logger.info("  Evaluating XGBoost (no defense) via surrogate transfer and query-based black-box...")
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
        include_random_control=True,
        include_query_blackbox=True,
        min_flows=min_flows,
    )

    # 5b. Undefended DNN
    logger.info("  Evaluating DNN (no defense) white-box...")
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
        include_random_control=True,
        min_flows=min_flows,
    )

    # 5c. FGSM-Defended DNN
    logger.info("  Evaluating DNN (FGSM-trained) white-box...")
    sweep_dnn_fgsm = run_epsilon_sweep(
        model=dnn_fgsm,
        X_test=X_test,
        y_test=y_test,
        attacker=attacker,
        epsilons=epsilons,
        attack_types=["fgsm", "pgd"],
        benign_label="Benign",
        model_name="DNN",
        defense_name="FGSM-trained",
        pgd_steps=10,
        include_random_control=True,
        min_flows=min_flows,
    )

    # 5d. PGD-7-Defended DNN
    logger.info("  Evaluating DNN (PGD-7-trained) white-box...")
    sweep_dnn_pgd = run_epsilon_sweep(
        model=dnn_pgd,
        X_test=X_test,
        y_test=y_test,
        attacker=attacker,
        epsilons=epsilons,
        attack_types=["fgsm", "pgd"],
        benign_label="Benign",
        model_name="DNN",
        defense_name="PGD-7-trained",
        pgd_steps=10,
        include_random_control=True,
        min_flows=min_flows,
    )

    combined_results_df = pd.concat(
        [sweep_xgb, sweep_dnn_base, sweep_dnn_fgsm, sweep_dnn_pgd], ignore_index=True
    )
    table_csv_path = output_dir / "robustness_sweep.csv"
    combined_results_df.to_csv(table_csv_path, index=False)
    logger.info(f"  Saved full robustness sweep table to: {table_csv_path}")

    # 6. Gradient Masking Verification
    logger.info("\n[Step 5/6] Verifying Gradient Masking Evidence...")
    # Evaluate transfer attack from XGBoost surrogate vs white-box
    dnn_pgd_wb_subset = sweep_dnn_pgd[
        (np.isclose(sweep_dnn_pgd["ε"], adv_epsilon)) & (sweep_dnn_pgd["Attack"].str.startswith("PGD"))
    ]
    wb_evasion = float(dnn_pgd_wb_subset["Evasion Rate"].iloc[0]) if not dnn_pgd_wb_subset.empty else 0.0

    # Transfer attack from undefended surrogate against PGD-defended model
    transfer_res = attacker.transfer_attack(
        surrogate_model=dnn_undefended.torch_model_,
        target_model=dnn_pgd,
        X=X_test,
        y=y_test,
        attack="pgd",
        epsilon=adv_epsilon,
        steps=10,
        benign_label="Benign",
    )
    bb_transfer_evasion = transfer_res["adv_evasion_rate"]
    masking_eval = check_gradient_masking(whitebox_evasion=wb_evasion, blackbox_evasion=bb_transfer_evasion)
    logger.info(f"  White-box PGD Evasion: {wb_evasion:.4f} | Black-box Transfer Evasion: {bb_transfer_evasion:.4f}")
    logger.info(f"  {masking_eval['evidence']}")

    # Defense improvement metrics
    imp_fgsm_vs_base = compute_defense_improvement(
        sweep_dnn_base, sweep_dnn_fgsm, target_epsilon=adv_epsilon, attack_type="FGSM"
    )
    imp_pgd_vs_base = compute_defense_improvement(
        sweep_dnn_base, sweep_dnn_pgd, target_epsilon=adv_epsilon, attack_type="PGD-10"
    )

    # 7. Cross-Dataset Adversarial Evaluation
    logger.info("\n[Step 6/6] Evaluating Cross-Dataset Impact of Defenses...")
    cross_results_pgd = evaluate_cross_dataset_robustness(
        baseline_model=dnn_undefended,
        defended_model=dnn_pgd,
        X_source=X_test,
        y_source=y_test,
        X_target=X_unsw,
        y_target=y_unsw,
        attacker=attacker,
        epsilon=adv_epsilon,
        benign_label="Benign",
    )

    summary_data = {
        "dataset_source": cfg.evaluation.primary_dataset,
        "dataset_target": cfg.evaluation.secondary_dataset,
        "defense_epsilon": adv_epsilon,
        "defense_ratio": adv_ratio,
        "defense_improvements": {
            "fgsm_defense": imp_fgsm_vs_base,
            "pgd_defense": imp_pgd_vs_base,
        },
        "gradient_masking_check": masking_eval,
        "cross_dataset_impact": cross_results_pgd,
        "citation_verification_notes": [
            "[MANUAL CITATION VERIFICATION REQUIRED] arXiv:2306.05494: verify black-box evasion claims.",
            "[MANUAL CITATION VERIFICATION REQUIRED] arXiv:2502.15561: verify empirical adversarial training gains.",
        ],
    }
    summary_json_path = output_dir / "robustness_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"  Saved summary metrics to: {summary_json_path}")

    report_md = _generate_report_markdown(
        combined_results_df,
        imp_fgsm_vs_base,
        imp_pgd_vs_base,
        cross_results_pgd,
        masking_eval,
    )
    report_path = output_dir / "robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"  Saved report to: {report_path}")

    logger.info("\n" + "=" * 72)
    logger.info("PHASE 5 RESULTS TABLE")
    logger.info("=" * 72)
    print(format_results_markdown_table(combined_results_df))
    logger.info("\nPhase 5 execution finished successfully.")
    return 0


def _generate_report_markdown(
    df: pd.DataFrame,
    imp_fgsm: dict,
    imp_pgd: dict,
    cross_results: dict,
    masking_eval: dict,
) -> str:
    md_table = format_results_markdown_table(df)

    content = f"""# Phase 5 — Adversarial Robustness & Threat Model Report

**Generated by:** `scripts/run_phase5.py`  
**Threat Model:** Endpoint attacker manipulating strictly controllable forward traffic & delays to evade into `Benign`.

---

> [!WARNING]
> ### Critical Scope & Scientific Framing Notice
> **Do NOT claim general adversarial robustness based on feature-space FGSM/PGD testing.**  
> Flow feature perturbations (e.g. modifying mathematical averages of packet length or IAT) do not translate directly into valid, routable IP packet streams without risking TCP stack corruption. All results represent **feature-space sensitivity and worst-case mathematical bounds**, not verified physical packet evasion.

---

## 1. Adversarial Evaluation Table

{md_table}

---

## 2. Adversarial Training Comparison (FGSM vs PGD-7)

- **Observed FGSM Training Impact (ε={imp_fgsm['epsilon']}):**
  - Baseline Adv ACC: `{imp_fgsm['baseline_adv_acc']:.4f}` → Defended Adv ACC: `{imp_fgsm['defended_adv_acc']:.4f}` (**Δ ACC: {imp_fgsm['defense_acc_gain']:+.4f}**)
  - Baseline Evasion Rate: `{imp_fgsm['baseline_evasion_rate']:.4f}` → Defended Evasion Rate: `{imp_fgsm['defended_evasion_rate']:.4f}` (**Δ Evasion: {imp_fgsm['evasion_reduction']:+.4f}**)
- **Observed PGD-7 Training Impact (ε={imp_pgd['epsilon']}):**
  - Baseline Adv ACC: `{imp_pgd['baseline_adv_acc']:.4f}` → Defended Adv ACC: `{imp_pgd['defended_adv_acc']:.4f}` (**Δ ACC: {imp_pgd['defense_acc_gain']:+.4f}**)
  - Baseline Evasion Rate: `{imp_pgd['baseline_evasion_rate']:.4f}` → Defended Evasion Rate: `{imp_pgd['defended_evasion_rate']:.4f}` (**Δ Evasion: {imp_pgd['evasion_reduction']:+.4f}**)

---

## 3. Gradient Masking Verification

- **White-Box PGD-10 Evasion Rate:** `{masking_eval['whitebox_evasion']:.4f}`
- **Black-Box Transfer Evasion Rate:** `{masking_eval['blackbox_evasion']:.4f}`
- **Evidence Verdict:** {masking_eval['evidence']}
- **Scientific Standard:** Gradient masking is only claimed when black-box/transfer attacks succeed where white-box attacks fail due to shattered or vanished gradients.

---

## 4. Cross-Dataset Generalization Impact

- **Source In-Domain Clean ACC:** Baseline = `{cross_results['source_in_domain']['baseline_clean_acc']:.4f}`, Defended = `{cross_results['source_in_domain']['defended_clean_acc']:.4f}`
- **Target Cross-Domain Clean ACC:** Baseline = `{cross_results['target_cross_domain']['baseline_clean_acc']:.4f}`, Defended = `{cross_results['target_cross_domain']['defended_clean_acc']:.4f}`
- **Cross-Domain Clean Accuracy Delta:** `{cross_results['target_cross_domain']['cross_clean_acc_change']:+.4f}`
- **Cross-Domain Generalization Gap:** Baseline = `{cross_results['target_cross_domain']['generalization_gap_baseline']:.4f}`, Defended = `{cross_results['target_cross_domain']['generalization_gap_defended']:.4f}`

---

## 5. Citations for Manual Verification

> [!IMPORTANT]
> The following citations are explicitly flagged for manual user verification:
> - **[MANUAL CITATION VERIFICATION REQUIRED]** `arXiv:2306.05494`: "Evasion Adversarial Attacks Remain Impractical Against ML-based Network Intrusion Detection Systems" — verify reported data availability and threat model assumptions.
> - **[MANUAL CITATION VERIFICATION REQUIRED]** `arXiv:2502.15561`: Feature-space adversarial defense gains — verify applicability to specific datasets and feature representations.
"""
    return content


if __name__ == "__main__":
    sys.exit(main())
