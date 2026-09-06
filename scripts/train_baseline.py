"""
scripts/train_baseline.py — Train and evaluate baseline models for Phase 1.
Usage: python scripts/train_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.reproducibility import set_global_seed
from src.utils.config import load_config
from src.utils.logging_setup import setup_logger
from src.data.loader import load_dataset
from src.data.preprocessor import Preprocessor
from src.data.label_audit import LabelAuditor
from src.models.baseline_xgb import XGBBaseline
from src.models.baseline_rf import RFBaseline
from src.evaluation.metrics import full_evaluation_report
import json


def main():
    logger = setup_logger("NIDS-Phase1")
    logger.info("=" * 60)
    logger.info("Phase 1 — Data Pipeline & In-Dataset Baseline")
    logger.info("=" * 60)

    # Load config
    cfg = load_config("configs/baseline.yaml")
    seed = cfg.reproducibility.global_seed
    set_global_seed(seed)
    logger.info(f"Global seed: {seed}")

    # ---- 1. Data Loading ----
    logger.info("\n[Step 1/5] Loading CIC-IDS2017 dataset...")
    df = load_dataset("cic-ids-2017")
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # ---- 2. Preprocessing ----
    logger.info("\n[Step 2/5] Preprocessing pipeline...")
    prep = Preprocessor(
        drop_ip_port=cfg.features.drop_ip_port,
        normalization=cfg.features.normalization,
    )
    splits = prep.fit_and_split(
        df,
        test_ratio=cfg.data.test_split_ratio,
        val_ratio=cfg.data.val_split_ratio,
        seed=seed,
    )
    logger.info(f"  Train: {splits['train']['X'].shape}")
    logger.info(f"  Val:   {splits['val']['X'].shape}")
    logger.info(f"  Test:  {splits['test']['X'].shape}")
    logger.info(f"  Features: {len(prep.feature_names_)}")
    logger.info(f"  Classes:  {prep.class_names_}")

    # ---- 3. Label Audit ----
    logger.info("\n[Step 3/5] Running label audit...")
    auditor = LabelAuditor(cv_folds=3, seed=seed)
    audit_report = auditor.audit(df)
    logger.info(f"  Noise rate: {audit_report['noise_rate']:.4f} "
                f"({audit_report['noise_rate'] * 100:.2f}%)")
    logger.info(f"  Suspected mislabels: {audit_report['num_issues']}")

    # Save audit report
    auditor.generate_report(audit_report, save_path="results/label_audit_report.md")

    # ---- 4. XGBoost Training ----
    logger.info("\n[Step 4/5] Training XGBoost baseline...")
    xgb_model = XGBBaseline(
        n_estimators=cfg.training.n_estimators,
        max_depth=cfg.training.max_depth,
        learning_rate=cfg.training.learning_rate,
        random_state=seed,
    )
    xgb_model.fit(splits["train"]["X"], splits["train"]["y"])

    logger.info("  Evaluating XGBoost on test set...")
    xgb_report = full_evaluation_report(
        xgb_model,
        splits["test"]["X"],
        splits["test"]["y"],
        class_names=prep.class_names_,
        save_results=True,
    )
    logger.info(f"  XGBoost Accuracy: {xgb_report['summary']['accuracy']:.4f}")
    logger.info(f"  XGBoost Macro-F1: {xgb_report['summary']['macro_f1']:.4f}")
    logger.info(f"  XGBoost Throughput: {xgb_report['latency']['full_test']['throughput_flows_sec']:,.0f} flows/sec")

    # ---- 5. Random Forest Training ----
    logger.info("\n[Step 5/5] Training Random Forest baseline...")
    rf_model = RFBaseline(
        n_estimators=cfg.training.n_estimators,
        max_depth=cfg.training.max_depth,
        random_state=seed,
    )
    rf_model.fit(splits["train"]["X"], splits["train"]["y"])

    logger.info("  Evaluating Random Forest on test set...")
    rf_report = full_evaluation_report(
        rf_model,
        splits["test"]["X"],
        splits["test"]["y"],
        class_names=prep.class_names_,
        save_results=False,
    )
    logger.info(f"  RF Accuracy: {rf_report['summary']['accuracy']:.4f}")
    logger.info(f"  RF Macro-F1: {rf_report['summary']['macro_f1']:.4f}")
    logger.info(f"  RF Throughput: {rf_report['latency']['full_test']['throughput_flows_sec']:,.0f} flows/sec")

    # ---- GO Gate Check ----
    logger.info("\n" + "=" * 60)
    logger.info("GO Gate G1 Check:")
    target_acc = cfg.evaluation.target_accuracy
    target_f1 = cfg.evaluation.target_macro_f1
    xgb_acc = xgb_report["summary"]["accuracy"]
    xgb_f1 = xgb_report["summary"]["macro_f1"]

    acc_pass = xgb_acc >= target_acc
    f1_pass = xgb_f1 >= target_f1

    logger.info(f"  Accuracy: {xgb_acc:.4f} {'PASS' if acc_pass else 'FAIL'} "
                f"(target >= {target_acc})")
    logger.info(f"  Macro-F1: {xgb_f1:.4f} {'PASS' if f1_pass else 'FAIL'} "
                f"(target >= {target_f1})")

    if acc_pass and f1_pass:
        logger.info("  >>> G1 GATE: PASS — Proceed to Phase 2")
    else:
        logger.info("  >>> G1 GATE: FAIL — Debug pipeline before Phase 2")

    # Save results summary
    results = {
        "xgboost": xgb_report["summary"],
        "random_forest": rf_report["summary"],
        "latency_xgb": xgb_report["latency"],
        "latency_rf": rf_report["latency"],
        "label_audit": {
            "noise_rate": audit_report["noise_rate"],
            "num_issues": audit_report["num_issues"],
        },
        "g1_gate": {
            "accuracy_pass": acc_pass,
            "f1_pass": f1_pass,
            "overall_pass": acc_pass and f1_pass,
        },
    }
    results_path = Path("results/phase1_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\nPhase 1 complete!")
    return results


if __name__ == "__main__":
    main()
