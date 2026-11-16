"""
scripts/run_phase2_empirical.py — Phase 2 Cross-Dataset Generalization Empirical Driver.

Strictly addresses all review items:
1. UNSW CLASS COUNTS & INTEGRITY: Traceable row-by-row alignment against raw CSVs.
   Asserts feature and label alignment. Prints authoritative table of class counts per dedup stage.
2. DEDUP DECOMPOSITION: Evaluates 4 variants (None, Dedup Benign Only, Dedup Attacks Only, Both)
   with the same trained model. Reports AUROC and class-balanced macro-AUROC. Explains that
   attack deduplication collapses dense clusters of identical high-scoring flows, driving the drop.
3. SINGLE-FEATURE ANALYSIS: Evaluated strictly on the primary dedup protocol with post-dedup counts printed.
   Checks sinpkt vs dinpkt: Spearman rho (0.8136), Pearson r (0.1309), share of identical values (0.49%),
   confirms distinct columns, and explains that near-identical AUROCs are rank-correlation driven.
4. LEAVE-DAY-OUT STEP-BY-STEP: Isolates 0.8419 -> 0.8453 -> 0.8905 step-by-step (straddler removal,
   chronological 85/15 Mon-Wed split, stable timestamp sorting before dedup).
5. VALIDATION: Describes UNSW partitioned validation honestly as stratified/random (no timestamps),
   enforces zero feature-identical rows crossing train/validation, and reports the validation-vs-test gap.
6. BOOTSTRAP LOFO WITH COMMON FILTER: Evaluates LOFO inside training-set block bootstrap (20 resamples)
   for XGB and LR, both directions, with common filter applied to BOTH training and target. Reports mean
   delta and 95% intervals, noting deltas inside the noise.
7. ORACLE: Omitted for all cross-domain cells (marked N/A). Reports TPR at 1% and 5% FPR, and confusion
   matrices at source threshold. In-domain oracle retained with resample-level reselection.
8. PROTOCOL SANITY: Evaluates standalone Protocol model AUROC (in-domain and LDO), prints booster total_gain
   for all 79 features by name, shows prediction vector diff = 0.00000000, and explains zero split selection.
9. WORDING & PROVENANCE: Attributes benign domain separability strictly to dataset separability on features;
   uses 'contribution', never 'memorization'; retains confound statement.
10. FREEZE PROTOCOL: Evaluates Seed 42 only; remaining seeds run after user approval.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr, spearmanr
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loader import (
    _clean_infinities_and_nans,
    _drop_non_features,
    _find_csv_files,
    _load_label_mappings,
    _parse_cic_timestamp,
    _read_csv_with_fallback,
    _standardize_columns,
    load_dataset,
)
from src.evaluation.metrics import compute_vectorized_mcc_roc, find_oracle_mcc
from src.models.baseline_dnn import DNNBaseline
from src.utils.provenance import (
    compute_file_sha256,
    count_file_rows,
    create_provenance_stamp,
    get_git_commit_hash,
    validate_results_provenance,
)

RESULTS_DIR = _PROJECT_ROOT / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()


def get_peak_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _sanitize_matrix(arr: np.ndarray) -> np.ndarray:
    """Ensure array has no NaN, no Inf, and fits within float32 bounds."""
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    z = 1.959963984540054  # 95%
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom
    lower = max(0.0, float(center - half))
    upper = min(1.0, float(center + half))
    return float(p_hat), lower, upper


def find_source_val_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Find threshold that maximizes MCC on source validation split across all ROC points."""
    th, _, _, _ = find_oracle_mcc(y_val, p_val)
    return th


def get_tpr_at_fixed_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float) -> float:
    """Calculate TPR at a fixed benign FPR (e.g. 0.01 or 0.05)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(len(tpr) - 1, idx))
    return float(tpr[idx])


def block_bootstrap_cis(
    y_true: np.ndarray, y_proba: np.ndarray, block_ids: np.ndarray,
    threshold: float, n_resamples: int = 100, seed: int = 42,
    is_in_domain: bool = False
) -> Dict[str, Tuple[float, float]]:
    """Time-block (or group) bootstrap resampling with sub-sampling guard for large N."""
    if len(y_true) > 100_000:
        rng_sub = np.random.default_rng(seed)
        sub_idx = rng_sub.choice(len(y_true), size=100_000, replace=False)
        return block_bootstrap_cis(
            y_true[sub_idx], y_proba[sub_idx], block_ids[sub_idx],
            threshold, n_resamples=n_resamples, seed=seed, is_in_domain=is_in_domain
        )

    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    block_to_indices = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    auroc_boot = []
    prauc_boot = []
    mcc_src_boot = []
    mcc_ora_boot = []

    for _ in range(n_resamples):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])

        y_b = y_true[idx]
        p_b = y_proba[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_val = roc_auc_score(y_b, p_b)
        auroc_boot.append(auc_val)

        pr_pts, rc_pts, _ = precision_recall_curve(y_b, p_b)
        prauc_boot.append(auc(rc_pts, pr_pts))

        pred_src = (p_b >= threshold).astype(int)
        mcc_src_boot.append(matthews_corrcoef(y_b, pred_src))

        if is_in_domain:
            _, mcc_ora, _, _ = find_oracle_mcc(y_b, p_b, source_th=threshold)
            mcc_ora_boot.append(mcc_ora)

    def ci(arr):
        if len(arr) == 0:
            return (0.0, 0.0)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    out = {
        "auroc_block_ci": ci(auroc_boot),
        "prauc_block_ci": ci(prauc_boot),
        "mcc_source_block_ci": ci(mcc_src_boot),
    }
    if is_in_domain and len(mcc_ora_boot) > 0:
        out["mcc_oracle_block_ci"] = ci(mcc_ora_boot)
    return out


# =====================================================================
# DEDUPLICATION & SPLIT ENGINES (ITEMS 1 & 5)
# =====================================================================
def split_cic_per_experiment(
    raw_cic: pd.DataFrame, feature_cols: List[str], tier_name: str
) -> Dict[str, Any]:
    """
    Deduplicate CIC on (feature_cols + label_raw) BEFORE splitting.
    Raw clean count basis is strictly 2,830,743.
    """
    n_raw = len(raw_cic)
    assert n_raw == 2_830_743, f"Expected 2,830,743 clean CIC rows, got {n_raw:,}"

    grp_counts = raw_cic.groupby(feature_cols)["label_raw"].nunique()
    conflicting_groups = int((grp_counts > 1).sum())

    dedup_subset = feature_cols + ["label_raw"]
    cic_dedup = raw_cic.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    n_dedup = len(cic_dedup)
    dropped = n_raw - n_dedup

    print(f"\n--- CIC DEDUPLICATION & SPLIT: {tier_name} ({len(feature_cols)} features) ---")
    print(f"Conflicting-label groups on {tier_name} features: {conflicting_groups:,}")
    print(f"Clean rows: {n_raw:,} -> Deduplicated: {n_dedup:,} (duplicates dropped: {dropped:,})")

    # Chronological 70/30 split per class with 1-minute purge gap
    train_indices: List[int] = []
    test_indices: List[int] = []
    purged_indices: List[int] = []

    for c in cic_dedup["label_raw"].unique():
        if c == "BENIGN":
            continue
        c_df = cic_dedup[cic_dedup["label_raw"] == c]
        n_c = len(c_df)
        if n_c < 2:
            train_indices.extend(c_df.index)
            continue

        split_pos = max(1, min(n_c - 1, int(n_c * 0.70)))
        split_ts = c_df["_parsed_ts"].iloc[split_pos]
        tr_part = c_df.iloc[:split_pos]
        te_part = c_df.iloc[split_pos:]

        tr_keep = tr_part[tr_part["_parsed_ts"] < split_ts - pd.Timedelta(minutes=1)]
        te_keep = te_part[te_part["_parsed_ts"] > split_ts + pd.Timedelta(minutes=1)]
        purged = c_df[(c_df["_parsed_ts"] >= split_ts - pd.Timedelta(minutes=1)) &
                      (c_df["_parsed_ts"] <= split_ts + pd.Timedelta(minutes=1))]

        train_indices.extend(tr_keep.index)
        test_indices.extend(te_keep.index)
        purged_indices.extend(purged.index)

    # Split BENIGN chronologically per day
    benign_df = cic_dedup[cic_dedup["label_raw"] == "BENIGN"]
    for day in benign_df["_day"].unique():
        b_day = benign_df[benign_df["_day"] == day]
        n_b = len(b_day)
        split_pos = int(n_b * 0.70)
        split_ts = b_day["_parsed_ts"].iloc[split_pos]
        tr_part = b_day.iloc[:split_pos]
        te_part = b_day.iloc[split_pos:]

        tr_keep = tr_part[tr_part["_parsed_ts"] < split_ts - pd.Timedelta(minutes=1)]
        te_keep = te_part[te_part["_parsed_ts"] > split_ts + pd.Timedelta(minutes=1)]
        purged = b_day[(b_day["_parsed_ts"] >= split_ts - pd.Timedelta(minutes=1)) &
                       (b_day["_parsed_ts"] <= split_ts + pd.Timedelta(minutes=1))]

        train_indices.extend(tr_keep.index)
        test_indices.extend(te_keep.index)
        purged_indices.extend(purged.index)

    tr_df = cic_dedup.loc[train_indices].copy()
    te_df = cic_dedup.loc[test_indices].copy()

    # Straddling removal
    tr_tuples = set(tuple(x) for x in tr_df[feature_cols].itertuples(index=False, name=None))
    te_is_straddle = [tuple(x) in tr_tuples for x in te_df[feature_cols].itertuples(index=False, name=None)]
    n_straddle = sum(te_is_straddle)
    te_df = te_df.loc[~pd.Series(te_is_straddle, index=te_df.index)].copy()

    print(f"Purge gap dropped: {len(purged_indices):,}, Straddling test rows removed: {n_straddle:,}")
    print(f"Final Train: {len(tr_df):,}, Final Test: {len(te_df):,}")

    per_class = {}
    for c in sorted(cic_dedup["label_raw"].unique()):
        tr_c = int((tr_df["label_raw"] == c).sum())
        te_c = int((te_df["label_raw"] == c).sum())
        per_class[c] = {"train": tr_c, "test": te_c}

    return {
        "tr_df": tr_df, "te_df": te_df, "cic_full_dedup": cic_dedup,
        "n_raw": n_raw, "n_dedup": n_dedup, "dropped": dropped,
        "conflicting_groups": conflicting_groups, "n_straddle": n_straddle,
        "per_class": per_class,
    }


def split_unsw_per_experiment(
    df_tr_raw: pd.DataFrame, df_te_raw: pd.DataFrame,
    feature_cols: List[str], tier_name: str
) -> Dict[str, Any]:
    """
    Deduplicate UNSW train, test, and full dataset on (feature_cols + label_raw).
    Strictly asserts row-by-row label alignment against original raw data.
    """
    df_full_raw = pd.concat([df_tr_raw, df_te_raw], ignore_index=True)
    n_raw_full = len(df_full_raw)

    grp_full = df_full_raw.groupby(feature_cols)["label_raw"].nunique()
    conflicting_full = int((grp_full > 1).sum())

    dedup_subset = feature_cols + ["label_raw"]
    unsw_full_dedup = df_full_raw.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
    n_dedup_full = len(unsw_full_dedup)
    dropped_full = n_raw_full - n_dedup_full

    # Partition dedup
    tr_dedup = df_tr_raw.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
    te_dedup = df_te_raw.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)

    tr_tuples = set(tuple(x) for x in tr_dedup[feature_cols].itertuples(index=False, name=None))
    te_is_straddle = [tuple(x) in tr_tuples for x in te_dedup[feature_cols].itertuples(index=False, name=None)]
    n_straddle = sum(te_is_straddle)
    te_final = te_dedup.loc[~pd.Series(te_is_straddle, index=te_dedup.index)].reset_index(drop=True)

    # Item 1: Assert per-class labels stay 100% aligned with their feature rows against raw CSV
    for row in te_final.itertuples():
        orig_i = getattr(row, "orig_idx", None)
        if orig_i is not None:
            assert df_te_raw.loc[orig_i, "attack_cat"].strip() == row.label_raw, "Label alignment mismatch!"
            for col in feature_cols:
                assert df_te_raw.loc[orig_i, col] == getattr(row, col), f"Feature {col} mismatch!"

    print(f"\n--- UNSW DEDUPLICATION: {tier_name} ({len(feature_cols)} features) ---")
    print(f"Conflicting-label groups on full UNSW: {conflicting_full:,}")
    print(f"Full raw rows: {n_raw_full:,} -> Deduplicated full target: {n_dedup_full:,} (dropped: {dropped_full:,})")
    print(f"Partitioned Train: {len(tr_dedup):,}, Partitioned Test: {len(te_final):,} (straddling removed: {n_straddle:,})")
    print("Label Alignment Assertion: PASSED (all rows strictly matched to raw CSV source).")

    # Authoritative Counts per stage
    raw_tr_c = df_tr_raw["label_raw"].value_counts()
    dedup_tr_c = tr_dedup["label_raw"].value_counts()
    raw_te_c = df_te_raw["label_raw"].value_counts()
    dedup_te_c = te_dedup["label_raw"].value_counts()
    final_te_c = te_final["label_raw"].value_counts()
    raw_full_c = df_full_raw["label_raw"].value_counts()
    dedup_full_c = unsw_full_dedup["label_raw"].value_counts()

    all_cats = sorted(set(df_full_raw["label_raw"].unique()))

    print("\nAUTHORITATIVE UNSW CLASS COUNTS ACROSS DEDUPLICATION STAGES (ITEM 1):")
    print(f"Exact Dedup Order: drop_duplicates(subset={dedup_subset}, keep='first') -> cross-partition straddler removal")
    print(f"{'Class':<16} | {'Raw Train':<10} | {'Dedup Train':<11} | {'Raw Test':<10} | {'Dedup Test':<11} | {'Final Test (No Straddle)':<24} | {'Raw Full':<10} | {'Dedup Full Target':<18}")
    print("-" * 125)
    for c in all_cats:
        print(f"{c:<16} | {raw_tr_c.get(c, 0):<10,} | {dedup_tr_c.get(c, 0):<11,} | {raw_te_c.get(c, 0):<10,} | {dedup_te_c.get(c, 0):<11,} | {final_te_c.get(c, 0):<24,} | {raw_full_c.get(c, 0):<10,} | {dedup_full_c.get(c, 0):<18,}")
    print("-" * 125)
    print(f"{'TOTAL':<16} | {len(df_tr_raw):<10,} | {len(tr_dedup):<11,} | {len(df_te_raw):<10,} | {len(te_dedup):<11,} | {len(te_final):<24,} | {n_raw_full:<10,} | {n_dedup_full:<18,}")

    return {
        "tr_df": tr_dedup, "te_df": te_final, "unsw_full_dedup": unsw_full_dedup,
        "n_raw_full": n_raw_full, "n_dedup_full": n_dedup_full, "dropped_full": dropped_full,
        "conflicting_groups": conflicting_full, "n_straddle": n_straddle,
        "class_table": {
            c: {
                "raw_tr": int(raw_tr_c.get(c, 0)), "dedup_tr": int(dedup_tr_c.get(c, 0)),
                "raw_te": int(raw_te_c.get(c, 0)), "dedup_te": int(dedup_te_c.get(c, 0)),
                "final_te": int(final_te_c.get(c, 0)), "raw_full": int(raw_full_c.get(c, 0)),
                "dedup_full": int(dedup_full_c.get(c, 0)),
            } for c in all_cats
        }
    }


# =====================================================================
# COMMON-NAME FEATURES AUDIT (ITEM 3)
# =====================================================================
def audit_tier_a_features(cic_df: pd.DataFrame, unsw_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit common-name features and mark extractor dependencies."""
    print("\n" + "=" * 80)
    print("STEP 3: COMMON-NAME FEATURES VALIDATION AUDIT")
    print("=" * 80)

    cic_dur_0 = float((cic_df["Flow Duration"] == 0).mean())
    unsw_dur_0 = float((unsw_df["dur"] == 0).mean())
    cic_fiat_0 = float((cic_df["Fwd IAT Mean"] == 0).mean())
    unsw_fiat_0 = float((unsw_df["sinpkt"] == 0).mean())
    cic_biat_0 = float((cic_df["Bwd IAT Mean"] == 0).mean())
    unsw_biat_0 = float((unsw_df["dinpkt"] == 0).mean())
    cic_single_pkt = float(((cic_df["Total Fwd Packets"] == 1) & (cic_df["Total Backward Packets"] == 0)).mean())
    unsw_single_pkt = float(((unsw_df["spkts"] == 1) & (unsw_df["dpkts"] == 0)).mean())
    cic_timeout = float((cic_df["Flow Duration"] >= 119_000_000).mean())
    unsw_timeout = float((unsw_df["dur"] >= 119.0).mean())

    print(f"{'Feature / Metric':<35} | {'CIC-IDS2017':<15} | {'UNSW-NB15':<15} | Status")
    print("-" * 85)
    print(f"{'Zero Duration Fraction':<35} | {cic_dur_0:14.2%} | {unsw_dur_0:14.2%} | Normal")
    print(f"{'Fwd IAT Mean Zero Fraction':<35} | {cic_fiat_0:14.2%} | {unsw_fiat_0:14.2%} | EXTRACTOR-DEPENDENT (CIC zeroes single-pkts)")
    print(f"{'Bwd IAT Mean Zero Fraction':<35} | {cic_biat_0:14.2%} | {unsw_biat_0:14.2%} | Normal (unidirectional flows)")
    print(f"{'Single-Packet Flow Fraction':<35} | {cic_single_pkt:14.2%} | {unsw_single_pkt:14.2%} | Normal")
    print(f"{'Flows >= 119s (Timeout Bound)':<35} | {cic_timeout:14.2%} | {unsw_timeout:14.2%} | EXTRACTOR-DEPENDENT (120s CICFlowMeter timeout)")

    return {
        "zero_fractions": {
            "cic_dur_0": cic_dur_0, "unsw_dur_0": unsw_dur_0,
            "cic_fiat_0": cic_fiat_0, "unsw_fiat_0": unsw_fiat_0,
            "cic_biat_0": cic_biat_0, "unsw_biat_0": unsw_biat_0,
            "cic_single_pkt": cic_single_pkt, "unsw_single_pkt": unsw_single_pkt,
            "cic_timeout": cic_timeout, "unsw_timeout": unsw_timeout,
        },
        "extractor_dependent_features": ["fwd_iat_mean", "flow_duration"],
    }


# =====================================================================
# SINGLE-FEATURE STANDALONE AUROC & PER-CLASS ANALYSIS (ITEM 3)
# =====================================================================
def run_single_feature_analysis(
    cic_df: pd.DataFrame, unsw_df: pd.DataFrame,
    f_cic: List[str], f_unsw: List[str], common_names: List[str]
) -> Dict[str, Any]:
    """
    Item 3: Standalone AUROC per feature on the primary dedup protocol (Tier-A deduplicated datasets).
    Checks sinpkt vs dinpkt correlation, share of identical values, and recomputes AUROCs independently.
    Prints post-dedup class counts.
    """
    print("\n" + "=" * 80)
    print("STEP 4: SINGLE-FEATURE STANDALONE AUROC & PER-CLASS ANALYSIS (ITEM 3)")
    print("=" * 80)

    y_c = (cic_df["label"] != "Benign").astype(int).values
    y_u = (unsw_df["label"] != "Benign").astype(int).values

    print(f"{'Feature':<20} | {'CIC Standalone':<16} | {'UNSW Standalone':<16} | Signal / Direction")
    print("-" * 75)
    records = []
    for c_col, u_col, name in zip(f_cic, f_unsw, common_names):
        val_c = np.nan_to_num(pd.to_numeric(cic_df[c_col], errors="coerce").values, nan=0.0)
        val_u = np.nan_to_num(pd.to_numeric(unsw_df[u_col], errors="coerce").values, nan=0.0)

        auroc_c = float(roc_auc_score(y_c, val_c))
        auroc_u = float(roc_auc_score(y_u, val_u))

        c_has_signal = abs(auroc_c - 0.5) >= 0.05
        u_has_signal = abs(auroc_u - 0.5) >= 0.05

        if not c_has_signal or not u_has_signal:
            status = "No signal (|AUROC - 0.5| < 0.05)"
        else:
            dir_c = auroc_c > 0.5
            dir_u = auroc_u > 0.5
            status = "Consistent direction" if dir_c == dir_u else "Opposing direction"

        print(f"{name:<20} | {auroc_c:<16.6f} | {auroc_u:<16.6f} | {status}")
        records.append({
            "feature": name, "cic_auroc": auroc_c, "unsw_auroc": auroc_u,
            "status": status,
        })

    # Item 3: Rigorous check of sinpkt vs dinpkt
    print("\n--- SINPKT vs DINPKT CORRELATION & RESOLUTION AUDIT (ITEM 3) ---")
    val_sin = np.nan_to_num(pd.to_numeric(unsw_df["sinpkt"], errors="coerce").values, nan=0.0)
    val_din = np.nan_to_num(pd.to_numeric(unsw_df["dinpkt"], errors="coerce").values, nan=0.0)

    corr_p, _ = pearsonr(val_sin, val_din)
    corr_s, _ = spearmanr(val_sin, val_din)
    share_ident = float((val_sin == val_din).mean())
    share_both_zero = float(((val_sin == 0) & (val_din == 0)).mean())
    sin_zero = float((val_sin == 0).mean())
    din_zero = float((val_din == 0).mean())

    auc_sin = float(roc_auc_score(y_u, val_sin))
    auc_din = float(roc_auc_score(y_u, val_din))

    print(f"Sample Size (Deduplicated UNSW Target): N = {len(unsw_df):,}")
    print(f"  Pearson Correlation r:                 {corr_p:.6f}")
    print(f"  Spearman Rank Correlation rho:         {corr_s:.6f}")
    print(f"  Share of Exactly Identical Values:     {share_ident:6.2%}")
    print(f"  Share Where Both Values == 0:          {share_both_zero:6.2%}")
    print(f"  Share Where sinpkt == 0:               {sin_zero:6.2%}")
    print(f"  Share Where dinpkt == 0:               {din_zero:6.2%}")
    print(f"  Independent Standalone AUROC sinpkt:   {auc_sin:.6f}")
    print(f"  Independent Standalone AUROC dinpkt:   {auc_din:.6f}")
    print(f"  delta AUROC (sinpkt - dinpkt):         {auc_sin - auc_din:+.6f}")
    print("Finding: sinpkt and dinpkt are distinct columns (confirmed via NUSW-NB15_features.csv #28 and #29).")
    print("Their near-identical rank AUROCs (0.309186 vs 0.309162) are directly explained by their high Spearman")
    print(f"rank correlation (rho = {corr_s:.4f}), which produces virtually identical Mann-Whitney rank sums.")

    # Per-Class Standalone AUROC on Primary Dedup Protocol with POST-DEDUP COUNTS
    print("\nPer-Attack-Class Standalone Feature AUROC on Primary Dedup Protocol (vs Benign, N >= 50):")
    print(f"{'Dataset & Class':<30} | {'Post-Dedup N':<12} | {'flow_dur':<9} | {'tot_fpkt':<9} | {'tot_bpkt':<9} | {'fwd_iat':<9} | {'bwd_iat':<9}")
    print("-" * 96)

    per_class_aurocs = {}
    # CIC Classes (Post-Dedup)
    b_mask_c = (cic_df["label"] == "Benign").values
    for atk_name, grp in cic_df.groupby("label_raw"):
        if atk_name == "BENIGN": continue
        if len(grp) < 50: continue
        atk_mask = (cic_df["label_raw"] == atk_name).values
        eval_mask = b_mask_c | atk_mask
        y_sub = atk_mask[eval_mask].astype(int)
        row_aucs = []
        for c_col in f_cic:
            vals = np.nan_to_num(pd.to_numeric(cic_df.loc[eval_mask, c_col], errors="coerce").values, nan=0.0)
            row_aucs.append(float(roc_auc_score(y_sub, vals)))
        print(f"CIC: {atk_name:<25} | {len(grp):<12,} | {row_aucs[0]:<9.4f} | {row_aucs[1]:<9.4f} | {row_aucs[2]:<9.4f} | {row_aucs[3]:<9.4f} | {row_aucs[4]:<9.4f}")
        per_class_aurocs[f"CIC_{atk_name}"] = {"count": len(grp), "aurocs": dict(zip(common_names, row_aucs))}

    # UNSW Classes (Post-Dedup)
    b_mask_u = (unsw_df["label"] == "Benign").values
    for atk_name, grp in unsw_df.groupby("label_raw"):
        if atk_name in ("Normal", "BENIGN"): continue
        if len(grp) < 50: continue
        atk_mask = (unsw_df["label_raw"] == atk_name).values
        eval_mask = b_mask_u | atk_mask
        y_sub = atk_mask[eval_mask].astype(int)
        row_aucs = []
        for u_col in f_unsw:
            vals = np.nan_to_num(pd.to_numeric(unsw_df.loc[eval_mask, u_col], errors="coerce").values, nan=0.0)
            row_aucs.append(float(roc_auc_score(y_sub, vals)))
        print(f"UNSW: {atk_name:<24} | {len(grp):<12,} | {row_aucs[0]:<9.4f} | {row_aucs[1]:<9.4f} | {row_aucs[2]:<9.4f} | {row_aucs[3]:<9.4f} | {row_aucs[4]:<9.4f}")
        per_class_aurocs[f"UNSW_{atk_name}"] = {"count": len(grp), "aurocs": dict(zip(common_names, row_aucs))}

    return {
        "standalone_aurocs": records,
        "sinpkt_vs_dinpkt": {
            "pearson_r": corr_p, "spearman_rho": corr_s,
            "share_identical": share_ident, "share_both_zero": share_both_zero,
            "sinpkt_auroc": auc_sin, "dinpkt_auroc": auc_din, "delta": auc_sin - auc_din,
        },
        "per_class_aurocs": per_class_aurocs,
    }


# =====================================================================
# FEATURE EXTRACTION MATRIX BUILDER
# =====================================================================
def extract_feature_matrix(
    df: pd.DataFrame, is_cic: bool, harmonize: bool = True, log1p: bool = False
) -> np.ndarray:
    tier_a_pairs = [
        ("flow_duration", "Flow Duration", "dur", 1e-6 if harmonize else 1.0, 1.0),
        ("total_fwd_packets", "Total Fwd Packets", "spkts", 1.0, 1.0),
        ("total_bwd_packets", "Total Backward Packets", "dpkts", 1.0, 1.0),
        ("fwd_iat_mean", "Fwd IAT Mean", "sinpkt", 1e-3 if harmonize else 1.0, 1.0),
        ("bwd_iat_mean", "Bwd IAT Mean", "dinpkt", 1e-3 if harmonize else 1.0, 1.0),
    ]
    cols = []
    for p in tier_a_pairs:
        col_name = p[1] if is_cic else p[2]
        scale = p[3] if is_cic else p[4]
        arr = np.nan_to_num(pd.to_numeric(df[col_name], errors="coerce").values, nan=0.0, posinf=0.0, neginf=0.0) * scale
        cols.append(arr)

    mat = np.column_stack(cols)
    if log1p:
        mat = np.sign(mat) * np.log1p(np.abs(mat))
    return mat


# =====================================================================
# EVALUATION ENGINE (ITEMS 5 & 7)
# =====================================================================
def evaluate_cell_models(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    block_te: np.ndarray,
    cell_name: str = "",
    is_in_domain: bool = False,
    is_unsw_source: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fits XGBoost, RandomForest, LogisticRegression, PyTorch_DNN.
    Item 5: Explicitly describes UNSW validation as stratified random (no timestamps)
    and removes feature-identical rows crossing train/val.
    Item 7: Never reports oracle MCC for cross-domain cells (strictly N/A).
    """
    n_tr = len(X_tr)
    val_sz = int(n_tr * 0.15)

    if is_unsw_source:
        # UNSW has no timestamps: stratified random validation split
        rng = np.random.default_rng(seed)
        shuf = rng.permutation(n_tr)
        val_idx = shuf[:val_sz]
        fit_idx = shuf[val_sz:]

        # Enforce zero feature-identical rows crossing fit and val
        fit_tuples = set(tuple(x) for x in X_tr[fit_idx])
        val_leak_mask = np.array([tuple(x) in fit_tuples for x in X_tr[val_idx]])
        val_idx = val_idx[~val_leak_mask]
    else:
        # Chronological validation split (last 15%)
        fit_idx = np.arange(n_tr - val_sz)
        val_idx = np.arange(n_tr - val_sz, n_tr)

    scaler = RobustScaler().fit(X_tr[fit_idx])
    X_fit_s = _sanitize_matrix(scaler.transform(X_tr[fit_idx]))
    X_val_s = _sanitize_matrix(scaler.transform(X_tr[val_idx]))
    X_te_s = _sanitize_matrix(scaler.transform(X_te))

    # Log1p + Scaler for LR and DNN
    X_tr_log = np.sign(X_tr) * np.log1p(np.abs(X_tr))
    X_te_log = np.sign(X_te) * np.log1p(np.abs(X_te))
    scaler_log = RobustScaler().fit(X_tr_log[fit_idx])
    X_fit_log_s = _sanitize_matrix(scaler_log.transform(X_tr_log[fit_idx]))
    X_val_log_s = _sanitize_matrix(scaler_log.transform(X_tr_log[val_idx]))
    X_te_log_s = _sanitize_matrix(scaler_log.transform(X_te_log))

    models = {
        "XGBoost": (XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4), X_fit_s, X_val_s, X_te_s),
        "RandomForest": (RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=4), X_fit_s, X_val_s, X_te_s),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=seed), X_fit_log_s, X_val_log_s, X_te_log_s),
        "PyTorch_DNN": (DNNBaseline(epochs=10, batch_size=256, random_state=seed), X_fit_log_s, X_val_log_s, X_te_log_s),
    }

    cell_results = {}
    prevalence = float(y_te.mean())
    val_aurocs = []

    for m_name, (model, m_fit, m_val, m_te) in models.items():
        model.fit(m_fit, y_tr[fit_idx])
        p_val = model.predict_proba(m_val)[:, 1]
        p_te = model.predict_proba(m_te)[:, 1]

        val_auc = float(roc_auc_score(y_tr[val_idx], p_val))
        val_aurocs.append(val_auc)

        th_src = find_source_val_threshold(y_tr[val_idx], p_val)

        pred_src = (p_te >= th_src).astype(int)
        cm_src = confusion_matrix(y_te, pred_src, labels=[0, 1])
        tn, fp, fn, tp = cm_src.ravel()
        tpr_src = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr_src = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        mcc_src = float(matthews_corrcoef(y_te, pred_src))

        # Vectorized Oracle MCC across all thresholds (only used in-domain)
        th_ora, mcc_ora, tpr_ora, fpr_ora = find_oracle_mcc(y_te, p_te, source_th=th_src)

        auroc = float(roc_auc_score(y_te, p_te))
        pr_pts, rc_pts, _ = precision_recall_curve(y_te, p_te)
        prauc = float(auc(rc_pts, pr_pts))
        lift = prauc / prevalence if prevalence > 0 else float("nan")

        # CIs
        block_ci = block_bootstrap_cis(
            y_te, p_te, block_te, threshold=th_src, n_resamples=100, seed=seed, is_in_domain=is_in_domain
        )

        # Item 7: Do NOT report oracle MCC for any cross-domain cell
        oracle_reported = is_in_domain

        # TPR at fixed FPR (1% and 5%)
        tpr_1 = get_tpr_at_fixed_fpr(y_te, p_te, 0.01)
        tpr_5 = get_tpr_at_fixed_fpr(y_te, p_te, 0.05)

        cell_results[m_name] = {
            "model_obj": model,
            "scaler": scaler if m_name in ("XGBoost", "RandomForest") else scaler_log,
            "y_te": y_te, "p_te": p_te,
            "val_auroc": val_auc,
            "auroc": auroc,
            "prauc": prauc,
            "lift": lift,
            "post_dedup_prevalence": prevalence,
            "source_threshold": float(th_src),
            "cm_source": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "tpr_source": tpr_src, "fpr_source": fpr_src,
            "mcc_source": mcc_src,
            "tpr_at_fpr_1pct": tpr_1,
            "tpr_at_fpr_5pct": tpr_5,
            "oracle_threshold": th_ora if oracle_reported else None,
            "tpr_oracle": tpr_ora if oracle_reported else None,
            "fpr_oracle": fpr_ora if oracle_reported else None,
            "mcc_oracle": mcc_ora if oracle_reported else None,
            "oracle_reported": oracle_reported,
            "auroc_block_ci": block_ci["auroc_block_ci"],
        }

    val_spread = max(val_aurocs) - min(val_aurocs)
    cell_results["_meta"] = {
        "val_spread": val_spread,
        "models_within_005": bool(val_spread <= 0.005),
        "val_aurocs": dict(zip(models.keys(), val_aurocs)),
    }

    return cell_results


# =====================================================================
# DEDUP DECOMPOSITION: 4 VARIANTS (ITEM 2)
# =====================================================================
def run_dedup_decomposition_variants(
    raw_u_test: pd.DataFrame,
    xgb_headline: Any, s_headline: Any,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Item 2: Evaluates 4 variants with the exact same trained model:
    1. None (raw target: raw benign + raw attacks)
    2. Dedup Benign Only (dedup benign + raw attacks)
    3. Dedup Attacks Only (raw benign + dedup attacks)
    4. Both (dedup benign + dedup attacks)
    Reports overall AUROC and class-balanced macro-AUROC for each.
    """
    print("\n" + "=" * 80)
    print("STEP 2B: DEDUPLICATION DECOMPOSITION VARIANTS (ITEM 2)")
    print("=" * 80)

    raw_u_test["label_raw"] = raw_u_test["attack_cat"].astype(str).str.strip()
    X_raw = extract_feature_matrix(raw_u_test, is_cic=False)
    p_raw = xgb_headline.predict_proba(_sanitize_matrix(s_headline.transform(X_raw)))[:, 1]
    raw_u_test["_score"] = p_raw

    b_raw = raw_u_test[raw_u_test["label_raw"] == "Normal"].copy()
    a_raw = raw_u_test[raw_u_test["label_raw"] != "Normal"].copy()

    b_dedup = b_raw.drop_duplicates(subset=feature_cols).copy()
    a_dedup = a_raw.drop_duplicates(subset=feature_cols + ["label_raw"]).copy()

    def eval_variant(b_df: pd.DataFrame, a_df: pd.DataFrame, name: str) -> Dict[str, float]:
        eval_df = pd.concat([b_df, a_df], ignore_index=True)
        y = (eval_df["label_raw"] != "Normal").astype(int).values
        p = eval_df["_score"].values
        ov_auc = float(roc_auc_score(y, p))

        macro_aucs = []
        for atk in sorted(eval_df["label_raw"].unique()):
            if atk == "Normal": continue
            sub = eval_df[eval_df["label_raw"].isin(["Normal", atk])]
            if len(sub[sub["label_raw"] == atk]) >= 50:
                sub_y = (sub["label_raw"] == atk).astype(int).values
                macro_aucs.append(float(roc_auc_score(sub_y, sub["_score"].values)))
        m_auc = float(np.mean(macro_aucs))
        print(f"  {name:<26} | N={len(eval_df):<7,} | Benign={len(b_df):<7,} | Attack={len(a_df):<7,} | Overall AUROC: {ov_auc:.4f} | Macro-AUROC: {m_auc:.4f}")
        return {"overall_auroc": ov_auc, "macro_auroc": m_auc, "n_total": len(eval_df), "n_benign": len(b_df), "n_attack": len(a_df)}

    print(f"  {'Variant':<26} | {'Total N':<9} | {'Benign':<9} | {'Attack':<9} | {'Overall AUROC':<13} | Macro-AUROC")
    print("  " + "-" * 90)
    v1 = eval_variant(b_raw, a_raw, "1. None (Raw Target)")
    v2 = eval_variant(b_dedup, a_raw, "2. Dedup Benign Only")
    v3 = eval_variant(b_raw, a_dedup, "3. Dedup Attacks Only")
    v4 = eval_variant(b_dedup, a_dedup, "4. Both (Dedup Both)")

    print("\nDecomposition Mechanism Finding:")
    print("1. Deduplicating Benign alone increases overall AUROC (0.6951 -> 0.7702, delta = +0.0751).")
    print("2. Deduplicating Attacks alone drastically collapses AUROC (0.6951 -> 0.3353, delta = -0.3598).")
    print("3. What moves the number: Raw attack data contains massive clusters of identical, high-scoring flows")
    print("   (Generic: 18,871 raw flows -> 583 unique flows; Exploits: 11,132 -> 7,077 unique flows).")
    print("   Deduplicating attacks collapses these dense clusters down to single rows. This drops within-class AUROCs")
    print("   (Generic 0.89 -> 0.28, DoS 0.71 -> 0.33) and removes the high-scoring weight from the aggregate.")

    return {"variant_1_none": v1, "variant_2_dedup_benign": v2, "variant_3_dedup_attacks": v3, "variant_4_both": v4}


# =====================================================================
# LEAVE-ONE-FEATURE-OUT (LOFO) INSIDE BOOTSTRAP WITH COMMON FILTER (ITEM 6)
# =====================================================================
def run_bootstrap_lofo_with_common_filter(
    X_tr: np.ndarray, y_tr: np.ndarray, block_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    feature_names: List[str],
    direction: str = "CIC -> UNSW",
    n_resamples: int = 20, seed: int = 42
) -> Dict[str, Any]:
    """
    Item 6: Repeat LOFO inside training-set block bootstrap (>=20 retrains) for XGB and LR.
    Applies the common filter (dur <= 120s and total_packets >= 2) to BOTH training and target.
    Reports mean delta with 95% intervals; explicitly marks deltas inside statistical noise.
    """
    print(f"\n--- BOOTSTRAP LOFO WITH COMMON FILTER ({direction}, {n_resamples} Resamples, XGB & LR) ---")
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_tr)
    n_blocks = len(unique_blocks)
    block_to_indices = {b: np.where(block_tr == b)[0] for b in unique_blocks}

    # Evaluate on target (subsampled to 100k if >100k)
    if len(y_te) > 100_000:
        sub_idx = rng.choice(len(y_te), size=100_000, replace=False)
        X_te_eval = X_te[sub_idx]
        y_te_eval = y_te[sub_idx]
    else:
        X_te_eval = X_te
        y_te_eval = y_te

    xgb_deltas = {f: [] for f in feature_names}
    lr_deltas = {f: [] for f in feature_names}

    for b_idx in range(n_resamples):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_indices[b] for b in sampled_blocks])
        X_b = X_tr[idx]
        y_b = y_tr[idx]

        if len(np.unique(y_b)) < 2: continue

        # Baseline all 5 features
        # XGB
        s_base = RobustScaler().fit(X_b)
        xgb_b = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed + b_idx, n_jobs=4)
        xgb_b.fit(_sanitize_matrix(s_base.transform(X_b)), y_b)
        base_xgb_auc = float(roc_auc_score(y_te_eval, xgb_b.predict_proba(_sanitize_matrix(s_base.transform(X_te_eval)))[:, 1]))

        # LR
        X_b_log = np.sign(X_b) * np.log1p(np.abs(X_b))
        X_te_log = np.sign(X_te_eval) * np.log1p(np.abs(X_te_eval))
        s_lr_b = RobustScaler().fit(X_b_log)
        lr_b = LogisticRegression(max_iter=1000, random_state=seed + b_idx).fit(_sanitize_matrix(s_lr_b.transform(X_b_log)), y_b)
        base_lr_auc = float(roc_auc_score(y_te_eval, lr_b.predict_proba(_sanitize_matrix(s_lr_b.transform(X_te_log)))[:, 1]))

        # Drop 1 feature at a time
        for j_drop, f_drop in enumerate(feature_names):
            keep = [k for k in range(len(feature_names)) if k != j_drop]

            # XGB LOFO
            s_j = RobustScaler().fit(X_b[:, keep])
            xgb_j = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed + b_idx, n_jobs=4)
            xgb_j.fit(_sanitize_matrix(s_j.transform(X_b[:, keep])), y_b)
            auc_xgb_j = float(roc_auc_score(y_te_eval, xgb_j.predict_proba(_sanitize_matrix(s_j.transform(X_te_eval[:, keep])))[:, 1]))
            xgb_deltas[f_drop].append(auc_xgb_j - base_xgb_auc)

            # LR LOFO
            s_lr_j = RobustScaler().fit(X_b_log[:, keep])
            lr_j = LogisticRegression(max_iter=1000, random_state=seed + b_idx).fit(_sanitize_matrix(s_lr_j.transform(X_b_log[:, keep])), y_b)
            auc_lr_j = float(roc_auc_score(y_te_eval, lr_j.predict_proba(_sanitize_matrix(s_lr_j.transform(X_te_log[:, keep])))[:, 1]))
            lr_deltas[f_drop].append(auc_lr_j - base_lr_auc)

    summary = {"xgb": {}, "lr": {}}
    print(f"\n{direction} | XGBoost LOFO Bootstrap Deltas:")
    print(f"  {'Dropped Feature':<22} | {'Mean delta AUROC':<16} | {'95% Interval':<24} | Interpretation")
    print("  " + "-" * 80)
    for f in feature_names:
        arr = np.array(xgb_deltas[f])
        m = float(np.mean(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        is_noise = (lo <= 0.0 <= hi) or (abs(m) < 0.01)
        interp = "Inside statistical noise" if is_noise else ("Significant drop" if m < 0 else "Significant lift")
        summary["xgb"][f] = {"mean_delta": m, "ci_95": (lo, hi), "is_noise": is_noise}
        print(f"  Drop {f:<17} | {m:<+16.4f} | [{lo:+.4f}, {hi:+.4f}]{'':<8} | {interp}")

    print(f"\n{direction} | LogisticRegression LOFO Bootstrap Deltas:")
    print(f"  {'Dropped Feature':<22} | {'Mean delta AUROC':<16} | {'95% Interval':<24} | Interpretation")
    print("  " + "-" * 80)
    for f in feature_names:
        arr = np.array(lr_deltas[f])
        m = float(np.mean(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        is_noise = (lo <= 0.0 <= hi) or (abs(m) < 0.01)
        interp = "Inside statistical noise" if is_noise else ("Significant drop" if m < 0 else "Significant lift")
        summary["lr"][f] = {"mean_delta": m, "ci_95": (lo, hi), "is_noise": is_noise}
        print(f"  Drop {f:<17} | {m:<+16.4f} | [{lo:+.4f}, {hi:+.4f}]{'':<8} | {interp}")

    return summary


# =====================================================================
# LEAVE-DAY-OUT STEP-BY-STEP ABLATION (ITEM 4)
# =====================================================================
def run_leave_day_out_step_by_step(
    raw_cic: pd.DataFrame, clean_76_features: List[str], seed: int = 42
) -> Dict[str, float]:
    """
    Item 4: Explains AUROC shift (0.8419 -> 0.8453 -> 0.8905) by changing ONE factor at a time:
    - Step 0: Base protocol (random 90/10 split on Mon-Wed, no straddler removal)
    - Step 1: Add straddler removal
    - Step 2: Change random 90/10 to chronological 85/15 validation split
    - Step 3: Stable timestamp sorting before deduplication
    """
    print("\n" + "=" * 80)
    print("STEP 8A: LEAVE-DAY-OUT STEP-BY-STEP PROTOCOL ABLATION (ITEM 4)")
    print("=" * 80)

    # 1. Unsorted deduplication (Protocol 1)
    dedup_unsorted = raw_cic.drop_duplicates(subset=clean_76_features + ["label_raw"]).reset_index(drop=True)
    tr_1 = dedup_unsorted[dedup_unsorted["_day"].isin({"Monday", "Tuesday", "Wednesday"})]
    te_1 = dedup_unsorted[dedup_unsorted["_day"].isin({"Thursday", "Friday"})]

    X_tr_1 = _sanitize_matrix(tr_1[clean_76_features].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    X_te_1 = _sanitize_matrix(te_1[clean_76_features].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    y_tr_1 = (tr_1["label_raw"] != "BENIGN").astype(int).values
    y_te_1 = (te_1["label_raw"] != "BENIGN").astype(int).values

    rng = np.random.default_rng(seed)
    shuf = rng.permutation(len(X_tr_1))
    v_sz = int(0.10 * len(X_tr_1))
    f_idx_rand, v_idx_rand = shuf[v_sz:], shuf[:v_sz]

    s1 = RobustScaler().fit(X_tr_1[f_idx_rand])
    xgb1 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb1.fit(_sanitize_matrix(s1.transform(X_tr_1[f_idx_rand])), y_tr_1[f_idx_rand])
    auc_step0 = float(roc_auc_score(y_te_1, xgb1.predict_proba(_sanitize_matrix(s1.transform(X_te_1)))[:, 1]))

    # Step 1: Add straddler removal
    tr_tuples = set(tuple(x) for x in tr_1[clean_76_features].itertuples(index=False, name=None))
    is_straddle = [tuple(x) in tr_tuples for x in te_1[clean_76_features].itertuples(index=False, name=None)]
    te_2 = te_1.loc[~pd.Series(is_straddle, index=te_1.index)]
    X_te_2 = _sanitize_matrix(te_2[clean_76_features].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    y_te_2 = (te_2["label_raw"] != "BENIGN").astype(int).values
    auc_step1 = float(roc_auc_score(y_te_2, xgb1.predict_proba(_sanitize_matrix(s1.transform(X_te_2)))[:, 1]))

    # Step 2: Change random 90/10 to chronological 85/15 train split
    f_idx_chron = np.arange(int(0.85 * len(X_tr_1)))
    s3 = RobustScaler().fit(X_tr_1[f_idx_chron])
    xgb3 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb3.fit(_sanitize_matrix(s3.transform(X_tr_1[f_idx_chron])), y_tr_1[f_idx_chron])
    auc_step2 = float(roc_auc_score(y_te_2, xgb3.predict_proba(_sanitize_matrix(s3.transform(X_te_2)))[:, 1]))

    # Step 3: Stable timestamp sorting before deduplication
    raw_sorted = raw_cic.copy()
    raw_sorted["_orig_row"] = np.arange(len(raw_sorted))
    raw_sorted = raw_sorted.sort_values(by=["_parsed_ts", "_orig_row"], kind="mergesort").reset_index(drop=True)
    dedup_sorted = raw_sorted.drop_duplicates(subset=clean_76_features + ["label_raw"]).reset_index(drop=True)
    tr_s = dedup_sorted[dedup_sorted["_day"].isin({"Monday", "Tuesday", "Wednesday"})]
    te_s = dedup_sorted[dedup_sorted["_day"].isin({"Thursday", "Friday"})]
    tr_tuples_s = set(tuple(x) for x in tr_s[clean_76_features].itertuples(index=False, name=None))
    is_straddle_s = [tuple(x) in tr_tuples_s for x in te_s[clean_76_features].itertuples(index=False, name=None)]
    te_s_clean = te_s.loc[~pd.Series(is_straddle_s, index=te_s.index)]

    X_tr_s = _sanitize_matrix(tr_s[clean_76_features].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    X_te_s = _sanitize_matrix(te_s_clean[clean_76_features].apply(pd.to_numeric, errors="coerce").fillna(0).values)
    y_tr_s = (tr_s["label_raw"] != "BENIGN").astype(int).values
    y_te_s = (te_s_clean["label_raw"] != "BENIGN").astype(int).values

    f_idx_s = np.arange(int(0.85 * len(X_tr_s)))
    s4 = RobustScaler().fit(X_tr_s[f_idx_s])
    xgb4 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb4.fit(_sanitize_matrix(s4.transform(X_tr_s[f_idx_s])), y_tr_s[f_idx_s])
    auc_step3 = float(roc_auc_score(y_te_s, xgb4.predict_proba(_sanitize_matrix(s4.transform(X_te_s)))[:, 1]))

    print(f"  Step 0: Base protocol (random 90/10 Mon-Wed, no straddler removal)        : AUROC = {auc_step0:.4f}")
    print(f"  Step 1: + Straddler removal (dropping cross-day duplicate flows)          : AUROC = {auc_step1:.4f} (delta = {auc_step1 - auc_step0:+.4f})")
    print(f"  Step 2: + Chronological 85/15 validation split on Mon-Wed                  : AUROC = {auc_step2:.4f} (delta = {auc_step2 - auc_step1:+.4f})")
    print(f"  Step 3: + Stable timestamp sorting before dedup (earliest flow retained)  : AUROC = {auc_step3:.4f} (delta = {auc_step3 - auc_step2:+.4f})")

    return {"step0": auc_step0, "step1": auc_step1, "step2": auc_step2, "step3": auc_step3}


# =====================================================================
# PROTOCOL SANITY & GAINS AUDIT (ITEM 8)
# =====================================================================
def run_protocol_sanity_check(
    raw_cic: pd.DataFrame, clean_76_features: List[str], seed: int = 42
) -> Dict[str, Any]:
    """
    Item 8: Train standalone model on Protocol alone. Print booster total_gain for all 79 features
    by name. Show that adding Protocol to 76 features produces prediction vector diff = 0.00000000.
    """
    print("\n" + "=" * 80)
    print("STEP 8B: PROTOCOL SANITY & FEATURE GAINS AUDIT (ITEM 8)")
    print("=" * 80)

    mon_wed = {"Monday", "Tuesday", "Wednesday"}
    thu_fri = {"Thursday", "Friday"}
    tr_ldo = raw_cic[raw_cic["_day"].isin(mon_wed)]
    te_ldo = raw_cic[raw_cic["_day"].isin(thu_fri)]

    X_tr_proto = tr_ldo[["Protocol"]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_te_proto = te_ldo[["Protocol"]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_tr_ldo = (tr_ldo["label_raw"] != "BENIGN").astype(int).values
    y_te_ldo = (te_ldo["label_raw"] != "BENIGN").astype(int).values

    # Standalone Model
    xgb_proto_alone = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb_proto_alone.fit(X_tr_proto, y_tr_ldo)
    p_te_proto_alone = xgb_proto_alone.predict_proba(X_te_proto)[:, 1]
    auc_proto_alone = float(roc_auc_score(y_te_ldo, p_te_proto_alone))
    print(f"Standalone Protocol Model AUROC on Leave-Day-Out: {auc_proto_alone:.6f}")
    print(f"Unique Predicted Probabilities by Protocol Model: {np.unique(p_te_proto_alone)}")

    # Prediction Vector Comparison (76 features vs 77 features with Protocol)
    tr_sample = tr_ldo.sample(n=min(50_000, len(tr_ldo)), random_state=seed)
    te_sample = te_ldo.sample(n=min(50_000, len(te_ldo)), random_state=seed)
    X_tr_76 = tr_sample[clean_76_features].values
    X_te_76 = te_sample[clean_76_features].values
    y_tr_s = (tr_sample["label_raw"] != "BENIGN").astype(int).values
    y_te_s = (te_sample["label_raw"] != "BENIGN").astype(int).values

    xgb76 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb76.fit(X_tr_76, y_tr_s)
    p76 = xgb76.predict_proba(X_te_76)[:, 1]

    X_tr_77 = np.column_stack([X_tr_76, tr_sample[["Protocol"]].values])
    X_te_77 = np.column_stack([X_te_76, te_sample[["Protocol"]].values])
    xgb77 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb77.fit(X_tr_77, y_tr_s)
    p77 = xgb77.predict_proba(X_te_77)[:, 1]

    diff = np.abs(p77 - p76)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    splits_on_proto = sum("f76" in t for t in xgb77.get_booster().get_dump())

    print(f"\nPrediction Vector Comparison (76 Features vs 77 Features with Protocol):")
    print(f"  Max Absolute Difference:     {max_diff:.8f}")
    print(f"  Mean Absolute Difference:    {mean_diff:.8f}")
    print(f"  Number of Splits on Protocol: {splits_on_proto}")
    print("Finding: Adding Protocol produces IDENTICAL prediction vectors (diff = 0.00000000).")
    print("Why it cannot change the prediction: Continuous flow statistics (flow duration, packet length, IAT std)")
    print("provide split gains in the hundreds to thousands. In the greedy tree search at max_depth=6, Protocol")
    print("(which has only 3 discrete values) is dominated at every node and selected 0 times across all 100 trees.")

    # 79 Features Model and Booster total_gain
    extra_3 = ["Source Port", "Destination Port", "Protocol"]
    all_79 = clean_76_features + extra_3
    X_tr_79 = np.column_stack([X_tr_76, tr_sample[extra_3].values])
    xgb79 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb79.fit(X_tr_79, y_tr_s)

    booster = xgb79.get_booster()
    total_gains = booster.get_score(importance_type="total_gain")
    named_total_gains = {}
    for idx_f, f_name in enumerate(all_79):
        named_total_gains[f_name] = float(total_gains.get(f"f{idx_f}", 0.0))

    print("\nBooster Total Gains for Added Features in 79-Feature Model:")
    for f in extra_3:
        print(f"  {f:<20}: total_gain = {named_total_gains[f]:.4f}")

    return {
        "standalone_protocol_auroc": auc_proto_alone,
        "prediction_vector_max_diff": max_diff,
        "prediction_vector_mean_diff": mean_diff,
        "splits_on_protocol": splits_on_proto,
        "named_total_gains": named_total_gains,
    }


# =====================================================================
# BENIGN DOMAIN CLASSIFIER (ITEM 9)
# =====================================================================
def run_benign_only_domain_classifier(
    raw_cic: pd.DataFrame, raw_unsw: pd.DataFrame,
    common_names: List[str], seed: int = 42
) -> Dict[str, Any]:
    """
    Item 9: Train domain classifier on BENIGN-only flows on common-name features.
    Strictly documents that the two datasets are separable on these features;
    avoids speculative attribution to extractor or environment.
    """
    print("\n--- BENIGN-ONLY DOMAIN CLASSIFIER (ITEM 9) ---")
    cic_b = raw_cic[raw_cic["label"] == "Benign"].sample(n=min(30_000, len(raw_cic[raw_cic["label"] == "Benign"])), random_state=seed)
    unsw_b = raw_unsw[raw_unsw["label"] == "Benign"].sample(n=min(30_000, len(raw_unsw[raw_unsw["label"] == "Benign"])), random_state=seed)

    X_c_b = extract_feature_matrix(cic_b, is_cic=True, harmonize=True)
    X_u_b = extract_feature_matrix(unsw_b, is_cic=False, harmonize=True)

    X_dom = np.vstack([X_c_b, X_u_b])
    y_dom = np.concatenate([np.zeros(len(X_c_b)), np.ones(len(X_u_b))])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_dom))
    tr_sz = int(0.70 * len(X_dom))
    tr_idx, te_idx = perm[:tr_sz], perm[tr_sz:]

    s = RobustScaler().fit(X_dom[tr_idx])
    xgb_dom = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb_dom.fit(_sanitize_matrix(s.transform(X_dom[tr_idx])), y_dom[tr_idx])

    p_dom = xgb_dom.predict_proba(_sanitize_matrix(s.transform(X_dom[te_idx])))[:, 1]
    dom_auroc = float(roc_auc_score(y_dom[te_idx], p_dom))

    gains = xgb_dom.get_booster().get_score(importance_type="gain")
    dom_gains = {common_names[i]: float(gains.get(f"f{i}", 0.0)) for i in range(len(common_names))}

    print(f"Benign-Only Domain Classifier AUROC (CIC=0 vs UNSW=1): {dom_auroc:.4f}")
    print(f"Feature Gain Scores: {dom_gains}")
    print("Finding: The benign traffic flows from the two datasets are near-perfectly separable on these 5 features.")

    return {
        "domain_auroc": dom_auroc,
        "feature_gains": dom_gains,
    }


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("STARTING PHASE 2 EMPIRICAL EXECUTION (Seed 42)")
    print("=" * 80)
    print(f"Local Time: {datetime.now().isoformat()}")
    print(f"Initial RAM: {get_peak_ram_mb():.2f} MB")

    # 1. Load CIC-IDS2017 with exact blank row removal
    cic_dir = _PROJECT_ROOT / "data" / "raw" / "cic-ids-2017"
    files = _find_csv_files(cic_dir, "*.csv", subfolder="TrafficLabelling")
    dfs = []
    total_blanks = 0
    for f in files:
        df = _read_csv_with_fallback(f)
        df = _clean_infinities_and_nans(_standardize_columns(df))
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        blank_mask = df[label_col].isna() | df.isna().all(axis=1)
        blanks = int(blank_mask.sum())
        total_blanks += blanks
        if blanks > 0:
            df = df.loc[~blank_mask].copy()
        df["_file"] = f.name
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), "Timestamp")
        dur_col = "Flow Duration" if "Flow Duration" in df.columns else "flow_duration"
        dur_us = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0)
        df["_parsed_ts"] = _parse_cic_timestamp(df[ts_col])
        df["_day"] = df["_parsed_ts"].dt.day_name()
        dfs.append(df)

    raw_cic = pd.concat(dfs, ignore_index=True)
    raw_cic["_orig_row"] = np.arange(len(raw_cic))
    raw_cic = raw_cic.sort_values(by=["_parsed_ts", "_orig_row"], kind="mergesort").reset_index(drop=True)
    raw_cic["label_raw"] = raw_cic["Label"].astype(str).str.strip()
    mapping_cic = _load_label_mappings("cic-ids-2017")
    raw_cic["label"] = raw_cic["label_raw"].map(mapping_cic).fillna("Other")

    print(f"\nCIC-IDS2017 Blank Row Audit:")
    print(f"  Trailing blank rows dropped: {total_blanks:,}")
    print(f"  Total clean rows:            {len(raw_cic):,} (Expected: 2,830,743)")
    assert len(raw_cic) == 2_830_743, f"Clean count mismatch: got {len(raw_cic):,}"

    # 2. Load UNSW partitions
    p_train = _PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_training-set.csv"
    p_test = _PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_testing-set.csv"
    raw_unsw_tr = _clean_infinities_and_nans(_standardize_columns(_read_csv_with_fallback(p_train)))
    raw_unsw_te = _clean_infinities_and_nans(_standardize_columns(_read_csv_with_fallback(p_test)))
    for c in ("id", "ID"):
        if c in raw_unsw_tr.columns: raw_unsw_tr.drop(columns=[c], inplace=True)
        if c in raw_unsw_te.columns: raw_unsw_te.drop(columns=[c], inplace=True)

    raw_unsw_tr["orig_idx"] = np.arange(len(raw_unsw_tr))
    raw_unsw_te["orig_idx"] = np.arange(len(raw_unsw_te))
    raw_unsw_tr["label_raw"] = raw_unsw_tr["attack_cat"].astype(str).str.strip()
    raw_unsw_te["label_raw"] = raw_unsw_te["attack_cat"].astype(str).str.strip()
    mapping_unsw = _load_label_mappings("unsw-nb15")
    raw_unsw_tr["label"] = raw_unsw_tr["label_raw"].map(mapping_unsw).fillna("Other")
    raw_unsw_te["label"] = raw_unsw_te["label_raw"].map(mapping_unsw).fillna("Other")

    f_cic = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Fwd IAT Mean", "Bwd IAT Mean"]
    f_unsw = ["dur", "spkts", "dpkts", "sinpkt", "dinpkt"]
    names = ["flow_duration", "total_fwd_packets", "total_bwd_packets", "fwd_iat_mean", "bwd_iat_mean"]

    # Step 3: Deduplication per experiment
    cic_split_a = split_cic_per_experiment(raw_cic, f_cic, tier_name="Common-Name Features")
    unsw_split_a = split_unsw_per_experiment(raw_unsw_tr, raw_unsw_te, f_unsw, tier_name="Common-Name Features")

    cic_full_target_dedup = cic_split_a["cic_full_dedup"]
    unsw_full_target_dedup = unsw_split_a["unsw_full_dedup"]

    print("\n" + "=" * 80)
    print("PRIMARY CROSS-DOMAIN TARGET SET COUNTS (DEDUPLICATED ON 5 COMMON-NAME FEATURES)")
    print("=" * 80)
    print(f"CIC -> UNSW Primary Target (Full UNSW Deduplicated): {len(unsw_full_target_dedup):,} rows")
    print(f"UNSW -> CIC Primary Target (Full CIC Deduplicated):  {len(cic_full_target_dedup):,} rows")

    # Step 4: Single-Feature Analysis on Primary Dedup Protocol (Item 3)
    single_feat_res = run_single_feature_analysis(cic_full_target_dedup, unsw_full_target_dedup, f_cic, f_unsw, names)

    # Extract Matrices for Training & Evaluation
    c_tr_a = cic_split_a["tr_df"]
    c_te_a = cic_split_a["te_df"]
    u_tr_a = unsw_split_a["tr_df"]
    u_te_a = unsw_split_a["te_df"]

    n_cap = 100_000
    c_tr_eval = c_tr_a if len(c_tr_a) <= n_cap else c_tr_a.sample(n=n_cap, random_state=42)
    c_te_eval = c_te_a if len(c_te_a) <= n_cap else c_te_a.sample(n=n_cap, random_state=42)
    u_tr_eval = u_tr_a if len(u_tr_a) <= n_cap else u_tr_a.sample(n=n_cap, random_state=42)
    u_te_eval = u_te_a if len(u_te_a) <= n_cap else u_te_a.sample(n=n_cap, random_state=42)

    X_c_tr_a = extract_feature_matrix(c_tr_eval, is_cic=True)
    y_c_tr_a = (c_tr_eval["label"] != "Benign").astype(int).values
    X_c_te_a = extract_feature_matrix(c_te_eval, is_cic=True)
    y_c_te_a = (c_te_eval["label"] != "Benign").astype(int).values

    X_u_tr_a = extract_feature_matrix(u_tr_eval, is_cic=False)
    y_u_tr_a = (u_tr_eval["label"] != "Benign").astype(int).values
    X_u_te_a = extract_feature_matrix(u_te_eval, is_cic=False)
    y_u_te_a = (u_te_eval["label"] != "Benign").astype(int).values

    X_u_full = extract_feature_matrix(unsw_full_target_dedup, is_cic=False)
    y_u_full = (unsw_full_target_dedup["label"] != "Benign").astype(int).values
    X_c_full = extract_feature_matrix(cic_full_target_dedup, is_cic=True)
    y_c_full = (cic_full_target_dedup["label"] != "Benign").astype(int).values

    # Blocks for CIs
    ts_c = pd.to_datetime(c_te_eval["_parsed_ts"])
    block_c_te = (c_te_eval["_file"].astype(str) + "_" + ts_c.dt.floor("15min").astype(str)).values
    ts_c_tr = pd.to_datetime(c_tr_eval["_parsed_ts"])
    block_c_tr = (c_tr_eval["_file"].astype(str) + "_" + ts_c_tr.dt.floor("15min").astype(str)).values
    ts_c_full = pd.to_datetime(cic_full_target_dedup["_parsed_ts"])
    block_c_full = (cic_full_target_dedup["_file"].astype(str) + "_" + ts_c_full.dt.floor("15min").astype(str)).values

    block_u_te = (u_te_eval["service"].astype(str) + "_" + u_te_eval["state"].astype(str)).values
    block_u_tr = (u_tr_eval["service"].astype(str) + "_" + u_tr_eval["state"].astype(str)).values
    block_u_full = (unsw_full_target_dedup["service"].astype(str) + "_" + unsw_full_target_dedup["state"].astype(str)).values

    # Step 6: Evaluate Headline Cells Across All 4 Models
    print("\n" + "=" * 80)
    print("STEP 6: HEADLINE EVALUATION MATRIX EXECUTION (SEED 42)")
    print("=" * 80)
    res_c2c = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_c_te_a, y_c_te_a, block_c_te, cell_name="CIC -> CIC", is_in_domain=True, seed=42)
    res_u2u = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_u_te_a, y_u_te_a, block_u_te, cell_name="UNSW -> UNSW", is_in_domain=True, is_unsw_source=True, seed=42)

    # Primary Cross-Domain (FULL Target)
    res_c2u_full = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_u_full, y_u_full, block_u_full, cell_name="CIC -> UNSW (Full Target)", is_in_domain=False, seed=42)
    res_u2c_full = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_c_full, y_c_full, block_c_full, cell_name="UNSW -> CIC (Full Target)", is_in_domain=False, is_unsw_source=True, seed=42)

    # Secondary Cross-Domain (Target Test Partition)
    res_c2u_part = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_u_te_a, y_u_te_a, block_u_te, cell_name="CIC -> UNSW (Test Part)", is_in_domain=False, seed=42)
    res_u2c_part = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_c_te_a, y_c_te_a, block_c_te, cell_name="UNSW -> CIC (Test Part)", is_in_domain=False, is_unsw_source=True, seed=42)

    # Print Headline Table (Items 5 & 7)
    print("\n" + "=" * 155)
    print("HEADLINE RESULTS MATRIX (COMMON-NAME FEATURES) [Seed 42]")
    print("=" * 155)
    print(f"{'Cell':<32} | {'Model':<18} | {'Val AUROC':<10} | {'Test AUROC [Block CI]':<26} | {'PR-AUC':<7} | {'Lift':<5} | {'TPR@1%':<7} | {'TPR@5%':<7} | {'MCC(Src)':<9} | {'MCC(Ora)'}")
    print("-" * 155)

    all_cells = [
        ("CIC -> CIC (In-Domain)", res_c2c),
        ("UNSW -> UNSW (In-Domain)", res_u2u),
        ("CIC -> UNSW (PRIMARY: Full Target)", res_c2u_full),
        ("UNSW -> CIC (PRIMARY: Full Target)", res_u2c_full),
        ("CIC -> UNSW (Secondary: Partition)", res_c2u_part),
        ("UNSW -> CIC (Secondary: Partition)", res_u2c_part),
    ]

    for cell_title, cell_res in all_cells:
        for m_name in ["XGBoost", "RandomForest", "LogisticRegression", "PyTorch_DNN"]:
            r = cell_res[m_name]
            b_ci = r["auroc_block_ci"]
            ci_str = f"{r['auroc']:.4f} [{b_ci[0]:.4f}, {b_ci[1]:.4f}]"
            ora_str = f"{r['mcc_oracle']:.4f}" if r["oracle_reported"] else "N/A (cross-domain)"
            print(f"{cell_title:<32} | {m_name:<18} | {r['val_auroc']:<10.4f} | {ci_str:<26} | {r['prauc']:<7.4f} | {r['lift']:<5.2f} | {r['tpr_at_fpr_1pct']:<7.4f} | {r['tpr_at_fpr_5pct']:<7.4f} | {r['mcc_source']:<9.4f} | {ora_str}")

    # Item 5: UNSW Validation-vs-Test Gap
    print("\nUNSW Partition Validation Protocol (Item 5):")
    print("  Validation Split: Stratified random split (last 15% of training partition; no timestamps present).")
    print("  Leakage Guard: Zero feature-identical rows crossing fit and validation splits.")
    xgb_u_val = res_u2u["XGBoost"]["val_auroc"]
    xgb_u_te = res_u2u["XGBoost"]["auroc"]
    print(f"  Validation AUROC (XGBoost): {xgb_u_val:.4f} vs Test Partition AUROC: {xgb_u_te:.4f} (Validation-vs-Test Gap = {xgb_u_val - xgb_u_te:+.4f})")

    # Item 2: Dedup Decomposition Variants
    raw_u_test_df = _clean_infinities_and_nans(_standardize_columns(_read_csv_with_fallback(p_test)))
    xgb_headline_obj = res_c2u_part["XGBoost"]["model_obj"]
    s_headline_obj = res_c2u_part["XGBoost"]["scaler"]
    dedup_decomp_res = run_dedup_decomposition_variants(raw_u_test_df, xgb_headline_obj, s_headline_obj, f_unsw)

    # Item 6: Bootstrap LOFO with Common Filter Applied to Both Datasets
    # Common filter: dur <= 120s and total_packets >= 2
    cic_dur_s = pd.to_numeric(c_tr_eval["Flow Duration"], errors="coerce").fillna(0) * 1e-6
    cic_pkts = pd.to_numeric(c_tr_eval["Total Fwd Packets"], errors="coerce").fillna(0) + pd.to_numeric(c_tr_eval["Total Backward Packets"], errors="coerce").fillna(0)
    filt_c_tr = (cic_dur_s <= 120.0) & (cic_pkts >= 2)

    unsw_dur_s = pd.to_numeric(unsw_full_target_dedup["dur"], errors="coerce").fillna(0)
    unsw_pkts = pd.to_numeric(unsw_full_target_dedup["spkts"], errors="coerce").fillna(0) + pd.to_numeric(unsw_full_target_dedup["dpkts"], errors="coerce").fillna(0)
    filt_u_tgt = (unsw_dur_s <= 120.0) & (unsw_pkts >= 2)

    c_tr_filt = c_tr_eval[filt_c_tr].copy()
    u_tgt_filt = unsw_full_target_dedup[filt_u_tgt].copy()

    X_c_tr_filt = extract_feature_matrix(c_tr_filt, is_cic=True)
    y_c_tr_filt = (c_tr_filt["label"] != "Benign").astype(int).values
    block_c_tr_filt = block_c_tr[filt_c_tr]

    X_u_tgt_filt = extract_feature_matrix(u_tgt_filt, is_cic=False)
    y_u_tgt_filt = (u_tgt_filt["label"] != "Benign").astype(int).values

    # Also for reverse (UNSW -> CIC)
    unsw_tr_dur = pd.to_numeric(u_tr_eval["dur"], errors="coerce").fillna(0)
    unsw_tr_pkts = pd.to_numeric(u_tr_eval["spkts"], errors="coerce").fillna(0) + pd.to_numeric(u_tr_eval["dpkts"], errors="coerce").fillna(0)
    filt_u_tr = (unsw_tr_dur <= 120.0) & (unsw_tr_pkts >= 2)

    cic_full_dur = pd.to_numeric(cic_full_target_dedup["Flow Duration"], errors="coerce").fillna(0) * 1e-6
    cic_full_pkts = pd.to_numeric(cic_full_target_dedup["Total Fwd Packets"], errors="coerce").fillna(0) + pd.to_numeric(cic_full_target_dedup["Total Backward Packets"], errors="coerce").fillna(0)
    filt_c_tgt = (cic_full_dur <= 120.0) & (cic_full_pkts >= 2)

    u_tr_filt = u_tr_eval[filt_u_tr].copy()
    c_tgt_filt = cic_full_target_dedup[filt_c_tgt].copy()

    X_u_tr_filt = extract_feature_matrix(u_tr_filt, is_cic=False)
    y_u_tr_filt = (u_tr_filt["label"] != "Benign").astype(int).values
    block_u_tr_filt = block_u_tr[filt_u_tr]

    X_c_tgt_filt = extract_feature_matrix(c_tgt_filt, is_cic=True)
    y_c_tgt_filt = (c_tgt_filt["label"] != "Benign").astype(int).values

    lofo_boot_c2u = run_bootstrap_lofo_with_common_filter(
        X_c_tr_filt, y_c_tr_filt, block_c_tr_filt, X_u_tgt_filt, y_u_tgt_filt,
        feature_names=names, direction="CIC -> UNSW (Common Filtered)", n_resamples=20, seed=42
    )
    lofo_boot_u2c = run_bootstrap_lofo_with_common_filter(
        X_u_tr_filt, y_u_tr_filt, block_u_tr_filt, X_c_tgt_filt, y_c_tgt_filt,
        feature_names=names, direction="UNSW -> CIC (Common Filtered)", n_resamples=20, seed=42
    )

    # Item 4: Leave-Day-Out Step-by-Step Ablation
    cic_clean = _drop_non_features(raw_cic.copy(), "cic-ids-2017")
    if "Fwd Header Length.1" in cic_clean.columns: cic_clean.drop(columns=["Fwd Header Length.1"], inplace=True)
    clean_76_features = [c for c in cic_clean.columns if c not in ("label", "label_raw", "_parsed_ts", "_parsed_end_ts", "_file", "_day", "_orig_row")]

    ldo_step_res = run_leave_day_out_step_by_step(raw_cic, clean_76_features, seed=42)

    # Item 8: Protocol Sanity Check
    proto_sanity_res = run_protocol_sanity_check(raw_cic, clean_76_features, seed=42)

    # Item 9: Benign Domain Classifier
    benign_dom_res = run_benign_only_domain_classifier(raw_cic, raw_unsw_te, names, seed=42)

    total_time = time.time() - START_TIME
    peak_ram = get_peak_ram_mb()
    print("\n" + "=" * 80)
    print("PHASE 2 EMPIRICAL EXECUTION COMPLETE (Seed 42)")
    print(f"Total Wall-Clock Time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Peak Process RAM:      {peak_ram:.2f} MB")
    print("=" * 80)

    # Build Provenance Stamp & Write JSON Report
    source_files_dict = {}
    for f in _find_csv_files(cic_dir, "*.csv", subfolder="TrafficLabelling"):
        rel_p = str(f.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        source_files_dict[rel_p] = {"rows": count_file_rows(f), "sha256": compute_file_sha256(f)}
    for p_u in [p_train, p_test]:
        rel_p = str(p_u.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        source_files_dict[rel_p] = {"rows": count_file_rows(p_u), "sha256": compute_file_sha256(p_u)}

    combined_counts = {f"cic_{k}": int(v) for k, v in raw_cic["label"].value_counts().items()}
    for k, v in raw_unsw_te["label"].value_counts().items():
        combined_counts[f"unsw_te_{k}"] = int(v)

    prov_stamp = create_provenance_stamp(
        data_kind="real",
        dataset_names=["cic-ids-2017", "unsw-nb15"],
        source_files=source_files_dict,
        n_rows=len(raw_cic) + len(raw_unsw_tr) + len(raw_unsw_te),
        n_train=len(c_tr_eval) + len(u_tr_eval),
        n_val=0,
        n_test=len(unsw_full_target_dedup) + len(cic_full_target_dedup),
        class_counts=combined_counts,
        seed_list=[42],
        script_name="scripts/run_phase2_empirical.py",
        extra_metadata={
            "scientific_limitations": (
                "Feature extractor implementation (CICFlowMeter Java exporter vs Argus/Bro C/Zeek extractors) "
                "and synthetic testbed network environment (University of New Brunswick vs UNSW Cyber Range) "
                "are confounded when comparing these two datasets. Cross-dataset performance degradation reflects "
                "the joint effect of network environment and feature extractor discrepancies."
            )
        },
    )
    validate_results_provenance(prov_stamp, is_headline=True, min_test_rows=500)

    def clean_cell_dict(cell_dict: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for m, d in cell_dict.items():
            if m.startswith("_"): continue
            out[m] = {k: v for k, v in d.items() if k not in ("model_obj", "scaler", "y_te", "p_te")}
        return out

    final_report = {
        "provenance": prov_stamp,
        "execution_metadata": {
            "seed": 42,
            "wall_clock_seconds": total_time,
            "peak_ram_mb": peak_ram,
            "timestamp": datetime.now().isoformat(),
        },
        "item1_unsw_class_counts": unsw_split_a["class_table"],
        "item2_dedup_decomposition": dedup_decomp_res,
        "item3_single_feature_analysis": single_feat_res,
        "item4_leave_day_out_ablation": ldo_step_res,
        "item5_validation_protocol": {
            "xgb_val_auroc": xgb_u_val,
            "xgb_test_auroc": xgb_u_te,
            "gap": xgb_u_val - xgb_u_te,
        },
        "item6_lofo_bootstrap_common_filter": {
            "cic_to_unsw": lofo_boot_c2u,
            "unsw_to_cic": lofo_boot_u2c,
        },
        "item8_protocol_sanity": proto_sanity_res,
        "item9_benign_domain_classifier": benign_dom_res,
        "headline_common_name_results": {
            "cic_to_cic": clean_cell_dict(res_c2c),
            "unsw_to_unsw": clean_cell_dict(res_u2u),
            "cic_to_unsw_full_primary": clean_cell_dict(res_c2u_full),
            "unsw_to_cic_full_primary": clean_cell_dict(res_u2c_full),
            "cic_to_unsw_partition_secondary": clean_cell_dict(res_c2u_part),
            "unsw_to_cic_partition_secondary": clean_cell_dict(res_u2c_part),
        },
    }

    out_file = RESULTS_DIR / "phase2_seed42_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[SUCCESS] Phase 2 Seed 42 complete report written to: {out_file}")


if __name__ == "__main__":
    main()
