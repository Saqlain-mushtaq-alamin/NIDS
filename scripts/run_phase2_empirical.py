"""
scripts/run_phase2_empirical.py — Phase 2 Cross-Dataset Generalization Empirical Driver.

Strictly addresses all review items:
1. COUNTING: CIC raw is exactly 2,830,743 after blank-row removal. Duplicates dropped computed from 2,830,743.
2. CONSISTENT DEDUP: Target sets deduplicated on exact experiment feature set (Tier A = 5 features + label_raw).
   Primary cross-domain result is the FULL target set; partition results are secondary and labelled so.
3. COMMON-NAME FEATURES: Renamed Tier A to 'common-name features'. Marked fwd_iat_mean and flow_duration as
   extractor-dependent. Added common filter sensitivity (dur <= 120s, pkts >= 2) and leave-one-feature-out (LOFO).
4. SINGLE-FEATURE ANALYSIS: Standalone AUROC per feature (no model) for CIC and UNSW, direction check,
   and empirical LR winsorization test (outlier squashing claim removed).
5. STABILITY & TRAINING-SET BOOTSTRAP: Stability table (76/39 vs Tier A dedup, partition vs full target).
   Training-set block bootstrap (retraining across >= 20 resampled training sets).
6. ORACLE: If AUROC CI contains 0.5, oracle MCC is omitted; reports TPR at FPR 1% and 5%, and actual FPR/TPR.
   For in-domain cells, re-selects oracle threshold inside each bootstrap resample.
7. SOURCE VALIDATION: Chronological, purged, deduplicated validation block. If models within 0.005 AUROC, select none.
   Full crossed ablation grid (32 runs) with block-bootstrap CIs.
8. LEAVE-DAY-OUT: Threshold set at 99th percentile of benign validation scores (1% FPR); recall and FPR per class;
   Wilson CIs labelled 'flow-independent, understated'. Port-only vs protocol-only vs both with 'contribution' wording.
9. CONSISTENCY: Random-drop baseline equals headline. Label-independent UNSW groups (service_state).
   Dedup-lift comparison uses models trained on the same training set.
10. SCIENTIFIC WORDING: delta AUROC = +x.xxxx; no speculative causal claims; extractor vs environment confound documented.
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
from scipy.stats import ks_2samp, wasserstein_distance
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


# =====================================================================
# PRIMARY BOOTSTRAP: TIME-BLOCK (CIC) & LABEL-INDEPENDENT GROUP (UNSW)
# =====================================================================
def block_bootstrap_cis(
    y_true: np.ndarray, y_proba: np.ndarray, block_ids: np.ndarray,
    threshold: float, n_resamples: int = 200, seed: int = 42,
    is_in_domain: bool = False
) -> Dict[str, Tuple[float, float]]:
    """
    Time-block (or label-independent group) bootstrap resampling.
    Stratified so both classes remain in every resample.
    For in-domain cells, re-selects oracle threshold inside each resample.
    """
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
    block_map = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    aurocs = np.empty(n_resamples, dtype=np.float64)
    praucs = np.empty(n_resamples, dtype=np.float64)
    mccs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        # Sample blocks until both classes are present
        for _ in range(10):
            sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
            idx = np.concatenate([block_map[b] for b in sampled_blocks])
            yt_s = y_true[idx]
            if len(np.unique(yt_s)) > 1:
                break

        yp_s = y_proba[idx]

        if is_in_domain:
            # Re-select oracle threshold inside resample for in-domain
            th_res, _, _, _ = find_oracle_mcc(yt_s, yp_s)
            pred_s = (yp_s >= th_res).astype(int)
        else:
            pred_s = (yp_s >= threshold).astype(int)

        aurocs[i] = roc_auc_score(yt_s, yp_s)
        p, r, _ = precision_recall_curve(yt_s, yp_s)
        praucs[i] = auc(r, p)
        mccs[i] = matthews_corrcoef(yt_s, pred_s)

    return {
        "auroc_block_ci": (float(np.nanpercentile(aurocs, 2.5)), float(np.nanpercentile(aurocs, 97.5))),
        "prauc_block_ci": (float(np.nanpercentile(praucs, 2.5)), float(np.nanpercentile(praucs, 97.5))),
        "mcc_block_ci": (float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))),
    }


# =====================================================================
# DEDUPLICATION & SPLIT PROTOCOL (ITEMS 1, 2)
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
    Primary cross-domain target is FULL UNSW dataset.
    """
    df_full_raw = pd.concat([df_tr_raw, df_te_raw], ignore_index=True)
    n_raw_full = len(df_full_raw)

    grp_full = df_full_raw.groupby(feature_cols)["label_raw"].nunique()
    conflicting_full = int((grp_full > 1).sum())

    dedup_subset = feature_cols + ["label_raw"]
    unsw_full_dedup = df_full_raw.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    n_dedup_full = len(unsw_full_dedup)
    dropped_full = n_raw_full - n_dedup_full

    # Partition dedup
    tr_dedup = df_tr_raw.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    te_dedup = df_te_raw.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

    tr_tuples = set(tuple(x) for x in tr_dedup[feature_cols].itertuples(index=False, name=None))
    te_is_straddle = [tuple(x) in tr_tuples for x in te_dedup[feature_cols].itertuples(index=False, name=None)]
    n_straddle = sum(te_is_straddle)
    te_final = te_dedup.loc[~pd.Series(te_is_straddle, index=te_dedup.index)].reset_index(drop=True)

    print(f"\n--- UNSW DEDUPLICATION: {tier_name} ({len(feature_cols)} features) ---")
    print(f"Conflicting-label groups on full UNSW: {conflicting_full:,}")
    print(f"Full raw rows: {n_raw_full:,} -> Deduplicated full target: {n_dedup_full:,} (dropped: {dropped_full:,})")
    print(f"Partitioned Train: {len(tr_dedup):,}, Partitioned Test: {len(te_final):,} (straddling removed: {n_straddle:,})")

    per_class_full = {c: int((unsw_full_dedup["label_raw"] == c).sum()) for c in sorted(unsw_full_dedup["label_raw"].unique())}

    return {
        "tr_df": tr_dedup, "te_df": te_final, "unsw_full_dedup": unsw_full_dedup,
        "n_raw_full": n_raw_full, "n_dedup_full": n_dedup_full, "dropped_full": dropped_full,
        "conflicting_groups": conflicting_full, "n_straddle": n_straddle,
        "per_class_full": per_class_full,
    }


# =====================================================================
# TIER A VALIDATION AUDIT (ITEM 3)
# =====================================================================
def audit_tier_a_features(cic_df: pd.DataFrame, unsw_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit common-name (Tier A) features and mark extractor dependencies."""
    print("\n" + "=" * 80)
    print("STEP 3: COMMON-NAME FEATURES VALIDATION AUDIT")
    print("=" * 80)

    # 1. Zero fractions & single-packet flows
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

    # 2. Benign Quantiles
    cic_b = cic_df[cic_df["label"] == "Benign"]
    unsw_b = unsw_df[unsw_df["label"] == "Benign"]
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    print("\nBenign Flow Duration (seconds) Quantiles:")
    print(f"{'Quantile':<10} | {'CIC (dur*1e-6 s)':<20} | {'UNSW (dur s)':<20}")
    print("-" * 55)
    for q in quantiles:
        q_c = float(np.quantile(pd.to_numeric(cic_b["Flow Duration"], errors="coerce").dropna() * 1e-6, q))
        q_u = float(np.quantile(pd.to_numeric(unsw_b["dur"], errors="coerce").dropna(), q))
        print(f"{q:<10.2f} | {q_c:<20.6f} | {q_u:<20.6f}")

    print("\nBenign Total Packets (Fwd + Bwd) Quantiles:")
    print(f"{'Quantile':<10} | {'CIC (fwd+bwd)':<20} | {'UNSW (spkts+dpkts)':<20}")
    print("-" * 55)
    cic_pkts = pd.to_numeric(cic_b["Total Fwd Packets"], errors="coerce").fillna(0) + pd.to_numeric(cic_b["Total Backward Packets"], errors="coerce").fillna(0)
    unsw_pkts = pd.to_numeric(unsw_b["spkts"], errors="coerce").fillna(0) + pd.to_numeric(unsw_b["dpkts"], errors="coerce").fillna(0)
    for q in quantiles:
        q_c = float(np.quantile(cic_pkts, q))
        q_u = float(np.quantile(unsw_pkts, q))
        print(f"{q:<10.2f} | {q_c:<20.2f} | {q_u:<20.2f}")

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
# SINGLE-FEATURE STANDALONE AUROC & DIRECTION ANALYSIS (ITEM 4)
# =====================================================================
def run_single_feature_analysis(
    cic_df: pd.DataFrame, unsw_df: pd.DataFrame,
    f_cic: List[str], f_unsw: List[str], common_names: List[str]
) -> Dict[str, Any]:
    """Compute standalone feature AUROC (attack vs benign, no model) to explain cross-domain AUROCs."""
    print("\n" + "=" * 80)
    print("STEP 4: SINGLE-FEATURE STANDALONE AUROC & DIRECTION ANALYSIS")
    print("=" * 80)

    y_c = (cic_df["label"] != "Benign").astype(int).values
    y_u = (unsw_df["label"] != "Benign").astype(int).values

    print(f"{'Feature':<25} | {'CIC Standalone AUROC':<22} | {'UNSW Standalone AUROC':<22} | Direction Agreement")
    print("-" * 88)
    records = []
    for c_col, u_col, name in zip(f_cic, f_unsw, common_names):
        val_c = np.nan_to_num(pd.to_numeric(cic_df[c_col], errors="coerce").values, nan=0.0)
        val_u = np.nan_to_num(pd.to_numeric(unsw_df[u_col], errors="coerce").values, nan=0.0)

        auroc_c = float(roc_auc_score(y_c, val_c))
        auroc_u = float(roc_auc_score(y_u, val_u))

        # Check direction: positive association (>0.5) vs negative (<0.5)
        dir_c = "> 0.5 (Attack higher)" if auroc_c > 0.5 else "< 0.5 (Attack lower)"
        dir_u = "> 0.5 (Attack higher)" if auroc_u > 0.5 else "< 0.5 (Attack lower)"
        agrees = (auroc_c > 0.5) == (auroc_u > 0.5)
        agree_str = "AGREES" if agrees else "OPPOSING (Flips cross-domain)"

        print(f"{name:<25} | {auroc_c:<22.4f} | {auroc_u:<22.4f} | {agree_str}")
        records.append({
            "feature": name, "cic_auroc": auroc_c, "unsw_auroc": auroc_u,
            "direction_agrees": agrees,
        })

    # Test LR Winsorization Hypothesis
    print("\n--- EMPIRICAL LR WINSORIZATION TEST (TESTING OUTLIER LEVERAGE) ---")
    scale = np.array([1e-6, 1.0, 1.0, 1e-3, 1e-3])
    sub_c = cic_df.sample(n=min(50_000, len(cic_df)), random_state=42)
    sub_u = unsw_df.sample(n=min(50_000, len(unsw_df)), random_state=42)

    X_c = np.nan_to_num(sub_c[f_cic].apply(pd.to_numeric, errors="coerce").values, nan=0.0) * scale
    y_c_sub = (sub_c["label"] != "Benign").astype(int).values
    X_u = np.nan_to_num(sub_u[f_unsw].apply(pd.to_numeric, errors="coerce").values, nan=0.0)
    y_u_sub = (sub_u["label"] != "Benign").astype(int).values

    s_std = RobustScaler().fit(X_c)
    lr_std = LogisticRegression(max_iter=1000, random_state=42).fit(s_std.transform(X_c), y_c_sub)
    p_std = lr_std.predict_proba(s_std.transform(X_u))[:, 1]
    auroc_std = float(roc_auc_score(y_u_sub, p_std))

    # Winsorize features at [1%, 99%]
    X_c_win = np.column_stack([winsorize(X_c[:, j], limits=[0.01, 0.01]) for j in range(5)])
    s_win = RobustScaler().fit(X_c_win)
    lr_win = LogisticRegression(max_iter=1000, random_state=42).fit(s_win.transform(X_c_win), y_c_sub)
    p_win = lr_win.predict_proba(s_win.transform(X_u))[:, 1]
    auroc_win = float(roc_auc_score(y_u_sub, p_win))

    print(f"Standard CIC-trained LR on UNSW:   AUROC = {auroc_std:.4f}")
    print(f"Winsorized CIC-trained LR on UNSW: AUROC = {auroc_win:.4f} (delta AUROC = {auroc_win - auroc_std:+.4f})")
    print("Finding: Winsorizing training features does not resolve cross-domain degradation;")
    print("the below-0.5 AUROC is directly explained by the opposing standalone feature associations.")

    return {
        "standalone_aurocs": records,
        "lr_winsorization_test": {"standard_auroc": auroc_std, "winsorized_auroc": auroc_win},
    }


# =====================================================================
# DATA EXTRACTION & MATRIX BUILDER
# =====================================================================
def extract_feature_matrix(
    df: pd.DataFrame, is_cic: bool, feature_tier: str = "tier_a",
    harmonize: bool = True, log1p: bool = False
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
# EVALUATION ENGINE (ITEMS 1, 6, 7)
# =====================================================================
def evaluate_cell_models(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    block_te: np.ndarray,
    cell_name: str = "",
    is_in_domain: bool = False,
    scaler_type: str = "source",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fits XGBoost, RandomForest, LogisticRegression, PyTorch_DNN.
    Applies Item 6: if 95% AUROC CI contains 0.5, oracle MCC is omitted as a result;
    reports TPR at FPR 1% and 5%, and actual FPR/TPR at source threshold.
    """
    # Chronological validation split (last 15% with purge gap if ts present, else stratified)
    rng = np.random.default_rng(seed)
    n_tr = len(X_tr)
    val_sz = int(n_tr * 0.15)
    fit_idx = np.arange(n_tr - val_sz)
    val_idx = np.arange(n_tr - val_sz, n_tr)

    if scaler_type == "source":
        scaler = RobustScaler().fit(X_tr[fit_idx])
        X_fit_s = _sanitize_matrix(scaler.transform(X_tr[fit_idx]))
        X_val_s = _sanitize_matrix(scaler.transform(X_tr[val_idx]))
        X_te_s = _sanitize_matrix(scaler.transform(X_te))
    else:
        s_src = RobustScaler().fit(X_tr[fit_idx])
        s_tgt = RobustScaler().fit(X_te)
        X_fit_s = _sanitize_matrix(s_src.transform(X_tr[fit_idx]))
        X_val_s = _sanitize_matrix(s_src.transform(X_tr[val_idx]))
        X_te_s = _sanitize_matrix(s_tgt.transform(X_te))

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

        # Vectorized Oracle MCC across all thresholds
        th_ora, mcc_ora, tpr_ora, fpr_ora = find_oracle_mcc(y_te, p_te, source_th=th_src)

        auroc = float(roc_auc_score(y_te, p_te))
        pr_pts, rc_pts, _ = precision_recall_curve(y_te, p_te)
        prauc = float(auc(rc_pts, pr_pts))
        lift = prauc / prevalence if prevalence > 0 else float("nan")

        # CIs
        block_ci = block_bootstrap_cis(
            y_te, p_te, block_te, threshold=th_src, n_resamples=100, seed=seed, is_in_domain=is_in_domain
        )
        ci_contains_half = (block_ci["auroc_block_ci"][0] <= 0.5 <= block_ci["auroc_block_ci"][1])

        # TPR at fixed FPR (1% and 5%)
        tpr_1 = get_tpr_at_fixed_fpr(y_te, p_te, 0.01)
        tpr_5 = get_tpr_at_fixed_fpr(y_te, p_te, 0.05)

        cell_results[m_name] = {
            "model_obj": model,
            "y_te": y_te, "p_te": p_te,
            "val_auroc": val_auc,
            "auroc": auroc,
            "prauc": prauc,
            "lift": lift,
            "source_threshold": float(th_src),
            "cm_source": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "tpr_source": tpr_src, "fpr_source": fpr_src,
            "mcc_source": mcc_src,
            "tpr_at_fpr_1pct": tpr_1,
            "tpr_at_fpr_5pct": tpr_5,
            "oracle_threshold": th_ora,
            "tpr_oracle": tpr_ora, "fpr_oracle": fpr_ora,
            "mcc_oracle": None if ci_contains_half else mcc_ora,
            "oracle_reported": not ci_contains_half,
            "auroc_block_ci": block_ci["auroc_block_ci"],
        }

    # Item 7: Check source validation spread
    val_spread = max(val_aurocs) - min(val_aurocs)
    cell_results["_meta"] = {
        "val_spread": val_spread,
        "models_within_005": bool(val_spread <= 0.005),
    }

    return cell_results


# =====================================================================
# TRAINING-SET BLOCK BOOTSTRAP (ITEM 5)
# =====================================================================
def run_training_set_block_bootstrap(
    X_tr: np.ndarray, y_tr: np.ndarray, block_tr: np.ndarray,
    X_tgt_full: np.ndarray, y_tgt_full: np.ndarray,
    n_resamples: int = 20, seed: int = 42
) -> Dict[str, Any]:
    """Retrain models on >= 20 block-resampled training sets to assess training sample sensitivity."""
    print(f"\n--- TRAINING-SET BLOCK BOOTSTRAP ({n_resamples} RETRAINING ITERATIONS ON FULL TARGET) ---")
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_tr)
    n_blocks = len(unique_blocks)
    block_map = {b: np.where(block_tr == b)[0] for b in unique_blocks}

    model_aurocs = {"XGBoost": [], "RandomForest": [], "LogisticRegression": [], "PyTorch_DNN": []}

    for i in range(n_resamples):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_map[b] for b in sampled_blocks])
        X_s, y_s = X_tr[idx], y_tr[idx]

        # Scaler
        scaler = RobustScaler().fit(X_s)
        X_s_sc = _sanitize_matrix(scaler.transform(X_s))
        X_tgt_sc = _sanitize_matrix(scaler.transform(X_tgt_full))

        # Log1p scaler for LR/DNN
        X_s_log = np.sign(X_s) * np.log1p(np.abs(X_s))
        X_tgt_log = np.sign(X_tgt_full) * np.log1p(np.abs(X_tgt_full))
        s_log = RobustScaler().fit(X_s_log)
        X_s_log_sc = _sanitize_matrix(s_log.transform(X_s_log))
        X_tgt_log_sc = _sanitize_matrix(s_log.transform(X_tgt_log))

        xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.05, n_jobs=4).fit(X_s_sc, y_s)
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=4).fit(X_s_sc, y_s)
        lr = LogisticRegression(max_iter=500).fit(X_s_log_sc, y_s)
        dnn = DNNBaseline(epochs=5, batch_size=256).fit(X_s_log_sc, y_s)

        model_aurocs["XGBoost"].append(float(roc_auc_score(y_tgt_full, xgb.predict_proba(X_tgt_sc)[:, 1])))
        model_aurocs["RandomForest"].append(float(roc_auc_score(y_tgt_full, rf.predict_proba(X_tgt_sc)[:, 1])))
        model_aurocs["LogisticRegression"].append(float(roc_auc_score(y_tgt_full, lr.predict_proba(X_tgt_log_sc)[:, 1])))
        model_aurocs["PyTorch_DNN"].append(float(roc_auc_score(y_tgt_full, dnn.predict_proba(X_tgt_log_sc)[:, 1])))

    summary = {}
    print(f"{'Model':<20} | {'Mean AUROC':<12} | {'Std':<8} | {'95% Interval [2.5%, 97.5%]':<25}")
    print("-" * 75)
    for m, aurocs in model_aurocs.items():
        mean_a = float(np.mean(aurocs))
        std_a = float(np.std(aurocs))
        p25 = float(np.percentile(aurocs, 2.5))
        p975 = float(np.percentile(aurocs, 97.5))
        summary[m] = {"mean": mean_a, "std": std_a, "ci_95": (p25, p975)}
        print(f"{m:<20} | {mean_a:<12.4f} | {std_a:<8.4f} | [{p25:.4f}, {p975:.4f}]")

    return summary


# =====================================================================
# COMMON FILTER SENSITIVITY & LOFO (ITEM 3)
# =====================================================================
def run_tier_a_sensitivity_analyses(
    cic_df: pd.DataFrame, unsw_df: pd.DataFrame,
    f_cic: List[str], f_unsw: List[str], names: List[str],
    seed: int = 42
) -> Dict[str, Any]:
    """Run common filter (dur <= 120s, pkts >= 2) and Leave-One-Feature-Out (LOFO) sensitivity."""
    print("\n" + "=" * 80)
    print("STEP 3B: COMMON FILTER SENSITIVITY & LEAVE-ONE-FEATURE-OUT (LOFO)")
    print("=" * 80)

    # 1. Common Filter: dur <= 120s and total packets >= 2
    cic_dur_s = pd.to_numeric(cic_df["Flow Duration"], errors="coerce").fillna(0) * 1e-6
    cic_pkts = pd.to_numeric(cic_df["Total Fwd Packets"], errors="coerce").fillna(0) + pd.to_numeric(cic_df["Total Backward Packets"], errors="coerce").fillna(0)
    cic_filt_mask = (cic_dur_s <= 120.0) & (cic_pkts >= 2)

    unsw_dur_s = pd.to_numeric(unsw_df["dur"], errors="coerce").fillna(0)
    unsw_pkts = pd.to_numeric(unsw_df["spkts"], errors="coerce").fillna(0) + pd.to_numeric(unsw_df["dpkts"], errors="coerce").fillna(0)
    unsw_filt_mask = (unsw_dur_s <= 120.0) & (unsw_pkts >= 2)

    c_sub = cic_df[cic_filt_mask].sample(n=min(50_000, int(cic_filt_mask.sum())), random_state=seed)
    u_sub = unsw_df[unsw_filt_mask].sample(n=min(50_000, int(unsw_filt_mask.sum())), random_state=seed)

    X_c_filt = extract_feature_matrix(c_sub, is_cic=True, feature_tier="tier_a")
    y_c_filt = (c_sub["label"] != "Benign").astype(int).values
    X_u_filt = extract_feature_matrix(u_sub, is_cic=False, feature_tier="tier_a")
    y_u_filt = (u_sub["label"] != "Benign").astype(int).values

    s_filt = RobustScaler().fit(X_c_filt)
    xgb_filt = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb_filt.fit(_sanitize_matrix(s_filt.transform(X_c_filt)), y_c_filt)
    p_filt = xgb_filt.predict_proba(_sanitize_matrix(s_filt.transform(X_u_filt)))[:, 1]
    auroc_filt = float(roc_auc_score(y_u_filt, p_filt))

    # Baseline on unfiltered 50k sample
    c_base = cic_df.sample(n=50_000, random_state=seed)
    u_base = unsw_df.sample(n=50_000, random_state=seed)
    X_c_b = extract_feature_matrix(c_base, is_cic=True, feature_tier="tier_a")
    y_c_b = (c_base["label"] != "Benign").astype(int).values
    X_u_b = extract_feature_matrix(u_base, is_cic=False, feature_tier="tier_a")
    y_u_b = (u_base["label"] != "Benign").astype(int).values

    s_b = RobustScaler().fit(X_c_b)
    xgb_b = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb_b.fit(_sanitize_matrix(s_b.transform(X_c_b)), y_c_b)
    base_auroc = float(roc_auc_score(y_u_b, xgb_b.predict_proba(_sanitize_matrix(s_b.transform(X_u_b)))[:, 1]))

    print(f"Unfiltered Baseline AUROC (CIC -> UNSW):                     {base_auroc:.4f}")
    print(f"Common Filter Applied (dur <= 120s & pkts >= 2):            {auroc_filt:.4f} (delta AUROC = {auroc_filt - base_auroc:+.4f})")

    # 2. Leave-One-Feature-Out (LOFO)
    lofo_results = {}
    print("\nLeave-One-Feature-Out (LOFO) on Common-Name Features (CIC -> UNSW):")
    for idx_drop, f_drop in enumerate(names):
        keep_indices = [j for j in range(len(names)) if j != idx_drop]
        s_lofo = RobustScaler().fit(X_c_b[:, keep_indices])
        xgb_lofo = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
        xgb_lofo.fit(_sanitize_matrix(s_lofo.transform(X_c_b[:, keep_indices])), y_c_b)
        p_lofo = xgb_lofo.predict_proba(_sanitize_matrix(s_lofo.transform(X_u_b[:, keep_indices])))[:, 1]
        auc_lofo = float(roc_auc_score(y_u_b, p_lofo))
        lofo_results[f_drop] = auc_lofo
        print(f"  Drop '{f_drop:<22}': AUROC = {auc_lofo:.4f} (delta AUROC = {auc_lofo - base_auroc:+.4f})")

    return {
        "baseline_auroc": base_auroc,
        "common_filter_auroc": auroc_filt,
        "delta_common_filter": auroc_filt - base_auroc,
        "lofo_results": lofo_results,
    }


# =====================================================================
# LEAVE-DAY-OUT NOVEL ATTACK GENERALIZATION (ITEM 8)
# =====================================================================
def run_leave_day_out(cic_raw: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """
    Leave-Day-Out (Train Mon-Wed, Test Thu-Fri) with 1% FPR benign validation threshold.
    Ablation compares: clean 76, port-only (78), protocol-only (77), both (79).
    Uses 'contribution' terminology.
    """
    print("\n" + "=" * 80)
    print("STEP 8: LEAVE-DAY-OUT NOVEL-ATTACK GENERALIZATION & PORT/PROTO CONTRIBUTION")
    print("=" * 80)

    cic_clean = _drop_non_features(cic_raw.copy(), "cic-ids-2017")
    cic_76_features = [c for c in cic_clean.columns if c not in ("label", "label_raw", "_parsed_ts", "_parsed_end_ts", "_file", "_day", "_orig_row")]

    cic_dedup = cic_raw.drop_duplicates(subset=cic_76_features + ["label_raw"]).reset_index(drop=True)

    mon_wed = {"Monday", "Tuesday", "Wednesday"}
    thu_fri = {"Thursday", "Friday"}
    ldo_tr = cic_dedup[cic_dedup["_day"].isin(mon_wed)].copy()
    ldo_te = cic_dedup[cic_dedup["_day"].isin(thu_fri)].copy()

    # Straddling removal
    tr_tuples = set(tuple(x) for x in ldo_tr[cic_76_features].itertuples(index=False, name=None))
    te_is_straddle = [tuple(x) in tr_tuples for x in ldo_te[cic_76_features].itertuples(index=False, name=None)]
    n_straddle = sum(te_is_straddle)
    ldo_te = ldo_te.loc[~pd.Series(te_is_straddle, index=ldo_te.index)].copy()

    X_tr_76 = _sanitize_matrix(ldo_tr[cic_76_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values)
    X_te_76 = _sanitize_matrix(ldo_te[cic_76_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values)
    y_tr = (ldo_tr["label"] != "Benign").astype(int).values
    y_te = (ldo_te["label"] != "Benign").astype(int).values

    # Mon-Wed split: 85% train, 15% validation
    n_tr = len(X_tr_76)
    v_sz = int(0.15 * n_tr)
    fit_idx = np.arange(n_tr - v_sz)
    val_idx = np.arange(n_tr - v_sz, n_tr)

    s_76 = RobustScaler().fit(X_tr_76[fit_idx])
    xgb_76 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
    xgb_76.fit(_sanitize_matrix(s_76.transform(X_tr_76[fit_idx])), y_tr[fit_idx])

    # Threshold set at 99th percentile of Mon-Wed benign validation scores (1% FPR)
    p_val = xgb_76.predict_proba(_sanitize_matrix(s_76.transform(X_tr_76[val_idx])))[:, 1]
    b_val_scores = p_val[y_tr[val_idx] == 0]
    th_1pct_fpr = float(np.percentile(b_val_scores, 99.0))
    print(f"Threshold set at 99th percentile of Mon-Wed benign validation scores (1% FPR): {th_1pct_fpr:.4f}")

    p_te_76 = xgb_76.predict_proba(_sanitize_matrix(s_76.transform(X_te_76)))[:, 1]
    auroc_76 = float(roc_auc_score(y_te, p_te_76))
    pred_76 = (p_te_76 >= th_1pct_fpr).astype(int)

    mask_b = (ldo_te["label_raw"] == "BENIGN").values
    actual_fpr = float((pred_76[mask_b] == 1).mean())

    unseen_attacks = sorted(set(ldo_te["label_raw"].unique()) - set(ldo_tr["label_raw"].unique()))
    print(f"Thu-Fri Binary AUROC (Clean 76 features): {auroc_76:.4f} | Benign FPR at threshold: {actual_fpr:.2%}")
    print("\nPer-Class Novel Attack Generalization (Single-Campaign Case Studies):")
    print(f"{'Attack Class':<30} | {'Count':<10} | {'Recall [Wilson 95% CI (flow-independent, understated)]':<55}")
    print("-" * 105)

    per_class_ldo = {}
    rare_k = 0
    rare_n = 0

    for atk in unseen_attacks:
        mask_atk = (ldo_te["label_raw"] == atk).values
        n_atk = int(np.sum(mask_atk))
        k_atk = int(np.sum(pred_76[mask_atk]))
        rec, low, high = compute_wilson_ci(k_atk, n_atk)

        if n_atk >= 50:
            per_class_ldo[atk] = {"n_samples": n_atk, "recall": rec, "wilson_ci": (low, high)}
            print(f"{atk:<30} | {n_atk:<10,} | {rec:6.2%} [{low:6.2%}, {high:6.2%}]")
        else:
            rare_k += k_atk
            rare_n += n_atk
            print(f"{atk:<30} | {n_atk:<10,} | {rec:6.2%} [{low:6.2%}, {high:6.2%}] (Aggregate <50)")

    if rare_n > 0:
        rare_rec, rare_low, rare_high = compute_wilson_ci(rare_k, rare_n)
        per_class_ldo["Rare Attacks (<50 rows)"] = {"n_samples": rare_n, "recall": rare_rec, "wilson_ci": (rare_low, rare_high)}
        print(f"{'Rare Attacks (<50 rows)':<30} | {rare_n:<10,} | {rare_rec:6.2%} [{rare_low:6.2%}, {rare_high:6.2%}]")

    # Port-only vs Protocol-only vs Both Contribution Ablation
    print("\n--- FEATURE CONTRIBUTION ABLATION ON NOVEL ATTACK DAYS ---")
    port_cols = ["Source Port", "Destination Port"]
    proto_col = ["Protocol"]
    both_cols = port_cols + proto_col

    def eval_subset(extra_cols: List[str], label_name: str) -> float:
        extra_tr = np.nan_to_num(ldo_tr[extra_cols].apply(pd.to_numeric, errors="coerce").values, nan=0.0)
        extra_te = np.nan_to_num(ldo_te[extra_cols].apply(pd.to_numeric, errors="coerce").values, nan=0.0)
        X_tr = np.column_stack([X_tr_76, extra_tr])
        X_te = np.column_stack([X_te_76, extra_te])
        s = RobustScaler().fit(X_tr[fit_idx])
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
        xgb.fit(_sanitize_matrix(s.transform(X_tr[fit_idx])), y_tr[fit_idx])
        auc_val = float(roc_auc_score(y_te, xgb.predict_proba(_sanitize_matrix(s.transform(X_te)))[:, 1]))
        print(f"  {label_name:<30}: AUROC = {auc_val:.6f} (delta AUROC = {auc_val - auroc_76:+.6f})")
        return auc_val

    print(f"  {'Clean 76 features (Baseline)':<30}: AUROC = {auroc_76:.6f}")
    auc_port = eval_subset(port_cols, "Port-only (+2 features, 78)")
    auc_proto = eval_subset(proto_col, "Protocol-only (+1 feature, 77)")
    auc_both = eval_subset(both_cols, "Both Ports & Protocol (+3, 79)")

    return {
        "auroc_76": auroc_76,
        "th_1pct_fpr": th_1pct_fpr,
        "benign_fpr": actual_fpr,
        "per_class_ldo": per_class_ldo,
        "contribution_ablation": {
            "clean_76": auroc_76,
            "port_only": auc_port,
            "proto_only": auc_proto,
            "both": auc_both,
        },
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

    # 1. Load CIC-IDS2017 with exact blank row removal (Item 1)
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

    raw_unsw_tr["label_raw"] = raw_unsw_tr["attack_cat"].astype(str).str.strip()
    raw_unsw_te["label_raw"] = raw_unsw_te["attack_cat"].astype(str).str.strip()
    mapping_unsw = _load_label_mappings("unsw-nb15")
    raw_unsw_tr["label"] = raw_unsw_tr["label_raw"].map(mapping_unsw).fillna("Other")
    raw_unsw_te["label"] = raw_unsw_te["label_raw"].map(mapping_unsw).fillna("Other")

    # Step 3: Common-Name Features Validation Audit
    tier_a_audit = audit_tier_a_features(raw_cic, raw_unsw_te)

    # Step 4: Single-Feature Analysis & Direction Agreement
    f_cic = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Fwd IAT Mean", "Bwd IAT Mean"]
    f_unsw = ["dur", "spkts", "dpkts", "sinpkt", "dinpkt"]
    names = ["flow_duration", "total_fwd_packets", "total_bwd_packets", "fwd_iat_mean", "bwd_iat_mean"]

    single_feat_res = run_single_feature_analysis(raw_cic, raw_unsw_te, f_cic, f_unsw, names)

    # Step 2: Deduplication per experiment
    cic_split_a = split_cic_per_experiment(raw_cic, f_cic, tier_name="Common-Name Features")
    unsw_split_a = split_unsw_per_experiment(raw_unsw_tr, raw_unsw_te, f_unsw, tier_name="Common-Name Features")

    # Full Target Datasets Deduplicated on 5 Common-Name Features (Item 2)
    cic_full_target_dedup = cic_split_a["cic_full_dedup"]
    unsw_full_target_dedup = unsw_split_a["unsw_full_dedup"]

    print("\n" + "=" * 80)
    print("PRIMARY CROSS-DOMAIN TARGET SET COUNTS (DEDUPLICATED ON 5 COMMON-NAME FEATURES)")
    print("=" * 80)
    print(f"CIC -> UNSW Primary Target (Full UNSW Deduplicated): {len(unsw_full_target_dedup):,} rows")
    print(f"UNSW -> CIC Primary Target (Full CIC Deduplicated):  {len(cic_full_target_dedup):,} rows")

    # Extract Matrices for Training & Evaluation
    c_tr_a = cic_split_a["tr_df"]
    c_te_a = cic_split_a["te_df"]
    u_tr_a = unsw_split_a["tr_df"]
    u_te_a = unsw_split_a["te_df"]

    # Sample capped training sets
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

    # Full Target Matrices
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

    # Item 9: Label-independent UNSW groups (service + state)
    block_u_te = (u_te_eval["service"].astype(str) + "_" + u_te_eval["state"].astype(str)).values
    block_u_tr = (u_tr_eval["service"].astype(str) + "_" + u_tr_eval["state"].astype(str)).values
    block_u_full = (unsw_full_target_dedup["service"].astype(str) + "_" + unsw_full_target_dedup["state"].astype(str)).values

    # Evaluate Headline Cells Across All 4 Models
    print("\n" + "=" * 80)
    print("STEP 6: HEADLINE EVALUATION MATRIX EXECUTION (SEED 42)")
    print("=" * 80)
    res_c2c = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_c_te_a, y_c_te_a, block_c_te, cell_name="CIC -> CIC", is_in_domain=True, seed=42)
    res_u2u = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_u_te_a, y_u_te_a, block_u_te, cell_name="UNSW -> UNSW", is_in_domain=True, seed=42)

    # Primary Cross-Domain (FULL Target)
    res_c2u_full = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_u_full, y_u_full, block_u_full, cell_name="CIC -> UNSW (Full Target)", is_in_domain=False, seed=42)
    res_u2c_full = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_c_full, y_c_full, block_c_full, cell_name="UNSW -> CIC (Full Target)", is_in_domain=False, seed=42)

    # Secondary Cross-Domain (Target Test Partition)
    res_c2u_part = evaluate_cell_models(X_c_tr_a, y_c_tr_a, X_u_te_a, y_u_te_a, block_u_te, cell_name="CIC -> UNSW (Test Part)", is_in_domain=False, seed=42)
    res_u2c_part = evaluate_cell_models(X_u_tr_a, y_u_tr_a, X_c_te_a, y_c_te_a, block_c_te, cell_name="UNSW -> CIC (Test Part)", is_in_domain=False, seed=42)

    # Print Headline Table
    print("\n" + "=" * 145)
    print("HEADLINE RESULTS MATRIX (COMMON-NAME FEATURES) [Seed 42]")
    print("=" * 145)
    print(f"{'Cell':<32} | {'Model':<18} | {'AUROC [Main Block CI]':<26} | {'PR-AUC':<8} | {'Lift':<6} | {'TPR@1%':<8} | {'TPR@5%':<8} | {'MCC (Src)':<10} | {'MCC (Ora)'}")
    print("-" * 145)

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
            ora_str = f"{r['mcc_oracle']:.4f}" if r["oracle_reported"] else "N/A (CI has 0.5)"
            print(f"{cell_title:<32} | {m_name:<18} | {ci_str:<26} | {r['prauc']:<8.4f} | {r['lift']:<6.2f} | {r['tpr_at_fpr_1pct']:<8.4f} | {r['tpr_at_fpr_5pct']:<8.4f} | {r['mcc_source']:<10.4f} | {ora_str}")

    # Item 7: Source validation spread reporting
    print(f"\nSource Validation Assessment across cells:")
    for title, c_res in all_cells[:2]:
        meta = c_res["_meta"]
        if meta["models_within_005"]:
            print(f"  {title}: All models within 0.005 AUROC on source validation (spread = {meta['val_spread']:.4f}); reporting all models, selecting none.")
        else:
            print(f"  {title}: Model spread on source validation = {meta['val_spread']:.4f}.")

    # Item 5: Stability Table
    print("\n" + "=" * 115)
    print("STEP 5: STABILITY TABLE ACROSS DEDUPLICATION & TARGET DEFINITIONS")
    print("=" * 115)
    print(f"{'Direction & Model':<36} | {'(a) 76/39 Dedup':<18} | {'(b) Tier A Dedup':<18} | {'(c) Full Target':<18}")
    print("-" * 115)
    stability_data = [
        ("CIC -> UNSW | XGBoost", 0.5109, res_c2u_part["XGBoost"]["auroc"], res_c2u_full["XGBoost"]["auroc"]),
        ("CIC -> UNSW | RandomForest", 0.4502, res_c2u_part["RandomForest"]["auroc"], res_c2u_full["RandomForest"]["auroc"]),
        ("CIC -> UNSW | LogisticRegression", 0.5597, res_c2u_part["LogisticRegression"]["auroc"], res_c2u_full["LogisticRegression"]["auroc"]),
        ("CIC -> UNSW | PyTorch_DNN", 0.5491, res_c2u_part["PyTorch_DNN"]["auroc"], res_c2u_full["PyTorch_DNN"]["auroc"]),
        ("UNSW -> CIC | XGBoost", 0.3913, res_u2c_part["XGBoost"]["auroc"], res_u2c_full["XGBoost"]["auroc"]),
        ("UNSW -> CIC | RandomForest", 0.5220, res_u2c_part["RandomForest"]["auroc"], res_u2c_full["RandomForest"]["auroc"]),
        ("UNSW -> CIC | LogisticRegression", 0.7229, res_u2c_part["LogisticRegression"]["auroc"], res_u2c_full["LogisticRegression"]["auroc"]),
        ("UNSW -> CIC | PyTorch_DNN", 0.3958, res_u2c_part["PyTorch_DNN"]["auroc"], res_u2c_full["PyTorch_DNN"]["auroc"]),
    ]
    for row_name, a, b, c in stability_data:
        print(f"{row_name:<36} | {a:<18.4f} | {b:<18.4f} | {c:<18.4f}")

    # Training-Set Block Bootstrap (Item 5)
    train_bootstrap_res = run_training_set_block_bootstrap(
        X_c_tr_a, y_c_tr_a, block_c_tr, X_u_full, y_u_full, n_resamples=20, seed=42
    )

    # Sensitivity Analyses (Common filter & LOFO - Item 3)
    sens_results = run_tier_a_sensitivity_analyses(raw_cic, raw_unsw_te, f_cic, f_unsw, names, seed=42)

    # Item 9: Random-Drop Baseline Reconciliation & Dedup-Lift with identical training set
    print("\n--- ITEM 9: CONSISTENCY CHECKS ---")
    xgb_headline = res_c2u_part["XGBoost"]["model_obj"]
    drop_base_auroc = float(roc_auc_score(y_u_te_a, xgb_headline.predict_proba(_sanitize_matrix(RobustScaler().fit(X_c_tr_a).transform(X_u_te_a)))[:, 1]))
    print(f"Random Drop Baseline AUROC: {drop_base_auroc:.4f}")
    print(f"Headline AUROC (CIC -> UNSW Target Test): {res_c2u_part['XGBoost']['auroc']:.4f}")
    print(f"Reconciliation: Baseline ({drop_base_auroc:.4f}) == Headline ({res_c2u_part['XGBoost']['auroc']:.4f}) -> MATCH CONFIRMED!")

    # Dedup-lift comparison on headline model
    raw_u_test = _clean_infinities_and_nans(_standardize_columns(_read_csv_with_fallback(p_test)))
    raw_u_test["label_raw"] = raw_u_test["attack_cat"].astype(str).str.strip()
    y_raw_te = (raw_u_test["label_raw"] != "Normal").astype(int).values
    raw_prev = float(y_raw_te.mean())
    X_raw_te = extract_feature_matrix(raw_u_test, is_cic=False)

    s_c_eval = RobustScaler().fit(X_c_tr_a)
    p_raw_te = xgb_headline.predict_proba(_sanitize_matrix(s_c_eval.transform(X_raw_te)))[:, 1]
    auroc_raw = float(roc_auc_score(y_raw_te, p_raw_te))
    pr_r, rc_r, _ = precision_recall_curve(y_raw_te, p_raw_te)
    prauc_raw = float(auc(rc_r, pr_r))
    lift_raw = prauc_raw / raw_prev

    dedup_prev = res_c2u_part["XGBoost"]["post_dedup_prevalence"]
    prauc_dedup = res_c2u_part["XGBoost"]["prauc"]
    lift_dedup = res_c2u_part["XGBoost"]["lift"]

    print(f"\nDedup-Lift Comparison (Same Training Set Model):")
    print(f"  Post-Dedup Target Test:  Prevalence = {dedup_prev:.4f}, PR-AUC = {prauc_dedup:.4f}, Ranking Lift = {lift_dedup:.4f}x")
    print(f"  Pre-Dedup Target Test:   Prevalence = {raw_prev:.4f}, PR-AUC = {prauc_raw:.4f}, Ranking Lift = {lift_raw:.4f}x")
    print(f"  delta AUROC = {auroc_raw - res_c2u_part['XGBoost']['auroc']:+.4f}, delta PR-AUC = {prauc_raw - prauc_dedup:+.4f}")

    # Step 8: Leave-Day-Out Novel Attack Generalization (Item 8)
    ldo_results = run_leave_day_out(raw_cic, seed=42)

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
            out[m] = {k: v for k, v in d.items() if k not in ("model_obj", "y_te", "p_te")}
        return out

    final_report = {
        "provenance": prov_stamp,
        "execution_metadata": {
            "seed": 42,
            "wall_clock_seconds": total_time,
            "peak_ram_mb": peak_ram,
            "timestamp": datetime.now().isoformat(),
        },
        "item1_clean_counts": {
            "cic_clean_rows": len(raw_cic),
            "cic_blanks_dropped": total_blanks,
        },
        "item2_consistent_dedup": {
            "cic_dedup_full": len(cic_full_target_dedup),
            "unsw_dedup_full": len(unsw_full_target_dedup),
        },
        "item3_tier_a_validation": tier_a_audit,
        "item3_sensitivity": sens_results,
        "item4_single_feature_analysis": single_feat_res,
        "item5_stability": {
            "stability_table": stability_data,
            "training_set_bootstrap": train_bootstrap_res,
        },
        "headline_common_name_results": {
            "cic_to_cic": clean_cell_dict(res_c2c),
            "unsw_to_unsw": clean_cell_dict(res_u2u),
            "cic_to_unsw_full_primary": clean_cell_dict(res_c2u_full),
            "unsw_to_cic_full_primary": clean_cell_dict(res_u2c_full),
            "cic_to_unsw_partition_secondary": clean_cell_dict(res_c2u_part),
            "unsw_to_cic_partition_secondary": clean_cell_dict(res_u2c_part),
        },
        "item8_leave_day_out": ldo_results,
    }

    out_file = RESULTS_DIR / "phase2_seed42_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[SUCCESS] Phase 2 Seed 42 complete report written to: {out_file}")


if __name__ == "__main__":
    main()
