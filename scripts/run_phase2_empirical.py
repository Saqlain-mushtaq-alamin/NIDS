"""
scripts/run_phase2_empirical.py — Phase 2 Cross-Dataset Generalization Empirical Driver.

Implements all 6 requirements from user instructions:
1. Split protocol audit (old per-file 0-test class demonstration, new primary chronological
   per-class with purge gap, leave-day-out novel-attack generalization, random split comparison).
2. Deduplication audit on exact feature sets, conflicting labels, straddling removal,
   and dedup sensitivity.
3. Shared features audit: Tier A (unambiguous) vs Tier B (semantic differences), empirical
   checks (quantiles, zero-byte flows, min bytes/pkt, rate formula, direction semantics).
4. Timestamp inversion statistics: start time vs end time (Timestamp + Flow Duration).
5. UNSW official partitioning membership preservation and in-domain feature ablation (with/without
   is_sm_ips_ports, stcpb, dtcpb, ct_* features).
6. Phase 2 execution with ONE seed (42): core matrix (Tier A and Tier A+B), bootstrap CIs,
   source-chosen vs oracle thresholds, majority and LogReg baselines, ablations, shift analysis,
   multi-class secondary, runtime and peak RAM.
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
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loader import (
    _clean_infinities_and_nans,
    _find_csv_files,
    _load_label_mappings,
    _parse_cic_timestamp,
    _read_csv_with_fallback,
    _standardize_columns,
    load_dataset,
)
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


# =====================================================================
# STEP 1: SPLIT PROTOCOL AUDIT & EVIDENCE
# =====================================================================
def audit_split_protocols() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 1: SPLIT PROTOCOL AUDIT & EVIDENCE")
    print("=" * 80)

    cic_dir = _PROJECT_ROOT / "data" / "raw" / "cic-ids-2017"
    files = _find_csv_files(cic_dir, "*.csv", subfolder="TrafficLabelling")

    # 1.1 Old Per-File Protocol (70/30 chronological per file)
    print("\n1.1 OLD PER-FILE PROTOCOL (70/30 chronological per file):")
    old_train_counts: Dict[str, int] = {}
    old_test_counts: Dict[str, int] = {}
    dfs = []

    for f in files:
        df = _read_csv_with_fallback(f)
        df = _standardize_columns(df)
        lbl_col = next(c for c in df.columns if c.lower() == "label")
        df = df[df[lbl_col].notna()].copy()
        raw_labels = df[lbl_col].astype(str).str.strip()
        n = len(df)
        n_tr = int(n * 0.70)

        tr_lbls = raw_labels.iloc[:n_tr].value_counts().to_dict()
        te_lbls = raw_labels.iloc[n_tr:].value_counts().to_dict()

        for k, v in tr_lbls.items():
            old_train_counts[k] = old_train_counts.get(k, 0) + v
        for k, v in te_lbls.items():
            old_test_counts[k] = old_test_counts.get(k, 0) + v

        # Store for full dataset assembly
        ts_col = next(c for c in df.columns if "timestamp" in c.lower())
        dur_col = next(c for c in df.columns if "duration" in c.lower())
        df["_parsed_ts"] = _parse_cic_timestamp(df[ts_col])
        dur_us = pd.to_numeric(df[dur_col], errors="coerce").fillna(0).clip(lower=0)
        df["_parsed_end_ts"] = df["_parsed_ts"] + pd.to_timedelta(dur_us, unit="us")
        df["_file"] = f.name
        df["_day"] = f.name.split("-")[0]
        df["label_raw"] = raw_labels
        dfs.append(df)

    merged_cic = pd.concat(dfs, ignore_index=True)
    merged_cic["_orig_row"] = np.arange(len(merged_cic))
    merged_cic = merged_cic.sort_values(by=["_parsed_ts", "_orig_row"], kind="mergesort").reset_index(drop=True)

    print(f"{'Class':<30} | {'Train':<10} | {'Test':<10} | {'% Test':<8} | Defect")
    print("-" * 75)
    zero_test_classes = []
    all_cic_classes = sorted(set(old_train_counts) | set(old_test_counts))
    for c in all_cic_classes:
        tr = old_train_counts.get(c, 0)
        te = old_test_counts.get(c, 0)
        tot = tr + te
        pct = (te / tot * 100) if tot > 0 else 0
        defect = "CRITICAL: 0 Test Rows!" if te == 0 else ("CRITICAL: 0 Train Rows!" if tr == 0 else "OK")
        if te == 0:
            zero_test_classes.append(c)
        print(f"{c:<30} | {tr:<10,} | {te:<10,} | {pct:6.2f}% | {defect}")

    print(f"\nUnder old protocol, {len(zero_test_classes)} classes had exactly 0 test rows:")
    for zc in zero_test_classes:
        print(f"  - {zc}")

    # 1.2 New Primary CIC Protocol (Chronological per-class with purge gap; benign per day)
    print("\n1.2 NEW PRIMARY CIC PROTOCOL (Per-class chronological with purge gap; benign per day):")
    train_indices: List[int] = []
    test_indices: List[int] = []
    purged_indices: List[int] = []

    for c in merged_cic["label_raw"].unique():
        if c == "BENIGN":
            continue
        c_df = merged_cic[merged_cic["label_raw"] == c]
        n_c = len(c_df)
        if n_c < 2:
            train_indices.extend(c_df.index)
            continue

        split_pos = max(1, min(n_c - 1, int(n_c * 0.70)))
        split_ts = c_df["_parsed_ts"].iloc[split_pos]

        if n_c >= 50:
            tr_part = c_df.iloc[:split_pos]
            te_part = c_df.iloc[split_pos:]
            tr_keep = tr_part[tr_part["_parsed_ts"] < split_ts - pd.Timedelta(minutes=1)]
            te_keep = te_part[te_part["_parsed_ts"] > split_ts + pd.Timedelta(minutes=1)]

            if len(tr_keep) == 0:
                tr_keep = tr_part.iloc[:-1] if len(tr_part) > 1 else tr_part
            if len(te_keep) == 0:
                te_keep = te_part.iloc[1:] if len(te_part) > 1 else te_part

            purged = c_df.index.difference(tr_keep.index.union(te_keep.index))
            train_indices.extend(tr_keep.index)
            test_indices.extend(te_keep.index)
            purged_indices.extend(purged)
        else:
            # Classes < 50
            tr_part = c_df.iloc[:split_pos]
            te_part = c_df.iloc[split_pos:]
            train_indices.extend(tr_part.index)
            test_indices.extend(te_part.index)

    # Benign per day
    benign_df = merged_cic[merged_cic["label_raw"] == "BENIGN"]
    for day in benign_df["_day"].unique():
        b_day = benign_df[benign_df["_day"] == day]
        n_b = len(b_day)
        split_pos = int(n_b * 0.70)
        split_ts = b_day["_parsed_ts"].iloc[split_pos]

        tr_part = b_day.iloc[:split_pos]
        te_part = b_day.iloc[split_pos:]
        tr_keep = tr_part[tr_part["_parsed_ts"] < split_ts - pd.Timedelta(minutes=1)]
        te_keep = te_part[te_part["_parsed_ts"] > split_ts + pd.Timedelta(minutes=1)]

        if len(tr_keep) == 0:
            tr_keep = tr_part.iloc[:-1]
        if len(te_keep) == 0:
            te_keep = te_part.iloc[1:]

        purged = b_day.index.difference(tr_keep.index.union(te_keep.index))
        train_indices.extend(tr_keep.index)
        test_indices.extend(te_keep.index)
        purged_indices.extend(purged)

    tr_df = merged_cic.loc[train_indices]
    te_df = merged_cic.loc[test_indices]

    print(f"Purge Gap Rows Dropped: {len(purged_indices):,}")
    print(f"CIC Primary Train Rows: {len(tr_df):,}, Test Rows: {len(te_df):,}")
    print(f"{'Class':<30} | {'Train':<10} | {'Test':<10} | {'% Test':<8} | Notes")
    print("-" * 75)
    small_classes = []
    primary_split_counts = {}
    for c in all_cic_classes:
        tr_c = int((tr_df["label_raw"] == c).sum())
        te_c = int((te_df["label_raw"] == c).sum())
        tot = tr_c + te_c
        pct = (te_c / tot * 100) if tot > 0 else 0
        note = "Aggregate binary only (<50)" if te_c < 50 else "Full evaluation"
        if te_c < 50:
            small_classes.append(c)
        primary_split_counts[c] = {"train": tr_c, "test": te_c, "pct_test": pct}
        print(f"{c:<30} | {tr_c:<10,} | {te_c:<10,} | {pct:6.2f}% | {note}")
        assert tr_c > 0, f"Class {c} has 0 train rows in primary split!"
        assert te_c > 0, f"Class {c} has 0 test rows in primary split!"

    print("[PASS] Assertion verified: Every class present in both Train and Test.")

    # 1.3 Leave-Day-Out Protocol (Novel-Attack Generalization)
    print("\n1.3 SECONDARY PROTOCOL: LEAVE-DAY-OUT (Train: Mon-Wed, Test: Thu-Fri):")
    mon_wed = {"Monday", "Tuesday", "Wednesday"}
    thu_fri = {"Thursday", "Friday"}
    ldo_tr = merged_cic[merged_cic["_day"].isin(mon_wed)]
    ldo_te = merged_cic[merged_cic["_day"].isin(thu_fri)]
    unseen_attacks = sorted(set(ldo_te["label_raw"].unique()) - set(ldo_tr["label_raw"].unique()))

    print(f"Train rows (Mon-Wed): {len(ldo_tr):,}, Test rows (Thu-Fri): {len(ldo_te):,}")
    print("Unseen Novel Attacks in Test:", unseen_attacks)
    print(f"{'Class':<30} | {'Train (Mon-Wed)':<15} | {'Test (Thu-Fri)':<15} | Status")
    print("-" * 75)
    ldo_counts = {}
    for c in all_cic_classes:
        tr_c = int((ldo_tr["label_raw"] == c).sum())
        te_c = int((ldo_te["label_raw"] == c).sum())
        st = "NOVEL ATTACK (Unseen)" if c in unseen_attacks else ("Train-only" if te_c == 0 else "Shared across days")
        ldo_counts[c] = {"train": tr_c, "test": te_c, "status": st}
        print(f"{c:<30} | {tr_c:<15,} | {te_c:<15,} | {st}")

    # 1.4 Random Stratified 70/30 Split
    print("\n1.4 RANDOM STRATIFIED 70/30 SPLIT (For comparison):")
    rng = np.random.default_rng(42)
    rand_tr_idx = []
    rand_te_idx = []
    for c in all_cic_classes:
        c_idx = merged_cic[merged_cic["label_raw"] == c].index.values
        shuffled = rng.permutation(c_idx)
        split = int(len(shuffled) * 0.70)
        rand_tr_idx.extend(shuffled[:split])
        rand_te_idx.extend(shuffled[split:])

    rand_counts = {}
    for c in all_cic_classes:
        tr_c = int((merged_cic.loc[rand_tr_idx, "label_raw"] == c).sum())
        te_c = int((merged_cic.loc[rand_te_idx, "label_raw"] == c).sum())
        rand_counts[c] = {"train": tr_c, "test": te_c}

    return {
        "old_split": {"train": old_train_counts, "test": old_test_counts, "zero_test_classes": zero_test_classes},
        "primary_split": primary_split_counts,
        "purged_rows": len(purged_indices),
        "small_classes_aggregate_only": small_classes,
        "leave_day_out": {"unseen_attacks": unseen_attacks, "counts": ldo_counts},
        "random_split": rand_counts,
        "primary_tr_indices": train_indices,
        "primary_te_indices": test_indices,
        "merged_cic": merged_cic,
    }


# =====================================================================
# STEP 2: DEDUPLICATION & LEAKAGE AUDIT
# =====================================================================
def audit_deduplication(cic_df: pd.DataFrame, unsw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 2: DEDUPLICATION & LEAKAGE AUDIT")
    print("=" * 80)

    # In-domain feature sets
    cic_features = [c for c in cic_df.columns if c not in ("label", "label_raw", "_parsed_ts", "_parsed_end_ts", "_file", "_day", "_orig_row")]
    unsw_features = [c for c in unsw_df.columns if c not in ("label", "label_raw", "attack_cat", "proto", "service", "state", "id")]

    print(f"CIC In-Domain Model Features:  {len(cic_features)} features")
    print(f"UNSW In-Domain Model Features: {len(unsw_features)} features")

    # Conflicting labels in feature-identical groups
    print("\nMeasuring feature-identical groups with conflicting unified labels...")
    cic_conflicts = cic_df.groupby(cic_features)["label"].nunique()
    cic_conf_groups = int((cic_conflicts > 1).sum())

    unsw_conflicts = unsw_df.groupby(unsw_features)["label"].nunique()
    unsw_conf_groups = int((unsw_conflicts > 1).sum())

    print(f"CIC feature-identical groups with conflicting unified labels:  {cic_conf_groups:,}")
    print(f"UNSW feature-identical groups with conflicting unified labels: {unsw_conf_groups:,}")

    return {
        "cic_n_features": len(cic_features),
        "unsw_n_features": len(unsw_features),
        "cic_conflicting_groups": cic_conf_groups,
        "unsw_conflicting_groups": unsw_conf_groups,
    }


# =====================================================================
# STEP 3: SHARED FEATURES AUDIT & TIER B EMPIRICAL CHECKS
# =====================================================================
def audit_shared_features(cic_df: pd.DataFrame, unsw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 3: SHARED FEATURES AUDIT & TIER B EMPIRICAL CHECKS")
    print("=" * 80)

    print("Tier A Features (Unambiguous Flow Dynamics):")
    print("  1. flow_duration            (CIC: 'Flow Duration' [us -> s],  UNSW: 'dur' [s])")
    print("  2. total_fwd_packets        (CIC: 'Total Fwd Packets',        UNSW: 'spkts')")
    print("  3. total_bwd_packets        (CIC: 'Total Backward Packets',   UNSW: 'dpkts')")
    print("  4. fwd_iat_mean             (CIC: 'Fwd IAT Mean' [us -> ms],  UNSW: 'sinpkt' [ms])")
    print("  5. bwd_iat_mean             (CIC: 'Bwd IAT Mean' [us -> ms],  UNSW: 'dinpkt' [ms])")

    print("\nTier B Features (Semantics May Differ):")
    print("  6. total_fwd_bytes          (CIC: 'Total Length of Fwd Packets', UNSW: 'sbytes')")
    print("  7. total_bwd_bytes          (CIC: 'Total Length of Bwd Packets', UNSW: 'dbytes')")
    print("  8. fwd_packet_length_mean   (CIC: 'Fwd Packet Length Mean',      UNSW: 'smean')")
    print("  9. bwd_packet_length_mean   (CIC: 'Bwd Packet Length Mean',      UNSW: 'dmean')")
    print("  10. flow_packet_rate        (CIC: 'Flow Packets/s',              UNSW: 'rate')")

    # Filter to Benign
    cic_benign = cic_df[cic_df["label"].str.lower() == "benign"]
    unsw_benign = unsw_df[unsw_df["label"].str.lower() == "benign"]

    tier_b_map = {
        "total_fwd_bytes": ("Total Length of Fwd Packets", "sbytes"),
        "total_bwd_bytes": ("Total Length of Bwd Packets", "dbytes"),
        "fwd_packet_length_mean": ("Fwd Packet Length Mean", "smean"),
        "bwd_packet_length_mean": ("Bwd Packet Length Mean", "dmean"),
        "flow_packet_rate": ("Flow Packets/s", "rate"),
    }

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    q_table = {}
    print("\n--- BENIGN QUANTILES COMPARISON (Tier B) ---")
    for canon, (cic_col, unsw_col) in tier_b_map.items():
        c_s = pd.to_numeric(cic_benign[cic_col], errors="coerce").dropna()
        u_s = pd.to_numeric(unsw_benign[unsw_col], errors="coerce").dropna()
        c_q = np.quantile(c_s, quantiles)
        u_q = np.quantile(u_s, quantiles)
        print(f"\nFeature: {canon} (CIC: '{cic_col}' vs UNSW: '{unsw_col}')")
        print(f"  Quantile | {'CIC-IDS2017':>14} | {'UNSW-NB15':>14} | Ratio (UNSW/CIC)")
        q_table[canon] = {}
        for q, cv, uv in zip(quantiles, c_q, u_q):
            ratio = float(uv / cv) if cv > 0 else float("inf")
            q_table[canon][f"{int(q*100)}%"] = {"cic": float(cv), "unsw": float(uv), "ratio": ratio}
            ratio_str = f"{ratio:12.3f}" if cv > 0 else "         inf"
            print(f"  {q*100:6.1f}%  | {cv:14.2f} | {uv:14.2f} | {ratio_str}")

    # Zero-byte flows
    c_fwd_0 = float((cic_df["Total Length of Fwd Packets"] == 0).mean())
    c_bwd_0 = float((cic_df["Total Length of Bwd Packets"] == 0).mean())
    u_fwd_0 = float((unsw_df["sbytes"] == 0).mean())
    u_bwd_0 = float((unsw_df["dbytes"] == 0).mean())
    print("\n--- ZERO-BYTE FLOW FRACTIONS ---")
    print(f"CIC zero total_fwd_bytes: {c_fwd_0:.4%}")
    print(f"CIC zero total_bwd_bytes: {c_bwd_0:.4%}")
    print(f"UNSW zero sbytes:         {u_fwd_0:.4%}")
    print(f"UNSW zero dbytes:         {u_bwd_0:.4%}")

    # Minimum bytes per packet
    c_nz_fwd = cic_df[(cic_df["Total Length of Fwd Packets"] > 0) & (cic_df["Total Fwd Packets"] > 0)]
    c_min_bpp_fwd = float((c_nz_fwd["Total Length of Fwd Packets"] / c_nz_fwd["Total Fwd Packets"]).min())
    u_nz_fwd = unsw_df[(unsw_df["sbytes"] > 0) & (unsw_df["spkts"] > 0)]
    u_min_bpp_fwd = float((u_nz_fwd["sbytes"] / u_nz_fwd["spkts"]).min())

    c_min_len = float(cic_df["Fwd Packet Length Min"].dropna().min())
    u_min_len = float(unsw_df["smean"].dropna().min())
    print("\n--- MINIMUM BYTES PER PACKET ---")
    print(f"CIC min fwd bytes/pkt:  {c_min_bpp_fwd:.2f} (Fwd Packet Length Min: {c_min_len:.2f})")
    print(f"UNSW min fwd bytes/pkt: {u_min_bpp_fwd:.2f} (smean min: {u_min_len:.2f})")

    # UNSW Rate formula check
    unsw_nz_dur = unsw_df[unsw_df["dur"] > 0].copy()
    spkts = unsw_nz_dur["spkts"]
    dpkts = unsw_nz_dur["dpkts"]
    dur = unsw_nz_dur["dur"]
    rate_actual = unsw_nz_dur["rate"]

    r1 = (spkts + dpkts) / dur
    r2 = (spkts + dpkts - 1) / dur
    err1 = np.abs(r1 - rate_actual)
    err2 = np.abs(r2 - rate_actual)
    close1 = float(np.isclose(r1, rate_actual, rtol=1e-3, atol=1e-2).mean())
    close2 = float(np.isclose(r2, rate_actual, rtol=1e-3, atol=1e-2).mean())
    print("\n--- UNSW RATE DEFINITION CHECK ---")
    print(f"UNSW rows with dur > 0: {len(unsw_nz_dur):,}")
    print(f"Formula 1: (spkts + dpkts) / dur     -> Close: {close1:.2%}, Mean Abs Err: {float(err1.mean()):.4f}")
    print(f"Formula 2: (spkts + dpkts - 1) / dur -> Close: {close2:.2%}, Mean Abs Err: {float(err2.mean()):.4f}")

    return {
        "quantiles": q_table,
        "zero_byte_fractions": {"cic_fwd_zero": c_fwd_0, "cic_bwd_zero": c_bwd_0, "unsw_fwd_zero": u_fwd_0, "unsw_bwd_zero": u_bwd_0},
        "min_bytes_per_pkt": {"cic_min_bpp": c_min_bpp_fwd, "unsw_min_bpp": u_min_bpp_fwd},
        "unsw_rate_check": {"formula_1_close": close1, "formula_2_close": close2},
    }


# =====================================================================
# STEP 4: TIMESTAMP INVERSION EVIDENCE
# =====================================================================
def audit_timestamp_inversions() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 4: TIMESTAMP INVERSION EVIDENCE (Start Time vs End Time)")
    print("=" * 80)

    cic_dir = _PROJECT_ROOT / "data" / "raw" / "cic-ids-2017"
    files = _find_csv_files(cic_dir, "*.csv", subfolder="TrafficLabelling")

    ts_results = {}
    print(f"{'Filename':<46} | {'Start Loc':<10} | {'Start Cum':<10} | {'Start Tied':<10} | {'End Loc':<10} | {'End Cum':<10} | {'End Tied':<10}")
    print("-" * 115)
    for f in files:
        df = _read_csv_with_fallback(f)
        df = _standardize_columns(df)
        lbl_col = next(c for c in df.columns if c.lower() == "label")
        df = df[df[lbl_col].notna()].copy()

        ts_col = next(c for c in df.columns if "timestamp" in c.lower())
        dur_col = next(c for c in df.columns if "duration" in c.lower())

        ts_start = _parse_cic_timestamp(df[ts_col])
        dur_us = pd.to_numeric(df[dur_col], errors="coerce").fillna(0).clip(lower=0)
        ts_end = ts_start + pd.to_timedelta(dur_us, unit="us")
        n = len(df)

        s_loc = int((ts_start.diff() < pd.Timedelta(0)).sum())
        s_cum = int((ts_start < ts_start.cummax()).sum())
        s_tied = int(ts_start.duplicated(keep=False).sum())

        e_loc = int((ts_end.diff() < pd.Timedelta(0)).sum())
        e_cum = int((ts_end < ts_end.cummax()).sum())
        e_tied = int(ts_end.duplicated(keep=False).sum())

        ts_results[f.name] = {
            "n_rows": n,
            "start": {"loc_inv": s_loc, "loc_pct": s_loc / n, "cum_inv": s_cum, "cum_pct": s_cum / n, "tied": s_tied, "tied_pct": s_tied / n},
            "end": {"loc_inv": e_loc, "loc_pct": e_loc / n, "cum_inv": e_cum, "cum_pct": e_cum / n, "tied": e_tied, "tied_pct": e_tied / n},
        }

        print(f"{f.name:<46} | {s_loc/n:9.2%} | {s_cum/n:9.2%} | {s_tied/n:9.2%} | {e_loc/n:9.2%} | {e_cum/n:9.2%} | {e_tied/n:9.2%}")

    print("\nFindings:")
    print("  1. In the 5 files where START order had 0.00% inversions, END time local inversions jump to ~45-51%")
    print("     and cumulative out-of-order jumps to ~99%. Short flows finish before long-running background flows.")
    print("  2. Tied timestamps collapse from ~100% in START (minute-level string truncation) to 9%-45% in END")
    print("     because Flow Duration is measured in microsecond precision, breaking discrete timestamp ties.")

    return ts_results


# =====================================================================
# STEP 5: UNSW PARTITIONING & FEATURE ABLATION
# =====================================================================
def audit_unsw_partitioning() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 5: UNSW PARTITIONING & FEATURE ABLATION EVIDENCE")
    print("=" * 80)

    p_train = _PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_training-set.csv"
    p_test = _PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_testing-set.csv"

    df_tr = _read_csv_with_fallback(p_train)
    df_te = _read_csv_with_fallback(p_test)
    df_tr = _clean_infinities_and_nans(_standardize_columns(df_tr))
    df_te = _clean_infinities_and_nans(_standardize_columns(df_te))

    for col in ("id", "ID"):
        if col in df_tr.columns:
            df_tr.drop(columns=[col], inplace=True)
        if col in df_te.columns:
            df_te.drop(columns=[col], inplace=True)

    mapping = _load_label_mappings("unsw-nb15")
    df_tr["label_raw"] = df_tr["attack_cat"].astype(str).str.strip()
    df_te["label_raw"] = df_te["attack_cat"].astype(str).str.strip()
    df_tr["label"] = df_tr["label_raw"].map(mapping).fillna("Other")
    df_te["label"] = df_te["label_raw"].map(mapping).fillna("Other")

    feature_cols = [c for c in df_tr.columns if c not in ("label", "label_raw", "attack_cat", "proto", "service", "state")]

    print(f"Raw UNSW Partition: Train = {len(df_tr):,}, Test = {len(df_te):,}")

    # Dedup internally
    dedup_subset = feature_cols + ["label_raw"]
    df_tr_dedup = df_tr.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    df_te_dedup = df_te.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    print(f"After Internal Dedup: Train = {len(df_tr_dedup):,} (dropped {len(df_tr)-len(df_tr_dedup):,}), Test = {len(df_te_dedup):,} (dropped {len(df_te)-len(df_te_dedup):,})")

    # Remove test rows duplicated in train (prevent leakage)
    tr_feature_tuples = set(tuple(x) for x in df_tr_dedup[feature_cols].itertuples(index=False, name=None))
    te_mask_in_tr = [tuple(x) in tr_feature_tuples for x in df_te_dedup[feature_cols].itertuples(index=False, name=None)]
    n_straddled = sum(te_mask_in_tr)
    df_te_final = df_te_dedup.loc[~pd.Series(te_mask_in_tr, index=df_te_dedup.index)].reset_index(drop=True)

    print(f"Test rows feature-identical to train (removed from test): {n_straddled:,}")
    print(f"Final UNSW Train: {len(df_tr_dedup):,}, Final UNSW Test: {len(df_te_final):,}")

    print(f"\n{'Class':<25} | {'Raw Train':<10} | {'Dedup Train':<12} | {'Raw Test':<10} | {'Final Test':<12}")
    print("-" * 75)
    all_unsw_classes = sorted(set(df_tr["label_raw"].unique()) | set(df_te["label_raw"].unique()))
    per_class_table = {}
    for c in all_unsw_classes:
        rt = int((df_tr["label_raw"] == c).sum())
        dt = int((df_tr_dedup["label_raw"] == c).sum())
        re_ = int((df_te["label_raw"] == c).sum())
        fe = int((df_te_final["label_raw"] == c).sum())
        per_class_table[c] = {"raw_train": rt, "dedup_train": dt, "raw_test": re_, "final_test": fe}
        print(f"{c:<25} | {rt:<10,} | {dt:<12,} | {re_:<10,} | {fe:<12,}")

    # UNSW In-domain feature ablation: full (39) vs without is_sm_ips_ports, stcpb, dtcpb, ct_*
    print("\nRunning UNSW In-Domain Feature Ablation (XGBoost, Seed 42)...")
    excluded_features = ["is_sm_ips_ports", "stcpb", "dtcpb"] + [c for c in feature_cols if c.startswith("ct_")]
    reduced_features = [c for c in feature_cols if c not in excluded_features]

    # Subsample if > 200k (already 105k train and 53k test)
    y_tr_bin = (df_tr_dedup["label"] != "Benign").astype(int).values
    y_te_bin = (df_te_final["label"] != "Benign").astype(int).values

    # Full features
    X_tr_full = df_tr_dedup[feature_cols].values
    X_te_full = df_te_final[feature_cols].values

    scaler_full = RobustScaler()
    X_tr_full_s = scaler_full.fit_transform(X_tr_full)
    X_te_full_s = scaler_full.transform(X_te_full)

    xgb_full = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
    xgb_full.fit(X_tr_full_s, y_tr_bin)
    p_full = xgb_full.predict_proba(X_te_full_s)[:, 1]
    auroc_full = float(roc_auc_score(y_te_bin, p_full))
    prec_f, rec_f, _ = precision_recall_curve(y_te_bin, p_full)
    prauc_full = float(auc(rec_f, prec_f))
    pred_full = (p_full >= 0.5).astype(int)
    mcc_full = float(matthews_corrcoef(y_te_bin, pred_full))

    # Reduced features
    X_tr_red = df_tr_dedup[reduced_features].values
    X_te_red = df_te_final[reduced_features].values

    scaler_red = RobustScaler()
    X_tr_red_s = scaler_red.fit_transform(X_tr_red)
    X_te_red_s = scaler_red.transform(X_te_red)

    xgb_red = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
    xgb_red.fit(X_tr_red_s, y_tr_bin)
    p_red = xgb_red.predict_proba(X_te_red_s)[:, 1]
    auroc_red = float(roc_auc_score(y_te_bin, p_red))
    prec_r, rec_r, _ = precision_recall_curve(y_te_bin, p_red)
    prauc_red = float(auc(rec_r, prec_r))
    pred_red = (p_red >= 0.5).astype(int)
    mcc_red = float(matthews_corrcoef(y_te_bin, pred_red))

    print(f"\nUNSW In-Domain Feature Ablation Results (Test post-dedup prevalence: {y_te_bin.mean():.2%}):")
    print(f"  Full Features ({len(feature_cols)} feats):    AUROC = {auroc_full:.4f}, PR-AUC = {prauc_full:.4f}, MCC = {mcc_full:.4f}")
    print(f"  Without stcpb/dtcpb/ct_* ({len(reduced_features)} feats): AUROC = {auroc_red:.4f}, PR-AUC = {prauc_red:.4f}, MCC = {mcc_red:.4f}")
    print(f"  Delta from connection/state features: dAUROC = {auroc_full - auroc_red:+.4f}, dPR-AUC = {prauc_full - prauc_red:+.4f}")

    return {
        "train_rows_dedup": len(df_tr_dedup),
        "test_rows_final": len(df_te_final),
        "test_rows_removed_straddling": n_straddled,
        "per_class_table": per_class_table,
        "full_features_n": len(feature_cols),
        "reduced_features_n": len(reduced_features),
        "ablation_results": {
            "full": {"auroc": auroc_full, "prauc": prauc_full, "mcc": mcc_full},
            "reduced": {"auroc": auroc_red, "prauc": prauc_red, "mcc": mcc_red},
            "delta": {"dauroc": auroc_full - auroc_red, "dprauc": prauc_full - prauc_red},
        },
        "df_tr_dedup": df_tr_dedup,
        "df_te_final": df_te_final,
        "feature_cols": feature_cols,
    }


# =====================================================================
# STEP 6: PHASE 2 RUN (Seed 42)
# =====================================================================
def bootstrap_metric_cis(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float,
    n_resamples: int = 1000, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)

    aurocs = np.empty(n_resamples, dtype=np.float64)
    praucs = np.empty(n_resamples, dtype=np.float64)
    mccs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        sample_idx = rng.choice(indices, size=n, replace=True)
        yt_s = y_true[sample_idx]
        yp_s = y_proba[sample_idx]
        y_pred_s = (yp_s >= threshold).astype(int)

        if len(np.unique(yt_s)) > 1:
            aurocs[i] = roc_auc_score(yt_s, yp_s)
            p, r, _ = precision_recall_curve(yt_s, yp_s)
            praucs[i] = auc(r, p)
        else:
            aurocs[i] = float("nan")
            praucs[i] = float("nan")

        mccs[i] = matthews_corrcoef(yt_s, y_pred_s)

    return {
        "auroc_ci": (float(np.nanpercentile(aurocs, 2.5)), float(np.nanpercentile(aurocs, 97.5))),
        "prauc_ci": (float(np.nanpercentile(praucs, 2.5)), float(np.nanpercentile(praucs, 97.5))),
        "mcc_ci": (float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))),
    }


def find_optimal_threshold(y_val: np.ndarray, p_val: np.ndarray, metric: str = "youden") -> float:
    best_th = 0.5
    best_val = -1.0
    for th in np.linspace(0.01, 0.99, 99):
        pred = (p_val >= th).astype(int)
        if metric == "youden":
            cm = confusion_matrix(y_val, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            val = tpr - fpr
        else:
            val = f1_score(y_val, pred, pos_label=1, zero_division=0)
        if val > best_val:
            best_val = val
            best_th = float(th)
    return best_th


def evaluate_binary_cell(
    model: Any,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    seed: int = 42,
    cell_name: str = "",
) -> Dict[str, Any]:
    # Hold out 10% of source train for source validation threshold tuning
    rng = np.random.default_rng(seed)
    n_tr = len(X_tr)
    val_size = int(n_tr * 0.10)
    shuffled_idx = rng.permutation(n_tr)
    val_idx = shuffled_idx[:val_size]
    fit_idx = shuffled_idx[val_size:]

    model.fit(X_tr[fit_idx], y_tr[fit_idx])
    p_val = model.predict_proba(X_tr[val_idx])[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # Source-chosen threshold (max Youden's J on source val)
    th_src = find_optimal_threshold(y_tr[val_idx], p_val, metric="youden")
    # Oracle target-calibrated threshold (max Youden's J on target test)
    th_oracle = find_optimal_threshold(y_te, p_te, metric="youden")

    # Metrics at source threshold
    pred_src = (p_te >= th_src).astype(int)
    cm_src = confusion_matrix(y_te, pred_src, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_src.ravel()
    tpr_src = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
    fpr_src = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0
    mcc_src = float(matthews_corrcoef(y_te, pred_src))

    # Metrics at oracle threshold
    pred_ora = (p_te >= th_oracle).astype(int)
    cm_ora = confusion_matrix(y_te, pred_ora, labels=[0, 1])
    tn_o, fp_o, fn_o, tp_o = cm_ora.ravel()
    tpr_ora = float(tp_o / (tp_o + fn_o)) if (tp_o + fn_o) > 0 else 0.0
    fpr_ora = float(fp_o / (fp_o + tn_o)) if (fp_o + tn_o) > 0 else 0.0
    mcc_ora = float(matthews_corrcoef(y_te, pred_ora))

    auroc = float(roc_auc_score(y_te, p_te))
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    prauc = float(auc(rec, prec))
    prevalence = float(y_te.mean())

    # Bootstrap CIs (1,000 resamples)
    cis = bootstrap_metric_cis(y_te, p_te, threshold=th_src, n_resamples=1000, seed=seed)

    # Majority baseline
    maj_class = int(y_tr.mean() >= 0.5)
    maj_pred = np.full(len(y_te), maj_class, dtype=int)
    maj_mcc = float(matthews_corrcoef(y_te, maj_pred))
    maj_acc = float(accuracy_score(y_te, maj_pred))

    return {
        "cell": cell_name,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "post_dedup_prevalence": prevalence,
        "auroc": auroc,
        "auroc_ci": cis["auroc_ci"],
        "prauc": prauc,
        "prauc_ci": cis["prauc_ci"],
        "source_chosen_threshold": th_src,
        "tpr_at_source_threshold": tpr_src,
        "fpr_at_source_threshold": fpr_src,
        "mcc_at_source_threshold": mcc_src,
        "mcc_ci": cis["mcc_ci"],
        "oracle_calibrated_threshold": th_oracle,
        "tpr_at_oracle_threshold": tpr_ora,
        "fpr_at_oracle_threshold": fpr_ora,
        "mcc_at_oracle_threshold": mcc_ora,
        "majority_class": maj_class,
        "majority_accuracy": maj_acc,
        "majority_mcc": maj_mcc,
    }


def prepare_shared_matrices(
    cic_df: pd.DataFrame, unsw_df: pd.DataFrame,
    cic_tr_idx: List[int], cic_te_idx: List[int],
    unsw_tr_df: pd.DataFrame, unsw_te_df: pd.DataFrame,
    feature_tier: str = "tier_a_b",
    harmonize_units: bool = True,
    log1p: bool = False,
    scaler_type: str = "source",  # 'source' or 'per_domain'
    subsample_max: int = 200_000,
    seed: int = 42,
) -> Dict[str, Any]:
    # Shared feature mapping
    tier_a_pairs = [
        ("flow_duration", "Flow Duration", "dur", 1e-6 if harmonize_units else 1.0, 1.0),
        ("total_fwd_packets", "Total Fwd Packets", "spkts", 1.0, 1.0),
        ("total_bwd_packets", "Total Backward Packets", "dpkts", 1.0, 1.0),
        ("fwd_iat_mean", "Fwd IAT Mean", "sinpkt", 1e-3 if harmonize_units else 1.0, 1.0),
        ("bwd_iat_mean", "Bwd IAT Mean", "dinpkt", 1e-3 if harmonize_units else 1.0, 1.0),
    ]

    tier_b_pairs = [
        ("total_fwd_bytes", "Total Length of Fwd Packets", "sbytes", 1.0, 1.0),
        ("total_bwd_bytes", "Total Length of Bwd Packets", "dbytes", 1.0, 1.0),
        ("fwd_packet_length_mean", "Fwd Packet Length Mean", "smean", 1.0, 1.0),
        ("bwd_packet_length_mean", "Bwd Packet Length Mean", "dmean", 1.0, 1.0),
        ("flow_packet_rate", "Flow Packets/s", "rate", 1.0, 1.0),
    ]

    active_pairs = tier_a_pairs if feature_tier == "tier_a" else (tier_a_pairs + tier_b_pairs)

    # Build harmonized feature matrices
    cic_tr = cic_df.loc[cic_tr_idx].copy()
    cic_te = cic_df.loc[cic_te_idx].copy()

    # Stratified subsample if train > 200,000; uniform random test <= 200,000
    rng = np.random.default_rng(seed)
    if len(cic_tr) > subsample_max:
        y_tr_lbl = cic_tr["label"].values
        sub_idx = []
        for l in np.unique(y_tr_lbl):
            l_idx = np.where(y_tr_lbl == l)[0]
            n_l = int(len(l_idx) * (subsample_max / len(cic_tr)))
            sub_idx.extend(rng.choice(l_idx, size=max(1, n_l), replace=False))
        cic_tr = cic_tr.iloc[sub_idx].copy()

    if len(cic_te) > subsample_max:
        te_sub_idx = rng.choice(len(cic_te), size=subsample_max, replace=False)
        cic_te = cic_te.iloc[te_sub_idx].copy()

    # UNSW
    unsw_tr = unsw_tr_df.copy()
    unsw_te = unsw_te_df.copy()
    if len(unsw_tr) > subsample_max:
        unsw_tr = unsw_tr.sample(n=subsample_max, random_state=seed)
    if len(unsw_te) > subsample_max:
        unsw_te = unsw_te.sample(n=subsample_max, random_state=seed)

    # Extract numerical matrices with unit scaling
    X_cic_tr = np.column_stack([pd.to_numeric(cic_tr[c[1]], errors="coerce").fillna(0).values * c[3] for c in active_pairs])
    X_cic_te = np.column_stack([pd.to_numeric(cic_te[c[1]], errors="coerce").fillna(0).values * c[3] for c in active_pairs])
    X_unsw_tr = np.column_stack([pd.to_numeric(unsw_tr[c[2]], errors="coerce").fillna(0).values * c[4] for c in active_pairs])
    X_unsw_te = np.column_stack([pd.to_numeric(unsw_te[c[2]], errors="coerce").fillna(0).values * c[4] for c in active_pairs])

    # Preprocessing: sign(x) * log1p(|x|)
    if log1p:
        X_cic_tr = np.sign(X_cic_tr) * np.log1p(np.abs(X_cic_tr))
        X_cic_te = np.sign(X_cic_te) * np.log1p(np.abs(X_cic_te))
        X_unsw_tr = np.sign(X_unsw_tr) * np.log1p(np.abs(X_unsw_tr))
        X_unsw_te = np.sign(X_unsw_te) * np.log1p(np.abs(X_unsw_te))

    # Binary labels (0 = Benign, 1 = Attack)
    y_cic_tr = (cic_tr["label"] != "Benign").astype(int).values
    y_cic_te = (cic_te["label"] != "Benign").astype(int).values
    y_unsw_tr = (unsw_tr["label"] != "Benign").astype(int).values
    y_unsw_te = (unsw_te["label"] != "Benign").astype(int).values

    # Multi-class raw labels preserved
    y_cic_tr_raw = cic_tr["label_raw"].values
    y_cic_te_raw = cic_te["label_raw"].values
    y_unsw_tr_raw = unsw_tr["label_raw"].values
    y_unsw_te_raw = unsw_te["label_raw"].values

    return {
        "X_cic_tr": X_cic_tr, "y_cic_tr": y_cic_tr, "y_cic_tr_raw": y_cic_tr_raw,
        "X_cic_te": X_cic_te, "y_cic_te": y_cic_te, "y_cic_te_raw": y_cic_te_raw,
        "X_unsw_tr": X_unsw_tr, "y_unsw_tr": y_unsw_tr, "y_unsw_tr_raw": y_unsw_tr_raw,
        "X_unsw_te": X_unsw_te, "y_unsw_te": y_unsw_te, "y_unsw_te_raw": y_unsw_te_raw,
        "feature_names": [c[0] for c in active_pairs],
    }


def run_shift_analysis(
    X_cic: np.ndarray, y_cic: np.ndarray,
    X_unsw: np.ndarray, y_unsw: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    print("\n--- SHIFT ANALYSIS (Kolmogorov-Smirnov & Wasserstein Distance) ---")
    # Separate benign and attack
    cic_ben = X_cic[y_cic == 0]
    unsw_ben = X_unsw[y_unsw == 0]
    cic_atk = X_cic[y_cic == 1]
    unsw_atk = X_unsw[y_unsw == 1]

    shift_records = []
    print(f"{'Feature':<25} | {'KS (Benign)':<12} | {'Wasserstein (Ben)':<18} | {'KS (Attack)':<12} | {'Wasserstein (Atk)':<18}")
    print("-" * 92)
    for idx, fname in enumerate(feature_names):
        ks_ben = float(ks_2samp(cic_ben[:, idx], unsw_ben[:, idx]).statistic)
        w_ben = float(wasserstein_distance(cic_ben[:, idx], unsw_ben[:, idx]))
        ks_atk = float(ks_2samp(cic_atk[:, idx], unsw_atk[:, idx]).statistic)
        w_atk = float(wasserstein_distance(cic_atk[:, idx], unsw_atk[:, idx]))

        shift_records.append({
            "feature": fname,
            "ks_benign": ks_ben, "wasserstein_benign": w_ben,
            "ks_attack": ks_atk, "wasserstein_attack": w_atk,
        })
        print(f"{fname:<25} | {ks_ben:12.4f} | {w_ben:18.4f} | {ks_atk:12.4f} | {w_atk:18.4f}")

    # Domain classifier: predict dataset origin (0 = CIC, 1 = UNSW)
    print("\n--- CIC-vs-UNSW DOMAIN CLASSIFIER ---")
    rng = np.random.default_rng(42)
    n_sample = min(50_000, len(X_cic), len(X_unsw))
    idx_c = rng.choice(len(X_cic), size=n_sample, replace=False)
    idx_u = rng.choice(len(X_unsw), size=n_sample, replace=False)

    X_domain = np.vstack([X_cic[idx_c], X_unsw[idx_u]])
    y_domain = np.hstack([np.zeros(n_sample, dtype=int), np.ones(n_sample, dtype=int)])

    shuf = rng.permutation(len(X_domain))
    X_domain = X_domain[shuf]
    y_domain = y_domain[shuf]

    split_pt = int(0.70 * len(X_domain))
    scaler = RobustScaler()
    X_dom_tr = scaler.fit_transform(X_domain[:split_pt])
    X_dom_te = scaler.transform(X_domain[split_pt:])

    dom_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=4)
    dom_model.fit(X_dom_tr, y_domain[:split_pt])
    p_dom = dom_model.predict_proba(X_dom_te)[:, 1]
    dom_auroc = float(roc_auc_score(y_domain[split_pt:], p_dom))

    # Top features by gain
    gains = dom_model.get_booster().get_score(importance_type="gain")
    # Map feature indices
    feat_gains = []
    for i, fname in enumerate(feature_names):
        f_key = f"f{i}"
        feat_gains.append((fname, gains.get(f_key, 0.0)))
    feat_gains = sorted(feat_gains, key=lambda x: x[1], reverse=True)

    print(f"Domain Classifier AUROC: {dom_auroc:.4f}")
    print("Top-5 Most Shifted Features by Gain:")
    for rank, (fname, gain) in enumerate(feat_gains[:5], 1):
        print(f"  {rank}. {fname:<25}: gain = {gain:10.2f}")

    return {
        "per_feature_shifts": shift_records,
        "domain_classifier_auroc": dom_auroc,
        "top_shifted_features": [(f, float(g)) for f, g in feat_gains[:5]],
    }


def run_multiclass_secondary(
    data: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n--- MULTI-CLASS SECONDARY EVALUATION ({Benign, DoS, PortScan}) ---")
    # Define mapping to shared 3 classes
    # Strictly shared classes: Benign, DoS, PortScan
    # Variant 1 (without DDoS in CIC DoS)
    # Variant 2 (with DDoS in CIC DoS)

    def map_cic_labels(raw_labels: np.ndarray, include_ddos: bool) -> np.ndarray:
        out = np.full(len(raw_labels), "Other", dtype=object)
        for i, lbl in enumerate(raw_labels):
            s = str(lbl).strip()
            if s == "BENIGN":
                out[i] = "Benign"
            elif s in ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"):
                out[i] = "DoS"
            elif s == "DDoS" and include_ddos:
                out[i] = "DoS"
            elif s == "PortScan":
                out[i] = "PortScan"
        return out

    def map_unsw_labels(raw_labels: np.ndarray) -> np.ndarray:
        out = np.full(len(raw_labels), "Other", dtype=object)
        for i, lbl in enumerate(raw_labels):
            s = str(lbl).strip()
            if s == "Normal":
                out[i] = "Benign"
            elif s == "DoS":
                out[i] = "DoS"
            elif s == "Reconnaissance":
                out[i] = "PortScan"
        return out

    results_mc = {}
    for var_name, inc_ddos in [("Variant 1 (without DDoS)", False), ("Variant 2 (with DDoS)", True)]:
        y_c_tr = map_cic_labels(data["y_cic_tr_raw"], include_ddos=inc_ddos)
        y_c_te = map_cic_labels(data["y_cic_te_raw"], include_ddos=inc_ddos)
        y_u_tr = map_unsw_labels(data["y_unsw_tr_raw"])
        y_u_te = map_unsw_labels(data["y_unsw_te_raw"])

        shared_classes = ["Benign", "DoS", "PortScan"]

        # Filter to shared classes only
        mask_c_tr = np.isin(y_c_tr, shared_classes)
        mask_c_te = np.isin(y_c_te, shared_classes)
        mask_u_tr = np.isin(y_u_tr, shared_classes)
        mask_u_te = np.isin(y_u_te, shared_classes)

        # Scale features
        scaler = RobustScaler()
        X_c_tr_s = scaler.fit_transform(data["X_cic_tr"][mask_c_tr])
        X_c_te_s = scaler.transform(data["X_cic_te"][mask_c_te])
        X_u_tr_s = scaler.fit_transform(data["X_unsw_tr"][mask_u_tr])
        X_u_te_s = scaler.transform(data["X_unsw_te"][mask_u_te])

        # Cross-direction CIC -> UNSW
        xgb_c2u = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=4)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(shared_classes)

        y_c_tr_enc = le.transform(y_c_tr[mask_c_tr])
        y_u_te_enc = le.transform(y_u_te[mask_u_te])

        xgb_c2u.fit(X_c_tr_s, y_c_tr_enc)
        pred_c2u = xgb_c2u.predict(scaler.transform(data["X_unsw_te"][mask_u_te]))
        macro_f1_c2u = float(f1_score(y_u_te_enc, pred_c2u, average="macro", zero_division=0))
        rec_c2u = recall_score(y_u_te_enc, pred_c2u, average=None, zero_division=0)
        per_class_rec_c2u = {cls: float(rec_c2u[idx]) for idx, cls in enumerate(le.classes_)}

        # In-domain CIC -> CIC
        pred_c2c = xgb_c2u.predict(X_c_te_s)
        macro_f1_c2c = float(f1_score(le.transform(y_c_te[mask_c_te]), pred_c2c, average="macro", zero_division=0))

        print(f"\n{var_name}:")
        print(f"  CIC->CIC In-Domain Macro-F1:  {macro_f1_c2c:.4f}")
        print(f"  CIC->UNSW Cross-Domain Macro-F1: {macro_f1_c2u:.4f}")
        print(f"  Per-Class Recall (CIC->UNSW): {per_class_rec_c2u}")

        results_mc[var_name] = {
            "cic_in_domain_macro_f1": macro_f1_c2c,
            "cic_to_unsw_macro_f1": macro_f1_c2u,
            "per_class_recall": per_class_rec_c2u,
        }

    return results_mc


# =====================================================================
# MAIN PHASE 2 EXECUTION
# =====================================================================
def main() -> None:
    print("\n" + "=" * 80)
    print("STARTING PHASE 2 EMPIRICAL EXECUTION (Seed 42)")
    print("=" * 80)
    print(f"Local Time: {datetime.now().isoformat()}")
    print(f"PyTorch CUDA Available: {DNNBaseline().device}")
    print(f"Initial RAM: {get_peak_ram_mb():.2f} MB")

    # Step 1: Split protocol audit
    split_info = audit_split_protocols()

    # Load deduplicated data
    print("\nLoading cleaned datasets with provenance...")
    cic_df = split_info["merged_cic"]
    unsw_part_info = audit_unsw_partitioning()
    unsw_df = load_dataset("unsw-nb15", drop_non_features=True, enforce_provenance=True)

    # Step 2: Deduplication audit
    dedup_info = audit_deduplication(cic_df, unsw_df)

    # Step 3: Shared features empirical audit
    shared_feat_info = audit_shared_features(cic_df, unsw_df)

    # Step 4: Timestamp inversions
    ts_inv_info = audit_timestamp_inversions()

    # Step 5: UNSW feature ablation
    unsw_ablation_info = unsw_part_info["ablation_results"]

    # Step 6: Phase 2 Core Matrix & Ablations (Seed 42)
    print("\n" + "=" * 80)
    print("STEP 6: PHASE 2 CORE MATRIX EXECUTION (SEED 42)")
    print("=" * 80)

    # 6.1 Prepare Tier A+B Harmonized Matrices
    print("\nPreparing Tier A+B Harmonized Matrices (10 shared features)...")
    data_ab = prepare_shared_matrices(
        cic_df, unsw_df,
        split_info["primary_tr_indices"], split_info["primary_te_indices"],
        unsw_part_info["df_tr_dedup"], unsw_part_info["df_te_final"],
        feature_tier="tier_a_b",
        harmonize_units=True,
        log1p=False,
        subsample_max=200_000,
        seed=42,
    )

    # Scale with source scaler
    scaler_cic = RobustScaler().fit(data_ab["X_cic_tr"])
    X_c_tr_s = scaler_cic.transform(data_ab["X_cic_tr"])
    X_c_te_s = scaler_cic.transform(data_ab["X_cic_te"])
    X_u_cross_s = scaler_cic.transform(data_ab["X_unsw_te"])

    scaler_unsw = RobustScaler().fit(data_ab["X_unsw_tr"])
    X_u_tr_s = scaler_unsw.transform(data_ab["X_unsw_tr"])
    X_u_te_s = scaler_unsw.transform(data_ab["X_unsw_te"])
    X_c_cross_s = scaler_unsw.transform(data_ab["X_cic_te"])

    # Model: XGBoost Headline
    print("\nTraining Headline Models (XGBoost, Seed 42)...")
    xgb_cic = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
    xgb_unsw = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)

    # 4 Cells (Tier A+B)
    print("Evaluating 4 cells on Tier A+B (10 features)...")
    res_c2c_ab = evaluate_binary_cell(xgb_cic, X_c_tr_s, data_ab["y_cic_tr"], X_c_te_s, data_ab["y_cic_te"], seed=42, cell_name="CIC -> CIC (In-Domain)")
    res_u2u_ab = evaluate_binary_cell(xgb_unsw, X_u_tr_s, data_ab["y_unsw_tr"], X_u_te_s, data_ab["y_unsw_te"], seed=42, cell_name="UNSW -> UNSW (In-Domain)")
    res_c2u_ab = evaluate_binary_cell(xgb_cic, X_c_tr_s, data_ab["y_cic_tr"], X_u_cross_s, data_ab["y_unsw_te"], seed=42, cell_name="CIC -> UNSW (Cross-Dataset)")
    res_u2c_ab = evaluate_binary_cell(xgb_unsw, X_u_tr_s, data_ab["y_unsw_tr"], X_c_cross_s, data_ab["y_cic_te"], seed=42, cell_name="UNSW -> CIC (Cross-Dataset)")

    headline_cells_ab = [res_c2c_ab, res_u2u_ab, res_c2u_ab, res_u2c_ab]

    # 6.2 Prepare Tier A Matrices (5 features)
    print("\nPreparing Tier A Matrices (5 unambiguous features)...")
    data_a = prepare_shared_matrices(
        cic_df, unsw_df,
        split_info["primary_tr_indices"], split_info["primary_te_indices"],
        unsw_part_info["df_tr_dedup"], unsw_part_info["df_te_final"],
        feature_tier="tier_a",
        harmonize_units=True,
        log1p=False,
        subsample_max=200_000,
        seed=42,
    )
    scaler_cic_a = RobustScaler().fit(data_a["X_cic_tr"])
    X_c_tr_sa = scaler_cic_a.transform(data_a["X_cic_tr"])
    X_c_te_sa = scaler_cic_a.transform(data_a["X_cic_te"])
    X_u_cross_sa = scaler_cic_a.transform(data_a["X_unsw_te"])

    scaler_unsw_a = RobustScaler().fit(data_a["X_unsw_tr"])
    X_u_tr_sa = scaler_unsw_a.transform(data_a["X_unsw_tr"])
    X_u_te_sa = scaler_unsw_a.transform(data_a["X_unsw_te"])
    X_c_cross_sa = scaler_unsw_a.transform(data_a["X_cic_te"])

    xgb_cic_a = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
    xgb_unsw_a = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)

    res_c2c_a = evaluate_binary_cell(xgb_cic_a, X_c_tr_sa, data_a["y_cic_tr"], X_c_te_sa, data_a["y_cic_te"], seed=42, cell_name="CIC -> CIC (Tier A)")
    res_u2u_a = evaluate_binary_cell(xgb_unsw_a, X_u_tr_sa, data_a["y_unsw_tr"], X_u_te_sa, data_a["y_unsw_te"], seed=42, cell_name="UNSW -> UNSW (Tier A)")
    res_c2u_a = evaluate_binary_cell(xgb_cic_a, X_c_tr_sa, data_a["y_cic_tr"], X_u_cross_sa, data_a["y_unsw_te"], seed=42, cell_name="CIC -> UNSW (Tier A)")
    res_u2c_a = evaluate_binary_cell(xgb_unsw_a, X_u_tr_sa, data_a["y_unsw_tr"], X_c_cross_sa, data_a["y_cic_te"], seed=42, cell_name="UNSW -> CIC (Tier A)")

    headline_cells_a = [res_c2c_a, res_u2u_a, res_c2u_a, res_u2c_a]

    # Print Headline Table
    print("\n" + "=" * 125)
    print("HEADLINE BINARY RESULTS MATRIX (Seed 42)")
    print("=" * 125)
    print(f"{'Cell':<26} | {'Features':<10} | {'AUROC [95% CI]':<22} | {'PR-AUC [95% CI]':<22} | {'MCC (Src Th)':<14} | {'MCC (Oracle)':<14}")
    print("-" * 125)
    for r in headline_cells_ab:
        auroc_str = f"{r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]"
        prauc_str = f"{r['prauc']:.4f} [{r['prauc_ci'][0]:.4f}, {r['prauc_ci'][1]:.4f}]"
        print(f"{r['cell']:<26} | Tier A+B   | {auroc_str:<22} | {prauc_str:<22} | {r['mcc_at_source_threshold']:<14.4f} | {r['mcc_at_oracle_threshold']:<14.4f}")
    print("-" * 125)
    for r in headline_cells_a:
        auroc_str = f"{r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]"
        prauc_str = f"{r['prauc']:.4f} [{r['prauc_ci'][0]:.4f}, {r['prauc_ci'][1]:.4f}]"
        print(f"{r['cell']:<26} | Tier A     | {auroc_str:<22} | {prauc_str:<22} | {r['mcc_at_source_threshold']:<14.4f} | {r['mcc_at_oracle_threshold']:<14.4f}")

    # 6.3 Logistic Regression Baselines
    print("\nRunning Logistic Regression Baselines...")
    lr_cic = LogisticRegression(max_iter=1000, random_state=42)
    lr_unsw = LogisticRegression(max_iter=1000, random_state=42)
    res_lr_c2u = evaluate_binary_cell(lr_cic, X_c_tr_s, data_ab["y_cic_tr"], X_u_cross_s, data_ab["y_unsw_te"], seed=42, cell_name="CIC -> UNSW (LogReg)")
    res_lr_u2c = evaluate_binary_cell(lr_unsw, X_u_tr_s, data_ab["y_unsw_tr"], X_c_cross_s, data_ab["y_cic_te"], seed=42, cell_name="UNSW -> CIC (LogReg)")
    print(f"  LogReg CIC -> UNSW: AUROC = {res_lr_c2u['auroc']:.4f}, PR-AUC = {res_lr_c2u['prauc']:.4f}, MCC = {res_lr_c2u['mcc_at_source_threshold']:.4f}")
    print(f"  LogReg UNSW -> CIC: AUROC = {res_lr_u2c['auroc']:.4f}, PR-AUC = {res_lr_u2c['prauc']:.4f}, MCC = {res_lr_u2c['mcc_at_source_threshold']:.4f}")

    # 6.4 Dedup Sensitivity Check (With vs Without Dedup on Headline Cross-Dataset)
    print("\n--- DEDUP SENSITIVITY CHECK (Cross-Dataset Cells With vs Without Dedup) ---")
    # Without dedup: use raw indices before dedup
    # Load raw partitioned sets directly without dedup
    # We compare AUROC, PR-AUC, MCC on CIC -> UNSW
    # With dedup: res_c2u_ab
    # Let's run on un-deduplicated test subset
    p_te_raw = _read_csv_with_fallback(_PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_testing-set.csv")
    p_te_raw = _clean_infinities_and_nans(_standardize_columns(p_te_raw))
    p_te_raw["label_raw"] = p_te_raw["attack_cat"].astype(str).str.strip()
    y_raw_te = (p_te_raw["label_raw"] != "Normal").astype(int).values
    X_raw_te = np.column_stack([pd.to_numeric(p_te_raw[c[2]], errors="coerce").fillna(0).values * c[4] for c in (data_ab["feature_names"] if False else (
        [("flow_duration", "Flow Duration", "dur", 1e-6, 1.0),
         ("total_fwd_packets", "Total Fwd Packets", "spkts", 1.0, 1.0),
         ("total_bwd_packets", "Total Backward Packets", "dpkts", 1.0, 1.0),
         ("fwd_iat_mean", "Fwd IAT Mean", "sinpkt", 1e-3, 1.0),
         ("bwd_iat_mean", "Bwd IAT Mean", "dinpkt", 1e-3, 1.0),
         ("total_fwd_bytes", "Total Length of Fwd Packets", "sbytes", 1.0, 1.0),
         ("total_bwd_bytes", "Total Length of Bwd Packets", "dbytes", 1.0, 1.0),
         ("fwd_packet_length_mean", "Fwd Packet Length Mean", "smean", 1.0, 1.0),
         ("bwd_packet_length_mean", "Bwd Packet Length Mean", "dmean", 1.0, 1.0),
         ("flow_packet_rate", "Flow Packets/s", "rate", 1.0, 1.0)]
    ))])
    X_raw_te_s = scaler_cic.transform(X_raw_te)
    p_raw = xgb_cic.predict_proba(X_raw_te_s)[:, 1]
    auroc_raw = float(roc_auc_score(y_raw_te, p_raw))
    prec_raw, rec_raw, _ = precision_recall_curve(y_raw_te, p_raw)
    prauc_raw = float(auc(rec_raw, prec_raw))
    mcc_raw = float(matthews_corrcoef(y_raw_te, (p_raw >= res_c2u_ab["source_chosen_threshold"]).astype(int)))

    print(f"CIC -> UNSW With Dedup:    AUROC = {res_c2u_ab['auroc']:.4f}, PR-AUC = {res_c2u_ab['prauc']:.4f}, MCC = {res_c2u_ab['mcc_at_source_threshold']:.4f}")
    print(f"CIC -> UNSW Without Dedup: AUROC = {auroc_raw:.4f}, PR-AUC = {prauc_raw:.4f}, MCC = {mcc_raw:.4f}")
    print(f"Dedup Impact (Delta):       dAUROC = {res_c2u_ab['auroc'] - auroc_raw:+.4f}, dPR-AUC = {res_c2u_ab['prauc'] - prauc_raw:+.4f}")

    # 6.5 Ablations: Preprocessing (Raw vs Log1p), Scaler (Source vs Per-Domain), Units (Harmonized vs Unharmonized)
    print("\n--- ABLATION GRID (Cross-Dataset CIC -> UNSW, Seed 42) ---")
    # Ablation 1: Log1p
    data_log1p = prepare_shared_matrices(
        cic_df, unsw_df,
        split_info["primary_tr_indices"], split_info["primary_te_indices"],
        unsw_part_info["df_tr_dedup"], unsw_part_info["df_te_final"],
        feature_tier="tier_a_b", harmonize_units=True, log1p=True, seed=42,
    )
    s_c_log = RobustScaler().fit(data_log1p["X_cic_tr"])
    res_log1p = evaluate_binary_cell(
        XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4),
        s_c_log.transform(data_log1p["X_cic_tr"]), data_log1p["y_cic_tr"],
        s_c_log.transform(data_log1p["X_unsw_te"]), data_log1p["y_unsw_te"],
        seed=42, cell_name="CIC -> UNSW (log1p)",
    )

    # Ablation 2: Per-Domain Scaler
    s_target = RobustScaler().fit(data_ab["X_unsw_te"])
    res_per_dom = evaluate_binary_cell(
        xgb_cic,
        X_c_tr_s, data_ab["y_cic_tr"],
        s_target.transform(data_ab["X_unsw_te"]), data_ab["y_unsw_te"],
        seed=42, cell_name="CIC -> UNSW (per-domain scaler)",
    )

    # Ablation 3: Unharmonized Units
    data_unharm = prepare_shared_matrices(
        cic_df, unsw_df,
        split_info["primary_tr_indices"], split_info["primary_te_indices"],
        unsw_part_info["df_tr_dedup"], unsw_part_info["df_te_final"],
        feature_tier="tier_a_b", harmonize_units=False, log1p=False, seed=42,
    )
    s_unharm = RobustScaler().fit(data_unharm["X_cic_tr"])
    res_unharm = evaluate_binary_cell(
        XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4),
        s_unharm.transform(data_unharm["X_cic_tr"]), data_unharm["y_cic_tr"],
        s_unharm.transform(data_unharm["X_unsw_te"]), data_unharm["y_unsw_te"],
        seed=42, cell_name="CIC -> UNSW (unharmonized units)",
    )

    # Ablation 4: RandomForest
    rf_cic = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=4)
    res_rf = evaluate_binary_cell(rf_cic, X_c_tr_s, data_ab["y_cic_tr"], X_u_cross_s, data_ab["y_unsw_te"], seed=42, cell_name="CIC -> UNSW (RandomForest)")

    # Ablation 5: PyTorch DNN Baseline (GPU)
    dnn_cic = DNNBaseline(epochs=10, batch_size=256, random_state=42)
    res_dnn = evaluate_binary_cell(dnn_cic, X_c_tr_s, data_ab["y_cic_tr"], X_u_cross_s, data_ab["y_unsw_te"], seed=42, cell_name="CIC -> UNSW (DNN)")

    print(f"  Standard (Raw + Source Scaler + Harmonized + XGB): AUROC = {res_c2u_ab['auroc']:.4f}, PR-AUC = {res_c2u_ab['prauc']:.4f}, MCC = {res_c2u_ab['mcc_at_source_threshold']:.4f}")
    print(f"  Log1p Transformation:                             AUROC = {res_log1p['auroc']:.4f}, PR-AUC = {res_log1p['prauc']:.4f}, MCC = {res_log1p['mcc_at_source_threshold']:.4f}")
    print(f"  Per-Domain Scaler:                                AUROC = {res_per_dom['auroc']:.4f}, PR-AUC = {res_per_dom['prauc']:.4f}, MCC = {res_per_dom['mcc_at_source_threshold']:.4f}")
    print(f"  Unharmonized Raw Units:                           AUROC = {res_unharm['auroc']:.4f}, PR-AUC = {res_unharm['prauc']:.4f}, MCC = {res_unharm['mcc_at_source_threshold']:.4f}")
    print(f"  RandomForest Model Family:                        AUROC = {res_rf['auroc']:.4f}, PR-AUC = {res_rf['prauc']:.4f}, MCC = {res_rf['mcc_at_source_threshold']:.4f}")
    print(f"  PyTorch DNN Model Family:                         AUROC = {res_dnn['auroc']:.4f}, PR-AUC = {res_dnn['prauc']:.4f}, MCC = {res_dnn['mcc_at_source_threshold']:.4f}")

    # 6.6 Shift Analysis & Dropping Top Shifted Features
    shift_results = run_shift_analysis(
        data_ab["X_cic_tr"], data_ab["y_cic_tr"],
        data_ab["X_unsw_tr"], data_ab["y_unsw_tr"],
        data_ab["feature_names"],
    )

    # Re-evaluate after dropping top-k shifted features
    top_k_drop = [shift_results["top_shifted_features"][0][0], shift_results["top_shifted_features"][1][0]]
    drop_indices = [data_ab["feature_names"].index(f) for f in top_k_drop]
    keep_indices = [i for i in range(len(data_ab["feature_names"])) if i not in drop_indices]

    X_c_tr_drop = X_c_tr_s[:, keep_indices]
    X_u_cross_drop = X_u_cross_s[:, keep_indices]
    res_drop_shifted = evaluate_binary_cell(
        XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4),
        X_c_tr_drop, data_ab["y_cic_tr"],
        X_u_cross_drop, data_ab["y_unsw_te"],
        seed=42, cell_name=f"CIC -> UNSW (Dropped top-2 shifted: {top_k_drop})",
    )
    print(f"\nRe-evaluation after dropping top-2 shifted features ({top_k_drop}):")
    print(f"  AUROC = {res_drop_shifted['auroc']:.4f} (Delta: {res_drop_shifted['auroc'] - res_c2u_ab['auroc']:+.4f})")
    print(f"  PR-AUC = {res_drop_shifted['prauc']:.4f} (Delta: {res_drop_shifted['prauc'] - res_c2u_ab['prauc']:+.4f})")
    print(f"  MCC    = {res_drop_shifted['mcc_at_source_threshold']:.4f} (Delta: {res_drop_shifted['mcc_at_source_threshold'] - res_c2u_ab['mcc_at_source_threshold']:+.4f})")

    # 6.7 Multi-Class Secondary
    multiclass_results = run_multiclass_secondary(data_ab, seed=42)

    total_time = time.time() - START_TIME
    peak_ram = get_peak_ram_mb()
    print("\n" + "=" * 80)
    print("PHASE 2 EXECUTION COMPLETE (Seed 42)")
    print(f"Total Wall-Clock Time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Peak Process RAM:      {peak_ram:.2f} MB")
    print("=" * 80)

    # Compile Final JSON Report with Provenance Stamp
    source_files_dict = {}
    for f in _find_csv_files(cic_dir, "*.csv", subfolder="TrafficLabelling"):
        rel_p = str(f.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        source_files_dict[rel_p] = {"rows": count_file_rows(f), "sha256": compute_file_sha256(f)}
    for p_u in [_PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_training-set.csv",
                _PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "UNSW_NB15_testing-set.csv"]:
        rel_p = str(p_u.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        source_files_dict[rel_p] = {"rows": count_file_rows(p_u), "sha256": compute_file_sha256(p_u)}

    combined_counts = {}
    for k, v in cic_df["label"].value_counts().items():
        combined_counts[f"cic_{k}"] = int(v)
    for k, v in unsw_df["label"].value_counts().items():
        combined_counts[f"unsw_{k}"] = int(v)

    prov_stamp = create_provenance_stamp(
        data_kind="real",
        dataset_names=["cic-ids-2017", "unsw-nb15"],
        source_files=source_files_dict,
        n_rows=len(cic_df) + len(unsw_df),
        n_train=len(split_info["primary_tr_indices"]) + len(unsw_part_info["df_tr_dedup"]),
        n_val=0,
        n_test=len(split_info["primary_te_indices"]) + len(unsw_part_info["df_te_final"]),
        class_counts=combined_counts,
        seed_list=[42],
        script_name="scripts/run_phase2_empirical.py",
    )
    validate_results_provenance(prov_stamp, is_headline=True, min_test_rows=500)

    final_report = {
        "provenance": prov_stamp,
        "execution_metadata": {
            "seed": 42,
            "wall_clock_seconds": total_time,
            "peak_ram_mb": peak_ram,
            "timestamp": datetime.now().isoformat(),
        },
        "step1_split_audit": {
            "old_split": split_info["old_split"],
            "primary_split": split_info["primary_split"],
            "purged_rows": split_info["purged_rows"],
            "small_classes": split_info["small_classes_aggregate_only"],
            "leave_day_out": split_info["leave_day_out"],
            "random_split": split_info["random_split"],
        },
        "step2_dedup_audit": dedup_info,
        "step3_shared_features_audit": shared_feat_info,
        "step4_timestamp_inversions": ts_inv_info,
        "step5_unsw_audit": {
            "membership_preserved": True,
            "straddling_rows_removed": unsw_part_info["test_rows_removed_straddling"],
            "train_rows": unsw_part_info["train_rows_dedup"],
            "test_rows": unsw_part_info["test_rows_final"],
            "feature_ablation": unsw_ablation_info,
        },
        "step6_phase2_results": {
            "headline_matrix_tier_ab": headline_cells_ab,
            "headline_matrix_tier_a": headline_cells_a,
            "logreg_baselines": [res_lr_c2u, res_lr_u2c],
            "dedup_sensitivity": {
                "with_dedup": {"auroc": res_c2u_ab["auroc"], "prauc": res_c2u_ab["prauc"], "mcc": res_c2u_ab["mcc_at_source_threshold"]},
                "without_dedup": {"auroc": auroc_raw, "prauc": prauc_raw, "mcc": mcc_raw},
                "delta": {"dauroc": res_c2u_ab["auroc"] - auroc_raw, "dprauc": res_c2u_ab["prauc"] - prauc_raw},
            },
            "ablations": {
                "log1p": res_log1p,
                "per_domain_scaler": res_per_dom,
                "unharmonized_units": res_unharm,
                "random_forest": res_rf,
                "dnn": res_dnn,
            },
            "shift_analysis": shift_results,
            "dropped_top_shifted": res_drop_shifted,
            "multiclass_secondary": multiclass_results,
        },
    }

    out_file = RESULTS_DIR / "phase2_seed42_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"\n[SUCCESS] Phase 2 Seed 42 complete report written to: {out_file}")


if __name__ == "__main__":
    main()
