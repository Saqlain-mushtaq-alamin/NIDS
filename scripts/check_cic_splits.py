import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset, _find_csv_files, _read_csv_with_fallback, _standardize_columns, _parse_cic_timestamp, _clean_infinities_and_nans

print("Testing Primary CIC Split Protocol...")

# Load raw CIC with timestamps and file/day tracking
p = Path("data/raw/cic-ids-2017")
files = _find_csv_files(p, "*.csv", subfolder="TrafficLabelling")

dfs = []
for f in files:
    df = _read_csv_with_fallback(f)
    df = _standardize_columns(df)
    lbl_col = next(c for c in df.columns if c.lower() == "label")
    df = df[df[lbl_col].notna()].copy()
    df["label_raw"] = df[lbl_col].astype(str).str.strip()
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    df["_parsed_ts"] = _parse_cic_timestamp(df[ts_col])
    df["_file"] = f.name
    # Extract day from filename
    day = f.name.split("-")[0] # e.g. Friday, Monday, Thursday, Tuesday, Wednesday
    df["_day"] = day
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged["_orig_row"] = np.arange(len(merged))
# Stably sort by parsed timestamp and orig_row
merged = merged.sort_values(by=["_parsed_ts", "_orig_row"], kind="mergesort").reset_index(drop=True)
print(f"Total merged clean CIC rows: {len(merged):,}")

# Primary split:
# For each label_raw class (except BENIGN):
# first 70% train, last 30% test, with purge gap
train_indices = []
test_indices = []
purged_indices = []

for c in merged["label_raw"].unique():
    if c == "BENIGN":
        continue
    c_df = merged[merged["label_raw"] == c]
    n_c = len(c_df)
    if n_c < 2:
        train_indices.extend(c_df.index)
        continue
    
    split_pos = int(n_c * 0.70)
    split_pos = max(1, min(n_c - 1, split_pos)) # ensure at least 1 in each
    split_ts = c_df["_parsed_ts"].iloc[split_pos]
    
    # Boundary minute window: 1 minute on either side
    # But do not purge all rows if class is tiny (n_c < 50)
    if n_c >= 50:
        boundary_mask = (c_df["_parsed_ts"] >= split_ts - pd.Timedelta(minutes=1)) & \
                        (c_df["_parsed_ts"] <= split_ts + pd.Timedelta(minutes=1))
        # ensure boundary mask doesn't eat the entire train or test
        tr_part = c_df.iloc[:split_pos]
        te_part = c_df.iloc[split_pos:]
        
        tr_keep = tr_part[tr_part["_parsed_ts"] < split_ts - pd.Timedelta(minutes=1)]
        te_keep = te_part[te_part["_parsed_ts"] > split_ts + pd.Timedelta(minutes=1)]
        
        # If purge window is too wide for a burst, fall back to pure index split with dropped exact boundary timestamp
        if len(tr_keep) == 0:
            tr_keep = tr_part.iloc[:-1] if len(tr_part) > 1 else tr_part
        if len(te_keep) == 0:
            te_keep = te_part.iloc[1:] if len(te_part) > 1 else te_part
            
        purged = c_df.index.difference(tr_keep.index.union(te_keep.index))
        train_indices.extend(tr_keep.index)
        test_indices.extend(te_keep.index)
        purged_indices.extend(purged)
    else:
        # For classes < 50, ensure at least 1 row in train and 1 in test
        tr_part = c_df.iloc[:split_pos]
        te_part = c_df.iloc[split_pos:]
        train_indices.extend(tr_part.index)
        test_indices.extend(te_part.index)

# For BENIGN: split chronologically per day
benign_df = merged[merged["label_raw"] == "BENIGN"]
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

tr_df = merged.loc[train_indices]
te_df = merged.loc[test_indices]

print(f"\nPurge Gap Rows Dropped: {len(purged_indices):,}")
print(f"CIC Primary Train Rows: {len(tr_df):,}, Test Rows: {len(te_df):,}")

print(f"\n{'Class':<30} | {'Train':<10} | {'Test':<10} | {'% Test':<8} | Notes")
print("-" * 75)
for c in sorted(merged["label_raw"].unique()):
    tr_c = (tr_df["label_raw"] == c).sum()
    te_c = (te_df["label_raw"] == c).sum()
    tot = tr_c + te_c
    pct = (te_c / tot * 100) if tot > 0 else 0
    note = "Aggregate binary only (<50)" if te_c < 50 else "Full evaluation"
    print(f"{c:<30} | {tr_c:<10,} | {te_c:<10,} | {pct:6.2f}% | {note}")
    assert tr_c > 0, f"Class {c} has 0 train rows!"
    assert te_c > 0, f"Class {c} has 0 test rows!"

print("\n[PASS] All classes present in both Train and Test!")

