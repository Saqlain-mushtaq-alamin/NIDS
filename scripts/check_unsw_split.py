import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import _read_csv_with_fallback, _standardize_columns, _clean_infinities_and_nans

p_train = Path("data/raw/unsw-nb15/UNSW_NB15_training-set.csv")
p_test = Path("data/raw/unsw-nb15/UNSW_NB15_testing-set.csv")

df_tr = _read_csv_with_fallback(p_train)
df_te = _read_csv_with_fallback(p_test)

df_tr = _clean_infinities_and_nans(_standardize_columns(df_tr))
df_te = _clean_infinities_and_nans(_standardize_columns(df_te))

if "id" in df_tr.columns:
    df_tr.drop(columns=["id"], inplace=True)
if "id" in df_te.columns:
    df_te.drop(columns=["id"], inplace=True)

# Map labels
from src.data.loader import _load_label_mappings
mapping = _load_label_mappings("unsw-nb15")

df_tr["label_raw"] = df_tr["attack_cat"].astype(str).str.strip()
df_te["label_raw"] = df_te["attack_cat"].astype(str).str.strip()
df_tr["label"] = df_tr["label_raw"].map(mapping).fillna("Other")
df_te["label"] = df_te["label_raw"].map(mapping).fillna("Other")

feature_cols = [c for c in df_tr.columns if c not in ("label", "label_raw", "attack_cat", "proto", "service", "state")]

print(f"Raw UNSW Train: {len(df_tr):,}, Raw UNSW Test: {len(df_te):,}")
print("Train raw label counts:\n", df_tr["label_raw"].value_counts())
print("\nTest raw label counts:\n", df_te["label_raw"].value_counts())

# Dedup train internally
dedup_subset = feature_cols + ["label_raw"]
df_tr_dedup = df_tr.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

# Dedup test internally
df_te_dedup = df_te.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

print(f"\nAfter internal dedup: Train={len(df_tr_dedup):,} (dropped {len(df_tr)-len(df_tr_dedup):,}), Test={len(df_te_dedup):,} (dropped {len(df_te)-len(df_te_dedup):,})")

# Remove test rows duplicated in train (prevent leakage)
# Using feature_cols for straddling check
tr_feature_tuples = set(tuple(x) for x in df_tr_dedup[feature_cols].itertuples(index=False, name=None))
te_mask_in_tr = [tuple(x) in tr_feature_tuples for x in df_te_dedup[feature_cols].itertuples(index=False, name=None)]
n_straddled = sum(te_mask_in_tr)
df_te_final = df_te_dedup.loc[~pd.Series(te_mask_in_tr, index=df_te_dedup.index)].reset_index(drop=True)

print(f"\nTest rows feature-identical to train (removed from test): {n_straddled:,}")
print(f"Final UNSW Train: {len(df_tr_dedup):,}, Final UNSW Test: {len(df_te_final):,}")
print("\nFinal Train per-class counts:\n", df_tr_dedup["label_raw"].value_counts())
print("\nFinal Test per-class counts:\n", df_te_final["label_raw"].value_counts())
