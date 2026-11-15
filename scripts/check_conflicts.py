import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import load_dataset

cic = load_dataset("cic-ids-2017", drop_non_features=True, enforce_provenance=True)
unsw = load_dataset("unsw-nb15", drop_non_features=True, enforce_provenance=True)

# 1. CIC conflicting labels
cic_feats = [c for c in cic.columns if c not in ("label", "label_raw")]
cic_conflicts = cic.groupby(cic_feats)["label"].nunique()
cic_conf_groups = (cic_conflicts > 1).sum()
print(f"CIC feature-identical groups with conflicting unified labels: {cic_conf_groups:,}")

# 2. UNSW conflicting labels
unsw_feats = [c for c in unsw.columns if c not in ("label", "label_raw")]
unsw_conflicts = unsw.groupby(unsw_feats)["label"].nunique()
unsw_conf_groups = (unsw_conflicts > 1).sum()
print(f"UNSW feature-identical groups with conflicting unified labels: {unsw_conf_groups:,}")

