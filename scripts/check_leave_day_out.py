import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import _find_csv_files, _read_csv_with_fallback, _standardize_columns

p = Path("data/raw/cic-ids-2017")
files = _find_csv_files(p, "*.csv", subfolder="TrafficLabelling")

mon_wed_days = {"Monday", "Tuesday", "Wednesday"}
thu_fri_days = {"Thursday", "Friday"}

tr_classes = set()
te_classes = set()
tr_counts = {}
te_counts = {}

for f in files:
    df = _read_csv_with_fallback(f)
    df = _standardize_columns(df)
    lbl_col = next(c for c in df.columns if c.lower() == "label")
    df = df[df[lbl_col].notna()].copy()
    raw_labels = df[lbl_col].astype(str).str.strip()
    day = f.name.split("-")[0]
    
    counts = raw_labels.value_counts().to_dict()
    if day in mon_wed_days:
        for k, v in counts.items():
            tr_counts[k] = tr_counts.get(k, 0) + v
            tr_classes.add(k)
    else:
        for k, v in counts.items():
            te_counts[k] = te_counts.get(k, 0) + v
            te_classes.add(k)

unseen = sorted(te_classes - tr_classes)
print("LEAVE-DAY-OUT (Train: Mon-Wed, Test: Thu-Fri)")
print("Unseen attack classes in Test (Novel Attacks):", unseen)
print(f"\n{'Class':<30} | {'Train (Mon-Wed)':<15} | {'Test (Thu-Fri)':<15} | Status")
print("-" * 75)
all_c = sorted(tr_classes | te_classes)
for c in all_c:
    tr = tr_counts.get(c, 0)
    te = te_counts.get(c, 0)
    st = "UNSEEN NOVEL ATTACK" if c in unseen else ("Train-only" if te == 0 else "Shared across days")
    print(f"{c:<30} | {tr:<15,} | {te:<15,} | {st}")

