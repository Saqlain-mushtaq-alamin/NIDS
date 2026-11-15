import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import _find_csv_files, _read_csv_with_fallback, _standardize_columns

p = Path('data/raw/cic-ids-2017')
files = _find_csv_files(p, '*.csv', subfolder='TrafficLabelling')

train_counts = {}
test_counts = {}

for f in files:
    df = _read_csv_with_fallback(f)
    df = _standardize_columns(df)
    lbl_col = next(c for c in df.columns if c.lower() == 'label')
    df = df[df[lbl_col].notna()].copy()
    raw_labels = df[lbl_col].astype(str).str.strip()
    
    n = len(df)
    n_tr = int(n * 0.70)
    
    tr_lbls = raw_labels.iloc[:n_tr].value_counts().to_dict()
    te_lbls = raw_labels.iloc[n_tr:].value_counts().to_dict()
    
    for k, v in tr_lbls.items():
        train_counts[k] = train_counts.get(k, 0) + v
    for k, v in te_lbls.items():
        test_counts[k] = test_counts.get(k, 0) + v

all_classes = sorted(set(train_counts) | set(test_counts))
print("OLD PER-FILE PROTOCOL (70/30 chronological per file):")
print(f"{'Class':<30} | {'Train':<10} | {'Test':<10} | {'% Test':<8}")
print("-" * 65)
for c in all_classes:
    tr = train_counts.get(c, 0)
    te = test_counts.get(c, 0)
    tot = tr + te
    pct = (te / tot * 100) if tot > 0 else 0
    print(f"{c:<30} | {tr:<10,} | {te:<10,} | {pct:6.2f}%")
