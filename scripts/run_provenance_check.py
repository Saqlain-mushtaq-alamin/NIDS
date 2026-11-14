"""
scripts/run_provenance_check.py — Direct provenance verification tool.

Checks data/raw/cic-ids-2017 and data/raw/unsw-nb15:
- Per-file row counts and SHA256
- Columns found (with whitespace stripped)
- Label distribution
- Validates against benchmark provenance guards without loosening checks.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loader import load_dataset, _find_csv_files, _read_csv_with_fallback, _standardize_columns
from src.utils.provenance import compute_file_sha256, count_file_rows


def audit_directory(dataset_name: str, target_dir: Path) -> dict:
    print(f"\n=======================================================")
    print(f"AUDITING PROVENANCE: {dataset_name} in {target_dir}")
    print(f"=======================================================")

    if not target_dir.exists():
        print(f"[FAIL] Directory does not exist: {target_dir}")
        return {"status": "FAILED", "error": f"Directory missing: {target_dir}"}

    csv_files = list(target_dir.glob("*.csv")) + list(target_dir.glob("**/*.csv"))
    if not csv_files:
        print(f"[FAIL] No CSV files found in {target_dir}")
        return {"status": "FAILED", "error": f"No CSV files found in {target_dir}"}

    print(f"Found {len(csv_files)} CSV file(s):")
    total_rows = 0
    file_stats = []
    for f in csv_files:
        sha = compute_file_sha256(f)
        rows = count_file_rows(f)
        total_rows += rows
        file_stats.append({"path": str(f.relative_to(_PROJECT_ROOT)), "rows": rows, "sha256": sha})
        print(f"  - {f.name}: {rows:,} data rows | sha256: {sha}")

    print(f"\nTotal raw data lines across files: {total_rows:,}")

    # Test loading and provenance validation
    try:
        print("\nRunning loader with enforce_provenance=True...")
        df = load_dataset(dataset_name, drop_non_features=False, enforce_provenance=True)
        print(f"[PASS] Successfully loaded '{dataset_name}' with {len(df):,} deduplicated rows!")
        print(f"Variant loaded: {df.attrs.get('variant', 'standard')}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)[:10]}...")
        if "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
            print("Unified label counts:")
            for lbl, cnt in counts.items():
                print(f"  {lbl}: {cnt:,}")
        return {"status": "PASSED", "rows": len(df), "files": file_stats}
    except Exception as e:
        print(f"\n[FAIL] Exact Provenance Failure for '{dataset_name}':")
        print(f"  Error: {type(e).__name__}: {e}")
        return {"status": "FAILED", "error": str(e), "files": file_stats}


def main():
    raw_root = _PROJECT_ROOT / "data" / "raw"
    cic_dir = raw_root / "cic-ids-2017"
    unsw_dir = raw_root / "unsw-nb15"

    res_cic = audit_directory("cic-ids-2017", cic_dir)
    res_unsw = audit_directory("unsw-nb15", unsw_dir)

    print("\n=======================================================")
    print("PROVENANCE AUDIT SUMMARY")
    print("=======================================================")
    print(f"CIC-IDS2017: {res_cic['status']}")
    if res_cic['status'] == "FAILED":
        print(f"  Reason: {res_cic.get('error')}")
    print(f"UNSW-NB15:   {res_unsw['status']}")
    if res_unsw['status'] == "FAILED":
        print(f"  Reason: {res_unsw.get('error')}")


if __name__ == "__main__":
    main()

