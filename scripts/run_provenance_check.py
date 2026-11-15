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

    # Canonical deduplication of files
    seen = set()
    csv_files = []
    for f in sorted(target_dir.rglob("*.csv")):
        key = str(f.resolve()).lower()
        if key not in seen and f.is_file():
            seen.add(key)
            csv_files.append(f.resolve())

    if not csv_files:
        print(f"[FAIL] No CSV files found in {target_dir}")
        return {"status": "FAILED", "error": f"No CSV files found in {target_dir}"}

    print(f"Found {len(csv_files)} unique CSV file(s):")
    total_raw_rows = 0
    file_stats = []
    for f in csv_files:
        sha = compute_file_sha256(f)
        rows = count_file_rows(f)
        total_raw_rows += rows
        file_stats.append({"path": str(f.relative_to(_PROJECT_ROOT)), "rows": rows, "sha256": sha})
        print(f"  - {f.name}: {rows:,} data rows | sha256: {sha}")

    print(f"\nTotal raw data lines before deduplication: {total_raw_rows:,}")

    # Test loading and provenance validation
    try:
        print("\nRunning loader with enforce_provenance=True...")
        df = load_dataset(dataset_name, drop_non_features=False, enforce_provenance=True)
        print(f"[PASS] Successfully loaded '{dataset_name}' with {len(df):,} deduplicated rows!")
        print(f"Variant loaded: {df.attrs.get('variant', 'standard')}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)[:10]}...")

        # Report timestamps if CIC
        if dataset_name == "cic-ids-2017":
            ts_ranges = df.attrs.get("file_ts_ranges", {})
            if ts_ranges:
                print("\nPer-file Timestamp Ranges (Hours 1-7 afternoon normalized):")
                for fn, (tmin, tmax) in sorted(ts_ranges.items()):
                    print(f"  - {fn}: {tmin} to {tmax}")

        # Label distribution
        if "label_raw" in df.columns:
            raw_counts = df["label_raw"].value_counts().to_dict()
            print("\nRaw label counts (label_raw):")
            for lbl, cnt in raw_counts.items():
                print(f"  {lbl}: {cnt:,}")

        if "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
            print("\nUnified label counts (label):")
            for lbl, cnt in counts.items():
                print(f"  {lbl}: {cnt:,}")

        # UNSW full mapping table and semantic check
        if dataset_name == "unsw-nb15" and "label_raw" in df.columns:
            print("\nUNSW Mapping Table (attack_cat -> unified label -> count):")
            mapping_df = df.groupby(["label_raw", "label"]).size().reset_index(name="count")
            for _, row in mapping_df.sort_values(by="count", ascending=False).iterrows():
                print(f"  {row['label_raw']:<16} -> {row['label']:<12} : {row['count']:,}")

            # Check strictly shared classes
            shared_classes = {"Benign", "DoS", "PortScan"}
            present_shared = set(counts.keys()) & shared_classes
            print(f"\nStrictly shared classes present for cross-dataset evaluation: {present_shared}")
            other_attacks = [k for k in counts.keys() if k not in shared_classes and k != "Benign"]
            print(f"Non-shared attack categories mapped to Other: {other_attacks}")

        return {"status": "PASSED", "rows": len(df), "files": file_stats}
    except Exception as e:
        import traceback
        print(f"\n[FAIL] Exact Provenance Failure for '{dataset_name}':")
        traceback.print_exc()
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

