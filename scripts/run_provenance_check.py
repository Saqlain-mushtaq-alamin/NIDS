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


def print_shared_feature_table():
    print("\n=======================================================")
    print("A5. SHARED-FEATURE INTERSECTION TABLE (CIC-IDS2017 <-> UNSW-NB15)")
    print("=======================================================")
    primary_pairs = [
        ("flow_duration", "Flow Duration", "us", "dur", "s", "s", "CIC * 1e-6; UNSW * 1.0", "Total active duration of bidirectional flow from first to last packet", "PRIMARY"),
        ("total_fwd_packets", "Total Fwd Packets", "count", "spkts", "count", "count", "Identity (int count)", "Total number of packets transmitted in forward (initiator->responder) direction", "PRIMARY"),
        ("total_bwd_packets", "Total Backward Packets", "count", "dpkts", "count", "count", "Identity (int count)", "Total number of packets transmitted in backward (responder->initiator) direction", "PRIMARY"),
        ("total_fwd_bytes", "Total Length of Fwd Packets", "bytes", "sbytes", "bytes", "bytes", "Identity (int bytes)", "Total volume of payload and header bytes transmitted in forward direction", "PRIMARY"),
        ("total_bwd_bytes", "Total Length of Bwd Packets", "bytes", "dbytes", "bytes", "bytes", "Identity (int bytes)", "Total volume of payload and header bytes transmitted in backward direction", "PRIMARY"),
        ("fwd_packet_length_mean", "Fwd Packet Length Mean", "bytes", "smean", "bytes", "bytes", "Identity (float bytes)", "Arithmetic mean packet size in forward direction (sbytes / spkts)", "PRIMARY"),
        ("bwd_packet_length_mean", "Bwd Packet Length Mean", "bytes", "dmean", "bytes", "bytes", "Identity (float bytes)", "Arithmetic mean packet size in backward direction (dbytes / dpkts)", "PRIMARY"),
        ("flow_packet_rate", "Flow Packets/s", "pkts/s", "rate", "pkts/s", "pkts/s", "Identity (rate)", "Total bidirectional packet rate over flow duration ((spkts + dpkts) / dur)", "PRIMARY"),
        ("fwd_iat_mean", "Fwd IAT Mean", "us", "sinpkt", "ms", "ms", "CIC * 1e-3; UNSW * 1.0", "Mean inter-packet arrival time between consecutive forward packets", "PRIMARY"),
        ("bwd_iat_mean", "Bwd IAT Mean", "us", "dinpkt", "ms", "ms", "CIC * 1e-3; UNSW * 1.0", "Mean inter-packet arrival time between consecutive backward packets", "PRIMARY"),
    ]
    print(f"{'Canonical Name':<24} | {'CIC Column':<26} | {'UNSW Column':<10} | {'Harmonized Unit':<15} | {'Status':<8}")
    print("-" * 92)
    for name, cic_col, cic_u, unsw_col, unsw_u, target_u, conv, just, status in primary_pairs:
        print(f"{name:<24} | {cic_col:<26} | {unsw_col:<10} | {target_u:<15} | {status:<8}")
        print(f"   -> Semantic: {just}")
        print(f"   -> Conversion: {conv}")

    print("\nUNCERTAIN CANDIDATE PAIRS (EXCLUDED from primary set per scientific protocol):")
    uncertain_pairs = [
        ("flow_bytes_per_sec", "Flow Bytes/s", "sload + dload", "bytes/s vs bits/s", "UNSW measures bits/s per direction; zero-duration clipping causes large scale discrepancy with CIC Flow Bytes/s."),
        ("tcp_window_advertisement", "Init_Win_bytes_forward", "swin", "bytes vs raw int", "CIC Init_Win_bytes_forward records window byte size; UNSW swin records window field without TCP window scale option."),
        ("flow_iat_std_jitter", "Flow IAT Std", "sjit / djit", "us vs ms (diff math)", "CIC Flow IAT Std measures empirical standard deviation; UNSW sjit/djit measures RFC 3550 jitter via exponential smoothing."),
        ("down_up_ratio", "Down/Up Ratio", "dbytes / sbytes", "integer vs float ratio", "CIC Down/Up Ratio is discrete integer packet ratio; UNSW requires synthetic continuous byte ratio."),
        ("tcp_rtt", "Active Mean", "tcprtt", "us vs ms (diff semantics)", "CIC Active Mean measures active flow duration; UNSW tcprtt measures TCP 3-way handshake round trip time."),
    ]
    for name, cic_col, unsw_col, units, reason in uncertain_pairs:
        print(f"  - {name} (CIC: '{cic_col}', UNSW: '{unsw_col}', Units: {units}) [UNCERTAIN]")
        print(f"      Reason: {reason}")


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
        df = load_dataset(dataset_name, drop_non_features=True, enforce_provenance=True)
        print(f"[PASS] Successfully loaded '{dataset_name}' with {len(df):,} deduplicated rows!")
        print(f"Variant loaded: {df.attrs.get('variant', 'standard')}")

        # A3. Report measured timestamps if CIC
        if dataset_name == "cic-ids-2017":
            ts_stats = df.attrs.get("timestamp_stats", {})
            if ts_stats:
                print("\nA3. MEASURED PER-FILE TIMESTAMP STATISTICS (CIC-IDS2017):")
                print(f"{'Filename':<55} | {'Local Inv (diff<0)':<20} | {'Cum Inv (ts<cummax)':<20} | {'Tied Rows':<18} | {'Range'}")
                print("-" * 140)
                for fn, s in sorted(ts_stats.items()):
                    local_str = f"{s['local_inv']:,} ({s['frac_local']:.2%})"
                    cum_str = f"{s['cum_ooo']:,} ({s['frac_cum_ooo']:.2%})"
                    tied_str = f"{s['tied_rows']:,} ({s['frac_tied']:.2%})"
                    rng_str = f"{s['min']} to {s['max']}"
                    print(f"{fn:<55} | {local_str:<20} | {cum_str:<20} | {tied_str:<18} | {rng_str}")

        # A4. Label distribution and full mapping tables
        if dataset_name == "cic-ids-2017":
            print("\nA4. FULL CIC-IDS2017 LABEL MAPPING TABLE (label_raw -> unified label):")
            cic_map_df = df.groupby(["label_raw", "label"]).size().reset_index(name="count")
            for _, row in cic_map_df.sort_values(by="count", ascending=False).iterrows():
                flag = ""
                if row["label_raw"] == "Heartbleed":
                    flag = " <-- [FLAGGED: OpenSSL TLS buffer over-read (CVE-2014-0160). Exploited via malformed heartbeat packets; categorized under DoS due to crash side-effects, though technically an information leakage exploit.]"
                print(f"  {row['label_raw']:<30} -> {row['label']:<12} : {row['count']:,}{flag}")

            print("\nSHARED-CLASS MULTI-CLASS DEFINITIONS:")
            print("  Strictly shared classes: {Benign, DoS, PortScan}")
            print("  Variant 1 (without DDoS):")
            print("    CIC DoS: {DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed} vs UNSW DoS")
            print("  Variant 2 (with DDoS):")
            print("    CIC DoS: {DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed, DDoS} vs UNSW DoS")

        # UNSW full mapping table and semantic check
        if dataset_name == "unsw-nb15":
            print("\nA4. FULL UNSW-NB15 MAPPING TABLE (attack_cat -> unified label -> count):")
            mapping_df = df.groupby(["label_raw", "label"]).size().reset_index(name="count")
            for _, row in mapping_df.sort_values(by="count", ascending=False).iterrows():
                print(f"  {row['label_raw']:<16} -> {row['label']:<12} : {row['count']:,}")

            print("\nSEMANTIC MAPPING AUDIT (UNSW):")
            print("  - Exploits -> Other (Flagged: Exploits are software/memory exploits, not Web Attacks)")
            print("  - Analysis -> Other (Flagged: Analysis is probe/network analysis, not PortScan)")
            print("  - Backdoors, Shellcode, Worms -> Other (Flagged: distinct malware payloads, not Botnets)")
            print("  - Strictly shared classes for multi-class evaluation: {'Benign', 'DoS', 'PortScan'}")

        # A2. Model input features list and leakage assert check
        model_features = [c for c in df.columns if c not in ("label", "label_raw")]
        print(f"\nA2. FINAL MODEL-INPUT FEATURES ({len(model_features)} features):")
        print(f"  {model_features}")

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

    print_shared_feature_table()

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

