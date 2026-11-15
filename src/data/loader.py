"""
src/data/loader.py — Unified data loader for NIDS datasets.
Supports real CIC-IDS2017, UNSW-NB15, NF-v2 benchmarks, and lab-simulated traffic.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

# Disable pyarrow backend to avoid access violations on Windows
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.pandas_compat import apply_pandas_compat

apply_pandas_compat()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

# Canonical non-feature metadata candidates that should be tracked separately
METADATA_COLUMN_PATTERNS = [
    "source ip", "destination ip", "timestamp", "source port", "destination port",
    "flow id", "src_ip", "dst_ip", "srcip", "dstip", "sport", "dsport", "dport",
    "stime", "ltime", "time", "date", "ipv4_src_addr", "ipv4_dst_addr",
    "l4_src_port", "l4_dst_port", "id", "protocol", "proto", "service", "state",
]

LEAKAGE_DENYLIST_EXACT = {
    "label", "label_raw", "attack_cat", "attack", "protocol", "proto",
    "service", "state", "id", "flow id", "flow_id", "timestamp", "stime",
    "ltime", "source ip", "destination ip", "source port", "destination port",
    "srcip", "dstip", "sport", "dsport", "dport", "l4_src_port", "l4_dst_port",
    "ipv4_src_addr", "ipv4_dst_addr",
}


def _load_taxonomy() -> set:
    """Return the unified taxonomy declared in configs/label_mappings.yaml."""
    with open(_CONFIGS_DIR / "label_mappings.yaml", "r", encoding="utf-8") as f:
        mappings = yaml.safe_load(f)
    return set(mappings.get("taxonomy", []))


_TAXONOMY = _load_taxonomy()


def _load_label_mappings(dataset_name: str) -> dict:
    """Load label mappings from configs/label_mappings.yaml for a given dataset."""
    config_path = _CONFIGS_DIR / "label_mappings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        mappings = yaml.safe_load(f)
    return mappings.get(dataset_name, {})


def _load_dataset_config(dataset_name: str) -> dict:
    """Load dataset metadata from configs/datasets.yaml."""
    config_path = _CONFIGS_DIR / "datasets.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        datasets = yaml.safe_load(f)
    return datasets.get("datasets", {}).get(dataset_name, {})


def _find_csv_files(local_dir: Path, file_pattern: str = "*.csv",
                    subfolder: Optional[str] = None) -> list[Path]:
    """Find all CSV files matching the pattern in local_dir or subfolders (deduplicating paths)."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {local_dir}")

    seen = set()
    unique_files = []

    def _add_if_new(p: Path):
        key = str(p.resolve()).lower()
        if key not in seen and p.is_file():
            seen.add(key)
            unique_files.append(p.resolve())

    # If subfolder (e.g. TrafficLabelling) exists, prioritize files inside it
    if subfolder:
        target_sub = local_dir / subfolder
        if target_sub.is_dir():
            for p in sorted(target_sub.glob(file_pattern)):
                _add_if_new(p)
            for p in sorted(target_sub.glob(f"**/{file_pattern}")):
                _add_if_new(p)
            if unique_files:
                return sorted(unique_files)

    for p in sorted(local_dir.glob(file_pattern)):
        _add_if_new(p)
    for p in sorted(local_dir.glob(f"**/{file_pattern}")):
        _add_if_new(p)

    if not unique_files:
        raise FileNotFoundError(f"No CSV files found in {local_dir} matching {file_pattern}")
    return sorted(unique_files)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df.columns = [col.strip() for col in df.columns]
    return df


def _extract_and_isolate_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and extract identity/metadata columns (IPs, ports, timestamps, flow IDs)
    into a separate metadata DataFrame, ensuring they are removed from the feature matrix.
    """
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    metadata_cols = []
    for pattern in METADATA_COLUMN_PATTERNS:
        if pattern in col_lower_map:
            metadata_cols.append(col_lower_map[pattern])

    # Also detect any IP-like string columns
    for col in df.columns:
        if col in metadata_cols or col.lower() in ("label", "attack", "attack_cat", "label_raw"):
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and sample.astype(str).str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").all():
                metadata_cols.append(col)

    metadata_cols = list(dict.fromkeys(metadata_cols))
    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    df = df.drop(columns=metadata_cols, errors="ignore")
    return df, metadata_df


def _clean_infinities_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive/negative infinities and infinite string representations with NaN."""
    # Replace in place across dataframe
    inf_representations = [np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf", "INF", "-INF"]
    df = df.replace(inf_representations, np.nan)
    return df


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """Read CSV with encoding fallback: utf-8 -> cp1252 -> latin-1."""
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, low_memory=False, engine="c", encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Could not read {file_path}")


def _parse_cic_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    Parse CIC-IDS2017 timestamps where 12-hour times have no AM/PM.
    Hours 1-7 are afternoon (+12 hours).
    """
    def convert_val(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        parts = s.split(" ")
        if len(parts) != 2:
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        date_part, time_part = parts[0], parts[1]
        time_tokens = time_part.split(":")
        try:
            h = int(time_tokens[0])
            m = int(time_tokens[1])
            sec = int(time_tokens[2]) if len(time_tokens) > 2 else 0
            if 1 <= h <= 7:
                h += 12
            return f"{date_part} {h:02d}:{m:02d}:{sec:02d}"
        except Exception:
            return s

    converted = ts_series.map(convert_val)
    return pd.to_datetime(converted, dayfirst=True, errors="coerce")


def _load_cic_ids_2017(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge CIC-IDS2017 CSV files (supporting TrafficLabelling folder structure).
    - Encoding fallback (utf-8 -> cp1252 -> latin-1)
    - Strip whitespace from column names BEFORE validation
    - Drop blank rows (e.g. 288,602 trailing NaNs in Thursday WebAttacks) BEFORE row count checks
    - Store raw label in 'label_raw'
    - Parse Timestamp (hours 1-7 -> 13-19)
    - Replace +/-inf with NaN
    - Handle duplicate 'Fwd Header Length' column
    """
    csv_files = _find_csv_files(data_dir, "*.csv", subfolder="TrafficLabelling")
    dfs = []
    total_raw_rows = 0
    total_blank_dropped = 0
    file_ts_ranges = {}
    file_ts_stats = {}

    print(f"\nLoading CIC-IDS2017 from {len(csv_files)} files...")
    for f in csv_files:
        df = _read_csv_with_fallback(f)
        raw_count = len(df)
        total_raw_rows += raw_count

        # Strip whitespace BEFORE validation
        df = _standardize_columns(df)

        # Locate label column case-insensitively
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        if not label_col:
            raise KeyError(f"Could not find 'Label' column in {f.name}. Found columns: {list(df.columns)[:5]}")

        # Drop trailing blank rows BEFORE +/-1% row count check and before anything else
        blank_mask = df[label_col].isna() | df.isna().all(axis=1)
        blank_count = int(blank_mask.sum())
        if blank_count > 0:
            df = df.loc[~blank_mask].copy()
        total_blank_dropped += blank_count

        # Store raw label in label_raw
        df["label_raw"] = df[label_col].astype(str).str.strip()

        # Parse Timestamp if present and sort stably with tie-break
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        if ts_col:
            parsed_ts = _parse_cic_timestamp(df[ts_col])
            df[ts_col] = parsed_ts
            ts_min = parsed_ts.min()
            ts_max = parsed_ts.max()
            n_rows = len(df)

            # Local inversions (diff < 0)
            local_inv = int((parsed_ts.diff() < pd.Timedelta(0)).sum())
            frac_local = local_inv / n_rows if n_rows > 0 else 0.0

            # Cumulative out-of-order (ts < cummax)
            cummax = parsed_ts.cummax()
            cum_ooo = int((parsed_ts < cummax).sum())
            frac_cum_ooo = cum_ooo / n_rows if n_rows > 0 else 0.0

            # Tied timestamps
            tied_rows = int(parsed_ts.duplicated(keep=False).sum())
            frac_tied = tied_rows / n_rows if n_rows > 0 else 0.0

            # Sort stably by timestamp with original row order as tie-break
            df["_orig_row"] = np.arange(n_rows)
            df = df.sort_values(by=[ts_col, "_orig_row"], kind="mergesort").reset_index(drop=True)
            df.drop(columns=["_orig_row"], inplace=True)

            file_ts_stats[f.name] = {
                "min": ts_min, "max": ts_max,
                "local_inv": local_inv, "frac_local": frac_local,
                "cum_ooo": cum_ooo, "frac_cum_ooo": frac_cum_ooo,
                "tied_rows": tied_rows, "frac_tied": frac_tied,
            }
            file_ts_ranges[f.name] = (ts_min, ts_max)
            print(f"  - {f.name}: clean={n_rows:,}, ts=[{ts_min} to {ts_max}], "
                  f"out-of-order={cum_ooo:,} ({frac_cum_ooo:.2%}), tied={tied_rows:,} ({frac_tied:.2%})")
        else:
            print(f"  - {f.name}: raw={raw_count:,}, dropped_blanks={blank_count:,}, clean={len(df):,}")

        # Replace infinities
        df = _clean_infinities_and_nans(df)

        # Handle duplicate "Fwd Header Length" column
        if "Fwd Header Length.1" in df.columns:
            df = df.drop(columns=["Fwd Header Length.1"])
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    clean_total = len(merged)
    ref_count = len(merged.drop_duplicates())
    print(f"\nCIC-IDS2017 Cleaning Summary:")
    print(f"  Total raw lines:                  {total_raw_rows:,}")
    print(f"  Total blank rows dropped:         {total_blank_dropped:,}")
    print(f"  Total clean rows:                 {clean_total:,} (Expected: 2,830,743)")
    print(f"  Metadata-inclusive dedup ref:     {ref_count:,}")

    merged.attrs["file_ts_ranges"] = file_ts_ranges
    merged.attrs["timestamp_stats"] = file_ts_stats
    merged.attrs["raw_reference_count"] = ref_count
    return merged


def _load_unsw_nb15(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Load and merge UNSW-NB15 CSV files.
    Supports either:
    1. Official partitioned sets: UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv (variant: 'partitioned')
    2. Raw 4 CSV files (UNSW-NB15_1..4.csv) with column names from NUSW-NB15_features.csv (variant: 'raw_full')
    """
    seen = set()
    def _dedup_paths(paths):
        res = []
        for p in paths:
            k = str(p.resolve()).lower()
            if k not in seen and p.is_file():
                seen.add(k)
                res.append(p.resolve())
        return sorted(res)

    # Check for partitioned variant first
    train_candidates = _dedup_paths(data_dir.glob("*training-set*.csv"))
    test_candidates = _dedup_paths(data_dir.glob("*testing-set*.csv"))

    if train_candidates and test_candidates:
        train_file = train_candidates[0]
        test_file = test_candidates[0]
        logger.info("Loading UNSW-NB15 partitioned set: %s and %s", train_file.name, test_file.name)
        df_train = _read_csv_with_fallback(train_file)
        df_test = _read_csv_with_fallback(test_file)

        raw_total = len(df_train) + len(df_test)
        print(f"\nUNSW-NB15 Partitioned variant:")
        print(f"  - {train_file.name}: {len(df_train):,} rows")
        print(f"  - {test_file.name}: {len(df_test):,} rows")
        print(f"  Total raw rows before deduplication: {raw_total:,}")

        df_train = _clean_infinities_and_nans(_standardize_columns(df_train))
        df_test = _clean_infinities_and_nans(_standardize_columns(df_test))
        merged = pd.concat([df_train, df_test], ignore_index=True)

        # Drop "id" column from features immediately
        id_cols = [c for c in merged.columns if c.strip().lower() == "id"]
        if id_cols:
            merged.drop(columns=id_cols, inplace=True)
            print(f"  Dropped dataset ID column: {id_cols}")

        # Store raw attack_cat in label_raw
        raw_lbl_col = next((c for c in merged.columns if c.strip().lower() in ("attack_cat", "attack", "label")), None)
        if raw_lbl_col:
            merged["label_raw"] = merged[raw_lbl_col].astype(str).str.strip()

        merged.attrs["raw_reference_count"] = raw_total
        return merged, "partitioned"

    # Otherwise load raw 4 CSVs
    raw_files = _dedup_paths(sorted(data_dir.glob("*UNSW-NB15_[1-4]*.csv")))
    if not raw_files:
        # Fallback to general CSV glob excluding features file
        raw_files = _dedup_paths([f for f in sorted(data_dir.glob("*.csv")) if "feature" not in f.name.lower()])

    if not raw_files:
        raise FileNotFoundError(f"No UNSW-NB15 data CSV files found in {data_dir}")

    # Check for NUSW-NB15_features.csv
    features_csv = data_dir / "NUSW-NB15_features.csv"
    if not features_csv.exists():
        found = list(data_dir.glob("*feature*.csv"))
        if found:
            features_csv = found[0]

    col_names = None
    if features_csv.exists():
        logger.info("Reading UNSW-NB15 column names from features file: %s", features_csv.name)
        fdf = _read_csv_with_fallback(features_csv)
        name_col = next((c for c in fdf.columns if c.strip().lower() in ("name", "feature name")), None)
        if name_col:
            col_names = [str(x).strip() for x in fdf[name_col].dropna().tolist()]
        elif len(fdf.columns) >= 2:
            col_names = [str(x).strip() for x in fdf.iloc[:, 1].dropna().tolist()]

    dfs = []
    for f in raw_files:
        df = _read_csv_with_fallback(f)
        df = _standardize_columns(df)
        if col_names and len(df.columns) == len(col_names):
            if not any(c.lower() in ("srcip", "dur", "proto") for c in df.columns):
                df.columns = col_names
        df = _clean_infinities_and_nans(df)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    # Drop id if present
    id_cols = [c for c in merged.columns if c.strip().lower() == "id"]
    if id_cols:
        merged.drop(columns=id_cols, inplace=True)

    raw_lbl_col = next((c for c in merged.columns if c.strip().lower() in ("attack_cat", "attack", "label")), None)
    if raw_lbl_col:
        merged["label_raw"] = merged[raw_lbl_col].astype(str).str.strip()

    return merged, "raw_full"


def _load_nf_v2(data_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load NetFlow-v2 format datasets sharing the 43-feature NetFlow schema."""
    csv_files = _find_csv_files(data_dir, "*.csv")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, engine="c")
        df = _standardize_columns(df)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def _apply_label_mapping(df: pd.DataFrame, dataset_name: str,
                         label_column: str, mappings: dict) -> pd.DataFrame:
    """
    Map raw labels to the unified taxonomy. Unmapped labels become 'Other'.
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame. "
                         f"Available: {list(df.columns)}")

    if not mappings:
        return df

    stripped = df[label_column].astype(str).str.strip()
    mapped = stripped.map(mappings)
    # Preserve already-unified labels so a partially-mapped dataset is not
    # silently collapsed into 'Other'.
    unmapped = mapped.isna() & stripped.isin(_TAXONOMY)
    df["label"] = mapped.where(~mapped.isna() | unmapped, "Other")

    if label_column != "label":
        df.drop(columns=[label_column], inplace=True)
    return df


def _drop_non_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop known non-feature columns (IP addresses, ports, protocols, timestamps)."""
    drop_cols = []
    col_lower = {c.lower(): c for c in df.columns}

    for pattern in METADATA_COLUMN_PATTERNS + ["protocol", "proto", "id"]:
        if pattern in col_lower:
            drop_cols.append(col_lower[pattern])

    for col in df.columns:
        if col in drop_cols or col.lower() in ("label", "attack", "label_raw"):
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and sample.astype(str).str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").all():
                drop_cols.append(col)

    drop_cols = list(set(drop_cols))
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df


def _verify_provenance(df: pd.DataFrame, dataset_name: str,
                       raw_columns: list[str], label_column: str,
                       variant: Optional[str] = None) -> None:
    """
    Verify provenance of real benchmark datasets per scientific validation standards:
    - Real column sets (CIC: Flow ID, Source IP, Timestamp, Label; UNSW: standard features)
    - Real label names (e.g. DoS Hulk, PortScan, Fuzzers, Exploits)
    - Total rows within 1% of documented benchmark counts:
        - CIC-IDS2017: ~2.83M (2,830,743)
        - UNSW-NB15 raw_full: ~2.54M (2,540,044)
        - UNSW-NB15 partitioned: 257,673 (82,332 train + 175,341 test)
    Refuses to run otherwise without loosening checks.
    """
    cols_lower = [c.lower().strip() for c in raw_columns]
    n_rows = len(df)

    if dataset_name == "cic-ids-2017":
        required_cols = {"flow id", "source ip", "timestamp", "label"}
        missing = required_cols - set(cols_lower)
        if missing:
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}': missing real columns {missing}. "
                f"Found columns: {raw_columns[:8]}... Must load the real CIC-IDS2017 TrafficLabelling CSVs."
            )
        # Check real labels using label_raw or case-insensitive search
        if "label_raw" in df.columns:
            lbl_series = df["label_raw"]
        elif label_column in df.columns:
            lbl_series = df[label_column]
        elif "label" in df.columns:
            lbl_series = df["label"]
        else:
            lbl_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
            lbl_series = df[lbl_col] if lbl_col else df[df.columns[-1]]

        unique_labels = set(lbl_series.astype(str).str.strip().unique())
        expected_sample = {"BENIGN", "DoS Hulk", "PortScan", "DDoS"}
        if not (unique_labels & expected_sample):
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}': expected real attack labels "
                f"like 'DoS Hulk', 'PortScan', got {list(unique_labels)[:5]}"
            )
        # Check row count within 1% of 2,830,743 (~2.83M)
        min_expected = int(2_830_743 * 0.99)
        max_expected = int(2_830_743 * 1.01)
        if not (min_expected <= n_rows <= max_expected):
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}': expected total rows within 1% "
                f"of ~2.83M ({min_expected:,} to {max_expected:,}), but got {n_rows:,} rows."
            )

    elif dataset_name == "unsw-nb15":
        required_cols = {"srcip", "sport", "dstip", "dsport", "proto", "dur", "sbytes", "dbytes"}
        has_req = required_cols.issubset(set(cols_lower)) or any(c in cols_lower for c in ("dur", "proto", "sbytes"))
        if not has_req and len(raw_columns) < 40:
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}': real UNSW-NB15 schema "
                f"missing. Found columns: {raw_columns[:8]}"
            )
        # Check label column
        if "label_raw" in df.columns:
            lbl_series = df["label_raw"]
        elif label_column in df.columns:
            lbl_series = df[label_column]
        elif "attack_cat" in df.columns:
            lbl_series = df["attack_cat"]
        elif "label" in df.columns:
            lbl_series = df["label"]
        else:
            lbl_col = next((c for c in df.columns if c.strip().lower() in ("attack_cat", "attack", "label")), None)
            lbl_series = df[lbl_col] if lbl_col else df[df.columns[-1]]

        unique_labels = set(lbl_series.astype(str).str.strip().unique())
        expected_sample = {"Normal", "Fuzzers", "Exploits", "Reconnaissance", "DoS"}
        if not (unique_labels & expected_sample):
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}': expected real attack categories "
                f"like 'Fuzzers', 'Exploits', got {list(unique_labels)[:5]}"
            )
        if variant == "partitioned":
            expected_base = 257_673
        else:
            expected_base = 2_540_044

        min_expected = int(expected_base * 0.99)
        max_expected = int(expected_base * 1.01)
        if not (min_expected <= n_rows <= max_expected):
            raise ValueError(
                f"Provenance verification failed for '{dataset_name}' (variant='{variant}'): "
                f"expected total rows within 1% of {expected_base:,} ({min_expected:,} to {max_expected:,}), "
                f"but got {n_rows:,} rows."
            )


def load_dataset(dataset_name: str, drop_non_features: bool = True,
                 min_rows: int = 5000, enforce_provenance: bool = True) -> pd.DataFrame:
    """
    Load a dataset by name and return a standardized DataFrame.

    Metadata (Source IP, Destination IP, Timestamp, Ports) is extracted and preserved
    in `df.attrs['metadata']`.
    Rows are deduplicated, infinities handled, and per-label counts logged.

    Args:
        dataset_name: One of 'cic-ids-2017', 'unsw-nb15', 'lycos-ids-2017',
                      'ctu-13', 'lab-simulated' (or alias 'lab-captured'),
                      or NetFlow-v2 keys ('nf-unsw-nb15-v2', 'nf-cse-cic-ids2018-v2', 'nf-ton-iot-v2').
        drop_non_features: If True, drops IP/port/protocol/timestamp columns from the feature matrix.
        min_rows: Minimum rows required; fails if dataset is smaller than this threshold.
        enforce_provenance: If True, verifies benchmark schemas, labels, and exact ~1% row counts.

    Returns:
        pd.DataFrame with standardized columns, unified 'label' column, and `df.attrs['metadata']`.
    """
    # Canonicalize legacy aliases
    normalized_name = dataset_name
    if dataset_name == "lab-captured":
        normalized_name = "lab-simulated"

    ds_config = _load_dataset_config(normalized_name)
    if not ds_config:
        # Fallback to literal lookup if not mapped
        ds_config = _load_dataset_config(dataset_name)
    if not ds_config:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available in configs/datasets.yaml")

    local_dir = _PROJECT_ROOT / ds_config.get("local_dir", "")
    label_column = ds_config.get("label_column", "label")
    file_pattern = ds_config.get("file_pattern", "*.csv")

    # If local_dir has no CSV files, allow test fixtures only when enforce_provenance=False
    fixture_dir = _PROJECT_ROOT / "tests" / "fixtures" / normalized_name
    has_local_csv = local_dir.exists() and (any(local_dir.glob("*.csv")) or any(local_dir.glob("**/*.csv")))
    if not has_local_csv and not enforce_provenance and fixture_dir.exists():
        if any(fixture_dir.glob("*.csv")) or any(fixture_dir.glob("**/*.csv")):
            logger.info("Using test fixture directory for '%s': %s", dataset_name, fixture_dir)
            local_dir = fixture_dir

    variant = "standard"
    # Load raw data based on dataset family
    if normalized_name == "cic-ids-2017":
        df = _load_cic_ids_2017(local_dir)
        variant = "traffic_labelling"
    elif normalized_name == "unsw-nb15":
        df, variant = _load_unsw_nb15(local_dir)
    elif normalized_name.startswith("nf-"):
        df = _load_nf_v2(local_dir, normalized_name)
        variant = "netflow_v2"
    else:
        csv_files = _find_csv_files(local_dir, file_pattern)
        dfs = [_read_csv_with_fallback(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df = _clean_infinities_and_nans(_standardize_columns(df))
        variant = "standard"

    raw_columns = list(df.columns)

    # Provenance verification for benchmark datasets
    if enforce_provenance and normalized_name in ("cic-ids-2017", "unsw-nb15"):
        _verify_provenance(df, normalized_name, raw_columns, label_column, variant=variant)

    df.attrs["variant"] = variant

    # 1. Deduplicate rows
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    n_after_dedup = len(df)
    if n_after_dedup < n_before_dedup:
        logger.info("Dataset '%s': dropped %d duplicate rows (%d -> %d)",
                    dataset_name, n_before_dedup - n_after_dedup, n_before_dedup, n_after_dedup)

    # 2. Extract and preserve metadata in df.attrs['metadata']
    df, metadata_df = _extract_and_isolate_metadata(df)
    df.attrs["metadata"] = metadata_df

    # 3. Handle inf / NaN
    df = _clean_infinities_and_nans(df)

    # 4. Apply label mapping
    mappings = _load_label_mappings(normalized_name) or _load_label_mappings(dataset_name)
    if "label_raw" in df.columns and mappings:
        stripped = df["label_raw"].astype(str).str.strip()
        mapped = stripped.map(mappings)
        unmapped = mapped.isna() & stripped.isin(_TAXONOMY)
        df["label"] = mapped.where(~mapped.isna() | unmapped, "Other")
    elif mappings and label_column in df.columns:
        df["label_raw"] = df[label_column].astype(str).str.strip()
        df = _apply_label_mapping(df, normalized_name, label_column, mappings)
    elif "label" not in df.columns:
        for col in df.columns:
            if col.lower() in ("label", "attack_cat", "attack", "class"):
                df["label_raw"] = df[col].astype(str).str.strip()
                df = _apply_label_mapping(df, normalized_name, col, mappings)
                break

    # 5. Drop non-feature columns from the feature matrix
    if drop_non_features:
        df = _drop_non_features(df, normalized_name)

    # 6. Log label counts
    if "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        logger.info("Dataset '%s' label counts (%d total):\n%s",
                    dataset_name, len(df), counts)

    # 7. Fail if dataset has fewer than min_rows
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset '{dataset_name}' has only {len(df)} rows; "
            f"minimum {min_rows} rows required for scientific validity."
        )

    return df


def split_dataset(df: pd.DataFrame, strategy: str = "time_aware",
                  test_ratio: float = 0.2, val_ratio: float = 0.1,
                  seed: int = 42, time_column: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Split a DataFrame into train, val, and test subsets using either a time-aware
    or stratified random protocol.

    Args:
        df: Input DataFrame with 'label' column. May carry `df.attrs['metadata']`.
        strategy: 'time_aware' (primary) or 'random' / 'stratified' (comparison).
        test_ratio: Ratio for test split.
        val_ratio: Ratio for validation split.
        seed: Random seed for random/stratified splits.
        time_column: Optional specific column name to use for time ordering.

    Returns:
        dict with keys 'train', 'val', 'test', each containing a subset DataFrame
        with preserved metadata in `.attrs['metadata']`.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame")

    metadata = df.attrs.get("metadata", None)

    if strategy in ("time_aware", "time_day_aware"):
        # Look for time column in df or metadata
        time_series = None
        if time_column:
            if time_column in df.columns:
                time_series = df[time_column]
            elif metadata is not None and time_column in metadata.columns:
                time_series = metadata[time_column]

        if time_series is None:
            # Search candidate names
            candidates = ["timestamp", "time", "stime", "ltime", "date", "day", "flow_start_time"]
            source = df if metadata is None else pd.concat([df, metadata], axis=1)
            lower_to_orig = {c.lower().strip(): c for c in source.columns}
            for cand in candidates:
                if cand in lower_to_orig:
                    time_series = source[lower_to_orig[cand]]
                    break

        if time_series is not None:
            # Parse time values and sort chronologically
            try:
                parsed_time = pd.to_datetime(time_series, errors="coerce")
                if parsed_time.notna().sum() > len(parsed_time) * 0.5:
                    sorted_order = parsed_time.argsort()
                else:
                    sorted_order = pd.to_numeric(time_series, errors="coerce").argsort()
            except Exception:
                sorted_order = np.arange(len(df))
        else:
            logger.warning(
                "No timestamp found in features or metadata; "
                "time_aware split falling back to chronological index order."
            )
            sorted_order = np.arange(len(df))

        n_total = len(df)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_test - n_val

        train_indices = sorted_order[:n_train]
        val_indices = sorted_order[n_train:n_train + n_val]
        test_indices = sorted_order[n_train + n_val:]

    elif strategy in ("random", "stratified"):
        rows = np.arange(len(df))
        labels = df["label"].to_numpy() if "label" in df.columns else None

        # Check if stratification is possible
        can_stratify = False
        if labels is not None:
            _, counts = np.unique(labels, return_counts=True)
            can_stratify = int(counts.min()) >= 2

        strat_arg = labels if can_stratify else None
        temp_idx, test_indices = train_test_split(
            rows, test_size=test_ratio, random_state=seed, stratify=strat_arg
        )

        if val_ratio > 0.0:
            val_relative = val_ratio / (1.0 - test_ratio)
            temp_strat = labels[temp_idx] if can_stratify else None
            train_indices, val_indices = train_test_split(
                temp_idx, test_size=val_relative, random_state=seed, stratify=temp_strat
            )
        else:
            train_indices = temp_idx
            val_indices = np.array([], dtype=int)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. Use 'time_aware' or 'random'.")

    # Slice and attach metadata
    splits = {}
    for key, idxs in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        subset = df.iloc[idxs].copy()
        if metadata is not None:
            subset.attrs["metadata"] = metadata.iloc[idxs].copy()
        splits[key] = subset

    return splits
