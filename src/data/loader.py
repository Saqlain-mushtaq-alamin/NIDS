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
    "l4_src_port", "l4_dst_port",
]


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
    """Find all CSV files matching the pattern in local_dir or subfolders."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {local_dir}")

    # If subfolder (e.g. TrafficLabelling) exists, prioritize files inside it
    if subfolder:
        target_sub = local_dir / subfolder
        if target_sub.is_dir():
            csv_files = sorted(target_sub.glob(file_pattern))
            if csv_files:
                return csv_files

    # Try local directory
    csv_files = sorted(local_dir.glob(file_pattern))
    if not csv_files:
        # Try recursive search
        csv_files = sorted(local_dir.glob(f"**/{file_pattern}"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {local_dir} matching {file_pattern}")
    return csv_files


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df.columns = [col.strip() for col in df.columns]
    return df


def _extract_and_isolate_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and extract identity/metadata columns (IPs, ports, timestamps, flow IDs)
    into a separate metadata DataFrame.
    """
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    metadata_cols = []
    for pattern in METADATA_COLUMN_PATTERNS:
        if pattern in col_lower_map:
            metadata_cols.append(col_lower_map[pattern])

    # Also detect any IP-like string columns
    for col in df.columns:
        if col in metadata_cols or col.lower() in ("label", "attack", "attack_cat"):
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and sample.astype(str).str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").all():
                metadata_cols.append(col)

    metadata_cols = list(dict.fromkeys(metadata_cols))
    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    return df, metadata_df


def _clean_infinities_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive/negative infinities and infinite string representations with NaN."""
    # Replace in place across dataframe
    inf_representations = [np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf", "INF", "-INF"]
    df = df.replace(inf_representations, np.nan)
    return df


def _load_cic_ids_2017(data_dir: Path) -> pd.DataFrame:
    """Load and merge CIC-IDS2017 CSV files (supporting TrafficLabelling folder structure)."""
    csv_files = _find_csv_files(data_dir, "*.csv", subfolder="TrafficLabelling")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False, engine="c", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, low_memory=False, engine="c", encoding="latin1")
        df = _standardize_columns(df)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def _load_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Load and merge all UNSW-NB15 CSV files."""
    csv_files = _find_csv_files(data_dir, "*.csv")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False, engine="c", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, low_memory=False, engine="c", encoding="latin1")
        df = _standardize_columns(df)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


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

    for pattern in METADATA_COLUMN_PATTERNS + ["protocol", "proto"]:
        if pattern in col_lower:
            drop_cols.append(col_lower[pattern])

    for col in df.columns:
        if col in drop_cols or col.lower() in ("label", "attack"):
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and sample.astype(str).str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").all():
                drop_cols.append(col)

    drop_cols = list(set(drop_cols))
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df


def load_dataset(dataset_name: str, drop_non_features: bool = True,
                 min_rows: int = 5000) -> pd.DataFrame:
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

    # Load raw data based on dataset family
    if normalized_name == "cic-ids-2017":
        df = _load_cic_ids_2017(local_dir)
    elif normalized_name == "unsw-nb15":
        df = _load_unsw_nb15(local_dir)
    elif normalized_name.startswith("nf-"):
        df = _load_nf_v2(local_dir, normalized_name)
    else:
        csv_files = _find_csv_files(local_dir, file_pattern)
        dfs = [pd.read_csv(f, low_memory=False, engine="c") for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df = _standardize_columns(df)

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
    if mappings and label_column in df.columns:
        df = _apply_label_mapping(df, normalized_name, label_column, mappings)
    elif "label" not in df.columns:
        for col in df.columns:
            if col.lower() in ("label", "attack_cat", "attack", "class"):
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
