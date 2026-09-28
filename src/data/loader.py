"""
src/data/loader.py — Unified data loader for NIDS datasets.
Supports CIC-IDS2017, UNSW-NB15, LYCOS-IDS2017, CTU-13, and lab-captured traffic.
"""
from pathlib import Path
from typing import Optional
import os

# Disable pyarrow backend to avoid access violations on Windows
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import yaml
import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_DATA_DIR = _PROJECT_ROOT / "data" / "raw"


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


def _find_csv_files(local_dir: Path, file_pattern: str = "*.csv") -> list[Path]:
    """Find all CSV files matching the pattern in the given directory."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {local_dir}")
    csv_files = sorted(local_dir.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {local_dir} matching {file_pattern}")
    return csv_files


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df.columns = [col.strip() for col in df.columns]
    return df


def _load_cic_ids_2017(data_dir: Path) -> pd.DataFrame:
    """Load and merge all CIC-IDS2017 CSV files into a single DataFrame."""
    csv_files = _find_csv_files(data_dir, "*.csv")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, engine="c")
        df = _standardize_columns(df)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def _load_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Load and merge all UNSW-NB15 CSV files."""
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
    """Map raw labels to unified taxonomy. Unmapped labels become 'Other'."""
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame. "
                         f"Available: {list(df.columns)}")

    df[label_column] = df[label_column].astype(str).str.strip()
    df["label"] = df[label_column].map(mappings).fillna("Other")
    df.drop(columns=[label_column], inplace=True)
    return df


def _drop_non_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop known non-feature columns (IP addresses, ports, protocol as integer)."""
    drop_cols = []
    col_lower = {c.lower(): c for c in df.columns}

    ip_port_patterns = [
        "source ip", "src_ip", "srcip", "src ip",
        "destination ip", "dst_ip", "dstip", "dst ip",
        "source port", "src_port", "sport", "src port",
        "destination port", "dst_port", "dsport", "dst port",
        "protocol", "proto",
    ]
    for pattern in ip_port_patterns:
        if pattern in col_lower:
            drop_cols.append(col_lower[pattern])

    # Also drop any column with IP-like content if it contains dots
    for col in df.columns:
        if col in drop_cols:
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(5)
            if len(sample) > 0 and sample.astype(str).str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").all():
                drop_cols.append(col)

    drop_cols = list(set(drop_cols))
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df


def load_dataset(dataset_name: str, drop_non_features: bool = True) -> pd.DataFrame:
    """
    Load a dataset by name and return a standardized DataFrame.

    The returned DataFrame always has a 'label' column with the unified taxonomy.
    IP/port/protocol columns are dropped by default to prevent dataset memorization.

    Args:
        dataset_name: One of 'cic-ids-2017', 'unsw-nb15', 'lycos-ids-2017',
                      'ctu-13', 'lab-captured'.
        drop_non_features: If True, drops IP/port/protocol columns.

    Returns:
        pd.DataFrame with standardized columns and unified 'label' column.
    """
    ds_config = _load_dataset_config(dataset_name)
    if not ds_config:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available in configs/datasets.yaml")

    local_dir = _PROJECT_ROOT / ds_config.get("local_dir", "")
    label_column = ds_config.get("label_column", "label")
    file_pattern = ds_config.get("file_pattern", "*.csv")

    # Load raw data based on dataset type
    if dataset_name == "cic-ids-2017":
        df = _load_cic_ids_2017(local_dir)
    elif dataset_name == "unsw-nb15":
        df = _load_unsw_nb15(local_dir)
    else:
        csv_files = _find_csv_files(local_dir, file_pattern)
        dfs = [pd.read_csv(f, low_memory=False, engine="c") for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df = _standardize_columns(df)

    # Apply label mapping
    mappings = _load_label_mappings(dataset_name)
    if mappings and label_column in df.columns:
        df = _apply_label_mapping(df, dataset_name, label_column, mappings)
    elif "label" not in df.columns:
        # Try to find a label-like column
        for col in df.columns:
            if col.lower() in ("label", "attack_cat", " class"):
                df = _apply_label_mapping(df, dataset_name, col, mappings)
                break

    # Drop non-feature columns
    if drop_non_features:
        df = _drop_non_features(df, dataset_name)

    return df
