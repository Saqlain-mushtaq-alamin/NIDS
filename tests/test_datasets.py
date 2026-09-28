"""
tests/test_datasets.py — Dataset availability and structure verification tests.
"""
from pathlib import Path
import pandas as pd
import yaml

DATA_ROOT = Path(r"d:\canvas\NIDS\data\raw")
CONFIG_PATH = Path(r"d:\canvas\NIDS\configs\datasets.yaml")

def test_datasets_config_exists():
    """Verify dataset registry configuration exists and can be parsed."""
    assert CONFIG_PATH.is_file(), "configs/datasets.yaml missing"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "datasets" in cfg
    assert "cic-ids-2017" in cfg["datasets"]
    assert "unsw-nb15" in cfg["datasets"]

def test_cic_ids_2017_exists():
    """Verify CIC-IDS2017 directory and data files exist."""
    cic_dir = DATA_ROOT / "cic-ids-2017"
    assert cic_dir.is_dir(), "CIC-IDS2017 raw data directory missing"
    csv_files = list(cic_dir.glob("*.csv"))
    assert len(csv_files) >= 1, "No CSV files found in CIC-IDS2017"

def test_unsw_nb15_exists():
    """Verify UNSW-NB15 directory and data files exist."""
    unsw_dir = DATA_ROOT / "unsw-nb15"
    assert unsw_dir.is_dir(), "UNSW-NB15 raw data directory missing"
    csv_files = list(unsw_dir.glob("*.csv"))
    assert len(csv_files) >= 1, "No CSV files found in UNSW-NB15"

def test_cic_ids_loadable():
    """Verify at least one CIC-IDS file can be loaded with expected label column."""
    pd.options.mode.string_storage = "python"
    cic_dir = DATA_ROOT / "cic-ids-2017"
    first_csv = next(cic_dir.glob("*.csv"), None)
    assert first_csv is not None
    df = pd.read_csv(first_csv, nrows=100, dtype=str)
    assert len(df) > 0, "Could not load rows from CIC-IDS2017"
    assert " Label" in df.columns or "Label" in df.columns or "label" in df.columns, "Label column not found"
