"""
tests/test_provenance.py — Tests for provenance stamping and scientific validation.
"""
import pytest
from pathlib import Path
from src.utils.provenance import (
    create_provenance_stamp,
    validate_results_provenance,
    save_stamped_json,
    save_stamped_csv,
)
import pandas as pd
import json


def test_create_provenance_stamp_valid():
    stamp = create_provenance_stamp(
        data_kind="real",
        dataset_names=["cic-ids-2017"],
        source_files={"data/raw/cic-ids-2017/Monday.csv": {"rows": 500000, "sha256": "abc"}},
        n_rows=500000,
        n_train=350000,
        n_val=50000,
        n_test=100000,
        class_counts={"Benign": 80000, "Attack": 20000},
        seed_list=[42, 123, 456, 789, 1024],
        script_name="run_phase1.py",
    )
    assert stamp["data_kind"] == "real"
    assert stamp["git_commit"] != ""
    assert stamp["n_test"] == 100000


def test_validate_rejects_simulated_for_headline():
    stamp = create_provenance_stamp(
        data_kind="simulated",
        dataset_names=["cic-ids-2017"],
        source_files={},
        n_rows=5000,
        n_train=3000,
        n_val=1000,
        n_test=1000,
        class_counts={"Benign": 800, "Attack": 200},
        seed_list=[42],
        script_name="run_phase2.py",
    )
    with pytest.raises(ValueError, match="Headline experiment results cannot be generated from simulated data"):
        validate_results_provenance(stamp, is_headline=True)


def test_validate_rejects_small_test_set():
    stamp = create_provenance_stamp(
        data_kind="real",
        dataset_names=["cic-ids-2017"],
        source_files={},
        n_rows=1000,
        n_train=600,
        n_val=200,
        n_test=200,  # < 500
        class_counts={"Benign": 150, "Attack": 50},
        seed_list=[42],
        script_name="run_phase2.py",
    )
    with pytest.raises(ValueError, match="Test set has 200 rows; minimum 500 required"):
        validate_results_provenance(stamp, is_headline=False)


def test_validate_phase5_class_balance_guard():
    stamp = create_provenance_stamp(
        data_kind="real",
        dataset_names=["cic-ids-2017"],
        source_files={},
        n_rows=2000,
        n_train=1000,
        n_val=200,
        n_test=800,
        class_counts={"Benign": 600, "Attack": 200},  # Attack < 500
        seed_list=[42],
        script_name="run_phase5.py",
    )
    with pytest.raises(ValueError, match="Phase 5 requires >= 500 attack and >= 500 benign flows"):
        validate_results_provenance(stamp, is_phase5=True)


def test_provenance_validator_cic_row_count_check():
    from src.data.loader import _verify_provenance
    cols = ["Flow ID", "Source IP", "Timestamp", "Label", "Fwd Packet Length Max"]
    # Only 10,000 rows instead of ~2.83M -> must raise
    df = pd.DataFrame({c: [0]*10000 for c in cols})
    df["Label"] = "BENIGN"
    with pytest.raises(ValueError, match=r"expected total rows within 1% of ~2.83M"):
        _verify_provenance(df, "cic-ids-2017", cols, "Label")


def test_provenance_validator_unsw_partitioned_variant():
    from src.data.loader import _verify_provenance
    cols = ["srcip", "sport", "dstip", "dsport", "proto", "dur", "sbytes", "dbytes", "attack_cat"]
    # 257,673 rows is standard partitioned set
    n_part = 257673
    df = pd.DataFrame({c: [0]*10 for c in cols})
    df["attack_cat"] = "Normal"
    # Test wrong row count for partitioned variant
    with pytest.raises(ValueError, match=r"expected total rows within 1% of 257,673"):
        _verify_provenance(df, "unsw-nb15", cols, "attack_cat", variant="partitioned")


def test_cic_duplicate_fwd_header_length_handling(tmp_path):
    from src.data.loader import _load_cic_ids_2017
    # Create CSV with duplicate 'Fwd Header Length'
    csv_file = tmp_path / "test_dup.csv"
    csv_file.write_text(
        "Flow ID, Source IP, Timestamp, Label, Fwd Header Length, Fwd Header Length.1\n"
        "1,192.168.1.1,2017-07-07 08:00:00,BENIGN,20,20\n",
        encoding="utf-8"
    )
    df = _load_cic_ids_2017(tmp_path)
    assert "Fwd Header Length.1" not in df.columns
    assert list(df.columns).count("Fwd Header Length") == 1


