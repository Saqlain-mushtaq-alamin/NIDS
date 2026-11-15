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
import numpy as np
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


def test_leakage_guard_feature_list():
    """A2. Assert none of the denylisted columns exist in model input features."""
    from src.data.loader import LEAKAGE_DENYLIST_EXACT, _drop_non_features
    # Sample DataFrame with all forbidden columns
    forbidden_cols = [
        "label", "label_raw", "attack_cat", "attack", "protocol", "proto",
        "service", "state", "id", "flow id", "timestamp", "source ip", "destination ip",
        "source port", "destination port", "srcip", "dstip", "sport", "dsport",
        "fwd_packets", "bwd_packets"
    ]
    df = pd.DataFrame({c: [1, 2] for c in forbidden_cols})
    df["label"] = ["Benign", "Attack"]
    df["label_raw"] = ["Normal", "DoS"]
    df_clean = _drop_non_features(df, "unsw-nb15")

    model_features = [c for c in df_clean.columns if c not in ("label", "label_raw")]
    for feat in model_features:
        f_low = feat.strip().lower()
        assert f_low not in LEAKAGE_DENYLIST_EXACT
        assert not any(f_low == d for d in ["ip", "port", "timestamp", "flow id", "id", "proto", "protocol", "service", "state", "attack_cat"])
    assert "fwd_packets" in model_features
    assert "bwd_packets" in model_features


def test_shared_feature_harmonization_magnitudes():
    """A5. Assert harmonized shared features have sensible physical ranges."""
    from src.data.feature_extractor import align_to_canonical, load_feature_schema
    schema = load_feature_schema()
    primary = schema.get("shared_primary", [])
    assert len(primary) == 10

    # Test CIC mock conversion
    cic_mock = pd.DataFrame({
        "Flow Duration": [1_000_000, 500_000],   # 1s, 0.5s in us
        "Total Fwd Packets": [10, 5],
        "Total Backward Packets": [8, 4],
        "Total Length of Fwd Packets": [1500, 750],
        "Total Length of Bwd Packets": [3000, 1500],
        "Fwd Packet Length Mean": [150.0, 150.0],
        "Bwd Packet Length Mean": [375.0, 375.0],
        "Flow Packets/s": [18.0, 18.0],
        "Fwd IAT Mean": [100_000.0, 50_000.0],   # 100ms, 50ms in us
        "Bwd IAT Mean": [125_000.0, 62_500.0],   # 125ms, 62.5ms in us
    })
    aligned_cic = align_to_canonical(cic_mock, "cic-ids-2017", primary)
    assert aligned_cic["flow_duration"].iloc[0] == pytest.approx(1.0)       # 1 sec
    assert aligned_cic["fwd_iat_mean"].iloc[0] == pytest.approx(100.0)      # 100 ms
    assert aligned_cic["total_fwd_packets"].iloc[0] == 10
    assert (aligned_cic > 0).all().all()

    # Test UNSW mock conversion
    unsw_mock = pd.DataFrame({
        "dur": [1.0, 0.5],                       # 1s, 0.5s in sec
        "spkts": [10, 5],
        "dpkts": [8, 4],
        "sbytes": [1500, 750],
        "dbytes": [3000, 1500],
        "smean": [150.0, 150.0],
        "dmean": [375.0, 375.0],
        "rate": [18.0, 18.0],
        "sinpkt": [100.0, 50.0],                 # 100ms, 50ms in ms
        "dinpkt": [125.0, 62.5],                 # 125ms, 62.5ms in ms
    })
    aligned_unsw = align_to_canonical(unsw_mock, "unsw-nb15", primary)
    assert aligned_unsw["flow_duration"].iloc[0] == pytest.approx(1.0)      # 1 sec
    assert aligned_unsw["fwd_iat_mean"].iloc[0] == pytest.approx(100.0)     # 100 ms
    assert (aligned_unsw > 0).all().all()

    # Exact equality across aligned pairs
    for col in primary:
        np.testing.assert_allclose(aligned_cic[col], aligned_unsw[col])


