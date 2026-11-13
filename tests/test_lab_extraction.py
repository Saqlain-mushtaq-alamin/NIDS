"""
tests/test_lab_extraction.py — Tests for pcap feature extraction and lab capture pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.data.feature_extractor import extract_lab_features, pcap_to_flows, CANONICAL_FEATURES


class TestLabExtraction:
    def test_extract_lab_features_on_real_pcap_end_to_end(self):
        pcap_path = Path("data/lab_captures/20260901_sess_002_DoS.pcap")
        if not pcap_path.is_file():
            pytest.skip(f"Capture {pcap_path} not found")

        df = extract_lab_features(pcap_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "label" in df.columns
        assert "flow_duration" in df.columns
        assert "total_fwd_packets" in df.columns
        assert (df["label"] == "DoS").any()

        # Check canonical features presence
        present_canonical = [c for c in CANONICAL_FEATURES if c in df.columns]
        assert len(present_canonical) >= 20

    def test_extract_lab_features_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_lab_features(Path("nonexistent_dir_or_file_12345.pcap"))

    def test_pcap_to_flows_direct(self):
        pcap_path = Path("data/lab_captures/20260901_sess_003_PortScan.pcap")
        if not pcap_path.is_file():
            pytest.skip(f"Capture {pcap_path} not found")

        flows = pcap_to_flows([pcap_path])
        assert isinstance(flows, pd.DataFrame)
        assert len(flows) > 0
        assert "src_ip" in flows.columns
        assert "dst_ip" in flows.columns

