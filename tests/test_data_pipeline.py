"""
tests/test_data_pipeline.py — Tests for Phase 1 data loading and preprocessing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestDataLoading:
    """Tests for src/data/loader.py (Section 1.1)."""

    def test_load_cic_ids_2017(self):
        import pandas as pd
        from src.data.loader import load_dataset
        df = load_dataset("cic-ids-2017")
        assert len(df) > 0, "CIC-IDS2017 should have rows"
        assert "label" in df.columns, "Standardized 'label' column required"
        assert df["label"].nunique() >= 2, "Need at least benign + 1 attack class"

    def test_load_unsw_nb15(self):
        from src.data.loader import load_dataset
        df = load_dataset("unsw-nb15")
        assert len(df) > 0, "UNSW-NB15 should have rows"
        assert "label" in df.columns

    def test_no_whitespace_columns(self):
        from src.data.loader import load_dataset
        df = load_dataset("cic-ids-2017")
        for col in df.columns:
            assert col == col.strip(), f"Column '{col}' has whitespace"

    def test_label_mapping_coverage(self):
        import yaml
        from src.data.loader import load_dataset
        config_path = Path(r"D:\canvas\NIDS\configs\label_mappings.yaml")
        with open(config_path) as f:
            mappings = yaml.safe_load(f)
        df = load_dataset("cic-ids-2017")
        raw_labels = set(df["label"].unique())
        mapped_labels = set()
        for v in mappings.get("cic-ids-2017", {}).values():
            mapped_labels.add(v)
        unmapped = raw_labels - mapped_labels - {"Other"}
        assert not unmapped, f"Unmapped labels: {unmapped}"


class TestPreprocessing:
    """Tests for src/data/preprocessor.py (Section 1.2)."""

    def test_no_infinities_after_preprocessing(self):
        import numpy as np
        from src.data.loader import load_dataset
        from src.data.preprocessor import Preprocessor
        df = load_dataset("cic-ids-2017")
        prep = Preprocessor()
        X, y = prep.fit_transform(df)
        assert not np.any(np.isinf(X)), "Infinities remain after preprocessing"
        assert not np.any(np.isnan(X)), "NaN values remain after preprocessing"

    def test_no_ip_port_features(self):
        import numpy as np
        from src.data.loader import load_dataset
        from src.data.preprocessor import Preprocessor
        df = load_dataset("cic-ids-2017")
        prep = Preprocessor(drop_ip_port=True)
        X, y = prep.fit_transform(df)
        forbidden = {"src_ip", "dst_ip", "src_port", "dst_port", "source_ip",
                     "destination_ip", "source_port", "destination_port",
                     "source ip", "destination ip", "source port", "destination port",
                     "srcip", "dstip", "sport", "dsport"}
        remaining = set(prep.feature_names_) & forbidden
        assert not remaining, f"IP/port features not removed: {remaining}"

    def test_stratified_split(self):
        import pandas as pd
        from src.data.loader import load_dataset
        from src.data.preprocessor import Preprocessor
        df = load_dataset("cic-ids-2017")
        prep = Preprocessor()
        splits = prep.split(df, test_ratio=0.2, val_ratio=0.1, seed=42)
        train_dist = pd.Series(splits["train"]["y"]).value_counts(normalize=True)
        test_dist = pd.Series(splits["test"]["y"]).value_counts(normalize=True)
        for cls in train_dist.index:
            if cls in test_dist.index:
                diff = abs(train_dist[cls] - test_dist[cls])
                assert diff < 0.10, f"Class '{cls}' distribution skew > 10% between train/test"

    def test_preprocessor_serialization(self):
        import joblib
        from src.data.loader import load_dataset
        from src.data.preprocessor import Preprocessor
        df = load_dataset("cic-ids-2017")
        prep = Preprocessor()
        prep.fit_transform(df)
        save_path = Path(r"D:\canvas\NIDS\experiments\test_preprocessor.joblib")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(prep, save_path)
        loaded = joblib.load(save_path)
        assert hasattr(loaded, "transform")
        assert loaded.feature_names_ == prep.feature_names_
        save_path.unlink(missing_ok=True)


class TestLabelAudit:
    """Tests for src/data/label_audit.py (Section 1.3)."""

    def test_label_audit_runs(self):
        from src.data.loader import load_dataset
        from src.data.label_audit import LabelAuditor
        df = load_dataset("cic-ids-2017")
        auditor = LabelAuditor(cv_folds=3, seed=42)
        report = auditor.audit(df)
        assert "noise_rate" in report
        assert 0.0 <= report["noise_rate"] <= 1.0
        assert "per_class_noise" in report
        assert len(report["per_class_noise"]) > 0

    def test_label_audit_report_generation(self):
        from src.data.loader import load_dataset
        from src.data.label_audit import LabelAuditor
        df = load_dataset("cic-ids-2017")
        auditor = LabelAuditor(cv_folds=3, seed=42)
        report = auditor.audit(df)
        content = auditor.generate_report(report)
        assert "Label Audit Report" in content
        assert "Noise Rate" in content
