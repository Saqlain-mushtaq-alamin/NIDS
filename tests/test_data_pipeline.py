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
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "label_mappings.yaml"
        with open(config_path) as f:
            mappings = yaml.safe_load(f)
        df = load_dataset("cic-ids-2017")
        raw_labels = set(df["label"].unique())
        mapped_labels = set()
        for v in mappings.get("cic-ids-2017", {}).values():
            mapped_labels.add(v)
        unmapped = raw_labels - mapped_labels - {"Other"}
        assert not unmapped, f"Unmapped labels: {unmapped}"

    def test_metadata_isolation(self):
        from src.data.loader import load_dataset
        df = load_dataset("cic-ids-2017", drop_non_features=True)
        assert hasattr(df, "attrs") and "metadata" in df.attrs
        metadata = df.attrs["metadata"]
        assert len(metadata) == len(df)
        # Identity columns should not be in features
        for col in ["source ip", "destination ip", "timestamp", "src_ip", "dst_ip"]:
            assert col not in [c.lower() for c in df.columns]

    def test_minimum_rows_failure(self):
        import pandas as pd
        from src.data.loader import load_dataset
        with pytest.raises(ValueError, match=r"minimum \d+ rows required"):
            load_dataset("cic-ids-2017", min_rows=1_000_000)

    def test_split_dataset_strategies(self):
        from src.data.loader import load_dataset, split_dataset
        df = load_dataset("cic-ids-2017")
        # Time-aware split
        splits_time = split_dataset(df, strategy="time_aware", test_ratio=0.2, val_ratio=0.1)
        assert len(splits_time["train"]) + len(splits_time["val"]) + len(splits_time["test"]) == len(df)
        assert len(splits_time["test"]) >= 500
        # Random split
        splits_rand = split_dataset(df, strategy="random", test_ratio=0.2, val_ratio=0.1, seed=42)
        assert len(splits_rand["train"]) + len(splits_rand["val"]) + len(splits_rand["test"]) == len(df)
        assert len(splits_rand["test"]) >= 500

    def test_harmonized_units_comparable_magnitude(self):
        from src.data.loader import load_dataset
        from src.data.feature_extractor import align_to_canonical
        cic_df = load_dataset("cic-ids-2017", drop_non_features=False)
        unsw_df = load_dataset("unsw-nb15", drop_non_features=False)
        
        cic_canon = align_to_canonical(cic_df, "cic-ids-2017")
        unsw_canon = align_to_canonical(unsw_df, "unsw-nb15")
        
        cic_dur = cic_canon["flow_duration"].dropna()
        unsw_dur = unsw_canon["flow_duration"].dropna()
        
        assert len(cic_dur) > 0 and len(unsw_dur) > 0
        cic_median = float(cic_dur.median())
        unsw_median = float(unsw_dur.median())
        
        # After harmonization, durations should be in seconds with comparable magnitude
        # (Ratio between 0.001 and 1000, NOT 10^6 which happens if microseconds vs seconds)
        ratio = cic_median / max(unsw_median, 1e-6)
        assert 0.001 <= ratio <= 1000.0, f"Harmonized durations have divergent magnitudes: cic={cic_median}, unsw={unsw_median}, ratio={ratio}"


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
        save_path = Path(__file__).resolve().parent.parent / "experiments" / "test_preprocessor.joblib"
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

    def test_label_audit_method_label(self):
        from src.data.loader import load_dataset
        from src.data.label_audit import LabelAuditor
        df = load_dataset("cic-ids-2017")
        auditor = LabelAuditor(cv_folds=3, seed=42)
        report = auditor.audit(df)
        assert report["method"] in ["cv_heuristic", "cleanlab"]
        try:
            import cleanlab  # noqa: F401
        except ImportError:
            assert report["method"] == "cv_heuristic"
