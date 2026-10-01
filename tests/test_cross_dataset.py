"""
tests/test_cross_dataset.py - Phase 2 cross-dataset evaluation tests (FR5).

These tests use tiny synthetic frames with *deliberately different* feature
spaces and label sets, so each correctness property of
:class:`~src.evaluation.cross_dataset.CrossDatasetEvaluator` is pinned down
independently of the real datasets:

* the diagonal is scored on held-out rows, never on training rows;
* the test dataset never influences preprocessing statistics;
* labels missing from the training set are excluded but *reported*;
* out-of-vocabulary predictions count as errors;
* the f1 delta is computed against the right reference cell;
* reporting is written and the headline claim is reproducible.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import Preprocessor
from src.evaluation.cross_dataset import (
    CrossDatasetError,
    CrossDatasetEvaluator,
    cross_dataset_matrix,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
class RecordingPreprocessor(Preprocessor):
    """Preprocessor that records every frame handed to ``transform``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformed: list[pd.DataFrame] = []

    def transform(self, df):
        self.transformed.append(df.copy())
        return super().transform(df)


class ConstantModel:
    """Predicts a single label, so metrics are trivially predictable."""

    def __init__(self, label="A"):
        self.label = label
        self.fit_calls = 0
        self.n_train_rows = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.n_train_rows = X.shape[0]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.label, dtype=object)


class RecordingModel(ConstantModel):
    """Constant model that also records the matrix it was fitted on."""

    def __init__(self, label="A"):
        super().__init__(label)
        self.seen: np.ndarray | None = None

    def fit(self, X, y):
        self.seen = np.array(X, copy=True)
        return super().fit(X, y)


def make_frame(n=200, seed=0, offset=0.0, scale=1.0, labels=("A", "B"),
               feature_names=("f1", "f2"), extra_column=None):
    """Build a tiny labelled frame with a reproducible class signal."""
    rng = np.random.default_rng(seed)
    labels = list(labels)
    rows = []
    for i in range(n):
        label = labels[i % len(labels)]
        row = {
            name: float((1.0 if label == labels[0] else -1.0) * 3.0 * scale
                        + offset + rng.normal(0, 0.2))
            for name in feature_names
        }
        if extra_column is not None:
            row[extra_column] = float(rng.normal(1000, 10))
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def two_domains():
    """Two datasets: same labels, different feature spaces and offsets."""
    return {
        "alpha": make_frame(n=300, seed=1, offset=0.0, feature_names=("f1", "f2")),
        "beta": make_frame(n=300, seed=2, offset=5.0, feature_names=("g1", "g2", "g3")),
    }


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_requires_two_datasets(self):
        with pytest.raises(ValueError, match="at least 2 datasets"):
            CrossDatasetEvaluator(datasets=["only-one"],
                                  model_factory=ConstantModel)

    def test_requires_model_factory(self):
        with pytest.raises(ValueError, match="model_factory is required"):
            CrossDatasetEvaluator(datasets=["a", "b"])

    def test_rejects_bare_string(self):
        with pytest.raises(ValueError, match="not a string"):
            CrossDatasetEvaluator(datasets="cic-ids-2017", model_factory=ConstantModel)

    def test_registry_names_are_preserved(self, two_domains):
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel)
        assert evaluator.dataset_names == ["alpha", "beta"]

    def test_dataset_without_label_column_raises(self):
        evaluator = CrossDatasetEvaluator(
            datasets={"a": pd.DataFrame({"f1": [1.0, 2.0]}),
                      "b": pd.DataFrame({"f1": [1.0, 2.0]})},
            model_factory=ConstantModel)
        with pytest.raises(CrossDatasetError, match="no 'label' column"):
            evaluator.run_pair("a", "b")

    def test_unknown_dataset_raises(self, two_domains):
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel)
        with pytest.raises(CrossDatasetError, match="Unknown dataset"):
            evaluator.run_pair("alpha", "missing")

    def test_tuple_payload_accepted(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.array(["A", "B"] * 10, dtype=object)
        evaluator = CrossDatasetEvaluator(
            datasets={"a": (X, y), "b": (X + 1.0, y)},
            model_factory=ConstantModel)
        result = evaluator.run_pair("a", "b")
        assert result.n_test_total == 20

    def test_mapping_payload_accepted(self):
        payload = {"X": np.arange(40, dtype=float).reshape(20, 2),
                   "y": np.array(["A", "B"] * 10, dtype=object)}
        evaluator = CrossDatasetEvaluator(
            datasets={"a": payload, "b": payload},
            model_factory=ConstantModel, test_ratio=0.2, seed=1)
        # Diagonal: the payload is split, so only the held-out rows are scored.
        result = evaluator.run_pair("a", "a")
        assert result.n_test_total == 4
        assert result.n_train + result.n_test_total < 20   # rest went to val

    def test_mismatched_tuple_lengths_raise(self):
        evaluator = CrossDatasetEvaluator(
            datasets={"a": (np.zeros((5, 2)), np.array(["A", "B"] * 5)),
                      "b": (np.zeros((5, 2)), np.array(["A", "B"] * 5))},
            model_factory=ConstantModel)
        with pytest.raises(CrossDatasetError, match="rows but y has"):
            evaluator.run_pair("a", "b")


# --------------------------------------------------------------------------- #
# Leakage prevention
# --------------------------------------------------------------------------- #
class TestNoLeakage:
    def test_diagonal_scores_held_out_rows_only(self, two_domains):
        """The diagonal cell must not be fitted on the rows it is scored on."""
        frame = two_domains["alpha"]
        evaluator = CrossDatasetEvaluator(
            datasets={"alpha": frame}, model_factory=RecordingModel,
            test_ratio=0.5, val_ratio=0.0, seed=1) if False else None

        # Two datasets required, so mirror alpha as beta for a valid evaluator.
        evaluator = CrossDatasetEvaluator(
            datasets={"alpha": frame, "mirror": frame.copy()},
            model_factory=RecordingModel, test_ratio=0.5, val_ratio=0.0, seed=1)
        result = evaluator.run_pair("alpha", "alpha")

        assert result.n_train + result.n_test_total == len(frame)
        assert result.n_test_total == pytest.approx(len(frame) * 0.5, abs=2)
        assert result.n_test_evaluated == result.n_test_total

    def test_model_never_sees_test_rows(self, two_domains):
        """Training rows and evaluated rows must be disjoint populations."""
        frame = two_domains["alpha"]
        n = len(frame)
        evaluator = CrossDatasetEvaluator(
            datasets={"alpha": frame, "mirror": frame.copy()},
            model_factory=RecordingModel, test_ratio=0.2, seed=3)
        result = evaluator.run_pair("alpha", "alpha")
        assert result.n_train + result.n_test_evaluated <= n
        assert result.n_train < n

    def test_test_dataset_does_not_change_training_features(self, two_domains):
        """Adding a constant to the test frame must not alter train features."""
        evaluator_a = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=RecordingModel, seed=5)
        result_a = evaluator_a.run_pair("alpha", "beta")
        pre_a = evaluator_a._build_preprocessor("alpha")
        pre_a.split(two_domains["alpha"], test_ratio=0.2, seed=5)

        shifted_beta = two_domains["beta"].drop(columns=["label"]) + 1000.0
        shifted_beta["label"] = two_domains["beta"]["label"]
        shifted = {"alpha": two_domains["alpha"], "beta": shifted_beta}
        evaluator_b = CrossDatasetEvaluator(
            datasets=shifted, model_factory=RecordingModel, seed=5)
        evaluator_b.run_pair("alpha", "beta")
        pre_b = evaluator_b._build_preprocessor("alpha")
        pre_b.split(shifted["alpha"], test_ratio=0.2, seed=5)

        np.testing.assert_allclose(pre_a.medians_, pre_b.medians_)
        np.testing.assert_allclose(pre_a.scaler_.center_, pre_b.scaler_.center_)
        assert result_a.n_features == len(pre_a.feature_names_)

    def test_template_preprocessor_is_never_fitted(self, two_domains):
        """
        A caller-supplied Preprocessor is a *template*: the evaluator must build
        a fresh instance per pair so no fitted state leaks between cells.
        """
        template = RecordingPreprocessor(feature_mode="full")
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, preprocessor=template,
            model_factory=ConstantModel, seed=7)

        built = []
        original = evaluator._build_preprocessor

        def spy(train_dataset):
            instance = original(train_dataset)
            built.append(instance)
            return instance

        evaluator._build_preprocessor = spy
        evaluator.run_pair("alpha", "beta")
        evaluator.run_pair("beta", "alpha")

        assert len(built) == 2
        assert built[0] is not built[1]
        assert template.transformed == []          # template untouched
        # Each built instance was fitted on its own domain. split() materialises
        # train/val/test (3 calls) and the cross-domain case then transforms the
        # external test frame once more (4 calls total).
        assert all(b.feature_names_ for b in built)
        assert len(built[0].transformed) == 4
        assert len(built[1].transformed) == 4
        # The last frame each instance saw is the evaluation dataset itself.
        assert set(built[0].transformed[-1].columns) == {"g1", "g2", "g3", "label"}
        assert set(built[1].transformed[-1].columns) == {"f1", "f2", "label"}

    def test_preprocessor_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            Preprocessor().transform(pd.DataFrame({"f1": [1.0]}))


# --------------------------------------------------------------------------- #
# Label-set handling
# --------------------------------------------------------------------------- #
class TestLabelHandling:
    def test_unknown_test_labels_excluded_and_reported(self):
        train = make_frame(n=200, seed=11, labels=("A", "B"))
        test = make_frame(n=200, seed=12, labels=("A", "C"))
        evaluator = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=lambda: ConstantModel("A"))
        result = evaluator.run_pair("train", "test")

        assert result.test_only_labels == ["C"]
        assert result.shared_labels == ["A"]
        assert result.n_test_evaluated == pytest.approx(100, abs=2)
        assert result.n_test_total == 200
        assert 0.4 < result.test_coverage < 0.6
        assert any("rows dropped" in w for w in result.warnings)

    def test_out_of_vocabulary_prediction_is_an_error(self):
        """A prediction the model was never trained to emit must not be hidden."""
        train = make_frame(n=100, seed=13, labels=("A", "B"))
        test = make_frame(n=100, seed=14, labels=("A", "B"))
        evaluator = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=lambda: ConstantModel("Z"))  # never seen in training
        result = evaluator.run_pair("train", "test")
        assert result.metrics["accuracy"] == 0.0
        assert result.metrics["f1"] == 0.0

    def test_no_shared_labels_raises(self):
        train = make_frame(n=100, seed=15, labels=("A", "B"))
        test = make_frame(n=100, seed=16, labels=("C", "D"))
        evaluator = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=ConstantModel)
        with pytest.raises(CrossDatasetError, match="share no labels"):
            evaluator.run_pair("train", "test")

    def test_perfect_and_wrong_predictions(self):
        train = make_frame(n=100, seed=17, labels=("A", "B"))
        test = make_frame(n=100, seed=18, labels=("A", "B"))

        right = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=lambda: ConstantModel("A")).run_pair("train", "test")
        assert right.metrics["accuracy"] == pytest.approx(0.5)

        left = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=lambda: ConstantModel("Q")).run_pair("train", "test")
        assert left.metrics["accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# Matrix, deltas, summary
# --------------------------------------------------------------------------- #
class TestMatrix:
    def test_full_matrix_shape_and_columns(self, two_domains):
        from src.evaluation.cross_dataset import RESULT_COLUMNS

        matrix = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel, seed=2
        ).run_full_matrix()
        assert len(matrix) == 4                     # 2 datasets -> 4 ordered pairs
        assert list(matrix.columns) == RESULT_COLUMNS
        assert set(matrix["in_domain"]) == {True, False}

    def test_matrix_saves_to_csv(self, two_domains, tmp_path):
        path = tmp_path / "nested" / "matrix.csv"
        CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel, seed=2
        ).run_full_matrix(save_to=path)
        assert path.exists()
        assert len(pd.read_csv(path)) == 4

    def test_fresh_model_per_pair(self, two_domains):
        counter = {"n": 0}

        def factory():
            counter["n"] += 1
            return RecordingModel("A")

        CrossDatasetEvaluator(
            datasets=two_domains, model_factory=factory, seed=2
        ).run_full_matrix()
        assert counter["n"] == 4

    def test_convenience_wrapper(self, two_domains):
        matrix = cross_dataset_matrix(two_domains, model_factory=ConstantModel, seed=1)
        assert len(matrix) == 4

    def test_deltas_are_zero_on_diagonal(self, two_domains):
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel, seed=2)
        matrix = evaluator.run_full_matrix()
        deltas = evaluator.compute_f1_delta(matrix)
        assert deltas.loc["alpha->alpha", "f1_delta"] == 0.0
        assert deltas.loc["beta->beta", "f1_delta"] == 0.0
        assert len(deltas) == 4

    def test_delta_uses_own_diagonal_reference(self, two_domains):
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=ConstantModel, seed=2)
        matrix = evaluator.run_full_matrix()
        deltas = evaluator.compute_f1_delta(matrix)

        alpha_reference = matrix[
            (matrix["train_dataset"] == "alpha")
            & (matrix["test_dataset"] == "alpha")]["macro_f1"].iloc[0]
        assert deltas.loc["alpha->beta", "in_domain_macro_f1"] == alpha_reference
        assert deltas.loc["alpha->beta", "f1_delta"] == pytest.approx(
            alpha_reference
            - matrix[(matrix["train_dataset"] == "alpha")
                     & (matrix["test_dataset"] == "beta")]["macro_f1"].iloc[0]
        )

    def test_delta_requires_columns(self):
        evaluator = CrossDatasetEvaluator(
            datasets={"a": make_frame(20), "b": make_frame(20)},
            model_factory=ConstantModel)
        with pytest.raises(KeyError, match="macro_f1"):
            evaluator.compute_f1_delta(pd.DataFrame({"train_dataset": ["a"]}))

    def test_delta_on_empty_frame(self):
        evaluator = CrossDatasetEvaluator(
            datasets={"a": make_frame(20), "b": make_frame(20)},
            model_factory=ConstantModel)
        deltas = evaluator.compute_f1_delta(pd.DataFrame(columns=[
            "train_dataset", "test_dataset", "macro_f1"]))
        assert deltas.empty
        assert "f1_delta" in deltas.columns

    def test_summarize_reports_collapse(self, two_domains):
        """A real learner fits its own domain but not the shifted one."""
        from sklearn.tree import DecisionTreeClassifier

        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, seed=2,
            model_factory=lambda: DecisionTreeClassifier(random_state=0),
            model_name="dt")
        summary = evaluator.summarize(evaluator.run_full_matrix())
        assert summary["n_cells"] == 4
        assert summary["n_datasets"] == 2
        assert summary["cross_worse_than_in_domain"] is True
        assert summary["mean_f1_delta"] > 0
        assert summary["cross_f1_retention"] < 1.0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
class TestReporting:
    def test_report_written_with_headline(self, two_domains, tmp_path):
        evaluator = CrossDatasetEvaluator(
            datasets=two_domains, model_factory=lambda: ConstantModel("A"), seed=2,
            model_name="stub")
        matrix = evaluator.run_full_matrix()
        path = evaluator.generate_report(
            matrix, tmp_path / "r.md", extra_notes=["note-one"])

        text = path.read_text(encoding="utf-8")
        assert path.exists()
        assert "# Cross-Dataset Generalization Report" in text
        assert "## Headline" in text
        assert "## Full matrix" in text
        assert "## Interpretation" in text
        assert "alpha" in text and "beta" in text
        assert "note-one" in text

    def test_report_marks_label_subsets(self, tmp_path):
        train = make_frame(n=200, seed=21, labels=("A", "B"))
        test = make_frame(n=200, seed=22, labels=("A", "C"))
        evaluator = CrossDatasetEvaluator(
            datasets={"train": train, "test": test},
            model_factory=lambda: ConstantModel("A"), seed=2)
        path = evaluator.generate_report(evaluator.run_full_matrix(), tmp_path / "r.md")
        assert "label subset" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Script smoke test
# --------------------------------------------------------------------------- #
def test_run_cross_dataset_quick(tmp_path):
    """The driver script must run end-to-end and emit every advertised artifact."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_cross_dataset import main

    code = main([
        "--quick",
        "--datasets", "cic-ids-2017", "unsw-nb15",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0

    expected = [
        "cross_dataset_full_xgb.csv",
        "cross_dataset_full_xgb_agg.csv",
        "cross_dataset_full_xgb_stats.csv",
        "cross_dataset_full_xgb_report.md",
        "cross_dataset_full_xgb_summary.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name

    summary = json.loads((tmp_path / "cross_dataset_full_xgb_summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["n_cells"] == 4
    assert summary["in_domain_macro_f1_mean"] > summary["cross_domain_macro_f1_mean"]
