"""
src/evaluation/cross_dataset.py — Cross-dataset generalization harness
(PHASE_2 section 2.3, FR5: the core module of the project).

Purpose
-------
Measure, rigorously and without leakage, how much a classifier trained on one
network dataset loses when evaluated on a *different* one.  The headline metric
is the cross-domain F1 delta::

    dF1 = F1_in_domain - F1_cross_domain

Correctness rules enforced here (each one has a dedicated test):

1. **Training data only.**  The :class:`~src.data.preprocessor.Preprocessor` is
   fitted on the training dataset's *training split* and reused verbatim for the
   test dataset.  Test columns absent from the training feature space are
   dropped; training columns absent from the test frame are imputed with
   training medians.  No test statistic ever reaches the model.
2. **No diagonal shortcut.**  ``train_dataset == test_dataset`` is evaluated on a
   held-out split of the same dataset, never on the training rows.
3. **Label-set mismatch is explicit.**  Metrics are computed over the labels the
   model was actually trained to emit (the *shared* labels).  Test rows whose
   label the model can never produce are excluded and reported through
   ``test_coverage``.  Predictions that fall outside the shared set are counted
   as errors, never filtered away.
4. **Fresh model per pair.**  ``model_factory`` is called for every ordered pair,
   so no state leaks between cells of the matrix.
5. **Metric definitions are stated.**  ``f1`` is weighted-F1; ``macro_f1`` is the
   unweighted mean over shared labels.  Both are reported.

Typical use::

    evaluator = CrossDatasetEvaluator(
        datasets=["cic-ids-2017", "unsw-nb15", "lab-captured"],
        feature_mode="full",
        model_factory=lambda: XGBBaseline(n_estimators=150, random_state=42),
        seed=42,
    )
    matrix = evaluator.run_full_matrix()
    deltas = evaluator.compute_f1_delta(matrix)
    evaluator.generate_report(matrix, "results/cross_dataset_report.md")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessor import Preprocessor
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    per_class_report,
)
from src.utils.pandas_compat import apply_pandas_compat, as_object_array
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

apply_pandas_compat()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"

_DATASET_PAYLOAD = Union[str, pd.DataFrame, Mapping[str, np.ndarray],
                        Sequence[np.ndarray]]

RESULT_COLUMNS = [
    "train_dataset",
    "test_dataset",
    "in_domain",
    "model",
    "feature_mode",
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "mcc",
    "binary_f1",
    "binary_recall",
    "binary_fpr",
    "binary_auroc",
    "binary_mcc",
    "majority_class",
    "majority_accuracy",
    "majority_macro_f1",
    "majority_binary_f1",
    "n_train",
    "n_test_total",
    "n_test_evaluated",
    "n_train_features_raw",
    "n_test_features_raw",
    "n_features_shared",
    "n_features_dropped_train",
    "n_features_dropped_test",
    "features_dropped",
    "n_train_features",
    "n_shared_labels",
    "test_coverage",
    "shared_labels",
    "test_only_labels",
    "train_only_labels",
]


class CrossDatasetError(RuntimeError):
    """Raised when a cross-dataset configuration cannot be evaluated."""


def _as_array(labels) -> np.ndarray:
    """Coerce labels to a 1-D object array of ``str``."""
    array = np.asarray(labels)
    if array.ndim == 0:
        array = array.reshape(1)
    return as_object_array(array.astype(str))


def _bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        shared_labels: Sequence[str],
                        n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Row-level bootstrap 95% confidence interval for macro-F1 over shared labels."""
    if len(y_true) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    boot_scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_scores[i] = f1_score(
            y_true[sample_idx], y_pred[sample_idx],
            labels=shared_labels, average="macro", zero_division=0,
        )
    return float(np.percentile(boot_scores, 2.5)), float(np.percentile(boot_scores, 97.5))


@dataclass
class CrossDatasetResult:
    """Metrics and bookkeeping for a single ordered (train, test) pair."""

    train_dataset: str
    test_dataset: str
    metrics: dict
    shared_labels: list[str]
    test_only_labels: list[str]
    train_only_labels: list[str]
    n_train: int
    n_test_total: int
    n_test_evaluated: int
    n_features: int
    n_train_features_raw: int = 0
    n_test_features_raw: int = 0
    n_features_shared: int = 0
    n_features_dropped_train: int = 0
    n_features_dropped_test: int = 0
    features_dropped: list[str] = field(default_factory=list)
    macro_f1_ci_low: float = float("nan")
    macro_f1_ci_high: float = float("nan")
    binary_metrics: dict = field(default_factory=dict)
    majority_baseline: dict = field(default_factory=dict)
    per_class: Optional[pd.DataFrame] = None
    confusion_matrix: Optional[np.ndarray] = None
    model: str = "model"
    feature_mode: str = "full"
    seed: int = 42
    warnings: list[str] = field(default_factory=list)

    @property
    def test_coverage(self) -> float:
        """Fraction of test rows whose label the model could ever emit."""
        if self.n_test_total == 0:
            return 0.0
        return self.n_test_evaluated / self.n_test_total

    def to_row(self) -> dict:
        """Flatten to a single :data:`RESULT_COLUMNS`-shaped dict."""
        bin_m = self.binary_metrics
        maj_b = self.majority_baseline
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "in_domain": self.train_dataset == self.test_dataset,
            "model": self.model,
            "feature_mode": self.feature_mode,
            "seed": self.seed,
            "accuracy": self.metrics.get("accuracy", float("nan")),
            "precision": self.metrics.get("precision", float("nan")),
            "recall": self.metrics.get("recall", float("nan")),
            "f1": self.metrics.get("f1", float("nan")),
            "macro_f1": self.metrics.get("macro_f1", float("nan")),
            "macro_f1_ci_low": self.macro_f1_ci_low,
            "macro_f1_ci_high": self.macro_f1_ci_high,
            "mcc": self.metrics.get("mcc", float("nan")),
            "binary_f1": bin_m.get("binary_f1", float("nan")),
            "binary_recall": bin_m.get("binary_recall", float("nan")),
            "binary_fpr": bin_m.get("binary_fpr", float("nan")),
            "binary_auroc": bin_m.get("binary_auroc", float("nan")),
            "binary_mcc": bin_m.get("binary_mcc", float("nan")),
            "majority_class": maj_b.get("majority_class", ""),
            "majority_accuracy": maj_b.get("majority_accuracy", float("nan")),
            "majority_macro_f1": maj_b.get("majority_macro_f1", float("nan")),
            "majority_binary_f1": maj_b.get("majority_binary_f1", float("nan")),
            "n_train": self.n_train,
            "n_test_total": self.n_test_total,
            "n_test_evaluated": self.n_test_evaluated,
            "n_train_features_raw": self.n_train_features_raw,
            "n_test_features_raw": self.n_test_features_raw,
            "n_features_shared": self.n_features_shared,
            "n_features_dropped_train": self.n_features_dropped_train,
            "n_features_dropped_test": self.n_features_dropped_test,
            "features_dropped": "|".join(self.features_dropped),
            "n_train_features": self.n_features,
            "n_shared_labels": len(self.shared_labels),
            "test_coverage": self.test_coverage,
            "shared_labels": "|".join(self.shared_labels),
            "test_only_labels": "|".join(self.test_only_labels),
            "train_only_labels": "|".join(self.train_only_labels),
        }


class CrossDatasetEvaluator:
    """
    Central evaluation engine for cross-domain generalization experiments.

    Args:
        datasets: Either a list of registry keys (loaded via
            :func:`src.data.loader.load_dataset`) or a dict mapping a name to a
            ready payload.  A payload may be a ``DataFrame`` with a ``label``
            column, a ``(X, y)`` tuple, or a ``{"X": ..., "y": ...}`` mapping.
        preprocessor: An optional pre-configured :class:`Preprocessor`.  When
            ``None`` one is constructed per pair from ``feature_mode``,
            ``normalization`` and ``dataset``.  Pass an explicit instance only
            when the caller wants to control construction; the evaluator still
            calls ``fit`` on the training split.
        model_factory: Zero-argument callable returning a fresh estimator with
            ``fit`` / ``predict``.
        feature_mode: ``"full"``, ``"behavior_only"`` or ``"behavior_extended"``.
        normalization: Scaler mode for the built-in preprocessor.
        seed: Split / model seed.
        test_ratio: Fraction of each dataset reserved as its evaluation split.
        val_ratio: Fraction reserved as validation (unused by the harness beyond
            mirroring Phase 1's split contract).
        model_name: Label recorded in the results table.
        class_weight: Optional ``"balanced"`` passed to estimators that accept
            ``class_weight``.  Useful for the strongly imbalanced ``lab-captured``
            domain.  ``None`` keeps estimator defaults.
        normalize: Shorthand for weighted/unweighted macro reporting; kept for
            signature compatibility with earlier drafts.
        logger: Optional ``logging.Logger``.

    Attributes:
        dataset_names: Ordered dataset names participating in the matrix.
        model_factory: The estimator factory in use.
        feature_mode: Active feature mode.
        seed: Active seed.
    """

    def __init__(
        self,
        datasets: Union[Mapping[str, _DATASET_PAYLOAD], Sequence[str]],
        preprocessor: Optional[Preprocessor] = None,
        model_factory: Optional[Callable[[], object]] = None,
        feature_mode: str = "full",
        normalization: str = "robust",
        seed: int = 42,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        model_name: str = "model",
        class_weight: Optional[str] = None,
        normalize: bool = True,
        log1p: bool = False,
        per_domain_scaler: bool = False,
        min_test_rows: int = 0,
        logger=None,
    ):
        if model_factory is None:
            raise ValueError("model_factory is required")

        self.datasets = self._normalise_datasets(datasets)
        self.dataset_names = list(self.datasets)
        if len(self.dataset_names) < 2:
            raise ValueError(
                "Cross-dataset evaluation needs at least 2 datasets, got "
                f"{len(self.dataset_names)}"
            )

        self.preprocessor_template = preprocessor
        self.model_factory = model_factory
        self.feature_mode = feature_mode
        self.normalization = normalization
        self.seed = int(seed)
        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.model_name = model_name
        self.class_weight = class_weight
        self.normalize = normalize
        self.log1p = bool(log1p)
        self.per_domain_scaler = bool(per_domain_scaler)
        self.min_test_rows = int(min_test_rows)
        self.logger = logger

    # -- setup --------------------------------------------------------------
    @staticmethod
    def _normalise_datasets(datasets) -> dict:
        if isinstance(datasets, Mapping):
            return {str(name): payload for name, payload in datasets.items()}
        if isinstance(datasets, (str, bytes)):
            raise ValueError("datasets must be a sequence or mapping, not a string")
        return {str(name): name for name in datasets}

    def _load(self, name: str) -> pd.DataFrame:
        """Return dataset ``name`` as a DataFrame with a ``label`` column."""
        payload = self.datasets[name]

        if isinstance(payload, pd.DataFrame):
            frame = payload
        elif isinstance(payload, str):
            frame = load_dataset(payload)
        elif isinstance(payload, Mapping):
            if "X" not in payload or "y" not in payload:
                raise CrossDatasetError(
                    f"Dataset '{name}' mapping payload needs 'X' and 'y' keys."
                )
            X = np.asarray(payload["X"])
            y = _as_array(payload["y"])
            frame = pd.DataFrame(X)
            frame["label"] = y
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            X = np.asarray(payload[0])
            y = _as_array(payload[1])
            if X.shape[0] != y.shape[0]:
                raise CrossDatasetError(
                    f"Dataset '{name}': X has {X.shape[0]} rows but y has {y.shape[0]}."
                )
            frame = pd.DataFrame(X)
            frame["label"] = y
        else:
            raise CrossDatasetError(
                f"Unsupported payload for dataset '{name}' of type "
                f"{type(payload).__name__}."
            )

        if "label" not in frame.columns:
            raise CrossDatasetError(
                f"Dataset '{name}' has no 'label' column; cross-dataset "
                "evaluation needs the unified taxonomy."
            )
        return frame

    def _build_preprocessor(self, train_dataset: str) -> Preprocessor:
        """
        Build a *fresh* preprocessor for one pair, from the caller's template.

        A fresh instance per pair is what keeps fitted state (medians, scaler,
        feature list) from leaking between cells of the matrix.  The template's
        class is preserved so a caller can subclass ``Preprocessor`` to record or
        validate behaviour; if that subclass does not accept the standard
        arguments we fall back to the base class.
        """
        if self.preprocessor_template is None:
            return Preprocessor(
                drop_ip_port=True,
                normalization=self.normalization,
                feature_mode=self.feature_mode,
                dataset=train_dataset,
                log1p=self.log1p,
            )

        template = self.preprocessor_template
        factory = type(template)
        for candidate in (factory, Preprocessor):
            try:
                return candidate(
                    drop_ip_port=template.drop_ip_port,
                    normalization=template.normalization,
                    feature_mode=template.feature_mode,
                    variance_threshold=template.variance_threshold,
                    dataset=train_dataset,
                    log1p=getattr(template, "log1p", self.log1p),
                )
            except TypeError:
                continue
        raise CrossDatasetError(
            f"Cannot construct a preprocessor from template of type {factory.__name__}"
        )

    def _build_model(self):
        model = self.model_factory()
        if self.class_weight is not None:
            params = getattr(model, "get_params", None)
            if callable(params):
                try:
                    model.set_params(class_weight=self.class_weight)
                except (ValueError, TypeError):
                    if self.logger:
                        self.logger.debug(
                            "Model does not accept class_weight='%s'; ignoring.",
                            self.class_weight,
                        )
        return model

    def _log(self, message: str, *args) -> None:
        if self.logger:
            self.logger.info(message, *args)

    # -- evaluation ---------------------------------------------------------
    def run_pair(self, train_dataset: str, test_dataset: str,
                 seed: Optional[int] = None) -> CrossDatasetResult:
        """
        Train on ``train_dataset`` and evaluate on ``test_dataset``.

        When the two names are equal, evaluation uses the held-out test split of
        that dataset; the model never sees those rows.

        Args:
            train_dataset: Key of the training dataset.
            test_dataset: Key of the evaluation dataset.
            seed: Override the evaluator seed for this pair.

        Returns:
            A populated :class:`CrossDatasetResult`.

        Raises:
            CrossDatasetError: on an unknown dataset name or a degenerate split.
        """
        for name in (train_dataset, test_dataset):
            if name not in self.datasets:
                raise CrossDatasetError(
                    f"Unknown dataset '{name}'. Available: {self.dataset_names}"
                )

        seed = self.seed if seed is None else int(seed)

        train_frame = self._load(train_dataset)
        test_frame = self._load(test_dataset)

        # Feature intersection protocol:
        # Determine available features in train and test frame
        p_train = self._build_preprocessor(train_dataset)
        train_proj = p_train.project_features(train_frame, dataset=train_dataset)
        train_feat_raw = [
            c for c in train_proj.columns
            if pd.api.types.is_numeric_dtype(train_proj[c]) and train_proj[c].notna().sum() > 0
        ]

        p_test = self._build_preprocessor(test_dataset)
        test_proj = p_test.project_features(test_frame, dataset=test_dataset)
        test_feat_raw = [
            c for c in test_proj.columns
            if pd.api.types.is_numeric_dtype(test_proj[c]) and test_proj[c].notna().sum() > 0
        ]

        if train_dataset == test_dataset:
            shared_features = list(train_feat_raw)
            dropped_train = []
            dropped_test = []
        else:
            shared_features = [c for c in train_feat_raw if c in set(test_feat_raw)]
            dropped_train = [c for c in train_feat_raw if c not in set(shared_features)]
            dropped_test = [c for c in test_feat_raw if c not in set(shared_features)]

        if not shared_features:
            raise CrossDatasetError(
                f"'{train_dataset}' and '{test_dataset}' share no features "
                f"(train={len(train_feat_raw)}, test={len(test_feat_raw)}); "
                "cross-dataset evaluation requires non-empty feature intersection."
            )

        preprocessor = self._build_preprocessor(train_dataset)
        train_X, train_y, test_X, test_y = self._prepare(
            preprocessor, train_frame, test_frame,
            in_domain=(train_dataset == test_dataset), seed=seed,
            shared_features=shared_features, test_dataset=test_dataset,
        )

        warnings: list[str] = []
        if dropped_train:
            warnings.append(
                f"{len(dropped_train)} features dropped from '{train_dataset}' absent in test: {dropped_train}"
            )
        if dropped_test:
            warnings.append(
                f"{len(dropped_test)} features dropped from '{test_dataset}' absent in train: {dropped_test}"
            )
        if preprocessor.missing_features_:
            warnings.append(
                f"training dataset '{train_dataset}' cannot supply "
                f"{len(preprocessor.missing_features_)} requested features: "
                f"{preprocessor.missing_features_}"
            )

        train_labels = sorted(set(_as_array(train_y).tolist()))
        test_labels_all = sorted(set(test_y.tolist()))
        shared = [label for label in train_labels if label in set(test_labels_all)]
        test_only = [label for label in test_labels_all if label not in set(train_labels)]

        if not shared:
            raise CrossDatasetError(
                f"'{train_dataset}' and '{test_dataset}' share no labels "
                f"(train={train_labels}, test={test_labels_all}); cross-dataset "
                "metrics are undefined."
            )

        keep = np.isin(test_y, np.array(shared, dtype=object))
        n_total = int(test_y.shape[0])
        n_eval = int(keep.sum())
        if n_eval == 0:
            raise CrossDatasetError(
                f"No '{test_dataset}' rows carry a shared label; cannot evaluate."
            )
        if self.min_test_rows > 0 and n_eval < self.min_test_rows:
            raise ValueError(
                f"Evaluated test rows ({n_eval}) is smaller than min_test_rows "
                f"({self.min_test_rows}). Failing loudly per scientific rules."
            )

        if n_eval < n_total:
            warnings.append(
                f"{n_total - n_eval} test rows dropped: labels "
                f"{test_only} are absent from the training set "
                f"(coverage {n_eval / n_total:.1%})"
            )

        model = self._build_model()
        model.fit(train_X, train_y)

        y_true = test_y[keep]
        y_pred = _as_array(model.predict(test_X[keep]))

        # Multi-class metrics (macro_f1 computed strictly over shared labels)
        metrics = compute_metrics(y_true, y_pred, average="weighted")
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, labels=shared, average="macro", zero_division=0))
        boot_ci_low, boot_ci_high = _bootstrap_macro_f1(y_true, y_pred, shared, seed=seed)

        # Binary Benign-vs-Attack metrics
        has_benign = any(str(l).strip().lower() == "benign" for l in shared)
        if has_benign:
            y_true_bin = np.array([0 if str(l).strip().lower() == "benign" else 1 for l in y_true], dtype=int)
            y_pred_bin = np.array([0 if str(l).strip().lower() == "benign" else 1 for l in y_pred], dtype=int)
        else:
            neg_l = str(shared[0])
            y_true_bin = np.array([0 if str(l) == neg_l else 1 for l in y_true], dtype=int)
            y_pred_bin = np.array([0 if str(l) == neg_l else 1 for l in y_pred], dtype=int)

        bin_f1 = float(f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
        bin_rec = float(recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        bin_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        bin_mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))

        bin_auroc = float("nan")
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(test_X[keep])
                classes = [str(c) for c in getattr(model, "classes_", [])]
                if has_benign:
                    atk_idx = [i for i, c in enumerate(classes) if c.strip().lower() != "benign"]
                else:
                    atk_idx = [i for i, c in enumerate(classes) if c != neg_l]
                if atk_idx and proba.ndim == 2:
                    p_atk = proba[:, atk_idx].sum(axis=1)
                    if len(np.unique(y_true_bin)) > 1:
                        bin_auroc = float(roc_auc_score(y_true_bin, p_atk))
            except Exception:
                pass

        binary_metrics = {
            "binary_f1": bin_f1,
            "binary_recall": bin_rec,
            "binary_fpr": bin_fpr,
            "binary_auroc": bin_auroc,
            "binary_mcc": bin_mcc,
        }

        # Majority-class baseline (from training set majority)
        maj_counts = pd.Series(_as_array(train_y)).value_counts()
        majority_class = str(maj_counts.index[0]) if not maj_counts.empty else str(shared[0])
        maj_pred = np.full(len(y_true), majority_class, dtype=object)
        maj_acc = float(accuracy_score(y_true, maj_pred))
        maj_macro_f1 = float(f1_score(y_true, maj_pred, labels=shared, average="macro", zero_division=0))
        if has_benign:
            maj_pred_bin = np.array([0 if str(l).strip().lower() == "benign" else 1 for l in maj_pred], dtype=int)
        else:
            maj_pred_bin = np.array([0 if str(l) == neg_l else 1 for l in maj_pred], dtype=int)
        maj_bin_f1 = float(f1_score(y_true_bin, maj_pred_bin, pos_label=1, zero_division=0))

        majority_baseline = {
            "majority_class": majority_class,
            "majority_accuracy": maj_acc,
            "majority_macro_f1": maj_macro_f1,
            "majority_binary_f1": maj_bin_f1,
        }

        result = CrossDatasetResult(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            metrics=metrics,
            shared_labels=shared,
            test_only_labels=test_only,
            train_only_labels=[l for l in train_labels if l not in set(shared)],
            n_train=int(train_X.shape[0]),
            n_test_total=n_total,
            n_test_evaluated=n_eval,
            n_features=int(train_X.shape[1]),
            n_train_features_raw=len(train_feat_raw),
            n_test_features_raw=len(test_feat_raw),
            n_features_shared=len(shared_features),
            n_features_dropped_train=len(dropped_train),
            n_features_dropped_test=len(dropped_test),
            features_dropped=dropped_train + dropped_test,
            macro_f1_ci_low=boot_ci_low,
            macro_f1_ci_high=boot_ci_high,
            binary_metrics=binary_metrics,
            majority_baseline=majority_baseline,
            model=self.model_name,
            feature_mode=self.feature_mode,
            seed=seed,
            warnings=warnings,
        )
        result.per_class = per_class_report(y_true, y_pred)
        result.confusion_matrix = compute_confusion_matrix(y_true, y_pred)
        return result

    def _prepare(self, preprocessor: Preprocessor, train_frame: pd.DataFrame,
                 test_frame: pd.DataFrame, in_domain: bool, seed: int,
                 shared_features: Optional[Sequence[str]] = None,
                 test_dataset: Optional[str] = None):
        """
        Fit on the training split, then transform the evaluation data.

        For the in-domain case (``train_dataset == test_dataset``) the
        evaluation data is the preprocessor's own held-out test split, so the
        diagonal cell of the matrix is scored on rows the model never saw.  For
        the cross-domain case the test dataset is transformed with the
        training-fitted preprocessor and never influences fitting.

        Returns:
            ``(train_X, train_y, test_X, test_y)``.
        """
        splits = preprocessor.split(
            train_frame, test_ratio=self.test_ratio,
            val_ratio=self.val_ratio, seed=seed,
            allowed_features=shared_features,
        )
        train = splits["train"]
        if in_domain:
            return train["X"], train["y"], splits["test"]["X"], splits["test"]["y"]

        try:
            test_X, test_y = preprocessor.transform(
                test_frame, dataset=test_dataset, per_domain_scaler=self.per_domain_scaler
            )
        except TypeError:
            test_X, test_y = preprocessor.transform(test_frame)
        return train["X"], train["y"], test_X, test_y

    def run_full_matrix(self, seed: Optional[int] = None,
                        save_to: Optional[Path] = None) -> pd.DataFrame:
        """
        Run every ordered (train, test) permutation, including the diagonal.

        Args:
            seed: Optional seed override.
            save_to: Optional CSV path for the matrix.

        Returns:
            DataFrame with :data:`RESULT_COLUMNS`, one row per ordered pair.
        """
        seed = self.seed if seed is None else int(seed)
        rows = []

        for train_name in self.dataset_names:
            for test_name in self.dataset_names:
                self._log("[%s -> %s] training...", train_name, test_name)
                result = self.run_pair(train_name, test_name, seed=seed)
                row = result.to_row()
                rows.append(row)
                for warning in result.warnings:
                    self._log("  ! %s", warning)

        matrix = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            matrix.to_csv(save_to, index=False)
        return matrix

    # -- analysis -----------------------------------------------------------
    def compute_f1_delta(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ``F1_in_domain - F1_cross_domain`` for every ordered pair.

        The in-domain reference for a pair ``A -> B`` is the diagonal cell
        ``A -> A``: the same model family and the same feature space, evaluated
        on held-out data from the training domain.  For diagonal pairs the delta
        is 0 by definition.

        Args:
            results: Matrix from :meth:`run_full_matrix`.

        Returns:
            DataFrame indexed by ``"A->B"`` (ASCII, so the index survives CSV
            round-trips and console printing on Windows) with the train cell's
            in-domain reference and the resulting delta.
        """
        required = {"train_dataset", "test_dataset", "macro_f1"}
        missing = required - set(results.columns)
        if missing:
            raise KeyError(
                f"results is missing required column(s): {sorted(missing)}"
            )
        if results.empty:
            return pd.DataFrame(
                columns=["train_macro_f1", "in_domain_macro_f1",
                         "cross_domain_macro_f1", "f1_delta", "in_domain"]
            )

        diagonal = {}
        for _, row in results.iterrows():
            if row["train_dataset"] == row["test_dataset"]:
                diagonal[row["train_dataset"]] = float(row["macro_f1"])

        records = []
        for _, row in results.iterrows():
            train_name = row["train_dataset"]
            test_name = row["test_dataset"]
            cross_f1 = float(row["macro_f1"])
            reference = diagonal.get(train_name)
            delta = (reference - cross_f1) if reference is not None else float("nan")
            records.append({
                "pair": f"{train_name}->{test_name}",
                "train_dataset": train_name,
                "test_dataset": test_name,
                "in_domain": train_name == test_name,
                "train_macro_f1": reference,
                "in_domain_macro_f1": cross_f1 if train_name == test_name else reference,
                "cross_domain_macro_f1": cross_f1,
                "f1_delta": delta,
                "accuracy": float(row["accuracy"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            })

        return pd.DataFrame(records).set_index("pair")

    def summarize(self, results: pd.DataFrame) -> dict:
        """
        Aggregate the matrix into the headline numbers of the experiment.

        Returns:
            Dict with in-domain / cross-domain means, the average F1 delta,
            per-test-dataset detail and the retrieval fraction against the
            documented cross-dataset collapse.
        """
        deltas = self.compute_f1_delta(results).reset_index()

        in_domain = results[results["train_dataset"] == results["test_dataset"]]
        cross_domain = results[results["train_dataset"] != results["test_dataset"]]

        in_f1 = float(in_domain["macro_f1"].mean()) if len(in_domain) else float("nan")
        cross_f1 = float(cross_domain["macro_f1"].mean()) if len(cross_domain) else float("nan")

        return {
            "n_cells": int(len(results)),
            "n_datasets": len(self.dataset_names),
            "datasets": list(self.dataset_names),
            "feature_mode": self.feature_mode,
            "model": self.model_name,
            "seed": self.seed,
            "in_domain_macro_f1_mean": in_f1,
            "cross_domain_macro_f1_mean": cross_f1,
            "mean_f1_delta": float(in_f1 - cross_f1) if len(results) else float("nan"),
            "mean_f1_delta_rowwise": float(deltas["f1_delta"].mean()),
            "cross_worse_than_in_domain": bool(
                len(cross_domain) and len(in_domain) and cross_f1 < in_f1
            ),
            # Literature (research proposal section 3.2) reports cross-dataset F1
            # collapsing to ~0.01-0.05.  Retention above 1.0 means the model kept
            # *more* F1 under transfer than the documented worst case.
            "cross_f1_retention": (
                float(cross_f1 / in_f1) if in_f1 not in (0.0,) and not np.isnan(in_f1)
                else float("nan")
            ),
            "mean_test_coverage": float(results["test_coverage"].mean()),
        }

    # -- reporting ----------------------------------------------------------
    def _report_table(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse a (possibly multi-seed) matrix to one row per ordered pair.

        Cross-domain cells are scored once per seed, so the per-pair row reports
        the mean and spread across seeds together with the in-domain reference
        and the resulting delta.
        """
        seeds = sorted(results["seed"].unique().tolist())
        rows = []
        for (train_name, test_name), group in results.groupby(
            ["train_dataset", "test_dataset"], sort=False
        ):
            rows.append({
                "pair": f"{train_name}->{test_name}",
                "train_dataset": train_name,
                "test_dataset": test_name,
                "in_domain": train_name == test_name,
                "n_seeds": int(group["seed"].nunique()),
                "seeds": seeds,
                "accuracy": float(group["accuracy"].mean()),
                "precision": float(group["precision"].mean()),
                "recall": float(group["recall"].mean()),
                "f1": float(group["f1"].mean()),
                "macro_f1": float(group["macro_f1"].mean()),
                "macro_f1_std": (float(group["macro_f1"].std(ddof=1))
                                 if group["macro_f1"].size > 1 else 0.0),
                "macro_f1_ci_low": float(group["macro_f1_ci_low"].mean()) if "macro_f1_ci_low" in group else float("nan"),
                "macro_f1_ci_high": float(group["macro_f1_ci_high"].mean()) if "macro_f1_ci_high" in group else float("nan"),
                "binary_f1": float(group["binary_f1"].mean()) if "binary_f1" in group else float("nan"),
                "binary_recall": float(group["binary_recall"].mean()) if "binary_recall" in group else float("nan"),
                "binary_fpr": float(group["binary_fpr"].mean()) if "binary_fpr" in group else float("nan"),
                "binary_auroc": float(group["binary_auroc"].mean()) if "binary_auroc" in group else float("nan"),
                "binary_mcc": float(group["binary_mcc"].mean()) if "binary_mcc" in group else float("nan"),
                "majority_class": str(group["majority_class"].iloc[0]) if "majority_class" in group else "",
                "majority_macro_f1": float(group["majority_macro_f1"].mean()) if "majority_macro_f1" in group else float("nan"),
                "majority_binary_f1": float(group["majority_binary_f1"].mean()) if "majority_binary_f1" in group else float("nan"),
                "test_coverage": float(group["test_coverage"].mean()),
                "n_features": int(group["n_train_features"].iloc[0]),
                "n_features_shared": int(group["n_features_shared"].iloc[0]) if "n_features_shared" in group else int(group["n_train_features"].iloc[0]),
                "n_features_dropped_train": int(group["n_features_dropped_train"].iloc[0]) if "n_features_dropped_train" in group else 0,
                "n_features_dropped_test": int(group["n_features_dropped_test"].iloc[0]) if "n_features_dropped_test" in group else 0,
                "features_dropped": str(group["features_dropped"].iloc[0]) if "features_dropped" in group else "",
                "n_train": int(group["n_train"].mean()),
                "n_test_evaluated": int(group["n_test_evaluated"].mean()),
                "n_test_total": int(group["n_test_total"].mean()),
                "shared_labels": str(group["shared_labels"].iloc[0]),
                "test_only_labels": str(group["test_only_labels"].iloc[0]),
            })

        table = pd.DataFrame(rows)
        diagonal = {row["train_dataset"]: row["macro_f1"]
                    for _, row in table.iterrows() if row["in_domain"]}
        table["in_domain_macro_f1"] = table["train_dataset"].map(diagonal)
        table["f1_delta"] = np.where(
            table["in_domain"], 0.0, table["in_domain_macro_f1"] - table["macro_f1"]
        )
        return table

    def generate_report(self, results: pd.DataFrame, output_path: Path | str,
                        extra_notes: Optional[Sequence[str]] = None) -> Path:
        """
        Write a markdown report with the matrix, deltas and interpretation.

        Multi-seed inputs are aggregated to one row per ordered pair (mean over
        seeds plus the spread), so the table stays readable.

        Args:
            results: Matrix from :meth:`run_full_matrix`.
            output_path: Destination ``.md`` path.
            extra_notes: Extra lines appended under "Notes".

        Returns:
            The written path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = self._report_table(results)
        summary = self.summarize(results)
        seeds = table["seeds"].iloc[0]
        n_seeds = table["n_seeds"].iloc[0]
        diagonal = table[table["in_domain"]]["macro_f1"].mean()
        off = table[~table["in_domain"]]["macro_f1"].mean()
        delta = diagonal - off
        retention = f"{off / diagonal:.1%}" if diagonal else "n/a"
        retention_sentence = (
            f"({retention} of in-domain F1 retained)" if diagonal else ""
        )
        n_off = int((~table["in_domain"]).sum())
        n_off_worse = int((table.loc[~table["in_domain"], "f1_delta"] > 0).sum())
        mean_maj_f1 = float(table["majority_macro_f1"].mean()) if "majority_macro_f1" in table else float("nan")
        mean_bin_f1 = float(table["binary_f1"].mean()) if "binary_f1" in table else float("nan")

        n_datasets = len(summary['datasets'])
        n_pairs = len(table)

        lines = [
            f"# Cross-Dataset Generalization Report - `{self.feature_mode}` features",
            "",
            f"* Model: `{summary['model']}`  |  Feature mode: `{self.feature_mode}`  |  "
            f"Datasets: {', '.join(summary['datasets'])}",
            f"* Seeds: {', '.join(str(s) for s in seeds)} ({n_seeds} independent passes)",
            f"* Pairs: {n_pairs} ordered pairs ({n_datasets} datasets, diagonal included)",
            "",
            "> [!NOTE]",
            f"> Scientific inference is limited strictly to the evaluated datasets (N={n_datasets} datasets, "
            f"{n_pairs} directed pairs); pair observations share dataset marginals and are not independent.",
            "",
            "## Headline",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Mean in-domain macro-F1 | {diagonal:.4f} |",
            f"| Mean cross-domain macro-F1 | {off:.4f} |",
            f"| Mean majority-baseline macro-F1 | {mean_maj_f1:.4f} |",
            f"| Mean binary Benign-vs-Attack F1 | {mean_bin_f1:.4f} |",
            f"| Mean F1 delta (in - cross) | **{delta:.4f}** |",
            f"| Cross-domain F1 retention | {retention} |",
            f"| Mean test-label coverage | {summary['mean_test_coverage']:.1%} |",
            f"| Cross-domain worse than in-domain | "
            f"{'YES' if summary['cross_worse_than_in_domain'] else 'NO'} |",
            f"| Cross-domain pairs with a negative delta | {n_off_worse} / {n_off} |",
            "",
            "Values are means over seeds; `+/-` is the sample standard deviation "
            "across seeds.",
            "",
            "## Full Matrix (Multi-Class & Baseline)",
            "",
            "| Train | Test | In-domain | Macro-F1 (Mean +/- Std) | Row 95% CI | Maj-Class Macro-F1 | In-domain Ref | dF1 | Shared Feat. | Test Cov. |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for _, row in table.iterrows():
            delta_text = f"{row['f1_delta']:+.4f}" if not row["in_domain"] else "ref"
            ci_text = f"[{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}]" if not np.isnan(row['macro_f1_ci_low']) else "n/a"
            lines.append(
                f"| {row['train_dataset']} | {row['test_dataset']} | "
                f"{'yes' if row['in_domain'] else 'no'} | "
                f"{row['macro_f1']:.4f} +/-{row['macro_f1_std']:.4f} | "
                f"{ci_text} | {row['majority_macro_f1']:.4f} | "
                f"{row['in_domain_macro_f1']:.4f} | {delta_text} | "
                f"{int(row['n_features_shared'])} | {row['test_coverage']:.0%} |"
            )

        lines += [
            "",
            "## Binary Benign-vs-Attack Performance",
            "",
            "| Train | Test | In-domain | Binary F1 | Recall | FPR | AUROC | MCC | Majority Binary F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for _, row in table.iterrows():
            auroc_text = f"{row['binary_auroc']:.4f}" if not np.isnan(row['binary_auroc']) else "n/a"
            lines.append(
                f"| {row['train_dataset']} | {row['test_dataset']} | "
                f"{'yes' if row['in_domain'] else 'no'} | "
                f"{row['binary_f1']:.4f} | {row['binary_recall']:.4f} | "
                f"{row['binary_fpr']:.4f} | {auroc_text} | "
                f"{row['binary_mcc']:.4f} | {row['majority_binary_f1']:.4f} |"
            )

        lines += [
            "",
            "## Feature Harmonization & Drop Detail",
            "",
            "| Train | Test | Shared Features | Train Features Dropped | Test Features Dropped | Dropped Names |",
            "|---|---|---|---|---|---|",
        ]

        for _, row in table.iterrows():
            dropped_names = row['features_dropped'].replace("|", ", ") or "-"
            lines.append(
                f"| {row['train_dataset']} | {row['test_dataset']} | "
                f"{int(row['n_features_shared'])} | {int(row['n_features_dropped_train'])} | "
                f"{int(row['n_features_dropped_test'])} | {dropped_names} |"
            )

        lines += [
            "",
            "## Interpretation",
            "",
            f"The mean macro-F1 falls from {diagonal:.4f} in-domain to {off:.4f} "
            f"cross-domain, a delta of {delta:.4f} {retention_sentence}.",
            "",
            "The in-domain reference for a pair `A->B` is the diagonal cell `A->A`: "
            "the same model family and the same feature space, evaluated on "
            "held-out data from the training domain. dF1 isolates the "
            "cost of transfer from the quality of the in-domain fit.",
            "",
            "Multi-class macro-F1 is evaluated strictly over shared labels with `labels=shared_labels`.",
            "Evaluation uses the strict feature intersection between train and test domains, with zero dummy imputation.",
            "",
        ]

        if extra_notes:
            lines.append("## Notes")
            lines.append("")
            lines.extend(f"* {note}" for note in extra_notes)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


def run_ablation_grid(
    datasets: Sequence[str] = ("cic-ids-2017", "unsw-nb15"),
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
    feature_modes: Sequence[str] = ("full", "behavior_only"),
    transforms: Sequence[str] = ("raw", "log1p"),
    scalers: Sequence[str] = ("train_fitted", "per_domain"),
    models: Sequence[str] = ("xgb", "rf", "dnn", "lr"),
    output_dir: Optional[Union[Path, str]] = None,
    min_test_rows: int = 500,
    logger=None,
) -> pd.DataFrame:
    """
    Run the Phase 2 ablation grid across:
      {full, behavior_only} x {raw, log1p} x {train-fitted scaler, per-domain scaler} x {xgb, rf, dnn, lr}.

    Identifies which factor (features, log-scaling, domain adaptation, or architecture)
    recovers or explains cross-dataset transfer collapse.
    """
    out_dir = Path(output_dir) if output_dir is not None else _RESULTS_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _factory(name: str, s: int):
        name_lower = str(name).lower()
        if name_lower == "xgb":
            from src.models.baseline_xgb import XGBBaseline
            return lambda: XGBBaseline(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=s, n_jobs=4)
        elif name_lower == "rf":
            from src.models.baseline_rf import RFBaseline
            return lambda: RFBaseline(n_estimators=100, max_depth=12, random_state=s, n_jobs=4)
        elif name_lower == "dnn":
            from src.models.baseline_dnn import DNNBaseline
            return lambda: DNNBaseline(epochs=15, batch_size=128, random_state=s)
        elif name_lower == "lr":
            from sklearn.linear_model import LogisticRegression
            return lambda: LogisticRegression(max_iter=1000, random_state=s, class_weight="balanced")
        else:
            raise ValueError(f"Unknown model in ablation grid: {name}")

    all_rows = []
    total_configs = len(feature_modes) * len(transforms) * len(scalers) * len(models)
    config_idx = 0

    for f_mode in feature_modes:
        for trans in transforms:
            for sc in scalers:
                for mod in models:
                    config_idx += 1
                    if logger:
                        logger.info(
                            "[%d/%d] Ablation: mode=%s, transform=%s, scaler=%s, model=%s",
                            config_idx, total_configs, f_mode, trans, sc, mod,
                        )
                    for s in seeds:
                        evaluator = CrossDatasetEvaluator(
                            datasets=datasets,
                            model_factory=_factory(mod, s),
                            feature_mode=f_mode,
                            seed=s,
                            log1p=(trans == "log1p"),
                            per_domain_scaler=(sc == "per_domain"),
                            min_test_rows=min_test_rows,
                            model_name=mod,
                        )
                        matrix = evaluator.run_full_matrix()
                        for _, row in matrix.iterrows():
                            row_dict = dict(row)
                            row_dict["ablation_feature_mode"] = f_mode
                            row_dict["ablation_transform"] = trans
                            row_dict["ablation_scaler"] = sc
                            row_dict["ablation_model"] = mod
                            all_rows.append(row_dict)

    ablation_df = pd.DataFrame(all_rows)
    ablation_csv = out_dir / "cross_dataset_ablation.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    if logger:
        logger.info("Saved ablation grid results to %s", ablation_csv)

    return ablation_df


def cross_dataset_matrix(datasets, model_factory: Callable[[], object], **kwargs) -> pd.DataFrame:
    """Convenience wrapper: build an evaluator, run the matrix, return it."""
    evaluator = CrossDatasetEvaluator(
        datasets=datasets, model_factory=model_factory, **kwargs
    )
    return evaluator.run_full_matrix()


__all__ = [
    "CrossDatasetError",
    "CrossDatasetEvaluator",
    "CrossDatasetResult",
    "RESULT_COLUMNS",
    "cross_dataset_matrix",
    "run_ablation_grid",
]
