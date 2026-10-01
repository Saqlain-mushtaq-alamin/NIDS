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
    "mcc",
    "n_train",
    "n_test_total",
    "n_test_evaluated",
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
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "in_domain": self.train_dataset == self.test_dataset,
            "model": self.model,
            "feature_mode": self.feature_mode,
            "seed": self.seed,
            "accuracy": self.metrics["accuracy"],
            "precision": self.metrics["precision"],
            "recall": self.metrics["recall"],
            "f1": self.metrics["f1"],
            "macro_f1": self.metrics["macro_f1"],
            "mcc": self.metrics["mcc"],
            "n_train": self.n_train,
            "n_test_total": self.n_test_total,
            "n_test_evaluated": self.n_test_evaluated,
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

        preprocessor = self._build_preprocessor(train_dataset)
        train_X, train_y, test_X, test_y = self._prepare(
            preprocessor, train_frame, test_frame,
            in_domain=(train_dataset == test_dataset), seed=seed,
        )

        warnings: list[str] = []
        if preprocessor.missing_features_:
            warnings.append(
                f"training dataset '{train_dataset}' cannot supply "
                f"{len(preprocessor.missing_features_)} requested features: "
                f"{preprocessor.missing_features_}"
            )

        model = self._build_model()
        model.fit(train_X, train_y)

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
        if n_eval < n_total:
            warnings.append(
                f"{n_total - n_eval} test rows dropped: labels "
                f"{test_only} are absent from the training set "
                f"(coverage {n_eval / n_total:.1%})"
            )

        y_true = test_y[keep]
        y_pred = _as_array(model.predict(test_X[keep]))

        # A prediction outside the shared set is an error, never filtered out.
        metrics = compute_metrics(y_true, y_pred, average="weighted")

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
            model=self.model_name,
            feature_mode=self.feature_mode,
            seed=seed,
            warnings=warnings,
        )
        result.per_class = per_class_report(y_true, y_pred)
        result.confusion_matrix = compute_confusion_matrix(y_true, y_pred)
        return result

    def _prepare(self, preprocessor: Preprocessor, train_frame: pd.DataFrame,
                 test_frame: pd.DataFrame, in_domain: bool, seed: int):
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
        )
        train = splits["train"]
        if in_domain:
            return train["X"], train["y"], splits["test"]["X"], splits["test"]["y"]

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
                "test_coverage": float(group["test_coverage"].mean()),
                "n_features": int(group["n_train_features"].iloc[0]),
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

        lines = [
            f"# Cross-Dataset Generalization Report - `{self.feature_mode}` features",
            "",
            f"* Model: `{summary['model']}`  |  Feature mode: `{self.feature_mode}`  |  "
            f"Datasets: {', '.join(summary['datasets'])}",
            f"* Seeds: {', '.join(str(s) for s in seeds)} ({n_seeds} independent passes)",
            f"* Pairs: {len(table)} ordered pairs "
            f"({len(summary['datasets'])} datasets, diagonal included)",
            "",
            "## Headline",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Mean in-domain macro-F1 | {diagonal:.4f} |",
            f"| Mean cross-domain macro-F1 | {off:.4f} |",
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
            "## Full matrix",
            "",
            "| Train | Test | In-domain | ACC | P | R | F1 | macro-F1 | "
            "In-domain ref | dF1 | Test cov. | Feat. |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for _, row in table.iterrows():
            delta_text = f"{row['f1_delta']:+.4f}" if not row["in_domain"] else "ref"
            lines.append(
                f"| {row['train_dataset']} | {row['test_dataset']} | "
                f"{'yes' if row['in_domain'] else 'no'} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['macro_f1']:.4f} +/-{row['macro_f1_std']:.4f} | "
                f"{row['in_domain_macro_f1']:.4f} | {delta_text} | "
                f"{row['test_coverage']:.0%} | {int(row['n_features'])} |"
            )

        lines += [
            "",
            "## Per-dataset detail",
            "",
            "| Train | Test | Shared labels | Test-only labels | Test rows evaluated | Note |",
            "|---|---|---|---|---|---|",
        ]

        for _, row in table.iterrows():
            shared = row["shared_labels"].replace("|", ", ") or "-"
            only = row["test_only_labels"].replace("|", ", ") or "-"
            note = "label subset excluded" if row["test_coverage"] < 1.0 else "-"
            lines.append(
                f"| {row['train_dataset']} | {row['test_dataset']} | {shared} | "
                f"{only} | {int(row['n_test_evaluated']):,} / "
                f"{int(row['n_test_total']):,} | {note} |"
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
            "held-out data from the training domain. dF1 therefore isolates the "
            "cost of transfer from the quality of the in-domain fit.",
            "",
            "Cells with `Test cov. < 100%` are scored only on labels the model was "
            "trained to emit; the excluded rows are listed above rather than "
            "silently dropped. Predictions that fall outside the trained label set "
            "are counted as errors.",
            "",
        ]

        if extra_notes:
            lines.append("## Notes")
            lines.append("")
            lines.extend(f"* {note}" for note in extra_notes)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


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
]
