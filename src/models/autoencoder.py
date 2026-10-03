"""
src/models/autoencoder.py — Benign-only autoencoder anomaly detector (FR4).

Purpose
-------
The Phase 1-2 supervised classifiers can only emit the labels they were trained
on.  This module adds the complementary *unsupervised* half of the system: a
reconstruction model fitted **only on benign traffic**, so that any traffic it
cannot reconstruct — including attack types that do not exist in any training
set — surfaces as an anomaly.

Correctness rules enforced here (each has a dedicated test in
``tests/test_autoencoder.py``):

1. **Benign-only training.**  :meth:`AnomalyDetector.fit` filters its training
   frame to ``label == "Benign"`` and records how many rows it discarded.  An
   attack row can never reach the optimiser.
2. **Threshold from validation benign only.**  The decision threshold is a
   percentile of the reconstruction error of *held-out benign validation* rows.
   Attack rows from the test set are never used to pick it, so the reported
   detection rate is not an artefact of tuning on the evaluation data.
3. **Honest in-domain accounting.**  :meth:`evaluate` reports the false-alarm
   rate on benign rows and the detection rate per attack class separately,
   together with the ROC-AUC of the raw reconstruction error.  A detector that
   simply flags everything would score a detection rate of 1.0 and is visible
   as ``f1`` far below that.
4. **Cross-dataset transfer is measured, not assumed.**
   :meth:`evaluate_cross_dataset` fits on one dataset's benign traffic and
   scores another dataset with the *training* scaler and threshold; it reports
   how far the benign-test distribution's error distribution has moved.
5. **Deterministic.**  Every random draw goes through an explicitly seeded
   :class:`numpy.random.Generator` and a seeded ``torch.manual_seed``, so the
   same seed reproduces the same losses, threshold and metrics.

The detector works in **scaled** feature space.  Fit it on the output of
:class:`src.data.preprocessor.Preprocessor` (RobustScaler), which keeps the
reconstruction error dimensionless and comparable across datasets; the
cross-dataset path reuses the *training* preprocessor for exactly that reason.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.pandas_compat import apply_pandas_compat

apply_pandas_compat()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BENIGN_LABEL = "Benign"
LABEL_COLUMN = "label"

#: Percentile presets for :meth:`AnomalyDetector.calibrate_threshold`.
THRESHOLD_PERCENTILES = (95.0, 99.0, 99.9)


class FlowAutoencoder(nn.Module):
    """
    Fully-connected encoder-decoder over a flow feature vector.

    Architecture (``PHASE_3`` section 3.1.2)::

        encoder:  Input(N) -> 64 -> 32 -> latent_dim
        decoder:  latent_dim -> 32 -> 64 -> Input(N)

    with ReLU activations and ``dropout`` between layers.  The final encoder
    layer is **linear** (no ReLU): the latent code is used directly by the
    reconstruction head, and squashing it through a non-linearity costs
    reconstruction fidelity for no benefit.

    Attributes:
        input_dim: Number of input features ``N``.
        latent_dim: Bottleneck width.
        dropout: Dropout probability applied after each hidden ReLU.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: Sequence[int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        hidden = [int(width) for width in hidden_dims if int(width) > 0]
        if not hidden:
            raise ValueError("hidden_dims must contain at least one positive width")
        if any(width < latent_dim for width in hidden):
            raise ValueError(
                f"hidden_dims {hidden} must all be >= latent_dim {latent_dim}"
            )

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)

        encoder_layers: list[nn.Module] = []
        previous = self.input_dim
        for width in hidden:
            encoder_layers += [nn.Linear(previous, width), nn.ReLU(),
                               nn.Dropout(self.dropout)]
            previous = width
        encoder_layers.append(nn.Linear(previous, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        widths = [self.latent_dim] + list(reversed(hidden))
        for index in range(len(widths) - 1):
            decoder_layers += [nn.Linear(widths[index], widths[index + 1]),
                               nn.ReLU(), nn.Dropout(self.dropout)]
        decoder_layers.append(nn.Linear(widths[-1], self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # -- forward passes ------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent code for ``x``."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction of ``x``."""
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor,
                             per_feature: bool = False) -> torch.Tensor:
        """
        Mean squared reconstruction error of ``x``.

        Args:
            x: Batch of shape ``(batch, input_dim)``.
            per_feature: Return the unreduced ``(batch, input_dim)`` squared
                error instead of the per-row mean.

        Returns:
            ``(batch,)`` (or ``(batch, input_dim)``) tensor of errors.

        .. note::
            The call runs under ``torch.no_grad()`` *and* temporarily switches
            the module to ``eval()`` mode, so dropout is inactive.  Scoring a
            model in train mode would make the threshold non-deterministic.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                squared = (x - self.forward(x)) ** 2
                return squared if per_feature else squared.mean(dim=1)
        finally:
            if was_training:
                self.train()

    # -- persistence --------------------------------------------------------
    def config(self) -> dict:
        """Return the constructor arguments needed to rebuild this module."""
        hidden = [layer.out_features for layer in self.encoder
                  if isinstance(layer, nn.Linear)][:-1]
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": hidden,
            "dropout": self.dropout,
        }

    def save(self, path: str | Path) -> Path:
        """Write weights plus a JSON sidecar describing the architecture."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config()},
                   path)
        path.with_suffix(".config.json").write_text(
            json.dumps(self.config(), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FlowAutoencoder":
        """Rebuild a model from :meth:`save` output."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


@dataclass
class TrainingHistory:
    """Per-epoch training trace, kept for reproducibility and overfitting checks."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    def final(self) -> dict:
        return {
            "epochs": len(self.train_loss),
            "initial_train_loss": self.train_loss[0] if self.train_loss else float("nan"),
            "final_train_loss": self.train_loss[-1] if self.train_loss else float("nan"),
            "initial_val_loss": self.val_loss[0] if self.val_loss else float("nan"),
            "final_val_loss": self.val_loss[-1] if self.val_loss else float("nan"),
            "train_loss_decreased": (
                bool(self.train_loss[-1] < self.train_loss[0])
                if self.train_loss else False
            ),
        }


@dataclass
class EvaluationResult:
    """Metrics for one benign-vs-attack evaluation of a fitted detector."""

    dataset: str
    percentile: float
    threshold: float
    n_benign: int
    n_attack: int
    metrics: dict
    per_class_detection: dict
    error_summary: dict
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "percentile": self.percentile,
            "threshold": self.threshold,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            **self.metrics,
            "error_summary": self.error_summary,
            "per_class_detection": self.per_class_detection,
            "notes": "; ".join(self.notes),
        }


class AutoencoderError(RuntimeError):
    """Raised when a detector cannot be fitted or evaluated."""


class AnomalyDetector:
    """
    Benign-only autoencoder anomaly detector.

    Typical use::

        detector = AnomalyDetector(latent_dim=16, percentile=99.0, seed=42)
        detector.fit(X_benign_train, X_benign_val)
        result = detector.evaluate(X_test, y_test, dataset="cic-ids-2017")

    Args:
        latent_dim: Bottleneck width.
        hidden_dims: Encoder widths, largest first.
        dropout: Dropout probability.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        percentile: Validation-benign error percentile used as the threshold.
        device: ``"cpu"``, ``"cuda"`` or ``None`` (auto-select CUDA if present).
        seed: Seed for weight initialisation and batch shuffling.
        grad_clip: Optional gradient-norm clipping threshold.
        early_stopping_patience: Stop when the validation loss stops improving
            for this many epochs.  ``0`` disables early stopping.
    """

    def __init__(self,
                 latent_dim: int = 16,
                 hidden_dims: Sequence[int] = (64, 32),
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 epochs: int = 100,
                 batch_size: int = 64,
                 percentile: float = 99.0,
                 device: Optional[str] = None,
                 seed: int = 42,
                 grad_clip: Optional[float] = 5.0,
                 early_stopping_patience: int = 0):
        if percentile <= 0.0 or percentile >= 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.percentile = float(percentile)
        self.seed = int(seed)
        self.grad_clip = grad_clip
        self.early_stopping_patience = int(early_stopping_patience)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_: Optional[FlowAutoencoder] = None
        self.threshold_: Optional[float] = None
        self.history_: TrainingHistory = TrainingHistory()
        self.n_train_rows_: int = 0
        self.n_train_benign_: int = 0
        self.n_train_rejected_: int = 0
        self.n_features_: Optional[int] = None
        self.feature_names_: list[str] = []
        self._fitted = False

    # -- internals ----------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted or self.model_ is None:
            raise AutoencoderError("AnomalyDetector must be fitted before use.")

    @staticmethod
    def _matrix(X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2:
            raise AutoencoderError(
                f"Expected a 2-D feature matrix, got shape {array.shape}"
            )
        if array.size == 0:
            raise AutoencoderError("Feature matrix is empty")
        return np.where(np.isfinite(array), array, 0.0)

    @staticmethod
    def _benign_mask(labels) -> np.ndarray:
        if labels is None:
            raise AutoencoderError(
                "Training labels are required so attack rows can be excluded."
            )
        values = pd.Series(list(labels)).astype(str).str.strip().str.lower()
        return (values == BENIGN_LABEL.lower()).to_numpy()

    def _iterate(self, X: np.ndarray, rng: np.random.Generator,
                 shuffle: bool) -> Iterable[np.ndarray]:
        n = X.shape[0]
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            yield order[start:start + self.batch_size]

    def _to_tensor(self, rows: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(rows, dtype=torch.float32, device=self.device)

    # -- training -----------------------------------------------------------
    def fit(self, X, y=None, X_val=None, y_val=None) -> "AnomalyDetector":
        """
        Fit the reconstruction model on benign traffic only.

        Args:
            X: Training feature matrix.
            y: Training labels.  When given, only rows labelled ``"Benign"`` are
                used; attack rows are dropped and counted.  When ``None``,
                ``X`` is assumed to be benign already and ``y`` must be omitted.
            X_val: Optional benign validation matrix used for the early-stopping
                signal and for threshold calibration.
            y_val: Labels for ``X_val``; attack rows are filtered out here too.

        Returns:
            ``self``.

        Raises:
            AutoencoderError: if no benign training row survives filtering, or
                if a validation matrix is supplied without labels.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        matrix = self._matrix(X)
        self.n_features_ = int(matrix.shape[1])

        if y is not None:
            mask = self._benign_mask(y)
            self.n_train_rows_ = int(mask.size)
            self.n_train_benign_ = int(mask.sum())
            self.n_train_rejected_ = self.n_train_rows_ - self.n_train_benign_
            train = matrix[mask]
        else:
            train = matrix
            self.n_train_rows_ = int(matrix.shape[0])
            self.n_train_benign_ = self.n_train_rows_
            self.n_train_rejected_ = 0

        if train.shape[0] == 0:
            raise AutoencoderError(
                "No benign training rows: the autoencoder must be fitted on "
                "benign traffic only (FR4)."
            )

        val: Optional[np.ndarray] = None
        if X_val is not None:
            if y_val is None:
                raise AutoencoderError("X_val requires y_val so attacks can be filtered out.")
            val_matrix = self._matrix(X_val)
            if val_matrix.shape[1] != self.n_features_:
                raise AutoencoderError(
                    f"X_val has {val_matrix.shape[1]} features but training had "
                    f"{self.n_features_}"
                )
            val_mask = self._benign_mask(y_val)
            val = val_matrix[val_mask] if val_mask.any() else None

        self.model_ = FlowAutoencoder(
            input_dim=self.n_features_, latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims, dropout=self.dropout,
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.history_ = TrainingHistory()
        best_val = math.inf
        best_state: Optional[dict] = None
        since_improved = 0

        for _ in range(self.epochs):
            self.model_.train()
            epoch_loss, n_batches = 0.0, 0
            for indices in self._iterate(train, rng, shuffle=True):
                batch = self._to_tensor(train[indices])
                reconstruction = self.model_(batch)
                loss = loss_fn(reconstruction, batch)
                optimiser.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                optimiser.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self.history_.train_loss.append(
                epoch_loss / max(n_batches, 1))

            if val is not None:
                val_loss = float(self._loss_on(val, loss_fn))
                self.history_.val_loss.append(val_loss)
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    since_improved = 0
                else:
                    since_improved += 1
                    if (self.early_stopping_patience > 0
                            and since_improved >= self.early_stopping_patience):
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        self._fitted = True
        if val is not None:
            self.calibrate_threshold(val, percentile=self.percentile)
        return self

    def _loss_on(self, matrix: np.ndarray, loss_fn) -> float:
        self.model_.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            tensor = self._to_tensor(matrix)
            total += float(loss_fn(self.model_(tensor), tensor).item())
            count += 1
        return total / max(count, 1)

    # -- scoring ------------------------------------------------------------
    def reconstruction_error(self, X) -> np.ndarray:
        """Return the per-row reconstruction error of ``X``."""
        self._require_fitted()
        matrix = self._matrix(X)
        if matrix.shape[1] != self.n_features_:
            raise AutoencoderError(
                f"X has {matrix.shape[1]} features but the model was fitted on "
                f"{self.n_features_}"
            )
        self.model_.eval()
        with torch.no_grad():
            tensor = self._to_tensor(matrix)
            errors = self.model_.reconstruction_error(tensor)
        return errors.detach().cpu().numpy().astype(np.float64)

    def per_feature_error(self, X) -> np.ndarray:
        """Return the ``(n, n_features)`` squared reconstruction error."""
        self._require_fitted()
        matrix = self._matrix(X)
        if matrix.shape[1] != self.n_features_:
            raise AutoencoderError(
                f"X has {matrix.shape[1]} features but the model was fitted on "
                f"{self.n_features_}"
            )
        self.model_.eval()
        with torch.no_grad():
            tensor = self._to_tensor(matrix)
            errors = self.model_.reconstruction_error(tensor, per_feature=True)
        return errors.detach().cpu().numpy().astype(np.float64)

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """Return a boolean anomaly flag per row (``error > threshold``)."""
        self._require_fitted()
        cutoff = self.threshold_ if threshold is None else float(threshold)
        if cutoff is None:
            raise AutoencoderError(
                "No threshold available: call calibrate_threshold() first or "
                "pass threshold= to predict()."
            )
        return self.reconstruction_error(X) > cutoff

    def calibrate_threshold(self, X_val, percentile: Optional[float] = None
                            ) -> float:
        """
        Set the decision threshold to a percentile of benign validation errors.

        The caller is responsible for passing **benign** rows; the honest paths
        in this module (:meth:`fit` with ``X_val`` and :meth:`fit_from_frame`)
        filter them first.
        """
        self._require_fitted()
        percentile = self.percentile if percentile is None else float(percentile)
        if percentile <= 0.0 or percentile >= 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        errors = self.reconstruction_error(X_val)
        self.threshold_ = float(np.percentile(errors, percentile))
        self.percentile = percentile
        return self.threshold_

    # -- frame-level convenience -------------------------------------------
    def fit_from_frame(self, df: pd.DataFrame, label_column: str = LABEL_COLUMN,
                       feature_columns: Optional[Sequence[str]] = None,
                       test_ratio: float = 0.2, val_ratio: float = 0.1,
                       seed: Optional[int] = None) -> "AnomalyDetector":
        """
        Fit from a labelled frame, splitting off validation rows internally.

        The split is stratified over the *whole* frame so benign and attack rows
        are separated consistently; only the **benign** rows of the training
        partition reach the network, and only the benign rows of the validation
        partition reach the threshold.
        """
        seed = self.seed if seed is None else int(seed)
        from sklearn.model_selection import train_test_split

        if label_column not in df.columns:
            raise AutoencoderError(f"'{label_column}' column missing from frame")

        columns = list(feature_columns) if feature_columns else [
            c for c in df.columns
            if c != label_column and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not columns:
            raise AutoencoderError("No numeric feature columns found in frame")
        self.feature_names_ = columns

        labels = df[label_column].astype(str).to_numpy()
        rows = np.arange(len(df))
        counts = pd.Series(labels).value_counts()

        train_idx, temp_idx = train_test_split(
            rows, test_size=test_ratio, random_state=seed, shuffle=True,
            stratify=labels if (len(counts) >= 2 and counts.min() >= 2) else None,
        )
        if val_ratio > 0:
            temp_labels = labels[temp_idx]
            temp_counts = pd.Series(temp_labels).value_counts()
            fit_idx, val_idx = train_test_split(
                temp_idx, test_size=val_ratio / (1.0 - test_ratio),
                random_state=seed, shuffle=True,
                stratify=temp_labels if (len(temp_counts) >= 2
                                         and temp_counts.min() >= 2) else None,
            )
        else:
            fit_idx, val_idx = temp_idx, np.array([], dtype=int)

        X_all = df[columns].to_numpy(dtype=np.float64, na_value=np.nan)
        X_all = np.where(np.isfinite(X_all), X_all, 0.0)

        val_X = X_all[val_idx] if val_idx.size else X_all[fit_idx][:1]
        val_y = labels[val_idx] if val_idx.size else labels[fit_idx][:1]

        return self.fit(X_all[fit_idx], labels[fit_idx], val_X, val_y)

    # -- evaluation ---------------------------------------------------------
    @staticmethod
    def _binary_metrics(y_is_attack: np.ndarray, flagged: np.ndarray) -> dict:
        """
        Benign-vs-attack metrics with "flagged" as the positive class.

        Recall is therefore the attack detection rate and the false-alarm rate
        is computed on benign rows only.  Threshold-free ranking metrics (ROC
        AUC, average precision) are added by :meth:`evaluate`, which has the
        reconstruction error available.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
        )

        truth = y_is_attack.astype(int)
        flags = flagged.astype(int)
        n_attack = int(truth.sum())
        n_benign = int(truth.size - n_attack)

        return {
            "precision": float(precision_score(truth, flags, zero_division=0)),
            "recall": float(recall_score(truth, flags, zero_division=0)),
            "f1": float(f1_score(truth, flags, zero_division=0)),
            "accuracy": float(accuracy_score(truth, flags)),
            "detection_rate": (float(flags[truth == 1].mean())
                               if n_attack else float("nan")),
            "false_alarm_rate": _far(y_is_attack, flagged),
            "flag_rate": float(flags.mean()) if flags.size else 0.0,
            "n_attack": n_attack,
            "n_benign": n_benign,
        }

    def evaluate(self, X, y, dataset: str = "test",
                 threshold: Optional[float] = None) -> EvaluationResult:
        """
        Score a held-out matrix, reporting benign and attack behaviour apart.

        Args:
            X: Evaluation features (transformed with the *training* preprocessor).
            y: Labels using the unified taxonomy.
            dataset: Name recorded in the result.
            threshold: Override the calibrated threshold (default: use it).

        Returns:
            A populated :class:`EvaluationResult`.
        """
        self._require_fitted()
        cutoff = self.threshold_ if threshold is None else float(threshold)
        if cutoff is None:
            raise AutoencoderError("No threshold calibrated; call fit() with X_val.")

        errors = self.reconstruction_error(X)
        labels = pd.Series(np.asarray(list(y))).astype(str).str.strip()
        is_attack = (labels.to_numpy() != BENIGN_LABEL)
        flagged = errors > cutoff
        # Ranking scores for ROC-AUC: a *larger* reconstruction error must mean
        # a *higher* anomaly score, so the raw error is passed through
        # unnegated.  (Negating it inverts the ranking and reports ~0 AUC for a
        # perfectly separating detector.)
        scores = errors

        metrics = self._binary_metrics(is_attack, flagged)
        metrics["roc_auc"] = _safe_auc(is_attack, scores)
        metrics["average_precision"] = _safe_average_precision(is_attack, scores)

        per_class: dict[str, dict] = {}
        for name in sorted(set(labels)):
            mask = (labels.to_numpy() == name)
            count = int(mask.sum())
            entry = {
                "n": count,
                "mean_error": float(errors[mask].mean()) if count else float("nan"),
                "median_error": float(np.median(errors[mask])) if count else float("nan"),
                "flag_rate": float(flagged[mask].mean()) if count else float("nan"),
            }
            if name != BENIGN_LABEL:
                entry["detection_rate"] = entry["flag_rate"]
                entry["is_attack"] = True
            else:
                entry["false_alarm_rate"] = entry["flag_rate"]
                entry["is_attack"] = False
            per_class[name] = entry

        notes: list[str] = []
        for name, entry in per_class.items():
            if name == BENIGN_LABEL or not np.isfinite(entry["flag_rate"]):
                continue
            if entry["flag_rate"] == 0.0:
                notes.append(f"{name}: 0% detected at the {self.percentile:g}th percentile")

        return EvaluationResult(
            dataset=dataset,
            percentile=self.percentile,
            threshold=cutoff,
            n_benign=int((~is_attack).sum()),
            n_attack=int(is_attack.sum()),
            metrics=metrics,
            per_class_detection=per_class,
            error_summary=_describe(errors, is_attack),
            notes=notes,
        )

    def evaluate_cross_dataset(self, X, y, dataset: str,
                               threshold: Optional[float] = None) -> EvaluationResult:
        """
        Score another dataset with **this** model's scaler and threshold.

        Identical arithmetic to :meth:`evaluate`; the separate name exists so the
        result is unambiguous at the call site and in the reports, because the
        honest question is not "does the detector work" but "does the threshold
        survive the domain change".
        """
        result = self.evaluate(X, y, dataset=dataset, threshold=threshold)
        if result.n_benign and result.metrics.get("false_alarm_rate", 0.0) > 0.25:
            result.notes.append(
                "benign false-alarm rate exceeds 25% under transfer: the "
                "calibrated threshold does not survive the domain change"
            )
        return result

    # -- persistence --------------------------------------------------------
    def state_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of the fitted detector."""
        self._require_fitted()
        return {
            "model_config": self.model_.config(),
            "threshold": self.threshold_,
            "percentile": self.percentile,
            "n_features": self.n_features_,
            "feature_names": list(self.feature_names_),
            "seed": self.seed,
            "epochs": self.epochs,
            "history": self.history_.final(),
            "n_train_rows": self.n_train_rows_,
            "n_train_benign": self.n_train_benign_,
            "n_train_rejected": self.n_train_rejected_,
        }

    def save(self, directory: str | Path, name: str = "autoencoder") -> dict:
        """
        Persist weights, threshold and provenance to ``directory``.

        Returns a dict of written paths.
        """
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        weights = self.model_.save(directory / f"{name}.pt")
        meta = self.state_dict()
        meta_path = directory / f"{name}_threshold.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"weights": str(weights), "metadata": str(meta_path)}

    @classmethod
    def load(cls, directory: str | Path, name: str = "autoencoder",
             device: Optional[str] = None) -> "AnomalyDetector":
        """Rebuild a detector written by :meth:`save`."""
        directory = Path(directory)
        meta = json.loads((directory / f"{name}_threshold.json").read_text(encoding="utf-8"))
        detector = cls(
            latent_dim=meta["model_config"]["latent_dim"],
            hidden_dims=meta["model_config"]["hidden_dims"],
            dropout=meta["model_config"]["dropout"],
            percentile=meta["percentile"], seed=meta["seed"], device=device,
        )
        detector.model_ = FlowAutoencoder.load(directory / f"{name}.pt").to(
            detector.device
        )
        detector.threshold_ = meta["threshold"]
        detector.n_features_ = meta["n_features"]
        detector.feature_names_ = list(meta.get("feature_names", []))
        detector._fitted = True
        return detector

    # -- introspection ------------------------------------------------------
    @property
    def n_features(self) -> int:
        return int(self.n_features_ or 0)

    @property
    def threshold(self) -> float:
        if self.threshold_ is None:
            raise AutoencoderError("Threshold has not been calibrated yet.")
        return float(self.threshold_)

    def parameter_count(self) -> int:
        self._require_fitted()
        return int(sum(p.numel() for p in self.model_.parameters()))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _far(is_attack: np.ndarray, flagged: np.ndarray) -> float:
    """False-alarm rate: fraction of *benign* rows flagged as anomalies."""
    benign = ~is_attack
    if not benign.any():
        return float("nan")
    return float(flagged[benign].mean())


def _safe_auc(is_attack: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if not is_attack.any() or is_attack.all():
        return float("nan")
    try:
        return float(roc_auc_score(is_attack.astype(int), scores))
    except ValueError:
        return float("nan")


def _safe_average_precision(is_attack: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if not is_attack.any() or is_attack.all():
        return float("nan")
    try:
        return float(average_precision_score(is_attack.astype(int), scores))
    except ValueError:
        return float("nan")


def _describe(errors: np.ndarray, is_attack: np.ndarray) -> dict:
    benign_errors = errors[~is_attack]
    attack_errors = errors[is_attack]
    summary = {
        "benign_mean": float(benign_errors.mean()) if benign_errors.size else float("nan"),
        "benign_median": float(np.median(benign_errors)) if benign_errors.size else float("nan"),
        "benign_p99": (float(np.percentile(benign_errors, 99.0))
                       if benign_errors.size else float("nan")),
        "attack_mean": float(attack_errors.mean()) if attack_errors.size else float("nan"),
        "attack_median": float(np.median(attack_errors)) if attack_errors.size else float("nan"),
    }
    if benign_errors.size and attack_errors.size:
        summary["median_separation_ratio"] = float(
            np.median(attack_errors) / max(np.median(benign_errors), 1e-12)
        )
    else:
        summary["median_separation_ratio"] = float("nan")
    return summary


def train_benign_autoencoder(train_frame: pd.DataFrame, label_column: str = LABEL_COLUMN,
                             feature_columns: Optional[Sequence[str]] = None,
                             test_ratio: float = 0.2, val_ratio: float = 0.1,
                             seed: int = 42, **kwargs) -> tuple:
    """
    Convenience wrapper returning ``(detector, preprocessor, frame)``.

    The frame is *not* preprocessed here — pass raw rows and let the returned
    :class:`~src.data.preprocessor.Preprocessor` handle scaling, which keeps the
    train/test boundary in one place (the preprocessor is fitted on the training
    rows only).

    Returns:
        ``(detector, preprocessor, train_frame)``.
    """
    from src.data.preprocessor import Preprocessor

    preprocessor = Preprocessor(
        feature_mode=kwargs.pop("feature_mode", "behavior_extended"),
        dataset=kwargs.pop("dataset", None),
    )
    splits = preprocessor.split(train_frame, test_ratio=test_ratio,
                                val_ratio=val_ratio, seed=seed)
    train_X, train_y = splits["train"]["X"], splits["train"]["y"]
    val_X, val_y = splits["val"]["X"], splits["val"]["y"]

    detector = AnomalyDetector(seed=seed, **kwargs)
    detector.fit(train_X, train_y, val_X, val_y)
    detector.feature_names_ = preprocessor.get_feature_names()
    return detector, preprocessor, train_frame


__all__ = [
    "AnomalyDetector",
    "AutoencoderError",
    "BENIGN_LABEL",
    "EvaluationResult",
    "FlowAutoencoder",
    "THRESHOLD_PERCENTILES",
    "TrainingHistory",
    "train_benign_autoencoder",
]