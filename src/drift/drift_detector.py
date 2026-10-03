"""
src/drift/drift_detector.py — concept-drift monitoring (PHASE_3 section 3.3).

Scope
-----
Concept drift is the case where a detector keeps running correctly at
*inference* time while the relationship it learned no longer holds.  This module
implements the three monitors the phase plan requires, all driven by
:mod:`river`'s ``ADWIN``:

1. **Feature-magnitude drift** — per-feature ADWIN on the top-K most variable
   features, plus Population Stability Index (PSI) as a second, interpretable
   signal.
2. **Prediction-confidence drift** — ADWIN over the detector's confidence.
   Mean confidence collapsing while inputs look unchanged is the signature of a
   model that has quietly stopped fitting.
3. **Detection-rate drift** — the operational consequence: a sudden change in
   the fraction of flows flagged.

Why ADWIN
---------
ADWIN maintains a window of variable size and splits it whenever a
statistically significant change in the mean is detected, so it needs no
retraining trigger threshold and no fixed window length.  Its ``delta`` is the
*false-positive rate bound* the maintainer accepts, not an effect size: a
smaller delta means more sensitivity and more alerts.

Honesty constraint
------------------
Drift is reported as **evidence to investigate**, never as a diagnosis.  A
feature may legitimately shift because the network was re-addressed or a new
business application went live.  Every alert therefore carries its direction,
magnitude and contributing features so a human can adjudicate; nothing here
automatically retrains a model or blocks traffic.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two samples of one feature.

    Args:
        expected: Reference (training) sample.
        actual: Current sample.
        bins: Number of equal-width quantile bins taken from ``expected``.

    Returns:
        PSI, or ``nan`` when the reference sample is degenerate.

    Note:
        PSI is only interpretable against its conventional thresholds
        (<0.1 stable, 0.1-0.25 moderate, >0.25 significant).  Empty bins are
        replaced with a small floor rather than dropped, because a dropped bin
        silently understates the divergence.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size < 2 or actual.size < 2:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 3:
        return float("nan")

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    floor = 1e-6
    # The denominators must be the bin *counts*, not the sum of the raw values.
    # For any feature whose values average near zero (most standardised flow
    # features) ``expected.sum()`` is ~0, the percentage vector stops being a
    # set of proportions, and PSI inflates by several orders of magnitude.
    expected_pct = np.maximum(
        expected_counts / max(float(expected_counts.sum()), 1.0), floor)
    actual_pct = np.maximum(
        actual_counts / max(float(actual_counts.sum()), 1.0), floor)

    return float(np.sum((actual_pct - expected_pct)
                        * np.log(actual_pct / expected_pct)))


@dataclass
class DriftAlert:
    """A single drift notification."""

    feature: str
    kind: str
    timestamp: Optional[object] = None
    window: Optional[int] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    psi: Optional[float] = None
    magnitude: Optional[float] = None
    z_score: Optional[float] = None
    direction: Optional[str] = None
    severity: str = "info"
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "window": self.window,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "psi": self.psi,
            "magnitude": self.magnitude,
            "z_score": self.z_score,
            "direction": self.direction,
            "severity": self.severity,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.kind}/{self.feature}: {self.message}"


def _severity_from_z(abs_z: float) -> str:
    """Map an absolute z-score onto the alert severity ladder."""
    if abs_z >= 3.0:
        return "high"
    if abs_z >= 1.0:
        return "medium"
    return "low"


@dataclass
class DriftMonitorReport:
    """Aggregated drift state over a monitored stream."""

    n_samples: int = 0
    n_alerts: int = 0
    alerts: list[DriftAlert] = field(default_factory=list)
    feature_stats: Optional[pd.DataFrame] = None
    detection_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_alerts": self.n_alerts,
            "detection_rate": self.detection_rate,
            "alerts": [alert.to_dict() for alert in self.alerts],
            "feature_stats": (self.feature_stats.to_dict("records")
                              if self.feature_stats is not None else None),
        }


class DriftDetector:
    """
    Per-feature ADWIN drift detection with PSI attribution.

    Args:
        top_k: Number of highest-variance features to monitor.  Monitoring every
            feature is noisy and slow; the most variable ones drift first and
            dominate the report.
        delta: ADWIN false-positive bound.  Smaller is more sensitive.
        clock: ADWIN bucket-exchange period.
        min_window_length: Minimum window before ADWIN will declare drift.
        grace_period: Samples to ignore before drift may be reported.
        psi_threshold: PSI above which a feature is flagged as significant.
        seed: Reserved for reproducibility of any tie-breaking.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> base = pd.DataFrame(rng.normal(0, 1, size=(400, 6)),
        ...                     columns=list("abcdef"))
        >>> detector = DriftDetector(top_k=3)
        >>> _ = detector.fit(base)
        >>> shifted = pd.DataFrame(rng.normal(4, 1, size=(400, 6)),
        ...                        columns=list("abcdef"))
        >>> report = detector.monitor(shifted)
        >>> report.n_alerts > 0
        True
    """

    def __init__(self, top_k: int = 10, delta: float = 0.002, clock: int = 32,
                 min_window_length: int = 5, grace_period: int = 10,
                 psi_threshold: float = 0.25, seed: int = 42):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        if psi_threshold < 0:
            raise ValueError(f"psi_threshold must be non-negative; got {psi_threshold}")
        self.top_k = int(top_k)
        self.delta = float(delta)
        self.clock = int(clock)
        self.min_window_length = int(min_window_length)
        self.grace_period = int(grace_period)
        self.psi_threshold = float(psi_threshold)
        self.seed = int(seed)

        self.reference_: Optional[pd.DataFrame] = None
        self.monitored_features_: list[str] = []
        self._detectors: dict[str, object] = {}
        self._reference_means: dict[str, float] = {}
        self._baseline: dict[str, np.ndarray] = {}
        self._baseline_detections: dict[str, int] = {}
        self._detection_counts: dict[str, int] = {}
        self._scalar_detections: dict[str, int] = {}
        self._n_seen = 0
        self._confidence_detector = None
        self._detection_detector = None
        self._fitted = False

    # -- setup --------------------------------------------------------------
    def _new_detector(self):
        from river.drift import ADWIN

        return ADWIN(delta=self.delta, clock=self.clock,
                     min_window_length=self.min_window_length,
                     grace_period=self.grace_period)

    def fit(self, features: pd.DataFrame) -> "DriftDetector":
        """
        Establish the reference distribution and select features to monitor.

        Args:
            features: Training-window feature matrix.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: when ``features`` is empty or has no numeric column.
        """
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty or len(numeric) == 0:
            raise ValueError(
                "fit() needs a non-empty frame with at least one numeric column"
            )

        self.reference_ = numeric.reset_index(drop=True).copy()
        variances = self.reference_.var(ddof=0).sort_values(ascending=False)
        # Drop all-constant features: ADWIN cannot detect a change in a series
        # that never varies, and they would otherwise dominate ``top_k``.
        variances = variances[variances > 0]
        self.monitored_features_ = list(variances.index[:self.top_k])

        self._detectors = {name: self._new_detector()
                           for name in self.monitored_features_}
        self._reference_means = {
            name: float(self.reference_[name].mean())
            for name in self.monitored_features_
        }
        self._reference_stds = {
            name: float(self.reference_[name].std(ddof=0))
            for name in self.monitored_features_
        }
        self._baseline = {
            name: self.reference_[name].to_numpy(dtype=np.float64)
            for name in self.monitored_features_
        }
        self._confidence_detector = self._new_detector()
        self._detection_detector = self._new_detector()
        self._n_seen = 0
        self._fitted = True

        # Prime ADWIN with the reference samples.  A fresh ADWIN only detects a
        # change *within* the values it observes, so an unprimed monitor handed
        # a stream that is already shifted sees a stationary window and stays
        # silent while PSI — which compares against the stored reference — fires.
        # Priming gives both monitors the same baseline.
        for name in self.monitored_features_:
            for value in self._baseline[name]:
                self._detectors[name].update(float(value))
        # Detections already accumulated during priming, so a later report can
        # attribute drift to the monitored batch rather than to the baseline.
        self._baseline_detections = {
            name: 0 for name in self.monitored_features_
        }
        self._detection_counts = {name: 0 for name in self.monitored_features_}
        self._scalar_detections = {"confidence": 0, "detection_rate": 0}
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DriftDetector.fit() must be called before monitor()")

    # -- monitoring ---------------------------------------------------------
    def monitor(self, features: pd.DataFrame,
                confidences: Optional[Sequence[float]] = None,
                detections: Optional[Sequence[int]] = None,
                timestamps: Optional[Sequence[object]] = None,
                window_size: int = 0) -> DriftMonitorReport:
        """
        Feed a batch of observations to the monitors.

        Args:
            features: Feature matrix for this batch; monitored columns are
                selected by name, so extra columns are ignored and missing ones
                raise.
            confidences: Per-sample detector confidence in ``[0, 1]``.  Enables
                the confidence-drift monitor.
            detections: Per-sample binary detection.  Enables the
                detection-rate monitor.
            timestamps: Optional per-sample timestamps copied onto alerts.
            window_size: When > 0, PSI is computed on the trailing
                ``window_size`` samples instead of the whole stream, so PSI stays
                responsive instead of being averaged over all history.

        Returns:
            A :class:`DriftMonitorReport` listing every alert raised in this
            batch.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
            ValueError: if a monitored column is absent from ``features``, or the
                optional sequences do not match the batch length.
        """
        self._require_fitted()

        missing = [name for name in self.monitored_features_
                   if name not in features.columns]
        if missing:
            raise ValueError(
                f"Monitored feature(s) absent from the batch: {missing[:5]}"
            )

        n_rows = len(features)
        for label, series in (("confidences", confidences),
                              ("detections", detections),
                              ("timestamps", timestamps)):
            if series is not None and len(series) != n_rows:
                raise ValueError(
                    f"{label} has {len(series)} entries but the batch has "
                    f"{n_rows} rows"
                )

        alerts: list[DriftAlert] = []
        recent: dict[str, list[float]] = {name: [] for name in self.monitored_features_}

        for position, (_, row) in enumerate(features.iterrows()):
            timestamp = timestamps[position] if timestamps is not None else None
            for name in self.monitored_features_:
                value = float(row[name])
                if not np.isfinite(value):
                    continue
                detector = self._detectors[name]
                detector.update(value)
                recent[name].append(value)
                self._n_seen += 1

                if getattr(detector, "drift_detected", False):
                    # Counted here rather than via River's ``n_detections``:
                    # on the 0.22 C-backed build that counter stays at 0 even
                    # when drift is raised.
                    self._detection_counts[name] += 1
                    reference = self._reference_means[name]
                    current = float(np.mean(recent[name][-256:]))
                    delta = current - reference
                    # Express the shift in reference standard deviations.  A
                    # percentage of the mean is meaningless here: standardised
                    # flow features routinely have a mean within 0.001 of zero,
                    # which turns a small real shift into a "100000%" figure.
                    std = self._reference_stds.get(name, 0.0)
                    z_score = delta / std if std > 0 else float("inf")
                    alerts.append(DriftAlert(
                        feature=name,
                        kind="feature_drift",
                        timestamp=timestamp,
                        window=int(getattr(detector, "width", 0)),
                        reference_mean=reference,
                        current_mean=current,
                        magnitude=float(delta),
                        z_score=float(z_score),
                        direction="increase" if delta > 0 else "decrease",
                        severity=_severity_from_z(abs(z_score)),
                        message=(
                            f"{name} shifted {delta:+.4g} "
                            f"({z_score:+.1f} reference SD, reference mean "
                            f"{reference:.4g} +/- {std:.4g}) over ADWIN width "
                            f"{int(getattr(detector, 'width', 0))}"
                        ),
                    ))

        for name in self.monitored_features_:
            tail = (np.asarray(recent[name][-window_size:], dtype=np.float64)
                    if window_size > 0 else np.asarray(recent[name], dtype=np.float64))
            psi = _psi(self._baseline[name], tail) if tail.size else float("nan")
            if np.isfinite(psi) and psi >= self.psi_threshold:
                alerts.append(DriftAlert(
                    feature=name,
                    kind="psi_shift",
                    timestamp=timestamps[-1] if timestamps is not None else None,
                    psi=float(psi),
                    severity=("high" if psi >= 0.5 else "medium"),
                    message=(
                        f"PSI({name}) = {psi:.3f} >= {self.psi_threshold:g} "
                        f"({'significant' if psi >= 0.5 else 'moderate'} shift)"
                    ),
                ))

        if confidences is not None:
            confidence_alert = self._monitor_scalar(
                self._confidence_detector, "confidence", confidences,
                timestamps, "Detection confidence")
            if confidence_alert is not None:
                alerts.append(confidence_alert)

        detection_rate = None
        if detections is not None:
            detection_rate = float(np.mean(np.asarray(detections, dtype=float)))
            detection_alert = self._monitor_scalar(
                self._detection_detector, "detection_rate", detections,
                timestamps, "Detection rate")
            if detection_alert is not None:
                alerts.append(detection_alert)

        stats = self._summary(recent, detection_rate)
        return DriftMonitorReport(n_samples=n_rows, n_alerts=len(alerts),
                                  alerts=alerts, feature_stats=stats,
                                  detection_rate=detection_rate)

    def _monitor_scalar(self, detector, name: str, values: Sequence[float],
                        timestamps, pretty: str) -> Optional[DriftAlert]:
        """
        Push one scalar stream through ADWIN and build an alert if it drifts.

        The ``drift_detected`` flag is inspected **inside** the update loop on
        purpose: ADWIN raises it only on the update that detects the change and
        clears it on the next one.  Sampling it once after the loop therefore
        reports "no drift" for a stream that did drift.
        """
        fired = False
        fired_at = None
        for index, value in enumerate(values):
            detector.update(float(value))
            if getattr(detector, "drift_detected", False):
                fired = True
                fired_at = index
        if not fired:
            return None
        self._scalar_detections[name] = self._scalar_detections.get(name, 0) + 1

        current = float(np.mean([float(v) for v in values])) if len(values) else float("nan")
        estimate = float(getattr(detector, "estimation", current))
        direction = "increase" if current > estimate else "decrease"
        return DriftAlert(
            feature=name,
            kind="prediction_drift" if name == "confidence" else "detection_rate_drift",
            timestamp=(timestamps[fired_at]
                       if timestamps is not None and fired_at is not None
                       else (timestamps[-1] if timestamps is not None else None)),
            window=int(getattr(detector, "width", 0)),
            current_mean=current,
            direction=direction,
            severity="medium",
            message=(f"{pretty} drifted {direction} (current mean {current:.4g}, "
                     f"ADWIN estimation {estimate:.4g}); "
                     f"investigate retraining need"),
        )

    def _summary(self, recent: dict[str, list[float]],
                 detection_rate: Optional[float]) -> pd.DataFrame:
        """Build the per-feature drift summary table."""
        rows = []
        for name in self.monitored_features_:
            values = np.asarray(recent[name], dtype=np.float64)
            if values.size == 0:
                continue
            detector = self._detectors[name]
            reference = self._reference_means[name]
            current = float(values[-256:].mean())
            tail = values[-100:] if values.size > 100 else values
            psi = _psi(self._baseline[name], tail)
            detections = self._detection_counts.get(name, 0)
            rows.append({
                "feature": name,
                "n_samples": int(values.size),
                "reference_mean": reference,
                "current_mean": current,
                "reference_std": float(self.reference_[name].std(ddof=0)),
                "current_std": float(values.std(ddof=0)),
                "shift": current - reference,
                "psi": psi,
                "adwin_drift_detected": detections > 0,
                "adwin_detections": detections,
                "adwin_width": int(getattr(detector, "width", 0)),
            })
        if detection_rate is not None:
            detections = self._scalar_detections.get("detection_rate", 0)
            rows.append({
                "feature": "detection_rate",
                "n_samples": 0,
                "reference_mean": float("nan"),
                "current_mean": detection_rate,
                "reference_std": float("nan"),
                "current_std": float("nan"),
                "shift": float("nan"),
                "psi": float("nan"),
                "adwin_drift_detected": detections > 0,
                "adwin_detections": detections,
                "adwin_width": int(getattr(self._detection_detector, "width", 0)),
            })
        return pd.DataFrame(rows)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the reference statistics, monitor selection and ADWIN state."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({
                "top_k": self.top_k,
                "delta": self.delta,
                "clock": self.clock,
                "min_window_length": self.min_window_length,
                "grace_period": self.grace_period,
                "psi_threshold": self.psi_threshold,
                "seed": self.seed,
                "reference": self.reference_,
                "monitored_features": self.monitored_features_,
                "reference_means": self._reference_means,
                "detectors": self._detectors,
                "detection_counts": self._detection_counts,
                "scalar_detections": self._scalar_detections,
                "confidence_detector": self._confidence_detector,
                "detection_detector": self._detection_detector,
                "n_seen": self._n_seen,
            }, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DriftDetector":
        """Restore a detector written by :meth:`save`."""
        import pickle

        with Path(path).open("rb") as handle:
            blob = pickle.load(handle)
        detector = cls(top_k=blob["top_k"], delta=blob["delta"], clock=blob["clock"],
                       min_window_length=blob["min_window_length"],
                       grace_period=blob["grace_period"],
                       psi_threshold=blob["psi_threshold"], seed=blob["seed"])
        detector.reference_ = blob["reference"]
        detector.monitored_features_ = blob["monitored_features"]
        detector._reference_means = blob["reference_means"]
        detector._detectors = blob["detectors"]
        detector._detection_counts = blob.get("detection_counts",
                                              {name: 0
                                               for name in detector.monitored_features_})
        detector._scalar_detections = blob.get("scalar_detections",
                                               {"confidence": 0,
                                                "detection_rate": 0})
        detector._baseline = {
            name: detector.reference_[name].to_numpy(dtype=np.float64)
            for name in detector.monitored_features_
        }
        detector._confidence_detector = blob["confidence_detector"]
        detector._detection_detector = blob["detection_detector"]
        detector._n_seen = blob["n_seen"]
        detector._fitted = True
        return detector


def simulate_concept_drift(n_samples: int = 600, n_features: int = 8,
                           drift_at: Optional[int] = None, seed: int = 42,
                           labels: Optional[Sequence[str]] = None
                           ) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a feature stream whose distribution shifts partway through.

    Args:
        n_samples: Number of rows.
        n_features: Number of features.
        drift_at: Row index where the mean shifts.  Defaults to the midpoint.
        seed: RNG seed.
        labels: Optional per-row label used to make the shift label-dependent.

    Returns:
        ``(features, labels)`` where ``features`` is a frame of numeric columns.

    Note:
        This produces a *synthetic* distribution shift.  It demonstrates that the
        monitors fire and that ADWIN attributes the change to the right
        features; it is not evidence about drift in real network traffic.
    """
    rng = np.random.default_rng(seed)
    if drift_at is None:
        drift_at = n_samples // 2

    columns = [f"f{index}" for index in range(n_features)]
    before = rng.normal(0.0, 1.0, size=(drift_at, n_features))
    # Half the features shift sharply, half stay put: a monitor that flags
    # everything would be indistinguishable from one that flags the right subset.
    shift = np.zeros(n_features)
    shift[: max(1, n_features // 2)] = 4.0
    after = rng.normal(0.0, 1.0, size=(n_samples - drift_at, n_features)) + shift

    frame = pd.DataFrame(np.vstack([before, after]), columns=columns)
    if labels is not None:
        frame.attrs["labels"] = list(labels)
    return frame, (np.arange(n_samples))


def run_drift_experiment(seed: int = 42, n_features: int = 8,
                         top_k: int = 4) -> DriftMonitorReport:
    """
    Fit on a stable segment, then monitor a stream that drifts.

    Returns:
        The report produced by monitoring the drifting stream.
    """
    features, _ = simulate_concept_drift(n_samples=600, n_features=n_features,
                                         seed=seed)
    stable = features.iloc[:300].reset_index(drop=True)
    drifting = features.iloc[300:].reset_index(drop=True)

    detector = DriftDetector(top_k=top_k, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector.fit(stable)
        return detector.monitor(drifting, window_size=200)


__all__ = [
    "DriftAlert",
    "DriftDetector",
    "DriftMonitorReport",
    "run_drift_experiment",
    "simulate_concept_drift",
]