"""
src/models/fusion.py — multi-detector fusion (PHASE_3 section 3.4).

Scope
-----
The autoencoder and the GNN are **not** redundant: one scores each flow in
isolation, the other scores a window of host-to-host structure.  Neither can
substitute for the other, so fusion combines them rather than picking a winner.

Why weighted soft voting, and not an accuracy comparison
-------------------------------------------------------
Choosing "the better model on the test set" is model selection on held-out data:
it inflates the reported number and reports an upper bound that a deployed system
will not reproduce.  Instead the fusion weights are **fixed a priori** from the
relative reliability of each view (the per-flow detector is more specific, the
structural detector is more sensitive) and are *never* fitted against labels.
The consequence is that the fused score is a genuine combination, and the
ablation reported in ``docs/phase3_summary.md`` is an honest comparison rather
than a best-of-two.

Escalation contract
-------------------
The unified taxonomy is preserved by construction:

* ``label`` / ``final_label`` — the escalation taxonomy label (``"Benign"``,
  ``"Attack"``, or ``"NeedsReview"``).
* ``is_anomaly`` — ``True`` whenever the decision is an escalation, including
  the ``NeedsReview`` band.
* ``needs_review`` — ``True`` when the detectors disagree and their combined
  confidence sits inside the ambiguous band.  These are the rows a human must
  adjudicate, so they are flagged rather than silently resolved.

Every row records which detector(s) drove the decision, so a fused verdict is
always traceable back to its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

#: Escalation outcomes.
LABEL_BENIGN = "Benign"
LABEL_ATTACK = "Attack"
LABEL_REVIEW = "NeedsReview"

#: Output column contract, shared with the reporting layer and the tests.
FUSION_COLUMNS = (
    "flow_id",
    "timestamp",
    "label",
    "final_label",
    "is_anomaly",
    "needs_review",
    "confidence",
    "detection_source",
    "anomaly_score",
    "graph_score",
    "reconstruction_error",
    "gnn_attack_probability",
)


def _normalise(values: np.ndarray) -> np.ndarray:
    """
    Min-max scale a view's scores into ``[0, 1]``.

    A view is passed through untouched when normalisation carries no
    information — a single sample, or a constant series.  Substituting a
    fabricated "undecided" 0.5 there would overwrite the caller's actual score,
    and because the fused weight is applied afterwards, one invented value
    silently moves the final decision.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return values
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-12:
        return values
    return (values - low) / (high - low)


@dataclass
class FusionResult:
    """Fused per-flow decisions with an audit trail."""

    predictions: pd.DataFrame
    weights: dict
    agreement: dict
    n_flows: int = 0
    n_anomalies: int = 0
    n_review: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_flows": self.n_flows,
            "n_anomalies": self.n_anomalies,
            "n_review": self.n_review,
            "weights": self.weights,
            "agreement": self.agreement,
            "notes": "; ".join(self.notes),
        }

    def __len__(self) -> int:
        return self.n_flows


class DetectionFusion:
    """
    Combine per-flow anomaly scores with window-level GNN evidence.

    Args:
        anomaly_weight: Weight of the per-flow (autoencoder) view.
        graph_weight: Weight of the structural (GNN) view.
        review_band: Half-width of the ambiguous confidence band, in units of
            the 0.5 decision boundary.  ``0.2`` means a fused score in
            ``(0.3, 0.7)`` is escalated as ``NeedsReview`` when the detectors
            disagree.
        escalate_review: When ``True`` (default) a ``NeedsReview`` row counts as
            an anomaly.  When ``False`` it is neither benign nor attack and is
            left for triage.

    Example:
        >>> import numpy as np
        >>> fusion = DetectionFusion()
        >>> result = fusion.fuse(
        ...     anomaly_score=np.array([0.1, 0.2, 0.9, 0.95]),
        ...     graph_score=np.array([0.0, 0.9, 0.9, 0.1]),
        ... )
        >>> result.predictions["needs_review"].any()
        np.True_
    """

    def __init__(self, anomaly_weight: float = 0.6, graph_weight: float = 0.4,
                 review_band: float = 0.2, escalate_review: bool = True):
        if anomaly_weight < 0 or graph_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if anomaly_weight + graph_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        if not 0.0 <= review_band <= 0.5:
            raise ValueError(f"review_band must lie in [0, 0.5]; got {review_band}")
        self.anomaly_weight = float(anomaly_weight)
        self.graph_weight = float(graph_weight)
        self.review_band = float(review_band)
        self.escalate_review = bool(escalate_review)

    # -- helpers ------------------------------------------------------------
    @property
    def weights(self) -> dict:
        """Normalised fusion weights, summing to 1."""
        total = self.anomaly_weight + self.graph_weight
        return {
            "anomaly": self.anomaly_weight / total,
            "graph": self.graph_weight / total,
        }

    def effective_weights(self, graph_available: bool = True) -> dict:
        """
        Weights renormalised over the views that are actually available.

        When the structural view is missing its weight is redistributed to the
        per-flow view instead of being left as a constant zero term. Otherwise
        the single available view is permanently capped below the decision
        boundary and every confident detection lands in the review band.
        """
        if graph_available:
            return self.weights
        return {"anomaly": 1.0, "graph": 0.0}

    def fused_score(self, anomaly_score: Sequence[float],
                    graph_score: Sequence[float],
                    normalise_anomaly: bool = True,
                    normalise_graph: bool = False,
                    graph_available: bool = True) -> np.ndarray:
        """
        Return the weighted combined score in ``[0, 1]``.

        Args:
            anomaly_score: Per-flow reconstruction-error scores.
            graph_score: Structural scores.
            normalise_anomaly: Min-max scale the per-flow view.  Reconstruction
                errors are unbounded and have no absolute meaning, so they must
                be calibrated against the batch before they can be compared to
                a probability.
            normalise_graph: Leave ``False``.  The GNN view already emits
                calibrated probabilities in ``[0, 1]``; rescaling them across a
                batch would make one anomalous window inflate the scores of
                every other window in the batch.
            graph_available: When ``False`` the structural view is treated as
                absent and all weight goes to the per-flow view.
        """
        anomaly = np.asarray(anomaly_score, dtype=np.float64)
        graph = np.asarray(graph_score, dtype=np.float64)
        if normalise_anomaly:
            anomaly = _normalise(anomaly)
        if normalise_graph:
            graph = _normalise(graph)
        if anomaly.shape != graph.shape:
            raise ValueError(
                f"score length mismatch: anomaly={anomaly.shape[0]} vs "
                f"graph={graph.shape[0]}"
            )
        weights = self.effective_weights(graph_available)
        return weights["anomaly"] * anomaly + weights["graph"] * graph

    # -- fusion -------------------------------------------------------------
    def fuse(self,
             anomaly_score: Sequence[float],
             graph_score: Optional[Sequence[float]] = None,
             flow_id: Optional[Sequence] = None,
             timestamps: Optional[Sequence] = None,
             reconstruction_error: Optional[Sequence[float]] = None,
             gnn_probability: Optional[Sequence[float]] = None,
             labels: Optional[Sequence] = None) -> FusionResult:
        """
        Produce one fused decision per flow.

        Args:
            anomaly_score: Per-flow reconstruction-error score (higher is more
                anomalous).
            graph_score: Per-flow structural score obtained by broadcasting the
                window's GNN probability to the flows inside that window.  When
                ``None`` the structural view is treated as neutral (0.0) and the
                fusion degrades gracefully to the per-flow view alone.
            flow_id: Flow identifiers; positional indices when omitted.
            timestamps: Optional timestamps.
            reconstruction_error: Raw reconstruction errors, carried through for
                auditability.
            gnn_probability: Raw GNN probabilities, carried through for
                auditability.
            labels: Optional ground-truth labels, carried through for scoring
                **after** the decision is made — never used to pick a decision.

        Returns:
            A :class:`FusionResult`.

        Raises:
            ValueError: on mismatched lengths or all-missing scores.
        """
        anomaly_raw = np.asarray(anomaly_score, dtype=np.float64).ravel()
        n_flows = anomaly_raw.size
        if n_flows == 0:
            raise ValueError("fuse() received no flows")
        if not np.isfinite(anomaly_raw).any():
            raise ValueError("anomaly_score contains no finite values")

        graph_available = graph_score is not None
        if not graph_available:
            graph_raw = np.zeros(n_flows, dtype=np.float64)
            notes = ["graph_score was not supplied; fused on the per-flow view only"]
        else:
            graph_raw = np.asarray(graph_score, dtype=np.float64).ravel()
            if graph_raw.size != n_flows:
                raise ValueError(
                    f"graph_score has {graph_raw.size} entries but anomaly_score "
                    f"has {n_flows}"
                )
            notes = []
            if not np.isfinite(graph_raw).any():
                notes.append("graph_score had no finite values; "
                             "treated the structural view as neutral")
                graph_available = False

        fused = self.fused_score(anomaly_raw, graph_raw,
                                 graph_available=graph_available)

        # The two views are compared against 0.5 using the same calibration as
        # the fused score, so a column and the decision it explains can never
        # disagree. With no structural view there is nothing to disagree with,
        # so agreement is assumed rather than manufactured.
        anomaly_flag = (_normalise(anomaly_raw) if anomaly_raw.size > 1
                        else anomaly_raw) >= 0.5
        graph_flag = graph_raw >= 0.5
        agree = (anomaly_flag == graph_flag) if graph_available else np.ones(
            n_flows, dtype=bool)

        lower = 0.5 - self.review_band
        upper = 0.5 + self.review_band
        disputed = (~agree) & (fused > lower) & (fused < upper)
        confident_attack = fused >= upper
        confident_benign = fused <= lower

        needs_review = disputed
        is_anomaly = confident_attack | (needs_review if self.escalate_review
                                         else False)
        label = np.where(needs_review, LABEL_REVIEW,
                         np.where(confident_attack, LABEL_ATTACK, LABEL_BENIGN))

        source = np.where(
            (anomaly_flag & graph_flag) | ~agree & confident_attack,
            "both",
            np.where(anomaly_flag, "anomaly_detector", "graph_detector"),
        )
        source = np.where(needs_review, "both_disputed", source)

        identifiers = (list(flow_id) if flow_id is not None
                       else list(range(n_flows)))
        frame = pd.DataFrame({
            "flow_id": identifiers,
            "timestamp": (list(timestamps) if timestamps is not None
                          else [None] * n_flows),
            "label": label,
            # ``final_label`` is retained as an explicit alias: downstream
            # consumers and the phase spec refer to the same column by both
            # names, and silently emitting only one of them breaks whichever
            # name the reader expected.
            "final_label": label,
            "is_anomaly": is_anomaly,
            "needs_review": needs_review,
            "confidence": fused,
            "detection_source": source,
            "anomaly_score": (_normalise(anomaly_raw) if anomaly_raw.size > 1
                              else anomaly_raw),
            "graph_score": graph_raw,
            "reconstruction_error": (
                np.asarray(reconstruction_error, dtype=np.float64).ravel()
                if reconstruction_error is not None else anomaly_raw),
            "gnn_attack_probability": (
                np.asarray(gnn_probability, dtype=np.float64).ravel()
                if gnn_probability is not None else graph_raw),
        })
        if labels is not None:
            frame["true_label"] = list(labels)

        if not self.escalate_review and int(needs_review.sum()):
            notes.append(
                f"{int(needs_review.sum())} disputed flows left un-escalated "
                "because escalate_review=False"
            )

        return FusionResult(
            predictions=frame,
            weights=self.effective_weights(graph_available),
            agreement={
                "rate": float(agree.mean()) if n_flows else float("nan"),
                "n_agree": int(agree.sum()),
                "n_disputed": int((~agree).sum()),
            },
            n_flows=n_flows,
            n_anomalies=int(is_anomaly.sum()),
            n_review=int(needs_review.sum()),
            notes=notes,
        )

    # -- scoring ------------------------------------------------------------
    @staticmethod
    def score(result: FusionResult, label_column: str = "true_label",
              attack_labels: Sequence[str] = ("Attack",)
              ) -> dict:
        """
        Score a fused result against ground truth.

        Args:
            result: Output of :meth:`fuse` that carries ``labels``.
            label_column: Column holding the ground truth.
            attack_labels: Ground-truth values that count as a positive.

        Returns:
            A dict of metrics, or ``{}`` when no ground truth is present.

        Note:
            Scoring happens strictly after fusion, so the labels never
            influence a decision.
        """
        frame = result.predictions
        if label_column not in frame.columns:
            return {}

        truth = frame[label_column].astype(str).isin(set(attack_labels)).to_numpy()
        predicted = frame["is_anomaly"].to_numpy().astype(bool)

        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        tn = int((~truth & ~predicted).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        return {
            "accuracy": float((truth == predicted).mean()) if truth.size else float("nan"),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_ground_truth": int(truth.size),
        }


def run_fusion_experiment(seed: int = 42, n_flows: int = 2000) -> FusionResult:
    """
    Build a synthetic stream and fuse the two detector views over it.

    The per-flow view reacts to high-volume single flows; the structural view
    reacts to collective behaviour spread across many individually-normal
    flows.  Both score the same stream and their fusion is returned with ground
    truth attached for scoring.
    """
    rng = np.random.default_rng(seed)
    n = int(n_flows)

    truth = np.zeros(n, dtype=int)
    # Exfiltration: individually unremarkable, collectively enormous.
    exfil = slice(0, int(n * 0.15))
    truth[exfil] = 1
    # Volumetric attack: individually large, easy for the per-flow view.
    volumetric = slice(int(n * 0.15), int(n * 0.30))
    truth[volumetric] = 1

    anomaly_score = rng.normal(0.2, 0.05, size=n)
    anomaly_score[volumetric] = rng.normal(0.9, 0.05, size=truth[volumetric].sum())
    # Exfil flows stay near-normal per-flow: this is exactly the case the
    # per-flow detector cannot see on its own.
    graph_score = rng.normal(0.2, 0.05, size=n)
    graph_score[exfil] = rng.normal(0.9, 0.05, size=truth[exfil].sum())
    graph_score[volumetric] = rng.normal(0.5, 0.15,
                                         size=truth[volumetric].sum())

    labels = np.where(truth == 1, "Attack", "Benign")
    return DetectionFusion().fuse(anomaly_score, graph_score,
                                  flow_id=list(range(n)), labels=labels)


__all__ = [
    "DetectionFusion",
    "FUSION_COLUMNS",
    "FusionResult",
    "LABEL_ATTACK",
    "LABEL_BENIGN",
    "LABEL_REVIEW",
    "run_fusion_experiment",
]