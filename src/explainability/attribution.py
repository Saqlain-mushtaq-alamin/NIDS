"""
src/explainability/attribution.py — Shared helpers for feature attributions.

Every explainer in this package (SHAP, DeepLIFT) produces a numeric attribution
vector per alert.  Ranking those vectors and turning them into text an analyst
can read is the same job regardless of which method produced them, so it lives
here once:

* :func:`rank_features` — top-K features by absolute contribution.
* :func:`feature_description` — one-line plain-English description of a feature
  contribution, used by the Tier-1 SOC triage summary.
* :func:`align_attributions` — coerce the array shapes the different libraries
  emit (``(n, f)``, ``(n, f, k)``, ``(k, n, f)``, lists of class matrices) onto
  the canonical ``(n, f)`` for one target class.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "align_attributions",
    "feature_description",
    "rank_features",
]


def align_attributions(values, n_samples: Optional[int] = None,
                       class_index: Optional[int] = None) -> np.ndarray:
    """
    Coerce explainer output onto the canonical ``(n_samples, n_features)`` shape.

    SHAP emits ``(n, f)`` for a single-output model, ``(n, f, k)`` for a
    ``k``-class model (and a ``list`` of ``k`` arrays in older versions), and a
    few paths produce ``(k, n, f)``.  This normalises all of them.

    Args:
        values: Raw explainer output.
        n_samples: Expected number of rows; used to disambiguate the class axis
            when ``values`` is three-dimensional.
        class_index: Class to keep when the output is multi-class.  When ``None``
            and the output is multi-class, a ``(n, f, k)`` cube is returned so
            the caller can pick a class later — except for a 2-class model,
            which is collapsed to its positive class (see below).

    Returns:
        ``(n, f)`` float array, or ``(n, f, k)`` when multi-class output was
        requested without a ``class_index``.

    A 2-class cube is always reduced to class 1 when no ``class_index`` is
    given: SHAP's tree path already returns a single ``(n, f)`` vector for a
    binary model (the contribution to the positive class), so a
    KernelExplainer wrapped around a 2-column ``predict_proba`` has to agree
    with it rather than hand the caller a shape it must special-case.
    """
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("Explainer returned an empty attribution list.")
        stacked = np.stack([np.asarray(v, dtype=float) for v in values], axis=-1)
        values = stacked

    def _binary_or_keep(cube: np.ndarray) -> np.ndarray:
        if class_index is None and cube.ndim == 3 and cube.shape[-1] == 2:
            return cube[:, :, 1]
        return cube

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported attribution shape {arr.ndim}-D; expected 1-D, 2-D or 3-D."
        )

    # 3-D: decide whether the middle or the last axis is the class axis.
    if n_samples is not None:
        if arr.shape[0] == n_samples:
            # (n, f, k) — SHAP's current layout.
            if class_index is None:
                return _binary_or_keep(arr)
            return arr[:, :, class_index]
        if arr.shape[1] == n_samples:
            # (n, k, f) — occasional legacy layout.
            if class_index is None:
                return _binary_or_keep(np.transpose(arr, (0, 2, 1)))
            return arr[:, class_index, :]
        raise ValueError(
            f"Cannot place class axis for shape {arr.shape} with {n_samples} samples."
        )

    if class_index is None:
        return _binary_or_keep(arr)
    if arr.shape[-1] <= class_index:
        raise IndexError(f"class_index {class_index} out of range for shape {arr.shape}.")
    return arr[:, :, class_index]


def rank_features(values: np.ndarray, feature_names: Sequence[str],
                  top_k: int = 5) -> list[dict]:
    """
    Return the ``top_k`` features ranked by absolute contribution.

    Accepts a single ``(f,)`` vector (one prediction) or a ``(1, f)`` matrix.
    Multi-row matrices are rejected: a "top features for this alert" list is
    meaningless for several alerts at once, so the caller must reduce first.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(
                "rank_features expects a single prediction; got shape "
                f"{arr.shape}. Use a (f,) or (1, f) array."
            )
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Unsupported attribution shape {arr.shape}.")
    if len(feature_names) != arr.shape[0]:
        raise ValueError(
            f"{len(feature_names)} feature names for {arr.shape[0]} attributions."
        )

    top_k = max(1, min(int(top_k), arr.shape[0]))
    order = np.argsort(np.abs(arr))[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "feature": str(feature_names[i]),
            "contribution": float(arr[i]),
            "direction": "increases risk" if arr[i] > 0 else "decreases risk",
        }
        for rank, i in enumerate(order)
    ]


def feature_description(feature: str, contribution: float,
                        value: Optional[float] = None,
                        baseline: Optional[float] = None,
                        unit: str = "") -> str:
    """
    Build a short plain-English line for a single feature contribution.

    ``value`` / ``baseline`` are optional; when both are present the line also
    states how far the flow sits from the reference (e.g. ``12.4x baseline``),
    which is what makes a Tier-1 summary actionable rather than decorative.
    """
    direction = "raised" if contribution > 0 else "lowered"
    strength = abs(contribution)
    text = f"{feature}: {direction} risk (SHAP {contribution:+.3f}, |weight| {strength:.3f})"

    if value is not None and baseline is not None:
        if abs(baseline) > 1e-12:
            ratio = abs(value / baseline)
            # "%.1f" would render a 200x drop as "0.0x", which reads like "no
            # change"; keep the precision where the ratio is small.
            ratio_text = f"{ratio:.1f}x" if ratio >= 0.1 else f"{ratio:.3g}x"
            text += f"; {ratio_text} baseline ({value:.4g} vs {baseline:.4g}{unit})"
        else:
            text += f"; observed {value:.4g}{unit} vs a zero baseline"
    elif value is not None:
        text += f"; observed {value:.4g}{unit}"

    # Tier-1 lines are read at a glance; keep them short enough to scan.
    if len(text) > 90:
        text = text[:87] + "..."
    return text
