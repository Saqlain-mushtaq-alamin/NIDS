"""
src/explainability/faithfulness.py — Do the attributions explain the model?

An attribution can *look* convincing and still be wrong: any method can return
a ranked feature list, and an analyst has no way to tell from the list alone
whether those features actually drive the prediction.  Faithfulness measures
answer that question by intervening on the input (PHASE_4 section 4.3).

**Deletion AUC** is the metric used here:

1. start with the full flow,
2. progressively mask the top-``k`` attributed features (replace them with the
   baseline value, which for robust-scaled features is the training median),
3. record the model's probability for the explained class,
4. take the area under the resulting curve.

Masking features the model actually relies on should collapse the predicted
probability, so **lower AUC = more faithful**.  A method whose top features are
decorative will keep the probability high all the way to full masking and score
close to 1.

Probabilities are clipped to ``[0, 1]`` before integration: a metric bounded by
the geometry of the model's output must not depend on a scorer that returns
out-of-range values (the plan's mock scorer does exactly that).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

__all__ = ["as_predict_proba", "compare_explainers", "deletion_auc",
           "deletion_curve"]

try:  # NumPy >= 2.0 renamed trapz
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - NumPy 1.x
    _trapezoid = np.trapz  # type: ignore[attr-defined]


def as_predict_proba(model, class_index: Optional[Union[int, Sequence[int]]] = None
                     ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap ``model`` so it scores ``(n, f)`` input into a 1-D probability vector.

    Accepts:

    * an estimator with ``predict_proba`` (returns ``(n, k)``),
    * a ``torch.nn.Module`` (returns logits; softmax is applied),
    * a bare callable returning ``(n,)``, ``(n, 1)`` or ``(n, k)``.

    ``class_index`` selects the explained class and may be a per-row sequence
    (each alert is usually explained as its own predicted class).  With
    ``None`` the convention is: 1 column → column 0, 2 columns → column 1
    (the positive class), more columns → per-row argmax of the *unmasked*
    input, so the compared class stays fixed as features are removed.
    """
    if hasattr(model, "predict_proba"):
        scorer = lambda X: np.asarray(model.predict_proba(X), dtype=float)  # noqa: E731
    elif hasattr(model, "predict") and not callable(getattr(model, "forward", None)):
        scorer = lambda X: np.asarray(model.predict(X), dtype=float)  # noqa: E731
    elif _is_torch_module(model):
        scorer = lambda X: _torch_proba(model, X)  # noqa: E731
    else:
        scorer = lambda X: np.asarray(model(X), dtype=float)  # noqa: E731

    def _fixed_class(X: np.ndarray) -> np.ndarray:
        full = scorer(X)
        if full.ndim == 1:
            return full
        if full.shape[1] == 1:
            return full[:, 0]
        if class_index is None:
            return full[:, 1] if full.shape[1] == 2 else full.argmax(axis=1)
        return _gather(full, class_index)

    return _fixed_class


def _is_torch_module(model) -> bool:
    try:
        import torch.nn as nn  # type: ignore
    except Exception:  # pragma: no cover
        return False
    return isinstance(model, nn.Module)


def _torch_proba(model, X: np.ndarray) -> np.ndarray:
    import torch  # type: ignore

    device = next(model.parameters(), torch.zeros(1)).device
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[-1] == 1:
            return torch.sigmoid(out).squeeze(-1).cpu().numpy()
        return torch.softmax(out, dim=-1).cpu().numpy()


def _gather(proba: np.ndarray, class_index) -> np.ndarray:
    """Select one column per row (``class_index`` may be an int or a vector)."""
    idx = np.asarray(class_index)
    if idx.ndim == 0:
        return proba[:, int(idx)]
    if idx.shape[0] != proba.shape[0]:
        raise ValueError(
            f"class_index has {idx.shape[0]} entries for {proba.shape[0]} rows."
        )
    return proba[np.arange(proba.shape[0]), idx.astype(int)]


def deletion_curve(model, X: np.ndarray, attributions: np.ndarray,
                   steps: int = 10,
                   class_index: Optional[Union[int, Sequence[int]]] = None,
                   baseline: Optional[np.ndarray] = None) -> dict:
    """
    Compute the deletion curve: mean probability vs. number of masked features.

    Args:
        model: Any scorer accepted by :func:`as_predict_proba`.
        X: ``(n, f)`` input matrix.
        attributions: ``(f,)`` shared across all rows, or ``(n, f)`` per row.
        steps: Number of masking increments (``steps + 1`` curve points).
        class_index: Explained class (int or per-row vector); see
            :func:`as_predict_proba`.
        baseline: Replacement value for masked features.  ``None`` = 0.0, which
            is the training median for robust-scaled features.

    Returns:
        dict with ``k_masked`` (list of feature counts), ``mean_probability``
        (list of floats, clipped to [0, 1]), ``rows`` (per-row AUCs) and
        ``auc`` (area under the mean curve, lower = more faithful).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, f); got shape {X.shape}.")

    attr = np.asarray(attributions, dtype=float)
    if attr.ndim == 1:
        if attr.shape[0] != X.shape[1]:
            raise ValueError(
                f"attributions has {attr.shape[0]} entries for {X.shape[1]} features."
            )
        ranking = np.argsort(np.abs(attr))[::-1]           # shared ranking
        ranks = np.tile(ranking, (X.shape[0], 1))          # (n, f)
    elif attr.ndim == 2:
        if attr.shape != X.shape:
            raise ValueError(
                f"attributions shape {attr.shape} does not match X shape {X.shape}."
            )
        ranks = np.argsort(np.abs(attr), axis=1)[:, ::-1]  # per-row ranking
    else:
        raise ValueError(f"Unsupported attribution shape {attr.shape}.")

    score = as_predict_proba(model, class_index=class_index)
    fill = 0.0 if baseline is None else float(baseline)

    n_features = X.shape[1]
    k_masked, mean_probability, rows = [], [], []

    for step in range(steps + 1):
        k = int(round(step * n_features / steps))
        masked = X.copy()
        if k > 0:
            # ``ranks[row, j]`` is the feature occupying position ``j`` of that
            # row's importance ordering, so the top-k features to mask are the
            # first k *positions*, not the features whose index happens to be
            # smaller than k.  Flatten to explicit (row, col) pairs: the
            # broadcast form ``masked[rows[:, None], ranks[:, :k]]`` silently
            # cross-multiplies (4,1) against (4,1,k) into (4,4,k) and masks
            # every row with *other* rows' features.
            row_idx = np.repeat(np.arange(X.shape[0]), k)
            col_idx = ranks[:, :k].reshape(-1)
            masked[row_idx, col_idx] = fill
        proba = np.clip(np.asarray(score(masked), dtype=float).reshape(-1), 0.0, 1.0)
        k_masked.append(k)
        mean_probability.append(float(proba.mean()))
        rows.append(proba)

    row_matrix = np.vstack(rows)                      # (steps+1, n)
    row_auc = _trapezoid(row_matrix, dx=1.0 / steps, axis=0)
    auc = float(_trapezoid(np.asarray(mean_probability), dx=1.0 / steps))
    return {
        "k_masked": k_masked,
        "mean_probability": mean_probability,
        "rows": row_auc,
        "auc": auc,
        "steps": int(steps),
    }


def deletion_auc(model, X: np.ndarray, attributions: np.ndarray,
                 steps: int = 10,
                 class_index: Optional[Union[int, Sequence[int]]] = None,
                 baseline: Optional[np.ndarray] = None) -> float:
    """
    Scalar deletion AUC for a batch of alerts (lower = more faithful).

    This is the metric named in PHASE_4 section 4.3; :func:`deletion_curve`
    exposes the underlying curve for plotting and for per-alert scores.
    """
    return deletion_curve(model, X, attributions, steps=steps,
                          class_index=class_index, baseline=baseline)["auc"]


def compare_explainers(model, X: np.ndarray,
                       attributions_a: np.ndarray,
                       attributions_b: np.ndarray,
                       steps: int = 10,
                       class_index=None,
                       baseline: Optional[np.ndarray] = None,
                       name_a: str = "shap",
                       name_b: str = "deeplift") -> dict:
    """
    Paired faithfulness comparison of two attribution methods on the same alerts.

    Both methods explain the *same* rows of the *same* model, so the per-row
    scores are paired and a paired t-test is the right test (PHASE_4 section
    4.3.3).  Lower AUC is better, so ``more_faithful`` names the method with
    the smaller mean.

    Returns a JSON-ready dict including both score vectors, means, the t
    statistic and p-value, and a plain-language verdict.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    curve_a = deletion_curve(model, X, attributions_a, steps=steps,
                             class_index=class_index, baseline=baseline)
    curve_b = deletion_curve(model, X, attributions_b, steps=steps,
                             class_index=class_index, baseline=baseline)

    scores_a = np.asarray(curve_a["rows"], dtype=float)
    scores_b = np.asarray(curve_b["rows"], dtype=float)

    t_stat = p_value = None
    verdict = "inconclusive"
    if scores_a.size >= 2:
        try:
            import warnings

            from scipy import stats  # type: ignore
            with warnings.catch_warnings():
                # Identical inputs give zero variance, and SciPy answers with a
                # nan statistic and a divide-by-zero warning.  A nan would leak
                # into the report and the dashboard, so it is treated as
                # "no test possible" rather than as a number.
                warnings.simplefilter("ignore")
                result = stats.ttest_rel(scores_a, scores_b)
            t_stat, p_value = float(result.statistic), float(result.pvalue)
            if not np.isfinite(t_stat) or not np.isfinite(p_value):
                t_stat = p_value = None
        except Exception:  # pragma: no cover - scipy always present in practice
            t_stat = p_value = None

    diff = float(scores_a.mean() - scores_b.mean())
    significant = p_value is not None and p_value < 0.05
    if abs(diff) < 1e-9:
        verdict = "tie"
    elif significant:
        winner = name_a if diff < 0 else name_b
        verdict = f"{winner} significantly more faithful (paired t-test p={p_value:.4g})"
    else:
        better = name_a if diff < 0 else name_b
        verdict = (f"{better} numerically lower AUC, but the difference is not "
                   f"significant (p={'n/a' if p_value is None else format(p_value, '.4g')})")

    return {
        "n_alerts": int(X.shape[0]),
        "steps": int(steps),
        "lower_auc_is_better": True,
        f"{name_a}_scores": scores_a.tolist(),
        f"{name_b}_scores": scores_b.tolist(),
        f"{name_a}_mean_deletion_auc": float(scores_a.mean()),
        f"{name_b}_mean_deletion_auc": float(scores_b.mean()),
        f"{name_a}_curve": curve_a["mean_probability"],
        f"{name_b}_curve": curve_b["mean_probability"],
        "k_masked": curve_a["k_masked"],
        "mean_difference": diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_0.05": bool(significant),
        "more_faithful": ("tie" if verdict == "tie" else
                          (name_a if diff < 0 else name_b)),
        "verdict": verdict,
    }
