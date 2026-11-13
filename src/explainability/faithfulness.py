"""
src/explainability/faithfulness.py — Do the attributions explain the model?

An attribution can *look* convincing and still be wrong: any method can return
a ranked feature list, and an analyst has no way to tell from the list alone
whether those features actually drive the prediction.  Faithfulness measures
answer that question by intervening on the input (PHASE_4 section 4.3).

Two complementary metrics are evaluated:
1. **Deletion AUC**:
   - Start with the full flow.
   - Progressively mask the top-``k`` attributed features with a matched baseline value
     (zeros or training feature means/medians).
   - Record model probability of the explained class.
   - **Lower AUC = more faithful** (masking critical features collapses confidence).

2. **Insertion AUC**:
   - Start from a baseline input (all features replaced by baseline values).
   - Progressively unmask/insert the top-``k`` attributed features back to their original values.
   - Record model probability of the explained class.
   - **Higher AUC = more faithful** (introducing critical features restores confidence).

Controls & Statistical Validity:
- **Random-ranking control**: Feature rankings randomly permuted per alert. Faithful explainers
  must significantly beat the random control (lower deletion AUC, higher insertion AUC).
- **Matched baselines**: The masking baseline matches the explainer's reference baseline
  (scalar 0.0 or 1-D feature mean/median vector).
- **Effect sizes with bootstrap CIs**: Evaluation alerts often come from the same capture session
  or traffic window and are correlated. Rather than reporting spuriously small p-values from
  independent-sample tests, effect sizes (mean difference and Cohen's d) are reported with
  95% bootstrap confidence intervals over alerts.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union
import numpy as np

__all__ = [
    "as_predict_proba",
    "compare_explainers",
    "deletion_auc",
    "deletion_curve",
    "insertion_auc",
    "insertion_curve",
    "random_ranking_control",
    "paired_effect_size_ci",
]

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


def _resolve_baseline_vector(baseline: Optional[Union[float, np.ndarray]], n_features: int) -> np.ndarray:
    """Ensure baseline is a 1-D float array of length `n_features`."""
    if baseline is None:
        return np.zeros(n_features, dtype=float)
    arr = np.asarray(baseline, dtype=float)
    if arr.ndim == 0:
        return np.full(n_features, float(arr), dtype=float)
    if arr.ndim == 1 and arr.shape[0] == n_features:
        return arr
    if arr.ndim == 2 and arr.shape[1] == n_features:
        return arr[0]
    raise ValueError(f"baseline shape {arr.shape} does not match {n_features} features.")


def _get_ranks(X: np.ndarray, attributions: np.ndarray) -> np.ndarray:
    """Compute per-row ranking indices from feature attributions."""
    attr = np.asarray(attributions, dtype=float)
    if attr.ndim == 1:
        if attr.shape[0] != X.shape[1]:
            raise ValueError(
                f"attributions has {attr.shape[0]} entries for {X.shape[1]} features."
            )
        ranking = np.argsort(np.abs(attr))[::-1]
        return np.tile(ranking, (X.shape[0], 1))
    elif attr.ndim == 2:
        if attr.shape != X.shape:
            raise ValueError(
                f"attributions shape {attr.shape} does not match X shape {X.shape}."
            )
        return np.argsort(np.abs(attr), axis=1)[:, ::-1]
    raise ValueError(f"Unsupported attribution shape {attr.shape}.")


def deletion_curve(model, X: np.ndarray, attributions: np.ndarray,
                   steps: int = 10,
                   class_index: Optional[Union[int, Sequence[int]]] = None,
                   baseline: Optional[Union[float, np.ndarray]] = None) -> dict:
    """
    Compute the deletion curve: mean probability vs. number of masked features.

    Args:
        model: Any scorer accepted by :func:`as_predict_proba`.
        X: ``(n, f)`` input matrix.
        attributions: ``(f,)`` shared across all rows, or ``(n, f)`` per row.
        steps: Number of masking increments (``steps + 1`` curve points).
        class_index: Explained class (int or per-row vector); see :func:`as_predict_proba`.
        baseline: Replacement value for masked features (scalar or 1D array of length f).
            ``None`` = 0.0, which is the training median for robust-scaled features.

    Returns:
        dict with ``k_masked``, ``mean_probability``, ``rows`` (per-row AUCs) and
        ``auc`` (area under the mean curve, lower = more faithful).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, f); got shape {X.shape}.")

    n_features = X.shape[1]
    ranks = _get_ranks(X, attributions)
    fill_vec = _resolve_baseline_vector(baseline, n_features)
    score = as_predict_proba(model, class_index=class_index)

    k_masked, mean_probability, rows = [], [], []

    for step in range(steps + 1):
        k = int(round(step * n_features / steps))
        masked = X.copy()
        if k > 0:
            row_idx = np.repeat(np.arange(X.shape[0]), k)
            col_idx = ranks[:, :k].reshape(-1)
            masked[row_idx, col_idx] = fill_vec[col_idx]
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
                 baseline: Optional[Union[float, np.ndarray]] = None) -> float:
    """Scalar deletion AUC for a batch of alerts (lower = more faithful)."""
    return deletion_curve(model, X, attributions, steps=steps,
                          class_index=class_index, baseline=baseline)["auc"]


def insertion_curve(model, X: np.ndarray, attributions: np.ndarray,
                    steps: int = 10,
                    class_index: Optional[Union[int, Sequence[int]]] = None,
                    baseline: Optional[Union[float, np.ndarray]] = None) -> dict:
    """
    Compute the insertion curve: mean probability vs. number of inserted features.

    Higher AUC = more faithful.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, f); got shape {X.shape}.")

    n_features = X.shape[1]
    ranks = _get_ranks(X, attributions)
    fill_vec = _resolve_baseline_vector(baseline, n_features)
    score = as_predict_proba(model, class_index=class_index)

    base = np.tile(fill_vec, (X.shape[0], 1))
    k_inserted, mean_probability, rows = [], [], []

    for step in range(steps + 1):
        k = int(round(step * n_features / steps))
        inserted = base.copy()
        if k > 0:
            row_idx = np.repeat(np.arange(X.shape[0]), k)
            col_idx = ranks[:, :k].reshape(-1)
            inserted[row_idx, col_idx] = X[row_idx, col_idx]
        proba = np.clip(np.asarray(score(inserted), dtype=float).reshape(-1), 0.0, 1.0)
        k_inserted.append(k)
        mean_probability.append(float(proba.mean()))
        rows.append(proba)

    row_matrix = np.vstack(rows)                      # (steps+1, n)
    row_auc = _trapezoid(row_matrix, dx=1.0 / steps, axis=0)
    auc = float(_trapezoid(np.asarray(mean_probability), dx=1.0 / steps))
    return {
        "k_inserted": k_inserted,
        "mean_probability": mean_probability,
        "rows": row_auc,
        "auc": auc,
        "steps": int(steps),
    }


def insertion_auc(model, X: np.ndarray, attributions: np.ndarray,
                  steps: int = 10,
                  class_index: Optional[Union[int, Sequence[int]]] = None,
                  baseline: Optional[Union[float, np.ndarray]] = None) -> float:
    """Scalar insertion AUC for a batch of alerts (higher = more faithful)."""
    return insertion_curve(model, X, attributions, steps=steps,
                           class_index=class_index, baseline=baseline)["auc"]


def random_ranking_control(model, X: np.ndarray, steps: int = 10,
                           class_index: Optional[Union[int, Sequence[int]]] = None,
                           baseline: Optional[Union[float, np.ndarray]] = None,
                           seed: int = 42) -> dict:
    """
    Evaluate deletion and insertion curves under uniform random feature ranking.

    A faithful explanation method must significantly outperform this random control:
    - Lower deletion AUC than random
    - Higher insertion AUC than random
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    rng = np.random.default_rng(seed)
    random_attr = rng.standard_normal(size=X.shape)
    del_curve = deletion_curve(model, X, random_attr, steps=steps,
                               class_index=class_index, baseline=baseline)
    ins_curve = insertion_curve(model, X, random_attr, steps=steps,
                                class_index=class_index, baseline=baseline)
    return {
        "random_attributions": random_attr,
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
        "deletion_auc": float(del_curve["auc"]),
        "insertion_auc": float(ins_curve["auc"]),
        "deletion_rows": del_curve["rows"],
        "insertion_rows": ins_curve["rows"],
    }


def paired_effect_size_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                          n_bootstraps: int = 1000, seed: int = 42,
                          alpha: float = 0.05) -> dict:
    """
    Compute paired effect size (mean difference and Cohen's d) with bootstrap 95% CIs.

    Alerts from the same evaluation batch or session are correlated; therefore,
    inference is based on bootstrap confidence intervals rather than standard
    independent-sample p-values.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a shape {a.shape} != scores_b shape {b.shape}")
    n = a.shape[0]
    if n == 0:
        return {"mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0, "cohens_d": 0.0}

    diffs = a - b
    mean_diff = float(np.mean(diffs))
    var_a = np.var(a, ddof=1) if n > 1 else 0.0
    var_b = np.var(b, ddof=1) if n > 1 else 0.0
    pooled_sd = float(np.sqrt(0.5 * (var_a + var_b))) if (var_a + var_b) > 0 else 1.0
    cohens_d = float(mean_diff / (pooled_sd if pooled_sd > 1e-12 else 1.0))

    if n < 3:
        return {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": mean_diff,
            "ci_diff_lower": mean_diff,
            "ci_diff_upper": mean_diff,
            "cohens_d": cohens_d,
            "ci_d_lower": cohens_d,
            "ci_d_upper": cohens_d,
        }

    rng = np.random.default_rng(seed)
    boot_diffs = []
    boot_ds = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_a = a[idx]
        boot_b = b[idx]
        b_diff = float(np.mean(boot_a - boot_b))
        v_a = np.var(boot_a, ddof=1) if n > 1 else 0.0
        v_b = np.var(boot_b, ddof=1) if n > 1 else 0.0
        b_pooled = float(np.sqrt(0.5 * (v_a + v_b))) if (v_a + v_b) > 0 else 1.0
        b_d = b_diff / (b_pooled if b_pooled > 1e-12 else 1.0)
        boot_diffs.append(b_diff)
        boot_ds.append(b_d)

    ci_low_pct = 100 * (alpha / 2.0)
    ci_high_pct = 100 * (1.0 - alpha / 2.0)

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "ci_diff_lower": float(np.percentile(boot_diffs, ci_low_pct)),
        "ci_diff_upper": float(np.percentile(boot_diffs, ci_high_pct)),
        "cohens_d": cohens_d,
        "ci_d_lower": float(np.percentile(boot_ds, ci_low_pct)),
        "ci_d_upper": float(np.percentile(boot_ds, ci_high_pct)),
    }


def compare_explainers(model, X: np.ndarray,
                       attributions_a: np.ndarray,
                       attributions_b: np.ndarray,
                       steps: int = 10,
                       class_index=None,
                       baseline: Optional[Union[float, np.ndarray]] = None,
                       name_a: str = "shap",
                       name_b: str = "deeplift",
                       include_random_control: bool = True,
                       seed: int = 42) -> dict:
    """
    Paired faithfulness comparison of two attribution methods on the same alerts.

    Computes:
    - Deletion curves and insertion curves for both methods.
    - Paired effect sizes (mean difference and Cohen's d) with bootstrap 95% CIs.
    - Optional random-ranking control baseline.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    curve_a = deletion_curve(model, X, attributions_a, steps=steps,
                             class_index=class_index, baseline=baseline)
    curve_b = deletion_curve(model, X, attributions_b, steps=steps,
                             class_index=class_index, baseline=baseline)

    ins_a = insertion_curve(model, X, attributions_a, steps=steps,
                            class_index=class_index, baseline=baseline)
    ins_b = insertion_curve(model, X, attributions_b, steps=steps,
                            class_index=class_index, baseline=baseline)

    scores_a = np.asarray(curve_a["rows"], dtype=float)
    scores_b = np.asarray(curve_b["rows"], dtype=float)

    effect_size = paired_effect_size_ci(scores_a, scores_b, seed=seed)

    # Calculate paired t-test for legacy compatibility, with explicit correlation note
    t_stat = p_value = None
    if scores_a.size >= 2:
        try:
            import warnings
            from scipy import stats  # type: ignore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = stats.ttest_rel(scores_a, scores_b)
            t_stat, p_value = float(result.statistic), float(result.pvalue)
            if not np.isfinite(t_stat) or not np.isfinite(p_value):
                t_stat = p_value = None
        except Exception:
            t_stat = p_value = None

    diff = float(scores_a.mean() - scores_b.mean())
    ci_excludes_zero = (effect_size["ci_diff_lower"] > 0) or (effect_size["ci_diff_upper"] < 0)

    if abs(diff) < 1e-9:
        verdict = "tie"
    elif ci_excludes_zero:
        winner = name_a if diff < 0 else name_b
        verdict = (f"{winner} significantly more faithful (deletion diff {diff:+.4f}, "
                   f"95% CI [{effect_size['ci_diff_lower']:.4f}, {effect_size['ci_diff_upper']:.4f}])")
    else:
        better = name_a if diff < 0 else name_b
        verdict = (f"{better} numerically lower deletion AUC, but 95% bootstrap CI "
                   f"crosses zero ([{effect_size['ci_diff_lower']:.4f}, {effect_size['ci_diff_upper']:.4f}])")

    out = {
        "n_alerts": int(X.shape[0]),
        "steps": int(steps),
        "lower_auc_is_better": True,
        "lower_auc_is_better_for_deletion": True,
        "higher_auc_is_better_for_insertion": True,
        f"{name_a}_scores": scores_a.tolist(),
        f"{name_b}_scores": scores_b.tolist(),
        f"{name_a}_mean_deletion_auc": float(scores_a.mean()),
        f"{name_b}_mean_deletion_auc": float(scores_b.mean()),
        f"{name_a}_mean_insertion_auc": float(ins_a["auc"]),
        f"{name_b}_mean_insertion_auc": float(ins_b["auc"]),
        f"{name_a}_curve": curve_a["mean_probability"],
        f"{name_b}_curve": curve_b["mean_probability"],
        f"{name_a}_insertion_curve": ins_a["mean_probability"],
        f"{name_b}_insertion_curve": ins_b["mean_probability"],
        "k_masked": curve_a["k_masked"],
        "mean_difference": diff,
        "effect_size": effect_size,
        "t_statistic": t_stat,
        "p_value": p_value,
        "statistical_inference_note": (
            "Alerts may be correlated across sessions. The paired t-test p-value assumes "
            "independent samples; prefer the bootstrap 95% CI of effect size."
        ),
        "significant_at_0.05": bool(ci_excludes_zero),
        "more_faithful": ("tie" if verdict == "tie" else
                          (name_a if diff < 0 else name_b)),
        "verdict": verdict,
    }

    if include_random_control:
        rand_ctrl = random_ranking_control(model, X, steps=steps,
                                           class_index=class_index,
                                           baseline=baseline, seed=seed)
        out["random_control"] = {
            "deletion_auc": rand_ctrl["deletion_auc"],
            "insertion_auc": rand_ctrl["insertion_auc"],
            "deletion_curve": rand_ctrl["deletion_curve"]["mean_probability"],
            "insertion_curve": rand_ctrl["insertion_curve"]["mean_probability"],
        }
        out[f"{name_a}_vs_random"] = paired_effect_size_ci(
            scores_a, np.asarray(rand_ctrl["deletion_rows"], dtype=float), seed=seed
        )
        out[f"{name_b}_vs_random"] = paired_effect_size_ci(
            scores_b, np.asarray(rand_ctrl["deletion_rows"], dtype=float), seed=seed
        )

    return out
