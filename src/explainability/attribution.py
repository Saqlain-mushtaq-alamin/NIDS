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
    "ground_truth_precision_at_k",
    "evaluate_generator_feature_recovery",
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


def ground_truth_precision_at_k(
    attributions: np.ndarray,
    ground_truth_indices: Sequence[int],
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    """
    Compute top-k precision and recall of feature attributions against known ground truth.

    Args:
        attributions: (n, f) or (f,) array of feature importance/attributions.
        ground_truth_indices: Collection of integer indices of true discriminative features.
        top_k: Number of highest-magnitude features to evaluate.
        seed: Random seed for bootstrap confidence interval.

    Returns:
        dict containing per-sample precision and recall, mean, std, and 95% bootstrap CI.
    """
    arr = np.asarray(attributions, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"attributions must be 1D or 2D; got shape {arr.shape}")

    gt_set = set(int(i) for i in ground_truth_indices)
    k = max(1, min(int(top_k), arr.shape[1]))
    n_samples = arr.shape[0]

    precisions = []
    recalls = []

    for i in range(n_samples):
        top_indices = np.argsort(np.abs(arr[i]))[::-1][:k]
        hits = len(set(top_indices) & gt_set)
        precisions.append(hits / k)
        recalls.append(hits / len(gt_set) if gt_set else 0.0)

    prec_arr = np.array(precisions, dtype=float)
    rec_arr = np.array(recalls, dtype=float)

    from src.evaluation.statistics import bootstrap_ci

    ci_prec = bootstrap_ci(prec_arr, seed=seed) if len(prec_arr) >= 2 else (float(prec_arr[0]), float(prec_arr[0]))
    ci_rec = bootstrap_ci(rec_arr, seed=seed) if len(rec_arr) >= 2 else (float(rec_arr[0]), float(rec_arr[0]))

    return {
        "top_k": k,
        "n_samples": n_samples,
        "ground_truth_count": len(gt_set),
        "mean_precision": float(np.mean(prec_arr)),
        "std_precision": float(np.std(prec_arr)),
        "ci_precision": [float(ci_prec[0]), float(ci_prec[1])],
        "mean_recall": float(np.mean(rec_arr)),
        "std_recall": float(np.std(rec_arr)),
        "ci_recall": [float(ci_rec[0]), float(ci_rec[1])],
        "sample_precisions": prec_arr.tolist(),
    }


def evaluate_generator_feature_recovery(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 5,
    seed: int = 42,
) -> dict:
    """
    Benchmark feature recovery of TreeSHAP, KernelSHAP, DeepLIFT, and Random Control
    on generator data with mathematically known ground-truth discriminative features.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import torch
    import torch.nn as nn

    # Generate synthetic dataset where informative features are features 0 .. n_informative - 1
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,  # preserves first n_informative as informative features
        random_state=seed,
    )
    feature_names = [f"f_{i}" for i in range(n_features)]
    ground_truth_features = list(range(n_informative))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # 1. Train XGBoost model
    xgb_clf = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1, random_state=seed, eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)

    # 2. Train small PyTorch MLP
    torch.manual_seed(seed)
    mlp = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    mlp.eval()
    t_X = torch.as_tensor(X_train, dtype=torch.float32)
    t_y = torch.as_tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(t_X, t_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    mlp.train()
    for _ in range(10):
        for bx, by in loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
    mlp.eval()

    # Select positive class alerts from test set
    test_sub = X_test[:min(100, len(X_test))]

    # A. TreeSHAP on XGBoost
    import shap
    tree_expl = shap.TreeExplainer(xgb_clf)
    tree_shap_vals = tree_expl.shap_values(test_sub)
    if isinstance(tree_shap_vals, list):
        tree_attr = tree_shap_vals[1]
    elif tree_shap_vals.ndim == 3:
        tree_attr = tree_shap_vals[:, :, 1]
    else:
        tree_attr = tree_shap_vals

    tree_res = ground_truth_precision_at_k(tree_attr, ground_truth_features, top_k=n_informative, seed=seed)

    # B. KernelSHAP on MLP
    def _mlp_predict(data):
        with torch.no_grad():
            t_in = torch.as_tensor(data, dtype=torch.float32)
            return torch.softmax(mlp(t_in), dim=-1).numpy()

    kernel_expl = shap.KernelExplainer(_mlp_predict, X_train[:30])
    k_vals = kernel_expl.shap_values(test_sub[:20], nsamples=50)
    if isinstance(k_vals, list):
        kernel_attr = k_vals[1]
    elif k_vals.ndim == 3:
        kernel_attr = k_vals[:, :, 1]
    else:
        kernel_attr = k_vals
    kernel_res = ground_truth_precision_at_k(kernel_attr, ground_truth_features, top_k=n_informative, seed=seed)

    # C. DeepLIFT on MLP
    from captum.attr import DeepLift
    dl = DeepLift(mlp)
    dl_tensor = torch.as_tensor(test_sub, dtype=torch.float32).requires_grad_()
    dl_baseline = torch.zeros((1, n_features), dtype=torch.float32)
    dl_attr = dl.attribute(dl_tensor, target=1, baselines=dl_baseline).detach().cpu().numpy()
    dl_res = ground_truth_precision_at_k(dl_attr, ground_truth_features, top_k=n_informative, seed=seed)

    # D. Random ranking control
    rng = np.random.default_rng(seed)
    random_attr = rng.standard_normal(size=test_sub.shape)
    rand_res = ground_truth_precision_at_k(random_attr, ground_truth_features, top_k=n_informative, seed=seed)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "ground_truth_indices": ground_truth_features,
        "treeshap_precision": tree_res,
        "kernelshap_precision": kernel_res,
        "deeplift_precision": dl_res,
        "random_control_precision": rand_res,
    }

