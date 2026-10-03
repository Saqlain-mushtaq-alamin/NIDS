"""
src/explainability/shap_explainer.py — SHAP feature attribution for NIDS alerts.

Two explainer back-ends (PHASE_4 section 4.1):

``tree``
    :class:`shap.TreeExplainer` — exact and fast for the XGBoost baseline that
    drives the deployed detector.  One call explains a whole batch of alerts in
    roughly the time it takes to score them.

``kernel``
    :class:`shap.KernelExplainer` — model-agnostic, used for the neural models
    (DNN classifier / autoencoder) where no exact tree path exists.  It needs a
    background sample and is one to two orders of magnitude slower, which is why
    :meth:`SHAPExplainer.explain` records per-alert timings: explanation latency
    is part of the product, not an implementation detail.

Output contract of :meth:`SHAPExplainer.explain`::

    {
        "shap_values":  np.ndarray,   # (n, f) or (n, f, k) for multi-class
        "feature_names": list[str],
        "base_value":   float | list[float],
        "class_index":  int | None,
        "n_samples":    int,
        "timing_ms":    {"total_ms", "ms_per_alert", "n_alerts"},
    }
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - exercised only in stripped environments
    shap = None  # type: ignore

from src.explainability.attribution import align_attributions, rank_features

__all__ = ["SHAPExplainer"]

#: KernelSHAP cost explodes with the background size; 100 rows is the standard
#: compromise and matches the phase plan (PHASE_4 section 4.1.3).
MAX_BACKGROUND_SAMPLES = 100


def _require_shap():
    if shap is None:
        raise ImportError(
            "shap is not installed. Install it with `pip install shap` "
            "(it is listed in requirements.txt)."
        )
    return shap


class SHAPExplainer:
    """SHAP-based feature attribution for NIDS alerts."""

    MODEL_TYPES = ("tree", "kernel", "auto")

    def __init__(self, model=None, model_type: str = "tree",
                 background=None, predict: Optional[Callable] = None,
                 max_background_samples: int = MAX_BACKGROUND_SAMPLES):
        """
        Args:
            model: Fitted estimator.  Required for ``tree``; for ``kernel`` it
                is used to derive a default scoring function when ``predict``
                is not supplied.
            model_type: ``"tree"``, ``"kernel"`` or ``"auto"`` (tree when the
                model exposes a booster, kernel otherwise).
            background: Optional ``(n, f)`` background sample for KernelSHAP.
            predict: Callable ``f(X) -> scores`` used by KernelSHAP.  When it
                returns ``(n, k)`` the full class distribution is explained and
                a class can be sliced out later via ``explain(class_index=...)``.
            max_background_samples: Cap applied to the background sample.
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.MODEL_TYPES}, got '{model_type}'."
            )

        self.model = model
        self.max_background_samples = int(max_background_samples)
        self.background_: Optional[np.ndarray] = None
        self.explainer = None
        self.benchmark_: list[dict] = []
        self.last_timing_: dict = {}

        if model_type == "auto":
            resolved = (self._resolve_model_type(model, "auto")
                        if model is not None else "kernel")
        else:
            resolved = model_type
        self.model_type = resolved
        self.predict_ = predict

        if resolved == "tree":
            if model is None:
                raise ValueError("model_type='tree' requires a fitted model.")
            self.explainer = _require_shap().TreeExplainer(model)

        if background is not None:
            self.set_background(background, predict)

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _resolve_model_type(model, model_type: str) -> str:
        if model_type != "auto":
            return model_type
        if hasattr(model, "get_booster") or hasattr(model, "booster_"):
            return "tree"
        return "kernel"

    @classmethod
    def for_xgboost(cls, model) -> "SHAPExplainer":
        """Exact TreeExplainer for the XGBoost baseline (preferred path)."""
        return cls(model, model_type="tree")

    @classmethod
    def for_neural(cls, background: np.ndarray, predict: Callable,
                   **kwargs) -> "SHAPExplainer":
        """KernelSHAP explainer for a DNN / autoencoder."""
        explainer = cls(None, model_type="kernel", **kwargs)
        explainer.set_background(background, predict)
        return explainer

    def set_background(self, background_data: np.ndarray,
                       model_predict: Optional[Callable] = None) -> "SHAPExplainer":
        """
        Set the background distribution for KernelSHAP (neural models).

        The background is capped at ``max_background_samples`` rows: KernelSHAP
        evaluates every coalition against every background row, so the cost is
        ``n_instances x n_background x n_coalitions`` and an uncapped reference
        set makes per-alert explanation time unusable for a SOC console.
        """
        _require_shap()
        background = np.asarray(background_data, dtype=float)
        if background.ndim != 2:
            raise ValueError(
                f"background_data must be 2-D (n, f); got shape {background.shape}."
            )
        if background.shape[0] == 0:
            raise ValueError("background_data must contain at least one row.")

        predict = model_predict or self.predict_
        if predict is None:
            if self.model is None:
                raise ValueError(
                    "KernelSHAP needs a scoring function: pass model_predict=..."
                )
            predict = self._default_predict(self.model)

        keep = min(background.shape[0], self.max_background_samples)
        self.background_ = background[:keep]
        self.predict_ = predict
        self.explainer = _require_shap().KernelExplainer(predict, self.background_)
        self.model_type = "kernel"
        return self

    @staticmethod
    def _default_predict(model) -> Callable:
        """Use ``predict_proba`` when the estimator offers it."""
        if hasattr(model, "predict_proba"):
            def _predict(X: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict_proba(X), dtype=float)
            return _predict
        if hasattr(model, "predict"):
            return lambda X: np.asarray(model.predict(X), dtype=float)
        raise ValueError(
            "Model exposes neither predict_proba nor predict; pass model_predict."
        )

    # -- explanation -------------------------------------------------------
    def _require_ready(self):
        if self.explainer is None:
            raise RuntimeError(
                "Explainer not initialised. Fit a tree model or call "
                "set_background(...) for KernelSHAP first."
            )

    def _expected_value(self):
        raw = getattr(self.explainer, "expected_value", None)
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw]
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return float(arr)
        return [float(v) for v in arr.reshape(-1)]

    def explain(self, X: np.ndarray, feature_names: Sequence[str],
                class_index: Optional[int] = None) -> dict:
        """
        Compute SHAP values for the given flows.

        Args:
            X: ``(n, f)`` matrix of alert flows (already preprocessed exactly
                as the model expects them).
            feature_names: Names matching the columns of ``X``.
            class_index: For multi-class models, the class to keep.  Leave
                ``None`` to receive the full ``(n, f, k)`` cube.

        Returns:
            dict as documented in the module docstring.
        """
        self._require_ready()
        feature_names = [str(f) for f in feature_names]
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, f); got shape {X.shape}.")
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"{len(feature_names)} feature names for X with {X.shape[1]} columns."
            )

        started = time.perf_counter()
        raw = self.explainer.shap_values(X)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        values = align_attributions(raw, n_samples=X.shape[0],
                                    class_index=class_index)
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"Explainer returned {values.shape[1]} attributions for "
                f"{len(feature_names)} features."
            )

        timing = {
            "total_ms": elapsed_ms,
            "ms_per_alert": elapsed_ms / max(X.shape[0], 1),
            "n_alerts": int(X.shape[0]),
            "method": f"shap/{self.model_type}",
        }
        self.last_timing_ = timing
        self.benchmark_.append(timing)

        return {
            "shap_values": values,
            "feature_names": feature_names,
            "base_value": self._expected_value(),
            "class_index": class_index,
            "n_samples": int(X.shape[0]),
            "timing_ms": timing,
        }

    def top_features(self, shap_values: np.ndarray,
                     feature_names: Sequence[str],
                     top_k: int = 5) -> list[dict]:
        """Top-K most influential features for a single prediction."""
        return rank_features(shap_values, feature_names, top_k=top_k)

    def global_importance(self, shap_values: np.ndarray,
                          feature_names: Sequence[str],
                          top_k: int = 20) -> list[dict]:
        """
        Mean |SHAP| across every explained alert — the global ranking used for
        the summary plot and the Tier-2 header.
        """
        values = align_attributions(shap_values, n_samples=None, class_index=None)
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        else:
            values = np.abs(values)
        mean_abs = values.mean(axis=0)
        top_k = max(1, min(int(top_k), mean_abs.shape[0]))
        order = np.argsort(mean_abs)[::-1][:top_k]
        total = float(mean_abs.sum()) or 1.0
        return [
            {
                "rank": rank + 1,
                "feature": str(feature_names[i]),
                "mean_abs_shap": float(mean_abs[i]),
                "share_of_total": float(mean_abs[i] / total),
            }
            for rank, i in enumerate(order)
        ]

    def benchmark(self) -> dict:
        """Aggregate explanation-latency statistics over every recorded call."""
        if not self.benchmark_:
            return {"n_calls": 0, "n_alerts": 0, "mean_ms_per_alert": None}
        per_alert = [b["ms_per_alert"] for b in self.benchmark_]
        n_alerts = sum(b["n_alerts"] for b in self.benchmark_)
        return {
            "n_calls": len(self.benchmark_),
            "n_alerts": n_alerts,
            "mean_ms_per_alert": float(np.mean(per_alert)),
            "median_ms_per_alert": float(np.median(per_alert)),
            "max_ms_per_alert": float(np.max(per_alert)),
            "method": self.benchmark_[0]["method"],
        }

    def save_summary_plot(self, shap_values: np.ndarray,
                          feature_names: Sequence[str],
                          path: Union[str, Path],
                          top_k: int = 15) -> Optional[Path]:
        """
        Write the global SHAP summary plot to ``path``.

        Returns the path written, or ``None`` when a plotting backend is not
        available (the numeric ranking in :meth:`global_importance` is the
        authoritative result; the figure is a convenience).
        """
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except Exception:
            return None

        values = align_attributions(shap_values, n_samples=None, class_index=None)
        if values.ndim == 3:
            values = values.mean(axis=2)

        try:
            plt.figure(figsize=(8, 6))
            shap.summary_plot(values, features=np.asarray(feature_names),
                              max_display=int(top_k), show=False)
            fig = plt.gcf()
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close("all")
            return out
        except Exception:
            plt.close("all")
            return None
