"""
src/explainability/deeplift_explainer.py — DeepLIFT attribution via Captum.

DeepLIFT (PHASE_4 section 4.2) compares each hidden unit's activation with a
reference activation and back-propagates the difference to the input, which
gives a single-pass attribution whose cost is independent of the number of
features.  It is the natural counterpart to KernelSHAP for the neural models:

===================  =========================  ======================
Property             KernelSHAP                 DeepLIFT
===================  =========================  ======================
Model access         black box (scoring calls)  gradients + hooks
Cost per alert       ~tens of ms                ~sub-ms
Reference            background distribution   explicit baseline
===================  =========================  ======================

Output contract of :meth:`DeepLIFTExplainer.explain`::

    {
        "attributions": np.ndarray,   # (n, f), same layout as the input
        "feature_names": list[str] | None,
        "target_class": int | None,
        "base_value": float | None,   # score of the baseline input
        "timing_ms": {"total_ms", "ms_per_alert", "n_alerts"},
    }
"""
from __future__ import annotations

import time
from typing import Optional, Sequence, Union

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.explainability.attribution import align_attributions, rank_features

__all__ = ["DeepLIFTExplainer"]


def _require_captum():
    try:
        from captum.attr import DeepLift  # type: ignore
    except Exception as error:  # pragma: no cover - stripped environments
        raise ImportError(
            "captum is required for DeepLIFT attribution. Install it with "
            "`pip install captum` (it is listed in requirements.txt)."
        ) from error
    return DeepLift


class DeepLIFTExplainer:
    """DeepLIFT-based feature attribution using Captum."""

    def __init__(self, model, baseline: Optional[np.ndarray] = None,
                 multiply_by_inputs: bool = True,
                 device: Optional[Union[str, "torch.device"]] = None):
        """
        Args:
            model: A ``torch.nn.Module`` in eval mode (it is put in eval mode
                here).  Dropout/BatchNorm must not be active or the attribution
                would explain a different function than the one deployed.
            baseline: Optional reference input of shape ``(f,)`` or ``(1, f)``.
                ``None`` means an all-zero baseline, which for robust-scaled
                features is the training median — a meaningful "average flow".
            multiply_by_inputs: Captum's ``multiplies_by_inputs`` flag; keep it
                on so attributions scale with the input (comparable to SHAP).
            device: Torch device; defaults to the model's own device.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for DeepLIFT attribution.")
        DeepLift = _require_captum()

        self.model = model
        self.model.eval()
        self.device = device or next(model.parameters(), torch.zeros(1)).device
        self.model.to(self.device)
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=float)
        self.deeplift = DeepLift(self.model, multiply_by_inputs=multiply_by_inputs)
        self.benchmark_: list[dict] = []
        self.last_timing_: dict = {}

    # -- helpers -----------------------------------------------------------
    def _to_tensor(self, X) -> "torch.Tensor":
        if isinstance(X, torch.Tensor):
            tensor = X.detach()
        else:
            tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"X must be 1-D (f,) or 2-D (n, f); got shape {tuple(tensor.shape)}."
            )
        return tensor.to(self.device, dtype=torch.float32)

    def _baseline_tensor(self, sample: "torch.Tensor") -> "torch.Tensor":
        if self.baseline is None:
            return torch.zeros_like(sample)
        base = torch.as_tensor(self.baseline, dtype=torch.float32, device=self.device)
        if base.ndim == 1:
            base = base.unsqueeze(0)
        if base.shape[-1] != sample.shape[-1]:
            raise ValueError(
                f"baseline has {base.shape[-1]} features, model expects {sample.shape[-1]}."
            )
        return base.expand_as(sample)

    def _output_width(self, sample: "torch.Tensor") -> int:
        with torch.no_grad():
            out = self.model(sample[:1])
        if isinstance(out, (tuple, list)):
            out = out[0]
        return int(out.shape[-1])

    def _resolve_target(self, target_class: Optional[int],
                        sample: "torch.Tensor") -> Optional[int]:
        """Single-output models must not be given a class index."""
        width = self._output_width(sample)
        if width == 1:
            if target_class not in (None, 0, 1):
                raise ValueError(
                    f"Model has a single output; target_class={target_class} is invalid."
                )
            return None
        if target_class is None:
            raise ValueError(
                f"Model has {width} outputs; pass target_class explicitly."
            )
        if not 0 <= int(target_class) < width:
            raise ValueError(
                f"target_class {target_class} out of range for {width} outputs."
            )
        return int(target_class)

    # -- explanation -------------------------------------------------------
    def explain(self, X: Union[np.ndarray, "torch.Tensor"],
                target_class: int = 1,
                feature_names: Optional[Sequence[str]] = None,
                with_baseline_score: bool = True) -> dict:
        """
        Compute DeepLIFT attributions for the given flows.

        Args:
            X: ``(n, f)`` feature matrix (or a torch tensor).
            target_class: Class whose logit is being explained.  Ignored (and
                set to ``None``) for single-output regression-style models.
            feature_names: Optional names, echoed back for downstream ranking.
            with_baseline_score: Also score the baseline input so Tier-2 can
                show "score at baseline vs score observed".

        Returns:
            dict as documented in the module docstring.
        """
        sample = self._to_tensor(X)
        target = self._resolve_target(target_class, sample)

        started = time.perf_counter()
        attributions = self.deeplift.attribute(
            sample,
            baselines=self._baseline_tensor(sample),
            target=target,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        values = align_attributions(attributions.detach().cpu().numpy(),
                                    n_samples=sample.shape[0])
        if feature_names is not None and values.shape[1] != len(feature_names):
            raise ValueError(
                f"{len(feature_names)} feature names for {values.shape[1]} attributions."
            )

        timing = {
            "total_ms": elapsed_ms,
            "ms_per_alert": elapsed_ms / max(sample.shape[0], 1),
            "n_alerts": int(sample.shape[0]),
            "method": "captum/deeplift",
        }
        self.last_timing_ = timing
        self.benchmark_.append(timing)

        base_value = None
        if with_baseline_score:
            base_value = self.score(self._baseline_tensor(sample), target)

        return {
            "attributions": values,
            "feature_names": None if feature_names is None else [str(f) for f in feature_names],
            "target_class": target,
            "base_value": base_value,
            "timing_ms": timing,
        }

    def score(self, X: Union[np.ndarray, "torch.Tensor"],
              target_class: Optional[int] = None) -> float:
        """Mean model output for ``target_class`` over ``X`` (used for baselines)."""
        sample = self._to_tensor(X)
        target = self._resolve_target(target_class, sample)
        with torch.no_grad():
            out = self.model(sample)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.shape[-1] == 1:
                return float(out.mean().item())
            if target is None:
                target = int(out.argmax(dim=-1).mode().item())
            return float(out[:, target].mean().item())

    def top_features(self, attributions: np.ndarray,
                     feature_names: Sequence[str],
                     top_k: int = 5) -> list[dict]:
        """Top-K most influential features for a single prediction."""
        return rank_features(attributions, feature_names, top_k=top_k)

    def benchmark(self) -> dict:
        """Aggregate attribution-latency statistics over every recorded call."""
        if not self.benchmark_:
            return {"n_calls": 0, "n_alerts": 0, "mean_ms_per_alert": None}
        per_alert = [b["ms_per_alert"] for b in self.benchmark_]
        return {
            "n_calls": len(self.benchmark_),
            "n_alerts": sum(b["n_alerts"] for b in self.benchmark_),
            "mean_ms_per_alert": float(np.mean(per_alert)),
            "median_ms_per_alert": float(np.median(per_alert)),
            "max_ms_per_alert": float(np.max(per_alert)),
            "method": "captum/deeplift",
        }
