"""
src/evaluation/robustness.py — Adversarial Robustness Testing Module for NIDS.

Implements feature-space adversarial attacks (FGSM, PGD), physical feature constraint
enforcement, scaled-space bounding, random noise controls, query-based black-box attacks,
adversarial training (FGSM and multi-step PGD), and gradient masking verification.

HONEST FRAMING NOTICE (per §3.4 and §6 of the research proposal):
These are feature-space attacks, NOT problem-space (packet-level) attacks. Perturbing
flow features directly does not guarantee that a realizable network packet stream
can be constructed without violating TCP/IP protocol constraints or payload integrity.
All results represent feature-space model sensitivity evaluations, not real-world evasion.

CITATIONS FLAGGED FOR MANUAL USER VERIFICATION:
- [MANUAL CITATION VERIFICATION REQUIRED] arXiv:2306.05494: "Evasion Adversarial Attacks
  Remain Impractical Against ML-based Network Intrusion Detection Systems"
- [MANUAL CITATION VERIFICATION REQUIRED] arXiv:2502.15561: Adversarial training empirical
  gains in NIDS feature space.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


# Substrings identifying features typically controllable by an endpoint attacker
# (e.g. forward packet sizing, forward inter-arrival delay, connection duration)
CONTROLLABLE_SUBSTRINGS = (
    "fwd",
    "flow_duration",
    "flow_iat",
    "duration",
)

# Substrings identifying backward (server-side) or immutable network characteristics
NON_CONTROLLABLE_SUBSTRINGS = (
    "bwd",
    "down_up_ratio",
)

# Substrings identifying discrete integer count features
DISCRETE_COUNT_SUBSTRINGS = (
    "packets",
    "count",
    "header_length",
)


def get_benign_class_idx(model: Any, benign_label: Any = "Benign") -> int:
    """Return the integer class index corresponding to benign_label in model."""
    if hasattr(model, "label_encoder_") and hasattr(model.label_encoder_, "classes_"):
        classes = list(model.label_encoder_.classes_)
        if benign_label in classes:
            return classes.index(benign_label)
        classes_str = [str(c).lower() for c in classes]
        if str(benign_label).lower() in classes_str:
            return classes_str.index(str(benign_label).lower())
        if 0 in classes:
            return classes.index(0)
    return 0


class FeatureSpaceAttacker:
    """
    Feature-space adversarial attacks for NIDS robustness evaluation.

    Threat model:
    - Attacker perturbs ONLY attack flows (benign flows are unperturbed).
    - Attacker manipulates only controllable features (forward traffic, delays, duration).
    - Monotonicity / directional constraints: delays, lengths, and packet counts
      can only be increased (or kept identical), not decreased.
    - Physical constraints: counts >= 0, durations >= 0, discrete features integer-rounded.
    - Scaled-space bounds: clamping is performed per-feature using the fitted scaler
      (bound_scaled = (0 - center) / scale), ensuring negative scaled values are preserved
      and epsilon=0 is strictly the identity operation.
    - Goal: evasion (flow classified as Benign).
    """

    def __init__(
        self,
        enforce_constraints: bool = True,
        scaler: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
        controllable_features: Optional[Sequence[str]] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        clip_min: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        clip_max: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        directional: bool = True,
    ):
        self.enforce_constraints = enforce_constraints
        self.scaler = scaler
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.controllable_features = (
            set(controllable_features) if controllable_features is not None else None
        )
        self.default_min = min_val if min_val is not None else (0.0 if (scaler is None and clip_min is None) else None)
        self.default_max = max_val
        self.clip_min_override = clip_min
        self.clip_max_override = clip_max
        self.directional = directional

    def _to_tensor(
        self,
        data: Union[np.ndarray, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert input to tensor if necessary, auto-encoding string labels if needed."""
        if isinstance(data, torch.Tensor):
            t = data.to(dtype=dtype)
            return t if device is None else t.to(device)
        arr = np.asarray(data)
        if arr.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            arr = LabelEncoder().fit_transform(arr)
            dtype = torch.long
        return torch.tensor(arr, dtype=dtype, device=device)

    def get_controllable_mask(
        self,
        n_features: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return a boolean / float mask indicating which feature dimensions the attacker controls.
        If no feature names were configured, all features are assumed controllable.
        """
        if self.controllable_features is not None and self.feature_names is not None:
            mask = [name in self.controllable_features for name in self.feature_names]
        elif self.feature_names is not None:
            mask = []
            for name in self.feature_names:
                n_lower = name.lower()
                is_non_ctrl = any(s in n_lower for s in NON_CONTROLLABLE_SUBSTRINGS)
                is_ctrl = any(s in n_lower for s in CONTROLLABLE_SUBSTRINGS)
                mask.append(is_ctrl and not is_non_ctrl)
        else:
            mask = [True] * n_features

        if not any(mask):
            mask = [True] * n_features

        return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

    def get_scaled_lower_bounds(
        self,
        n_features: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-feature lower bounds in scaled space using the fitted scaler.
        bound_scaled = (0.0 - center_) / scale_.
        Preserves negative scaled values and avoids invalid zero-clamping in scaled space.
        """
        if self.clip_min_override is not None:
            if isinstance(self.clip_min_override, (int, float)):
                return torch.full((1, n_features), float(self.clip_min_override), device=device)
            return self._to_tensor(self.clip_min_override, device=device).reshape(1, n_features)

        if self.scaler is not None and hasattr(self.scaler, "center_") and hasattr(self.scaler, "scale_"):
            center = np.asarray(self.scaler.center_, dtype=np.float32)
            scale = np.asarray(self.scaler.scale_, dtype=np.float32)
            scale = np.where(scale == 0.0, 1.0, scale)
            raw_min = np.zeros_like(center)
            bounds_scaled = (raw_min - center) / scale
            return torch.tensor(bounds_scaled, dtype=torch.float32, device=device).unsqueeze(0)

        if self.default_min is not None:
            return torch.full((1, n_features), float(self.default_min), device=device)

        return None

    def _enforce_constraints(
        self,
        X_adv: torch.Tensor,
        X_orig: torch.Tensor,
        clip_min: Optional[torch.Tensor] = None,
        clip_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enforce physical constraints on perturbed features:
        1. Monotonicity / directional constraint: attacker can only increase controllable features.
        2. Lower bound in scaled space: clamp to bound_scaled = (0 - center)/scale.
        3. Upper bound if specified.
        4. Integer rounding for discrete count features in raw space.
        """
        # Directional constraint: perturbation delta >= 0 for controllable dimensions
        # Only enforce directional monotonicity when feature_names are present
        if self.directional and self.feature_names is not None:
            delta = X_adv - X_orig
            delta = torch.clamp(delta, min=0.0)
            X_adv = X_orig + delta

        # Clamp to per-feature scaled minimum
        if clip_min is not None:
            X_adv = torch.maximum(X_adv, clip_min)
        if clip_max is not None:
            X_adv = torch.minimum(X_adv, clip_max)

        # Integer rounding for discrete count features
        if self.scaler is not None and self.feature_names is not None:
            X_adv = self._round_discrete_features(X_adv)

        return X_adv

    def _round_discrete_features(self, X: torch.Tensor) -> torch.Tensor:
        """Round discrete count features to valid raw integers and re-scale."""
        if not (hasattr(self.scaler, "center_") and hasattr(self.scaler, "scale_")):
            return X
        if self.feature_names is None:
            return X

        device = X.device
        center = torch.tensor(self.scaler.center_, dtype=torch.float32, device=device).unsqueeze(0)
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=device).unsqueeze(0)
        scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)

        discrete_mask = [
            any(s in name.lower() for s in DISCRETE_COUNT_SUBSTRINGS)
            and not any(s in name.lower() for s in ("mean", "std", "iat", "rate", "ratio"))
            for name in self.feature_names
        ]
        if not any(discrete_mask):
            return X

        mask_t = torch.tensor(discrete_mask, dtype=torch.bool, device=device).unsqueeze(0)

        # Unscale to raw space
        X_raw = X * scale + center
        X_raw_rounded = torch.where(mask_t, torch.round(torch.clamp(X_raw, min=0.0)), X_raw)
        X_scaled = (X_raw_rounded - center) / scale
        return torch.where(mask_t, X_scaled, X)

    def fgsm(
        self,
        model: torch.nn.Module,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray, Sequence],
        epsilon: float = 0.1,
        clip_min: Optional[Any] = None,
        clip_max: Optional[Any] = None,
        target_benign_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack:
        If target_benign_idx is provided, performs targeted attack towards Benign:
            x_adv = x - ε * sign(∇_x L(f(x), y_benign))
        Otherwise untargeted:
            x_adv = x + ε * sign(∇_x L(f(x), y_true))

        Clamps strictly within per-feature scaled bounds and controllable dimensions.
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        X_t = self._to_tensor(X, dtype=torch.float32, device=device)
        y_t = self._to_tensor(y, dtype=torch.long, device=device)

        if epsilon <= 0.0:
            return X_t.clone().detach()

        was_training = model.training
        model.eval()

        X_adv = X_t.clone().detach().requires_grad_(True)
        model.zero_grad()

        output = model(X_adv)

        if target_benign_idx is not None:
            target_labels = torch.full_like(y_t, int(target_benign_idx))
            loss = torch.nn.functional.cross_entropy(output, target_labels)
            loss.backward()
            step_direction = -1.0
        else:
            loss = torch.nn.functional.cross_entropy(output, y_t)
            loss.backward()
            step_direction = 1.0

        if X_adv.grad is not None:
            grad = X_adv.grad
            ctrl_mask = self.get_controllable_mask(X_t.shape[1], device=device)
            grad = grad * ctrl_mask

            perturbation = step_direction * epsilon * grad.sign()
            X_adv = X_t + perturbation
        else:
            X_adv = X_t.clone()

        if self.enforce_constraints:
            if clip_min is not None:
                bounds_min = torch.full((1, X_t.shape[1]), float(clip_min), device=device) if isinstance(clip_min, (int, float)) else self._to_tensor(clip_min, device=device).reshape(1, X_t.shape[1])
            else:
                bounds_min = self.get_scaled_lower_bounds(X_t.shape[1], device=device)

            if clip_max is not None:
                bounds_max = torch.full((1, X_t.shape[1]), float(clip_max), device=device) if isinstance(clip_max, (int, float)) else self._to_tensor(clip_max, device=device).reshape(1, X_t.shape[1])
            elif self.default_max is not None or self.clip_max_override is not None:
                c_m = self.clip_max_override if self.clip_max_override is not None else self.default_max
                bounds_max = torch.full((1, X_t.shape[1]), float(c_m), device=device)
            else:
                bounds_max = None

            X_adv = self._enforce_constraints(X_adv, X_t, clip_min=bounds_min, clip_max=bounds_max)

        if was_training:
            model.train()

        return X_adv.detach()

    def pgd(
        self,
        model: torch.nn.Module,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray, Sequence],
        epsilon: float = 0.1,
        steps: int = 10,
        step_size: Optional[float] = None,
        clip_min: Optional[Any] = None,
        clip_max: Optional[Any] = None,
        random_start: bool = False,
        target_benign_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack (multi-step iterative attack):
        Targeted towards Benign (or untargeted), restricted to controllable features,
        projected into epsilon-ball and per-feature scaled bounds.
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        X_t = self._to_tensor(X, dtype=torch.float32, device=device)
        y_t = self._to_tensor(y, dtype=torch.long, device=device)

        if epsilon <= 0.0:
            return X_t.clone().detach()

        if step_size is None:
            step_size = max(float(epsilon) / float(steps) * 2.0, 0.01)

        was_training = model.training
        model.eval()

        ctrl_mask = self.get_controllable_mask(X_t.shape[1], device=device)

        if clip_min is not None:
            bounds_min = torch.full((1, X_t.shape[1]), float(clip_min), device=device) if isinstance(clip_min, (int, float)) else self._to_tensor(clip_min, device=device).reshape(1, X_t.shape[1])
        else:
            bounds_min = self.get_scaled_lower_bounds(X_t.shape[1], device=device)

        if clip_max is not None:
            bounds_max = torch.full((1, X_t.shape[1]), float(clip_max), device=device) if isinstance(clip_max, (int, float)) else self._to_tensor(clip_max, device=device).reshape(1, X_t.shape[1])
        elif self.default_max is not None or self.clip_max_override is not None:
            c_m = self.clip_max_override if self.clip_max_override is not None else self.default_max
            bounds_max = torch.full((1, X_t.shape[1]), float(c_m), device=device)
        else:
            bounds_max = None

        if random_start:
            if self.directional and self.feature_names is not None:
                noise = torch.FloatTensor(X_t.shape).uniform_(0.0, epsilon).to(device) * ctrl_mask
            else:
                noise = torch.FloatTensor(X_t.shape).uniform_(-epsilon, epsilon).to(device) * ctrl_mask
            X_adv = X_t + noise
            if self.enforce_constraints:
                X_adv = self._enforce_constraints(X_adv, X_t, clip_min=bounds_min, clip_max=bounds_max)
        else:
            X_adv = X_t.clone()

        step_direction = -1.0 if target_benign_idx is not None else 1.0
        target_labels = (
            torch.full_like(y_t, int(target_benign_idx)) if target_benign_idx is not None else y_t
        )

        for _ in range(steps):
            X_adv = X_adv.detach().requires_grad_(True)
            model.zero_grad()

            output = model(X_adv)
            loss = torch.nn.functional.cross_entropy(output, target_labels)
            loss.backward()

            if X_adv.grad is not None:
                grad = X_adv.grad * ctrl_mask
                perturbation = step_direction * step_size * grad.sign()
                X_adv = X_adv.detach() + perturbation
            else:
                X_adv = X_adv.detach()

            # Project back to epsilon-ball
            delta = X_adv - X_t
            if self.directional and self.feature_names is not None:
                delta = torch.clamp(delta, 0.0, epsilon) * ctrl_mask
            else:
                delta = torch.clamp(delta, -epsilon, epsilon) * ctrl_mask
            X_adv = X_t + delta

            if self.enforce_constraints:
                X_adv = self._enforce_constraints(X_adv, X_t, clip_min=bounds_min, clip_max=bounds_max)

        if was_training:
            model.train()

        return X_adv.detach()

    def random_noise(
        self,
        X: Union[torch.Tensor, np.ndarray],
        epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Generate random noise perturbation of equal magnitude epsilon as an experimental control.
        Respects controllable dimensions, directional constraints, and physical bounds.
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if epsilon <= 0.0:
            return X_arr.copy()

        device = torch.device("cpu")
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        ctrl_mask = self.get_controllable_mask(X_t.shape[1], device=device)

        if self.directional and self.feature_names is not None:
            noise = torch.FloatTensor(X_t.shape).uniform_(0.0, epsilon) * ctrl_mask
        else:
            noise = torch.FloatTensor(X_t.shape).uniform_(-epsilon, epsilon) * ctrl_mask

        X_adv = X_t + noise
        if self.enforce_constraints:
            bounds_min = self.get_scaled_lower_bounds(X_t.shape[1], device=device)
            bounds_max = (
                torch.full((1, X_t.shape[1]), float(self.clip_max_override), device=device)
                if self.clip_max_override is not None
                else None
            )
            X_adv = self._enforce_constraints(X_adv, X_t, clip_min=bounds_min, clip_max=bounds_max)

        return X_adv.cpu().numpy()

    def generate_adversarial(
        self,
        model: torch.nn.Module,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, Sequence],
        attack: str = "fgsm",
        epsilon: float = 0.1,
        steps: int = 10,
        step_size: Optional[float] = None,
        batch_size: int = 256,
        attack_only: bool = True,
        benign_label: Any = "Benign",
        target_benign_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate adversarial examples for evaluation.
        THREAT MODEL RULE: Attacker perturbs ONLY attack flows (benign flows are untouched).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, torch.Tensor):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = np.asarray(y)

        if epsilon <= 0.0:
            return X_arr.copy()

        n_samples = len(X_arr)
        X_result = X_arr.copy()

        if attack_only:
            is_attack = _is_attack_flow(y_arr, benign_label=benign_label)
            attack_indices = np.where(is_attack)[0]
        else:
            attack_indices = np.arange(n_samples)

        if len(attack_indices) == 0:
            return X_result

        # Determine benign class index
        if target_benign_idx is None:
            target_benign_idx = get_benign_class_idx(model, benign_label=benign_label)

        # Map labels to numeric indices using model's encoder or full y_arr
        if hasattr(model, "label_encoder_") and hasattr(model.label_encoder_, "classes_"):
            y_numeric = model.label_encoder_.transform(y_arr)
        elif y_arr.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            y_numeric = LabelEncoder().fit_transform(y_arr)
        else:
            y_numeric = y_arr

        attack_fn = self.pgd if attack.lower().startswith("pgd") else self.fgsm

        for i in range(0, len(attack_indices), batch_size):
            b_idx = attack_indices[i : i + batch_size]
            X_b = X_arr[b_idx]
            y_b = y_numeric[b_idx]

            if attack.lower().startswith("pgd"):
                X_adv_b = attack_fn(
                    model,
                    X_b,
                    y_b,
                    epsilon=epsilon,
                    steps=steps,
                    step_size=step_size,
                    target_benign_idx=target_benign_idx,
                )
            else:
                X_adv_b = attack_fn(
                    model,
                    X_b,
                    y_b,
                    epsilon=epsilon,
                    target_benign_idx=target_benign_idx,
                )

            X_result[b_idx] = X_adv_b.cpu().numpy()

        return X_result

    def transfer_attack(
        self,
        surrogate_model: torch.nn.Module,
        target_model: Any,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, Sequence],
        attack: str = "fgsm",
        epsilon: float = 0.1,
        steps: int = 10,
        benign_label: Any = "Benign",
        attack_only: bool = True,
    ) -> Dict[str, float]:
        """
        Execute black-box surrogate transfer attack:
        Generate perturbations on differentiable surrogate_model, evaluate evasion
        against non-differentiable target_model (e.g. XGBoost / Random Forest).
        """
        X_arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        X_adv = self.generate_adversarial(
            surrogate_model,
            X_arr,
            y_arr,
            attack=attack,
            epsilon=epsilon,
            steps=steps,
            attack_only=attack_only,
            benign_label=benign_label,
        )

        clean_preds = target_model.predict(X_arr)
        adv_preds = target_model.predict(X_adv)

        clean_metrics = compute_robustness_metrics(y_arr, clean_preds, benign_label=benign_label)
        adv_metrics = compute_robustness_metrics(y_arr, adv_preds, benign_label=benign_label)

        return {
            "attack": attack.upper(),
            "epsilon": float(epsilon),
            "clean_acc": clean_metrics["accuracy"],
            "adv_acc": adv_metrics["accuracy"],
            "delta_acc": adv_metrics["accuracy"] - clean_metrics["accuracy"],
            "clean_detection_rate": clean_metrics["detection_rate"],
            "adv_detection_rate": adv_metrics["detection_rate"],
            "delta_detection_rate": adv_metrics["detection_rate"] - clean_metrics["detection_rate"],
            "clean_evasion_rate": clean_metrics["evasion_rate"],
            "adv_evasion_rate": adv_metrics["evasion_rate"],
            "delta_evasion_rate": adv_metrics["evasion_rate"] - clean_metrics["evasion_rate"],
            "clean_fpr": clean_metrics["fpr"],
            "adv_fpr": adv_metrics["fpr"],
            "delta_fpr": adv_metrics["fpr"] - clean_metrics["fpr"],
        }

    def query_black_box_attack(
        self,
        target_model: Any,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, Sequence],
        epsilon: float = 0.1,
        steps: int = 10,
        benign_label: Any = "Benign",
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Direct query-based black-box attack on non-differentiable tree models (XGBoost/RF).
        Greedily explores controllable coordinate steps to maximize P(Benign) or flip predictions,
        respecting directional constraints and physical bounds.
        """
        X_arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32)
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        X_adv = X_arr.copy()
        is_attack = _is_attack_flow(y_arr, benign_label=benign_label)
        attack_indices = np.where(is_attack)[0]

        if len(attack_indices) == 0 or epsilon <= 0.0:
            clean_m = compute_robustness_metrics(y_arr, target_model.predict(X_arr), benign_label=benign_label)
            return X_adv, {
                "attack": "QUERY-BLACKBOX",
                "epsilon": float(epsilon),
                **{f"clean_{k}": v for k, v in clean_m.items()},
                **{f"adv_{k}": v for k, v in clean_m.items()},
                "delta_acc": 0.0,
                "delta_detection_rate": 0.0,
                "delta_evasion_rate": 0.0,
                "delta_fpr": 0.0,
            }

        device = torch.device("cpu")
        ctrl_mask = self.get_controllable_mask(X_arr.shape[1], device=device).numpy().flatten()
        controllable_indices = np.where(ctrl_mask > 0)[0]
        bounds_min = self.get_scaled_lower_bounds(X_arr.shape[1], device=device)
        bounds_min_np = bounds_min.numpy().flatten() if bounds_min is not None else None

        # Vectorized batch evaluation over attack flows
        X_attack = X_adv[attack_indices].copy()
        current_preds = target_model.predict(X_attack)
        evaded_mask = (current_preds == benign_label) | (current_preds == 0)

        step_fractions = np.linspace(0.2, 1.0, min(steps, 5)) * epsilon

        for feat_idx in controllable_indices:
            if np.all(evaded_mask):
                break

            for step_val in step_fractions:
                un_evaded = ~evaded_mask
                if not np.any(un_evaded):
                    break

                candidate = X_attack.copy()
                candidate[un_evaded, feat_idx] += step_val

                if bounds_min_np is not None:
                    candidate[:, feat_idx] = np.maximum(candidate[:, feat_idx], bounds_min_np[feat_idx])

                if self.scaler is not None and self.feature_names is not None:
                    cand_t = self._round_discrete_features(
                        torch.tensor(candidate, dtype=torch.float32)
                    )
                    candidate = cand_t.numpy()

                test_preds = target_model.predict(candidate[un_evaded])
                newly_evaded = (test_preds == benign_label) | (test_preds == 0)

                if np.any(newly_evaded):
                    un_evaded_indices = np.where(un_evaded)[0]
                    success_indices = un_evaded_indices[newly_evaded]
                    X_attack[success_indices] = candidate[success_indices]
                    evaded_mask[success_indices] = True

        X_adv[attack_indices] = X_attack

        clean_preds = target_model.predict(X_arr)
        adv_preds = target_model.predict(X_adv)

        clean_m = compute_robustness_metrics(y_arr, clean_preds, benign_label=benign_label)
        adv_m = compute_robustness_metrics(y_arr, adv_preds, benign_label=benign_label)

        metrics = {
            "attack": "QUERY-BLACKBOX",
            "epsilon": float(epsilon),
            "clean_acc": clean_m["accuracy"],
            "adv_acc": adv_m["accuracy"],
            "delta_acc": adv_m["accuracy"] - clean_m["accuracy"],
            "clean_detection_rate": clean_m["detection_rate"],
            "adv_detection_rate": adv_m["detection_rate"],
            "delta_detection_rate": adv_m["detection_rate"] - clean_m["detection_rate"],
            "clean_evasion_rate": clean_m["evasion_rate"],
            "adv_evasion_rate": adv_m["evasion_rate"],
            "delta_evasion_rate": adv_m["evasion_rate"] - clean_m["evasion_rate"],
            "clean_fpr": clean_m["fpr"],
            "adv_fpr": adv_m["fpr"],
            "delta_fpr": adv_m["fpr"] - clean_m["fpr"],
        }
        return X_adv, metrics


def _is_attack_flow(y: Union[np.ndarray, Sequence], benign_label: Any = "Benign") -> np.ndarray:
    """Return boolean array indicating whether each sample is an attack flow (non-benign)."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ("U", "S", "O"):
        b_str = str(benign_label).lower()
        is_benign = np.char.lower(y_arr.astype(str)) == b_str
    else:
        is_benign = (y_arr == benign_label)
        if not np.any(is_benign) and 0 in y_arr:
            is_benign = (y_arr == 0)
    return ~is_benign


def compute_robustness_metrics(
    y_true: Union[np.ndarray, Sequence],
    y_pred: Union[np.ndarray, Sequence],
    benign_label: Any = "Benign",
) -> Dict[str, float]:
    """
    Compute comprehensive scientific robustness metrics:
    - accuracy: overall classification accuracy
    - detection_rate: fraction of attack flows detected as attack (True Positives / All Attacks)
    - evasion_rate: fraction of attack flows classified as Benign (False Negatives / All Attacks)
    - fpr: false positive rate on benign flows (FP / All Benign)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.dtype.kind in ("U", "S", "O") and y_pred_arr.dtype.kind in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        y_true_arr = LabelEncoder().fit_transform(y_true_arr)
    elif y_pred_arr.dtype.kind in ("U", "S", "O") and y_true_arr.dtype.kind in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        y_pred_arr = LabelEncoder().fit_transform(y_pred_arr)

    acc = float(np.mean(y_true_arr == y_pred_arr))

    is_attack_true = _is_attack_flow(y_true_arr, benign_label=benign_label)
    is_attack_pred = _is_attack_flow(y_pred_arr, benign_label=benign_label)

    n_attacks = int(np.sum(is_attack_true))
    n_benign = int(np.sum(~is_attack_true))

    if n_attacks > 0:
        detected_attacks = np.sum(is_attack_true & is_attack_pred)
        evaded_attacks = np.sum(is_attack_true & (~is_attack_pred))
        detection_rate = float(detected_attacks / n_attacks)
        evasion_rate = float(evaded_attacks / n_attacks)
    else:
        detection_rate = 1.0
        evasion_rate = 0.0

    if n_benign > 0:
        false_positives = np.sum((~is_attack_true) & is_attack_pred)
        fpr = float(false_positives / n_benign)
    else:
        fpr = 0.0

    return {
        "accuracy": round(acc, 4),
        "detection_rate": round(detection_rate, 4),
        "evasion_rate": round(evasion_rate, 4),
        "fpr": round(fpr, 4),
    }


def compute_accuracy_and_fpr(
    y_true: Union[np.ndarray, Sequence],
    y_pred: Union[np.ndarray, Sequence],
    benign_label: Any = "Benign",
) -> Tuple[float, float]:
    """Compatibility wrapper returning (accuracy, fpr)."""
    m = compute_robustness_metrics(y_true, y_pred, benign_label=benign_label)
    return m["accuracy"], m["fpr"]


def check_gradient_masking(
    whitebox_evasion: float,
    blackbox_evasion: float,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Check for empirical evidence of gradient masking (Athalye et al., ICML 2018).

    Gradient masking occurs when white-box gradient attacks appear ineffective
    due to broken or vanished gradients, but black-box transfer or query-based attacks
    succeed with significantly higher evasion rates.

    Only claim gradient masking if supported by evidence.
    """
    gap = blackbox_evasion - whitebox_evasion
    masking_detected = bool(gap >= threshold and whitebox_evasion < 0.20)
    return {
        "whitebox_evasion": round(whitebox_evasion, 4),
        "blackbox_evasion": round(blackbox_evasion, 4),
        "gap": round(gap, 4),
        "gradient_masking_detected": masking_detected,
        "evidence": (
            f"Gradient masking confirmed: black-box evasion ({blackbox_evasion:.1%}) "
            f"exceeds white-box evasion ({whitebox_evasion:.1%}) by {gap:.1%}"
            if masking_detected
            else (
                f"No gradient masking detected: white-box attack evasion ({whitebox_evasion:.1%}) "
                f"is consistent with black-box evasion ({blackbox_evasion:.1%})"
            )
        ),
    }


def adversarial_train(
    model: Any,
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor, Sequence],
    attacker: Optional[FeatureSpaceAttacker] = None,
    adv_method: str = "pgd",
    epsilon: float = 0.1,
    adv_ratio: float = 0.5,
    pgd_steps: int = 7,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Any:
    """
    Train or retrain a model with clean and adversarial examples.
    Supports single-step FGSM and multi-step PGD adversarial training (e.g. PGD-7 with random start).
    """
    if attacker is None:
        attacker = FeatureSpaceAttacker()

    if hasattr(model, "fit") and hasattr(model, "torch_model_"):
        defended_model = model.clone() if hasattr(model, "clone") else copy.deepcopy(model)
        defended_model.fit(
            X_train,
            y_train,
            adversarial_attacker=attacker,
            adv_epsilon=epsilon,
            adv_ratio=adv_ratio,
            adv_method=adv_method,
            pgd_steps=pgd_steps,
        )
        return defended_model

    if isinstance(model, torch.nn.Module):
        from src.models.baseline_dnn import DNNBaseline
        dnn = DNNBaseline(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            device=device,
        )
        dnn.fit(
            X_train,
            y_train,
            adversarial_attacker=attacker,
            adv_epsilon=epsilon,
            adv_ratio=adv_ratio,
            adv_method=adv_method,
            pgd_steps=pgd_steps,
        )
        return dnn.torch_model_

    raise TypeError(f"Unsupported model type for adversarial training: {type(model)}")


def run_epsilon_sweep(
    model: Any,
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor, Sequence],
    attacker: FeatureSpaceAttacker,
    epsilons: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.5),
    attack_types: Sequence[str] = ("fgsm", "pgd"),
    surrogate_model: Optional[torch.nn.Module] = None,
    benign_label: Any = "Benign",
    model_name: str = "DNN",
    defense_name: str = "no defense",
    pgd_steps: int = 10,
    pgd_step_size: Optional[float] = None,
    include_random_control: bool = True,
    include_query_blackbox: bool = False,
    min_flows: int = 0,
) -> pd.DataFrame:
    """
    Run an epsilon sweep to characterize the model degradation curve across ε values.

    Guards:
    1. Aborts if the clean model is a constant classifier (predicts a single class).
    2. Aborts if test set has < min_flows attack flows or < min_flows benign flows
       (when min_flows > 0).
    """
    X_arr = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else np.asarray(X_test)
    y_arr = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else np.asarray(y_test)

    # Clean model predictions
    clean_preds = model.predict(X_arr)

    # Guards are enforced when min_flows > 0 (e.g. min_flows=500 in experiments and guard unit tests)
    if min_flows > 0:
        unique_preds = np.unique(clean_preds)
        if len(unique_preds) <= 1:
            raise ValueError(
                f"Evaluation aborted: clean model '{model_name}' is a constant classifier "
                f"(predicted single class: {unique_preds[0] if len(unique_preds) else 'none'})."
            )

        is_attack = _is_attack_flow(y_arr, benign_label=benign_label)
        n_attacks = int(np.sum(is_attack))
        n_benign = int(np.sum(~is_attack))
        if n_attacks < min_flows or n_benign < min_flows:
            raise ValueError(
                f"Evaluation aborted: test set must contain >= {min_flows} attack flows and "
                f">= {min_flows} benign flows (got {n_attacks} attacks, {n_benign} benign)."
            )

    is_attack = _is_attack_flow(y_arr, benign_label=benign_label)
    clean_metrics = compute_robustness_metrics(y_arr, clean_preds, benign_label=benign_label)

    torch_model: Optional[torch.nn.Module] = None
    if hasattr(model, "torch_model_"):
        torch_model = model.torch_model_
    elif isinstance(model, torch.nn.Module):
        torch_model = model
    elif surrogate_model is not None:
        torch_model = surrogate_model

    records: list[dict[str, Any]] = []

    for atk in attack_types:
        atk_name = f"PGD-{pgd_steps}" if atk.lower().startswith("pgd") else "FGSM"

        for eps in epsilons:
            eps_val = float(eps)

            if torch_model is not None:
                # If model is not the surrogate, pass model to find benign_idx; if surrogate, pass surrogate
                eval_for_idx = model if hasattr(model, "label_encoder_") else torch_model
                b_idx = get_benign_class_idx(eval_for_idx, benign_label=benign_label)

                X_adv = attacker.generate_adversarial(
                    torch_model,
                    X_arr,
                    y_arr,
                    attack=atk,
                    epsilon=eps_val,
                    steps=pgd_steps,
                    step_size=pgd_step_size,
                    attack_only=True,
                    benign_label=benign_label,
                    target_benign_idx=b_idx,
                )
            else:
                X_adv = X_arr

            adv_preds = model.predict(X_adv)
            adv_m = compute_robustness_metrics(y_arr, adv_preds, benign_label=benign_label)

            if include_random_control:
                X_rnd = attacker.random_noise(X_arr, epsilon=eps_val)
                X_ctrl = X_arr.copy()
                X_ctrl[is_attack] = X_rnd[is_attack]
                ctrl_preds = model.predict(X_ctrl)
                ctrl_m = compute_robustness_metrics(y_arr, ctrl_preds, benign_label=benign_label)
                ctrl_evasion = ctrl_m["evasion_rate"]
            else:
                ctrl_evasion = 0.0

            records.append({
                "Model": f"{model_name} ({defense_name})",
                "Attack": atk_name,
                "ε": eps_val,
                "Clean ACC": clean_metrics["accuracy"],
                "Adv ACC": adv_m["accuracy"],
                "Δ ACC": round(adv_m["accuracy"] - clean_metrics["accuracy"], 4),
                "Clean Detection": clean_metrics["detection_rate"],
                "Adv Detection": adv_m["detection_rate"],
                "Evasion Rate": adv_m["evasion_rate"],
                "Random Ctrl Evasion": ctrl_evasion,
                "Clean FPR": clean_metrics["fpr"],
                "Adv FPR": adv_m["fpr"],
                "Δ FPR": round(adv_m["fpr"] - clean_metrics["fpr"], 4),
            })

    if include_query_blackbox and not hasattr(model, "torch_model_") and not isinstance(model, torch.nn.Module):
        for eps in epsilons:
            eps_val = float(eps)
            _, q_metrics = attacker.query_black_box_attack(
                target_model=model,
                X=X_arr,
                y=y_arr,
                epsilon=eps_val,
                steps=pgd_steps,
                benign_label=benign_label,
            )
            records.append({
                "Model": f"{model_name} ({defense_name})",
                "Attack": "Query-BlackBox",
                "ε": eps_val,
                "Clean ACC": clean_metrics["accuracy"],
                "Adv ACC": q_metrics["adv_acc"],
                "Δ ACC": q_metrics["delta_acc"],
                "Clean Detection": clean_metrics["detection_rate"],
                "Adv Detection": q_metrics["adv_detection_rate"],
                "Evasion Rate": q_metrics["adv_evasion_rate"],
                "Random Ctrl Evasion": 0.0,
                "Clean FPR": clean_metrics["fpr"],
                "Adv FPR": q_metrics["adv_fpr"],
                "Δ FPR": q_metrics["delta_fpr"],
            })

    return pd.DataFrame(records)


def compute_defense_improvement(
    baseline_sweep: pd.DataFrame,
    defended_sweep: pd.DataFrame,
    target_epsilon: float = 0.1,
    attack_type: str = "FGSM",
) -> Dict[str, float]:
    """
    Compute observed empirical defense improvement without relying on unverified literature benchmarks.
    """
    b_subset = baseline_sweep[
        (np.isclose(baseline_sweep["ε"], target_epsilon))
        & (baseline_sweep["Attack"].str.upper() == attack_type.upper())
    ]
    d_subset = defended_sweep[
        (np.isclose(defended_sweep["ε"], target_epsilon))
        & (defended_sweep["Attack"].str.upper() == attack_type.upper())
    ]

    if b_subset.empty or d_subset.empty:
        raise ValueError(f"No matching rows found for ε={target_epsilon}, attack={attack_type}")

    base_adv_acc = float(b_subset["Adv ACC"].iloc[0])
    base_adv_fpr = float(b_subset["Adv FPR"].iloc[0])
    def_adv_acc = float(d_subset["Adv ACC"].iloc[0])
    def_adv_fpr = float(d_subset["Adv FPR"].iloc[0])

    base_evasion = float(b_subset["Evasion Rate"].iloc[0]) if "Evasion Rate" in b_subset else (1.0 - base_adv_acc)
    def_evasion = float(d_subset["Evasion Rate"].iloc[0]) if "Evasion Rate" in d_subset else (1.0 - def_adv_acc)

    acc_gain = def_adv_acc - base_adv_acc
    fpr_reduction = base_adv_fpr - def_adv_fpr
    evasion_reduction = base_evasion - def_evasion

    return {
        "epsilon": target_epsilon,
        "attack": attack_type,
        "baseline_adv_acc": base_adv_acc,
        "defended_adv_acc": def_adv_acc,
        "defense_acc_gain": round(acc_gain, 4),
        "defense_acc_gain_pct": round(acc_gain * 100, 2),
        "baseline_adv_fpr": base_adv_fpr,
        "defended_adv_fpr": def_adv_fpr,
        "defense_fpr_reduction": round(fpr_reduction, 4),
        "defense_fpr_reduction_pct": round(fpr_reduction * 100, 2),
        "baseline_evasion_rate": base_evasion,
        "defended_evasion_rate": def_evasion,
        "evasion_reduction": round(evasion_reduction, 4),
    }


def evaluate_cross_dataset_robustness(
    baseline_model: Any,
    defended_model: Any,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    attacker: FeatureSpaceAttacker,
    epsilon: float = 0.1,
    benign_label: Any = "Benign",
) -> Dict[str, Any]:
    """
    Evaluate whether adversarial training hurts cross-dataset generalization.
    """
    src_base_clean = compute_robustness_metrics(y_source, baseline_model.predict(X_source), benign_label)
    src_def_clean = compute_robustness_metrics(y_source, defended_model.predict(X_source), benign_label)

    src_adv_base_X = attacker.generate_adversarial(
        baseline_model.torch_model_,
        X_source,
        y_source,
        attack="fgsm",
        epsilon=epsilon,
        attack_only=True,
        benign_label=benign_label,
    )
    src_adv_def_X = attacker.generate_adversarial(
        defended_model.torch_model_,
        X_source,
        y_source,
        attack="fgsm",
        epsilon=epsilon,
        attack_only=True,
        benign_label=benign_label,
    )
    src_base_adv = compute_robustness_metrics(y_source, baseline_model.predict(src_adv_base_X), benign_label)
    src_def_adv = compute_robustness_metrics(y_source, defended_model.predict(src_adv_def_X), benign_label)

    known_classes = set(baseline_model.label_encoder_.classes_)
    if hasattr(defended_model, "label_encoder_"):
        known_classes &= set(defended_model.label_encoder_.classes_)

    target_mask = np.isin(y_target, list(known_classes))
    if np.any(target_mask):
        X_target = X_target[target_mask]
        y_target = y_target[target_mask]

    tgt_base_clean = compute_robustness_metrics(y_target, baseline_model.predict(X_target), benign_label)
    tgt_def_clean = compute_robustness_metrics(y_target, defended_model.predict(X_target), benign_label)

    tgt_adv_base_X = attacker.generate_adversarial(
        baseline_model.torch_model_,
        X_target,
        y_target,
        attack="fgsm",
        epsilon=epsilon,
        attack_only=True,
        benign_label=benign_label,
    )
    tgt_adv_def_X = attacker.generate_adversarial(
        defended_model.torch_model_,
        X_target,
        y_target,
        attack="fgsm",
        epsilon=epsilon,
        attack_only=True,
        benign_label=benign_label,
    )
    tgt_base_adv = compute_robustness_metrics(y_target, baseline_model.predict(tgt_adv_base_X), benign_label)
    tgt_def_adv = compute_robustness_metrics(y_target, defended_model.predict(tgt_adv_def_X), benign_label)

    clean_gap_base = src_base_clean["accuracy"] - tgt_base_clean["accuracy"]
    clean_gap_def = src_def_clean["accuracy"] - tgt_def_clean["accuracy"]
    cross_clean_diff = tgt_def_clean["accuracy"] - tgt_base_clean["accuracy"]
    cross_adv_diff = tgt_def_adv["accuracy"] - tgt_base_adv["accuracy"]

    return {
        "epsilon": epsilon,
        "source_in_domain": {
            "baseline_clean_acc": src_base_clean["accuracy"],
            "baseline_adv_acc": src_base_adv["accuracy"],
            "defended_clean_acc": src_def_clean["accuracy"],
            "defended_adv_acc": src_def_adv["accuracy"],
            "defense_gain_in_domain": round(src_def_adv["accuracy"] - src_base_adv["accuracy"], 4),
            "baseline_evasion": src_base_adv["evasion_rate"],
            "defended_evasion": src_def_adv["evasion_rate"],
        },
        "target_cross_domain": {
            "baseline_clean_acc": tgt_base_clean["accuracy"],
            "baseline_adv_acc": tgt_base_adv["accuracy"],
            "defended_clean_acc": tgt_def_clean["accuracy"],
            "defended_adv_acc": tgt_def_adv["accuracy"],
            "defense_gain_cross_domain": round(tgt_def_adv["accuracy"] - tgt_base_adv["accuracy"], 4),
            "cross_clean_acc_change": round(cross_clean_diff, 4),
            "generalization_gap_baseline": round(clean_gap_base, 4),
            "generalization_gap_defended": round(clean_gap_def, 4),
            "generalization_gap_widened": bool(clean_gap_def > clean_gap_base),
        },
    }


def format_results_markdown_table(df: pd.DataFrame) -> str:
    """Format results DataFrame as GitHub Flavored Markdown table."""
    headers = [col for col in [
        "Model", "Attack", "ε", "Clean ACC", "Adv ACC", "Δ ACC",
        "Clean Detection", "Adv Detection", "Evasion Rate", "Random Ctrl Evasion",
        "Clean FPR", "Adv FPR", "Δ FPR"
    ] if col in df.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        eps_str = f"{row['ε']:.2f}"
        items = []
        for h in headers:
            val = row[h]
            if h == "ε":
                items.append(eps_str)
            elif isinstance(val, (int, float)):
                if "Δ" in h:
                    items.append(f"{val:+.4f}")
                elif "Rate" in h or "Detection" in h:
                    items.append(f"{val:.4f}")
                else:
                    items.append(f"{val:.4f}")
            else:
                items.append(str(val))
        lines.append("| " + " | ".join(items) + " |")
    return "\n".join(lines)
