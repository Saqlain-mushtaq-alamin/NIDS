"""
src/evaluation/robustness.py — Adversarial Robustness Testing Module for NIDS.

Implements feature-space adversarial attacks (FGSM, PGD), physical feature constraint
enforcement, epsilon degradation sweeps, basic adversarial training, black-box surrogate
transfer attacks, and cross-dataset adversarial evaluation.

HONEST FRAMING NOTICE (per §3.4 and §6 of the research proposal):
These are feature-space attacks, NOT problem-space (packet-level) attacks. Perturbing
flow features directly does not guarantee that a realizable network packet stream
can be constructed without violating TCP/IP protocol constraints or payload integrity.
All results represent feature-space model sensitivity evaluations, not real-world evasion.
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


class FeatureSpaceAttacker:
    """
    Feature-space adversarial attacks for NIDS robustness evaluation.

    IMPORTANT: These are feature-space attacks, not problem-space (packet-level)
    attacks. Per §3.4 of the research proposal, this is a known limitation —
    most published adversarial NIDS evaluations use this same approach, but the
    field's own surveys criticize it as insufficient for real adversarial
    robustness claims.
    """

    def __init__(
        self,
        enforce_constraints: bool = True,
        min_val: Optional[float] = 0.0,
        max_val: Optional[float] = None,
    ):
        self.enforce_constraints = enforce_constraints
        self.default_min = min_val
        self.default_max = max_val

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

    def fgsm(
        self,
        model: torch.nn.Module,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray, Sequence],
        epsilon: float = 0.1,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack:
            x_adv = x + ε * sign(∇_x L(f(x), y))
        Clamps to valid feature constraints.
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        X_t = self._to_tensor(X, dtype=torch.float32, device=device)
        y_t = self._to_tensor(y, dtype=torch.long, device=device)

        was_training = model.training
        model.eval()

        X_adv = X_t.clone().detach().requires_grad_(True)
        model.zero_grad()

        output = model(X_adv)
        loss = torch.nn.functional.cross_entropy(output, y_t)
        loss.backward()

        if X_adv.grad is not None:
            perturbation = epsilon * X_adv.grad.sign()
            X_adv = X_t + perturbation
        else:
            X_adv = X_t.clone()

        # Clamp to valid feature ranges
        if self.enforce_constraints:
            c_min = clip_min if clip_min is not None else self.default_min
            c_max = clip_max if clip_max is not None else self.default_max
            X_adv = self._enforce_constraints(X_adv, clip_min=c_min, clip_max=c_max)

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
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = None,
        random_start: bool = False,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack (iterative FGSM):
            x_adv^(t+1) = Proj_{x + S}(x_adv^t + step_size * sign(∇_x L))
        Clamps to valid feature constraints and projects back to epsilon-ball.
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        X_t = self._to_tensor(X, dtype=torch.float32, device=device)
        y_t = self._to_tensor(y, dtype=torch.long, device=device)

        if step_size is None:
            # Default step size chosen so steps * step_size comfortably spans the epsilon ball
            step_size = max(float(epsilon) / float(steps) * 2.0, 0.02)

        was_training = model.training
        model.eval()

        if random_start:
            noise = torch.FloatTensor(X_t.shape).uniform_(-epsilon, epsilon).to(device)
            X_adv = X_t + noise
            if self.enforce_constraints:
                c_min = clip_min if clip_min is not None else self.default_min
                c_max = clip_max if clip_max is not None else self.default_max
                X_adv = self._enforce_constraints(X_adv, clip_min=c_min, clip_max=c_max)
        else:
            X_adv = X_t.clone()

        for _ in range(steps):
            X_adv = X_adv.detach().requires_grad_(True)
            model.zero_grad()

            output = model(X_adv)
            loss = torch.nn.functional.cross_entropy(output, y_t)
            loss.backward()

            if X_adv.grad is not None:
                perturbation = step_size * X_adv.grad.sign()
                X_adv = X_adv.detach() + perturbation
            else:
                X_adv = X_adv.detach()

            # Project back to epsilon-ball
            delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
            X_adv = X_t + delta

            # Clamp to valid feature ranges
            if self.enforce_constraints:
                c_min = clip_min if clip_min is not None else self.default_min
                c_max = clip_max if clip_max is not None else self.default_max
                X_adv = self._enforce_constraints(X_adv, clip_min=c_min, clip_max=c_max)

        if was_training:
            model.train()

        return X_adv.detach()

    def _enforce_constraints(
        self,
        X: torch.Tensor,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Enforce physical constraints on perturbed features.
        Packet counts, byte counts, flag counts and durations must be non-negative.
        """
        if clip_min is not None:
            X = torch.clamp(X, min=float(clip_min))
        if clip_max is not None:
            X = torch.clamp(X, max=float(clip_max))
        return X

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
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate adversarial examples for a complete dataset in mini-batches.
        Returns a numpy array of perturbed features.
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, torch.Tensor):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = np.asarray(y)

        n_samples = len(X_arr)
        adv_batches: list[np.ndarray] = []

        attack_fn = self.pgd if attack.lower().startswith("pgd") else self.fgsm

        for i in range(0, n_samples, batch_size):
            X_b = X_arr[i : i + batch_size]
            y_b = y_arr[i : i + batch_size]

            if attack.lower().startswith("pgd"):
                X_adv = attack_fn(
                    model,
                    X_b,
                    y_b,
                    epsilon=epsilon,
                    steps=steps,
                    step_size=step_size,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                X_adv = attack_fn(
                    model,
                    X_b,
                    y_b,
                    epsilon=epsilon,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            adv_batches.append(X_adv.cpu().numpy())

        return np.concatenate(adv_batches, axis=0)

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
    ) -> Dict[str, float]:
        """
        Execute black-box transfer attack:
        Generate adversarial examples using differentiable surrogate_model,
        then evaluate evasion against non-differentiable target_model (e.g. XGBoost/RF).
        """
        X_arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        # Determine integer labels for surrogate gradient calculation
        if hasattr(surrogate_model, "label_encoder_"):
            y_indices = surrogate_model.label_encoder_.transform(y_arr)
        elif hasattr(target_model, "label_encoder_"):
            y_indices = target_model.label_encoder_.transform(y_arr)
        elif y_arr.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            y_indices = LabelEncoder().fit_transform(y_arr)
        else:
            y_indices = y_arr

        # Generate adversarial examples from surrogate
        X_adv = self.generate_adversarial(
            surrogate_model, X_arr, y_indices, attack=attack, epsilon=epsilon, steps=steps
        )

        # Evaluate target model on clean vs adv
        clean_preds = target_model.predict(X_arr)
        adv_preds = target_model.predict(X_adv)

        clean_acc, clean_fpr = compute_accuracy_and_fpr(y_arr, clean_preds, benign_label=benign_label)
        adv_acc, adv_fpr = compute_accuracy_and_fpr(y_arr, adv_preds, benign_label=benign_label)

        return {
            "attack": attack.upper(),
            "epsilon": float(epsilon),
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "delta_acc": adv_acc - clean_acc,
            "clean_fpr": clean_fpr,
            "adv_fpr": adv_fpr,
            "delta_fpr": adv_fpr - clean_fpr,
        }


def compute_accuracy_and_fpr(
    y_true: Union[np.ndarray, Sequence],
    y_pred: Union[np.ndarray, Sequence],
    benign_label: Any = "Benign",
) -> Tuple[float, float]:
    """
    Compute classification accuracy and False Positive Rate (FPR).

    FPR is defined as FP / (FP + TN), where benign instances are negative (0)
    and any anomaly/attack class is positive (1).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Harmonize dtypes if one is string/object and the other is numeric
    if y_true_arr.dtype.kind in ("U", "S", "O") and y_pred_arr.dtype.kind in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        y_true_arr = LabelEncoder().fit_transform(y_true_arr)
    elif y_pred_arr.dtype.kind in ("U", "S", "O") and y_true_arr.dtype.kind in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        y_pred_arr = LabelEncoder().fit_transform(y_pred_arr)

    acc = float(np.mean(y_true_arr == y_pred_arr))

    # Binary categorization: Benign vs Attack/Anomaly
    # Check if benign_label matches directly, case-insensitively, or numeric 0
    is_benign_true = (y_true_arr == benign_label)
    if not np.any(is_benign_true) and isinstance(benign_label, str):
        # Fallback check lowercase or integer 0
        if y_true_arr.dtype.kind in ("U", "S", "O"):
            is_benign_true = (
                np.char.lower(y_true_arr.astype(str)) == benign_label.lower()
            )
        elif 0 in y_true_arr:
            is_benign_true = (y_true_arr == 0)
    if not np.any(is_benign_true) and 0 in y_true_arr:
        is_benign_true = (y_true_arr == 0)

    total_benign = np.sum(is_benign_true)
    if total_benign > 0:
        is_benign_pred = (y_pred_arr == benign_label)
        if not np.any(is_benign_pred) and isinstance(benign_label, str):
            is_benign_pred = (
                np.char.lower(y_pred_arr.astype(str)) == benign_label.lower()
            )
        if not np.any(is_benign_pred) and 0 in y_pred_arr:
            is_benign_pred = (y_pred_arr == 0)

        false_positives = np.sum(is_benign_true & (~is_benign_pred))
        fpr = float(false_positives / total_benign)
    else:
        fpr = 0.0

    return acc, fpr


def adversarial_train(
    model: Any,
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor, Sequence],
    attacker: Optional[FeatureSpaceAttacker] = None,
    epsilon: float = 0.1,
    adv_ratio: float = 0.5,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Any:
    """
    Train or retrain a model with a mixture of clean and adversarial examples.
    Default: 50% clean + 50% FGSM-perturbed examples per Step 5.2.2.
    """
    if attacker is None:
        attacker = FeatureSpaceAttacker()

    # If already DNNBaseline instance
    if hasattr(model, "fit") and hasattr(model, "torch_model_"):
        defended_model = model.clone() if hasattr(model, "clone") else copy.deepcopy(model)
        defended_model.fit(
            X_train,
            y_train,
            adversarial_attacker=attacker,
            adv_epsilon=epsilon,
            adv_ratio=adv_ratio,
        )
        return defended_model

    # If it's a raw torch.nn.Module
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
) -> pd.DataFrame:
    """
    Run an epsilon sweep to characterize the model degradation curve across ε values.

    Returns DataFrame matching the Results Table Template in Section 5.2:
    | Model | Attack | ε | Clean ACC | Adv ACC | Δ ACC | Clean FPR | Adv FPR | Δ FPR |
    """
    X_arr = X_test.detach().cpu().numpy() if isinstance(X_test, torch.Tensor) else np.asarray(X_test)
    y_arr = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else np.asarray(y_test)

    # Determine underlying PyTorch module for gradient attacks
    torch_model: Optional[torch.nn.Module] = None
    if hasattr(model, "torch_model_"):
        torch_model = model.torch_model_
    elif isinstance(model, torch.nn.Module):
        torch_model = model
    elif surrogate_model is not None:
        torch_model = surrogate_model

    if torch_model is None and not (hasattr(model, "predict")):
        raise ValueError("Must provide either a PyTorch model or a surrogate model for tree baselines.")

    clean_preds = model.predict(X_arr)
    clean_acc, clean_fpr = compute_accuracy_and_fpr(y_arr, clean_preds, benign_label=benign_label)

    records: list[dict[str, Any]] = []

    for atk in attack_types:
        atk_name = f"PGD-{pgd_steps}" if atk.lower().startswith("pgd") else "FGSM"

        for eps in epsilons:
            eps_val = float(eps)

            # Generate adversarial examples using PyTorch model
            if torch_model is not None:
                # Convert string labels to class indices if needed
                if hasattr(model, "label_encoder_"):
                    y_indices = model.label_encoder_.transform(y_arr)
                elif hasattr(surrogate_model, "label_encoder_"):
                    y_indices = surrogate_model.label_encoder_.transform(y_arr)
                else:
                    # If labels are strings, encode internally
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_indices = le.fit_transform(y_arr)

                X_adv = attacker.generate_adversarial(
                    torch_model,
                    X_arr,
                    y_indices,
                    attack=atk,
                    epsilon=eps_val,
                    steps=pgd_steps,
                    step_size=pgd_step_size,
                )
            else:
                X_adv = X_arr

            adv_preds = model.predict(X_adv)
            adv_acc, adv_fpr = compute_accuracy_and_fpr(y_arr, adv_preds, benign_label=benign_label)

            records.append({
                "Model": f"{model_name} ({defense_name})",
                "Attack": atk_name,
                "ε": eps_val,
                "Clean ACC": round(clean_acc, 4),
                "Adv ACC": round(adv_acc, 4),
                "Δ ACC": round(adv_acc - clean_acc, 4),
                "Clean FPR": round(clean_fpr, 4),
                "Adv FPR": round(adv_fpr, 4),
                "Δ FPR": round(adv_fpr - clean_fpr, 4),
            })

    return pd.DataFrame(records)


def compute_defense_improvement(
    baseline_sweep: pd.DataFrame,
    defended_sweep: pd.DataFrame,
    target_epsilon: float = 0.1,
    attack_type: str = "FGSM",
) -> Dict[str, float]:
    """
    Compute improvement achieved by adversarial defense under attack (Step 5.2.4).
    Compares against literature target:
        +35% detection accuracy, -12.5% false positives (arXiv:2502.15561).
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

    acc_gain = def_adv_acc - base_adv_acc
    fpr_reduction = base_adv_fpr - def_adv_fpr  # Positive means reduction

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
    Evaluate whether adversarial training hurts cross-dataset generalization (Step 5.2.5).

    Returns comparison of source in-domain performance vs target out-of-domain performance
    for both the baseline and the defended model under clean and adversarial conditions.
    """
    # 1. Source (in-domain) evaluations
    src_base_clean = compute_accuracy_and_fpr(y_source, baseline_model.predict(X_source), benign_label)[0]
    src_def_clean = compute_accuracy_and_fpr(y_source, defended_model.predict(X_source), benign_label)[0]

    src_adv_base_X = attacker.generate_adversarial(
        baseline_model.torch_model_,
        X_source,
        baseline_model.label_encoder_.transform(y_source),
        attack="fgsm",
        epsilon=epsilon,
    )
    src_adv_def_X = attacker.generate_adversarial(
        defended_model.torch_model_,
        X_source,
        defended_model.label_encoder_.transform(y_source),
        attack="fgsm",
        epsilon=epsilon,
    )
    src_base_adv = compute_accuracy_and_fpr(y_source, baseline_model.predict(src_adv_base_X), benign_label)[0]
    src_def_adv = compute_accuracy_and_fpr(y_source, defended_model.predict(src_adv_def_X), benign_label)[0]

    # 2. Target (cross-domain) evaluations
    # Filter target to shared classes known by the model's label encoder
    known_classes = set(baseline_model.label_encoder_.classes_)
    if hasattr(defended_model, "label_encoder_"):
        known_classes &= set(defended_model.label_encoder_.classes_)

    target_mask = np.isin(y_target, list(known_classes))
    if np.any(target_mask):
        X_target = X_target[target_mask]
        y_target = y_target[target_mask]

    tgt_base_clean = compute_accuracy_and_fpr(y_target, baseline_model.predict(X_target), benign_label)[0]
    tgt_def_clean = compute_accuracy_and_fpr(y_target, defended_model.predict(X_target), benign_label)[0]

    tgt_adv_base_X = attacker.generate_adversarial(
        baseline_model.torch_model_,
        X_target,
        baseline_model.label_encoder_.transform(y_target),
        attack="fgsm",
        epsilon=epsilon,
    )
    tgt_adv_def_X = attacker.generate_adversarial(
        defended_model.torch_model_,
        X_target,
        defended_model.label_encoder_.transform(y_target),
        attack="fgsm",
        epsilon=epsilon,
    )
    tgt_base_adv = compute_accuracy_and_fpr(y_target, baseline_model.predict(tgt_adv_base_X), benign_label)[0]
    tgt_def_adv = compute_accuracy_and_fpr(y_target, defended_model.predict(tgt_adv_def_X), benign_label)[0]

    # Compute generalization gaps
    clean_gap_base = src_base_clean - tgt_base_clean
    clean_gap_def = src_def_clean - tgt_def_clean
    cross_clean_diff = tgt_def_clean - tgt_base_clean
    cross_adv_diff = tgt_def_adv - tgt_base_adv

    return {
        "epsilon": epsilon,
        "source_in_domain": {
            "baseline_clean_acc": round(src_base_clean, 4),
            "baseline_adv_acc": round(src_base_adv, 4),
            "defended_clean_acc": round(src_def_clean, 4),
            "defended_adv_acc": round(src_def_adv, 4),
            "defense_gain_in_domain": round(src_def_adv - src_base_adv, 4),
        },
        "target_cross_domain": {
            "baseline_clean_acc": round(tgt_base_clean, 4),
            "baseline_adv_acc": round(tgt_base_adv, 4),
            "defended_clean_acc": round(tgt_def_clean, 4),
            "defended_adv_acc": round(tgt_def_adv, 4),
            "defense_gain_cross_domain": round(tgt_def_adv - tgt_base_adv, 4),
            "cross_clean_acc_change": round(cross_clean_diff, 4),
            "generalization_gap_baseline": round(clean_gap_base, 4),
            "generalization_gap_defended": round(clean_gap_def, 4),
            "generalization_gap_widened": bool(clean_gap_def > clean_gap_base),
        },
    }


def format_results_markdown_table(df: pd.DataFrame) -> str:
    """Format results DataFrame as GitHub Flavored Markdown table."""
    headers = [
        "Model", "Attack", "ε", "Clean ACC", "Adv ACC", "Δ ACC",
        "Clean FPR", "Adv FPR", "Δ FPR"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        eps_str = f"{row['ε']:.2f}"
        row_str = (
            f"| {row['Model']} | {row['Attack']} | {eps_str} | "
            f"{row['Clean ACC']:.4f} | {row['Adv ACC']:.4f} | {row['Δ ACC']:+.4f} | "
            f"{row['Clean FPR']:.4f} | {row['Adv FPR']:.4f} | {row['Δ FPR']:+.4f} |"
        )
        lines.append(row_str)
    return "\n".join(lines)
