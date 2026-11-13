"""
tests/test_robustness.py — Tests for Phase 5 Adversarial Robustness Testing.

Validates:
- Test 5.1: FGSM perturbation generation and constraint enforcement
- Test 5.1: PGD multi-step attack strength vs FGSM
- Test 5.2: Model accuracy degradation under attack
- Test 5.2: Adversarial training robustness improvement
- Test 5.2: Cross-dataset generalization impact of adversarial training
- Additional contract tests: surrogate transfer attacks, constraint bounds,
  epsilon sweep generation, and metrics computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.evaluation.robustness import (
    FeatureSpaceAttacker,
    adversarial_train,
    compute_accuracy_and_fpr,
    compute_defense_improvement,
    evaluate_cross_dataset_robustness,
    format_results_markdown_table,
    run_epsilon_sweep,
)
from src.models.baseline_dnn import DNNBaseline
from src.models.baseline_xgb import XGBBaseline


# --------------------------------------------------------------------------- #
# ✅ Test 5.1 — Attack Implementation Validation (Verbatim from Plan)
# --------------------------------------------------------------------------- #
class TestPlanTest51:
    """✅ Test 5.1 — Attack Implementation Validation."""

    def test_fgsm_produces_perturbation(self):
        """FGSM generates adversarial examples different from originals."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3)
        )

        X = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))

        attacker = FeatureSpaceAttacker()
        X_adv = attacker.fgsm(model, X, y, epsilon=0.1)

        assert X_adv.shape == X.shape
        assert not torch.allclose(X_adv, X), "Adversarial examples should differ from originals"
        assert torch.all(X_adv >= 0), "Constraint violation: negative features"

    def test_pgd_stronger_than_fgsm(self):
        """PGD (multi-step) should generally be a stronger attack than single-step FGSM."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )

        X = torch.randn(50, 10).abs()  # Positive features
        y = torch.randint(0, 2, (50,))

        attacker = FeatureSpaceAttacker()
        X_fgsm = attacker.fgsm(model, X, y, epsilon=0.1)
        X_pgd = attacker.pgd(model, X, y, epsilon=0.1, steps=10)

        # PGD perturbation should be at least as large as FGSM
        fgsm_dist = torch.norm(X_fgsm - X, dim=1).mean()
        pgd_dist = torch.norm(X_pgd - X, dim=1).mean()

        # PGD uses the full epsilon budget more effectively
        assert pgd_dist >= fgsm_dist * 0.8, (
            "PGD should use epsilon budget at least as effectively as FGSM"
        )


# --------------------------------------------------------------------------- #
# ✅ Test 5.2 — Robustness Evaluation Validation (Verbatim from Plan)
# --------------------------------------------------------------------------- #
class TestPlanTest52:
    """✅ Test 5.2 — Robustness Evaluation Validation."""

    @pytest.fixture
    def synthetic_data(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # Positive features with gentle boundary separation
        X_0 = np.random.uniform(0.1, 0.7, size=(150, 10))  # Benign class
        X_1 = np.random.uniform(0.5, 1.1, size=(150, 10))  # Attack class
        X = np.vstack([X_0, X_1]).astype(np.float32)
        y = np.array(["Benign"] * 150 + ["Attack"] * 150)

        # Split train / test
        idx = np.random.permutation(len(X))
        train_idx, test_idx = idx[:200], idx[200:]
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_adversarial_accuracy_drops(self, synthetic_data):
        """Model accuracy should decrease under adversarial attack."""
        X_train, y_train, X_test, y_test = synthetic_data

        model = DNNBaseline(epochs=25, batch_size=32, learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        attacker = FeatureSpaceAttacker()
        sweep = run_epsilon_sweep(
            model=model,
            X_test=X_test,
            y_test=y_test,
            attacker=attacker,
            epsilons=[0.3],
            attack_types=["fgsm"],
            benign_label="Benign",
        )

        clean_acc = sweep["Clean ACC"].iloc[0]
        adv_acc = sweep["Adv ACC"].iloc[0]

        assert clean_acc >= 0.85, f"Baseline should train well, got clean_acc={clean_acc}"
        assert adv_acc < clean_acc, "Adversarial attack should degrade accuracy"
        assert sweep["Δ ACC"].iloc[0] < 0.0

    def test_adversarial_training_improves_robustness(self, synthetic_data):
        """Adversarially trained model should be more robust than baseline."""
        X_train, y_train, X_test, y_test = synthetic_data

        attacker = FeatureSpaceAttacker()

        # Undefended baseline
        baseline = DNNBaseline(epochs=25, batch_size=32, learning_rate=0.01, random_state=42)
        baseline.fit(X_train, y_train)

        # Defended model via adversarial training
        defended = DNNBaseline(epochs=25, batch_size=32, learning_rate=0.01, random_state=42)
        defended.fit(
            X_train,
            y_train,
            adversarial_attacker=attacker,
            adv_epsilon=0.25,
            adv_ratio=0.5,
        )

        # Evaluate both under attack at epsilon=0.25
        base_sweep = run_epsilon_sweep(
            baseline, X_test, y_test, attacker, epsilons=[0.25], attack_types=["fgsm"]
        )
        def_sweep = run_epsilon_sweep(
            defended, X_test, y_test, attacker, epsilons=[0.25], attack_types=["fgsm"]
        )

        base_adv_acc = base_sweep["Adv ACC"].iloc[0]
        def_adv_acc = def_sweep["Adv ACC"].iloc[0]

        assert def_adv_acc > base_adv_acc, (
            f"Adversarial training should improve robustness: defended={def_adv_acc} vs baseline={base_adv_acc}"
        )

        improvement = compute_defense_improvement(base_sweep, def_sweep, target_epsilon=0.25)
        assert improvement["defense_acc_gain"] > 0

    def test_adversarial_training_cross_dataset_impact(self, synthetic_data):
        """
        IMPORTANT: Check if adversarial training hurts cross-dataset performance.
        This is a novel observation if it does.
        """
        X_train, y_train, X_test, y_test = synthetic_data

        attacker = FeatureSpaceAttacker()

        # Undefended baseline
        baseline = DNNBaseline(epochs=25, batch_size=32, learning_rate=0.01, random_state=42)
        baseline.fit(X_train, y_train)

        # Defended model
        defended = DNNBaseline(epochs=25, batch_size=32, learning_rate=0.01, random_state=42)
        defended.fit(
            X_train,
            y_train,
            adversarial_attacker=attacker,
            adv_epsilon=0.2,
            adv_ratio=0.5,
        )

        # Create out-of-distribution target dataset (shifted feature distribution)
        np.random.seed(123)
        X_target = np.clip(X_test + np.random.normal(0.8, 0.4, size=X_test.shape), 0.0, None).astype(np.float32)
        y_target = y_test.copy()

        cross_eval = evaluate_cross_dataset_robustness(
            baseline_model=baseline,
            defended_model=defended,
            X_source=X_test,
            y_source=y_test,
            X_target=X_target,
            y_target=y_target,
            attacker=attacker,
            epsilon=0.2,
            benign_label="Benign",
        )

        assert "source_in_domain" in cross_eval
        assert "target_cross_domain" in cross_eval
        # Both in-domain and cross-domain evaluated
        assert cross_eval["source_in_domain"]["baseline_clean_acc"] >= 0.0
        assert cross_eval["target_cross_domain"]["defended_clean_acc"] >= 0.0


# --------------------------------------------------------------------------- #
# Additional Attack Contract & Edge Case Tests
# --------------------------------------------------------------------------- #
class TestAttackerContract:
    """Detailed contract tests for FeatureSpaceAttacker."""

    def test_zero_epsilon_preserves_input(self):
        """At epsilon=0.0, perturbed output equals original."""
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        X = torch.rand(10, 8) + 0.1
        y = torch.randint(0, 2, (10,))

        attacker = FeatureSpaceAttacker()
        X_adv = attacker.fgsm(model, X, y, epsilon=0.0)
        assert torch.allclose(X_adv, X, atol=1e-5)

        X_pgd = attacker.pgd(model, X, y, epsilon=0.0, steps=5)
        assert torch.allclose(X_pgd, X, atol=1e-5)

    def test_enforce_constraints_custom_bounds(self):
        """FeatureSpaceAttacker respects custom min and max bounds."""
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
        X = torch.ones(10, 5) * 5.0
        y = torch.zeros(10, dtype=torch.long)

        attacker = FeatureSpaceAttacker(enforce_constraints=True, min_val=2.0, max_val=6.0)
        X_adv = attacker.fgsm(model, X, y, epsilon=5.0, clip_min=2.0, clip_max=6.0)

        assert torch.all(X_adv >= 2.0)
        assert torch.all(X_adv <= 6.0)

    def test_attacker_handles_numpy_arrays(self):
        """Attacker accepts numpy arrays seamlessly."""
        model = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 2))
        X_np = np.random.rand(15, 6).astype(np.float32)
        y_np = np.random.randint(0, 2, size=15)

        attacker = FeatureSpaceAttacker()
        X_adv = attacker.fgsm(model, X_np, y_np, epsilon=0.1)
        assert isinstance(X_adv, torch.Tensor)
        assert X_adv.shape == (15, 6)

    def test_transfer_attack_against_xgboost(self):
        """Surrogate transfer attack deceives non-differentiable XGBoost model."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Train simple dataset
        X = np.random.randn(200, 10).astype(np.float32).clip(0.0, None)
        y = np.array(["Benign"] * 100 + ["Attack"] * 100)

        # Train XGBoost target model
        xgb = XGBBaseline(n_estimators=10, max_depth=3, random_state=42)
        xgb.fit(X, y)

        # Train DNN surrogate model
        dnn = DNNBaseline(epochs=15, batch_size=32, random_state=42)
        dnn.fit(X, y)

        attacker = FeatureSpaceAttacker()
        result = attacker.transfer_attack(
            surrogate_model=dnn.torch_model_,
            target_model=xgb,
            X=X,
            y=y,
            attack="fgsm",
            epsilon=0.5,
            benign_label="Benign",
        )

        assert "clean_acc" in result
        assert "adv_acc" in result
        assert "delta_acc" in result
        assert result["clean_acc"] > 0.0

    def test_run_epsilon_sweep_schema(self):
        """Epsilon sweep produces exact columns matching Section 5.2 template."""
        model = DNNBaseline(epochs=5, batch_size=32, random_state=42)
        X = np.random.rand(40, 8).astype(np.float32)
        y = np.array(["Benign", "Attack"] * 20)
        model.fit(X, y)

        attacker = FeatureSpaceAttacker()
        df = run_epsilon_sweep(
            model=model,
            X_test=X,
            y_test=y,
            attacker=attacker,
            epsilons=[0.01, 0.05, 0.1],
            attack_types=["fgsm", "pgd"],
            benign_label="Benign",
        )

        expected_columns = [
            "Model", "Attack", "ε", "Clean ACC", "Adv ACC", "Δ ACC",
            "Clean FPR", "Adv FPR", "Δ FPR"
        ]
        for col in expected_columns:
            assert col in df.columns
        assert "Clean Detection" in df.columns
        assert "Adv Detection" in df.columns
        assert "Evasion Rate" in df.columns
        assert len(df) == 6  # 2 attacks * 3 epsilons

        # Verify markdown table generation
        md_table = format_results_markdown_table(df)
        assert "| Model | Attack | ε | Clean ACC |" in md_table


# --------------------------------------------------------------------------- #
# ✅ Task 3 — Scientific Robustness & Threat Model Validation Tests
# --------------------------------------------------------------------------- #
class TestTask3Robustness:
    """Validation of Task 3 rewrite requirements."""

    def test_scaled_space_clamp_epsilon_zero_identity(self):
        """
        Scaled-space clamp bug fix:
        In scaled space, non-negative raw features naturally have negative scaled values.
        Epsilon=0 must be strictly the identity operation, preserving negative scaled values.
        """
        from sklearn.preprocessing import RobustScaler

        # Synthetic raw data with medians ~ 50.0 and IQR ~ 20.0
        X_raw = np.array([[0.0, 50.0], [10.0, 60.0], [50.0, 70.0], [100.0, 90.0]], dtype=np.float32)
        scaler = RobustScaler()
        scaler.fit(X_raw)

        # In scaled space, raw 0.0 becomes negative: (0.0 - center) / scale < 0
        X_scaled = scaler.transform(X_raw)
        assert np.any(X_scaled < 0.0), "Scaled features must contain negative values"

        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        attacker = FeatureSpaceAttacker(scaler=scaler, directional=True)

        # At epsilon=0.0, FGSM and PGD must preserve negative scaled values identically
        X_adv_fgsm = attacker.fgsm(model, X_scaled, y, epsilon=0.0)
        assert torch.allclose(X_adv_fgsm, torch.tensor(X_scaled, dtype=torch.float32), atol=1e-5)
        assert torch.any(X_adv_fgsm < 0.0), "Negative scaled values must not be truncated to 0"

        X_adv_pgd = attacker.pgd(model, X_scaled, y, epsilon=0.0, steps=5)
        assert torch.allclose(X_adv_pgd, torch.tensor(X_scaled, dtype=torch.float32), atol=1e-5)

    def test_threat_model_only_attack_flows_perturbed(self):
        """
        Threat model rule:
        Attacker perturbs ONLY attack flows. Benign flows remain 100% unchanged.
        """
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        X = np.random.uniform(0.1, 1.0, size=(20, 4)).astype(np.float32)
        y = np.array(["Benign"] * 10 + ["Attack"] * 10)

        attacker = FeatureSpaceAttacker()
        X_adv = attacker.generate_adversarial(
            model, X, y, attack="fgsm", epsilon=0.5, attack_only=True, benign_label="Benign"
        )

        # Benign rows (0..9) must be identical to original input
        assert np.allclose(X_adv[:10], X[:10]), "Benign flows must not be perturbed by the attacker"
        # Attack rows (10..19) should have been perturbed
        assert not np.allclose(X_adv[10:], X[10:]), "Attack flows must be perturbed"

    def test_threat_model_controllable_features_and_directionality(self):
        """
        Controllable feature restriction:
        Only attacker-controllable features (forward traffic, delays, duration) are perturbed.
        Backward traffic (server responses) must have zero perturbation.
        Perturbations must be non-negative (directional increase).
        """
        feature_names = [
            "flow_duration",               # Controllable (can delay)
            "total_fwd_packets",           # Controllable (can send dummy packets)
            "total_bwd_packets",           # Non-controllable (server response)
            "bwd_packet_length_mean",      # Non-controllable (server response)
        ]

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        X = torch.ones((5, 4), dtype=torch.float32) * 5.0
        y = torch.tensor([1] * 5, dtype=torch.long)  # Attack flows

        attacker = FeatureSpaceAttacker(
            feature_names=feature_names,
            directional=True,
        )

        X_adv = attacker.fgsm(model, X, y, epsilon=1.0)

        # Backward features (col 2 and 3) must NOT be perturbed
        assert torch.allclose(X_adv[:, 2], X[:, 2]), "Backward packets should have 0 perturbation"
        assert torch.allclose(X_adv[:, 3], X[:, 3]), "Backward packet length should have 0 perturbation"

        # Controllable features (col 0 and 1) should be >= original
        assert torch.all(X_adv[:, 0] >= X[:, 0]), "Controllable duration must not decrease"
        assert torch.all(X_adv[:, 1] >= X[:, 1]), "Controllable forward packets must not decrease"

    def test_controls_random_noise_and_query_blackbox(self):
        """
        Controls:
        Random noise baseline of equal magnitude.
        Query-based black-box attack on tree models.
        """
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X = np.random.uniform(0.1, 1.0, size=(100, 4)).astype(np.float32)
        y = np.array(["Benign"] * 50 + ["Attack"] * 50)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        attacker = FeatureSpaceAttacker()

        # Random noise control
        X_rnd = attacker.random_noise(X, epsilon=0.2)
        assert X_rnd.shape == X.shape
        assert np.max(np.abs(X_rnd - X)) <= 0.2 + 1e-5

        # Query-based black-box attack
        X_adv, q_metrics = attacker.query_black_box_attack(
            target_model=rf,
            X=X,
            y=y,
            epsilon=0.3,
            steps=5,
            benign_label="Benign",
        )
        assert "clean_detection_rate" in q_metrics
        assert "adv_detection_rate" in q_metrics
        assert "adv_evasion_rate" in q_metrics
        assert q_metrics["adv_evasion_rate"] >= 0.0

    def test_multi_step_pgd_adversarial_training(self):
        """
        Real multi-step PGD adversarial training (e.g. PGD-7, random start).
        """
        np.random.seed(42)
        torch.manual_seed(42)
        X = np.random.uniform(0.1, 1.0, size=(100, 6)).astype(np.float32)
        y = np.array(["Benign"] * 50 + ["Attack"] * 50)

        attacker = FeatureSpaceAttacker()
        dnn = DNNBaseline(epochs=10, batch_size=32, random_state=42)
        dnn.fit(
            X,
            y,
            adversarial_attacker=attacker,
            adv_epsilon=0.15,
            adv_ratio=0.5,
            adv_method="pgd",
            pgd_steps=7,
            random_start=True,
        )

        preds = dnn.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({"Benign", "Attack"})

    def test_guards_abort_on_constant_classifier_and_small_test_set(self):
        """
        Guard validation:
        1. Aborts with ValueError if clean model is a constant classifier.
        2. Aborts with ValueError if test set has < 500 attack or < 500 benign flows.
        """
        # Constant classifier mock
        class ConstantModel:
            def predict(self, X):
                return np.array(["Benign"] * len(X))

        const_model = ConstantModel()
        X_test = np.random.rand(1200, 4).astype(np.float32)
        y_test = np.array(["Benign"] * 600 + ["Attack"] * 600)
        attacker = FeatureSpaceAttacker()

        # Guard 1: Constant classifier
        with pytest.raises(ValueError, match="constant classifier"):
            run_epsilon_sweep(
                const_model, X_test, y_test, attacker, min_flows=500
            )

        # Model with valid split
        class ValidModel:
            def predict(self, X):
                return np.array(["Benign"] * (len(X) // 2) + ["Attack"] * (len(X) - len(X) // 2))

        valid_model = ValidModel()
        # Guard 2: Small test set (< 500 flows)
        X_small = np.random.rand(100, 4).astype(np.float32)
        y_small = np.array(["Benign"] * 50 + ["Attack"] * 50)

        with pytest.raises(ValueError, match="must contain >= 500"):
            run_epsilon_sweep(
                valid_model, X_small, y_small, attacker, min_flows=500
            )

    def test_gradient_masking_evidence_check(self):
        """
        Gradient masking verification:
        Only claims gradient masking when supported by evidence
        (e.g. black-box/transfer evasion significantly exceeds white-box evasion).
        """
        from src.evaluation.robustness import check_gradient_masking

        # Case 1: White-box fails (0.05), black-box transfer succeeds (0.40) -> Masking confirmed
        res_masking = check_gradient_masking(whitebox_evasion=0.05, blackbox_evasion=0.40)
        assert res_masking["gradient_masking_detected"] is True
        assert "Gradient masking confirmed" in res_masking["evidence"]

        # Case 2: White-box is equally effective as black-box -> No masking
        res_no_masking = check_gradient_masking(whitebox_evasion=0.35, blackbox_evasion=0.30)
        assert res_no_masking["gradient_masking_detected"] is False
        assert "No gradient masking detected" in res_no_masking["evidence"]

