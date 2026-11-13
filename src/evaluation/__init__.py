# Package initialization
from src.evaluation.metrics import compute_metrics, per_class_report
from src.evaluation.calibration import (
    expected_calibration_error,
    reliability_diagram_data,
    TemperatureScaler,
)
from src.evaluation.robustness import (
    FeatureSpaceAttacker,
    adversarial_train,
    run_epsilon_sweep,
    compute_defense_improvement,
    evaluate_cross_dataset_robustness,
    compute_accuracy_and_fpr,
)

__all__ = [
    "compute_metrics",
    "per_class_report",
    "expected_calibration_error",
    "reliability_diagram_data",
    "TemperatureScaler",
    "FeatureSpaceAttacker",
    "adversarial_train",
    "run_epsilon_sweep",
    "compute_defense_improvement",
    "evaluate_cross_dataset_robustness",
    "compute_accuracy_and_fpr",
]
