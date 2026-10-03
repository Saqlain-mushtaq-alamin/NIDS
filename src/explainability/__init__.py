"""
src.explainability — Tier-aware explainability for NIDS alerts (Phase 4).

Public API::

    from src.explainability import (
        SHAPExplainer,        # TreeSHAP / KernelSHAP attribution
        DeepLIFTExplainer,    # Captum DeepLIFT attribution for torch models
        TieredExplanation,    # Tier-1 SOC summary / Tier-2 full attribution
        build_alert,          # numeric attributions -> analyst-facing alert
        deletion_auc,         # faithfulness metric (lower = more faithful)
        compare_explainers,   # paired SHAP vs DeepLIFT faithfulness test
    )
"""
from src.explainability.attribution import (
    align_attributions,
    feature_description,
    rank_features,
)
from src.explainability.deeplift_explainer import DeepLIFTExplainer
from src.explainability.faithfulness import (
    as_predict_proba,
    compare_explainers,
    deletion_auc,
    deletion_curve,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.tiered_explanation import TieredExplanation, build_alert

__all__ = [
    "DeepLIFTExplainer",
    "SHAPExplainer",
    "TieredExplanation",
    "align_attributions",
    "as_predict_proba",
    "build_alert",
    "compare_explainers",
    "deletion_auc",
    "deletion_curve",
    "feature_description",
    "rank_features",
]
