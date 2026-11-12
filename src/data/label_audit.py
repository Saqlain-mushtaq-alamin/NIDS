"""
src/data/label_audit.py — Label noise detection and audit reporting.
Uses cleanlab for confident learning when available, falls back to heuristic methods.
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LabelAuditor:
    """
    Detect and report potential label noise in a dataset.

    Uses cross-validated predictions to estimate per-class noise rates,
    then generates a structured audit report.
    """

    def __init__(self, cv_folds: int = 3, seed: int = 42):
        self.cv_folds = cv_folds
        self.seed = seed

    def _estimate_noise_with_cleanlab(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Use cleanlab's confident learning to estimate noise."""
        try:
            import cleanlab  # noqa: F401
            from cleanlab.filter import find_label_issues
        except ImportError as exc:
            logger.warning(
                "cleanlab is unavailable (%s); falling back to cv_heuristic", exc
            )
            return {}

        le = LabelEncoder()
        y_int = le.fit_transform(y)
        classes = le.classes_

        clf = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=self.seed, n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        proba = np.zeros((len(y_int), len(classes)))

        for train_idx, val_idx in skf.split(X, y_int):
            clf.fit(X[train_idx], y_int[train_idx])
            proba[val_idx] = clf.predict_proba(X[val_idx])

        # Find label issues
        issues = find_label_issues(
            labels=y_int,
            pred_probs=proba,
            return_indices_ranked_by="self_confidence",
        )

        noise_rate = len(issues) / len(y_int) if len(issues) > 0 else 0.0

        # Per-class noise
        per_class = {}
        for idx, cls in enumerate(classes):
            cls_mask = y_int == idx
            cls_issues = np.intersect1d(issues, np.where(cls_mask)[0])
            per_class[str(cls)] = len(cls_issues) / max(cls_mask.sum(), 1)

        return {
            "method": "cleanlab",
            "noise_rate": float(noise_rate),
            "num_issues": int(len(issues)),
            "total_samples": len(y_int),
            "per_class_noise": per_class,
            "issue_indices": issues.tolist()[:100],  # First 100 for reference
        }

    def _estimate_noise_heuristic(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Estimate noise using cross-validation error as proxy."""
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=self.seed, n_jobs=-1
        )

        y_pred = cross_val_predict(clf, X, y, cv=self.cv_folds)

        # Misclassified samples are potential noise
        misclassified = y != y_pred
        noise_rate = misclassified.mean()

        # Per-class noise
        classes = np.unique(y)
        per_class = {}
        for cls in classes:
            cls_mask = y == cls
            cls_misclassified = misclassified & cls_mask
            per_class[str(cls)] = cls_misclassified.sum() / max(cls_mask.sum(), 1)

        return {
            "method": "cv_heuristic",
            "noise_rate": float(noise_rate),
            "num_issues": int(misclassified.sum()),
            "total_samples": len(y),
            "per_class_noise": per_class,
        }

    def audit(self, df: pd.DataFrame, label_column: str = "label") -> dict:
        """
        Run label audit on a DataFrame.

        Args:
            df: DataFrame with features and a label column.
            label_column: Name of the label column.

        Returns:
            Audit report dict with noise_rate, per_class_noise, etc.
        """
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found.")

        # Separate features and labels
        feature_cols = [c for c in df.columns if c != label_column]
        X = df[feature_cols].select_dtypes(include=[np.number]).values
        y = df[label_column].values

        # Impute any remaining NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        # Try cleanlab first, fall back to heuristic
        report = self._estimate_noise_with_cleanlab(X, y)
        if not report:
            report = self._estimate_noise_heuristic(X, y)

        # Add class distribution
        unique, counts = np.unique(y, return_counts=True)
        report["class_distribution"] = {str(u): int(c) for u, c in zip(unique, counts)}

        return report

    def generate_report(self, report: dict, save_path: Optional[str] = None) -> str:
        """Generate a markdown-formatted audit report."""
        lines = [
            "# Label Audit Report",
            "",
            f"**Method:** {report.get('method', 'unknown')}",
            f"**Overall Noise Rate:** {report.get('noise_rate', 0):.4f} "
            f"({report.get('noise_rate', 0) * 100:.2f}%)",
            f"**Suspected Mislabels:** {report.get('num_issues', 0)} / "
            f"{report.get('total_samples', 0)}",
            "",
            "## Per-Class Noise Rates",
            "",
            "| Class | Noise Rate |",
            "|-------|-----------|",
        ]
        for cls, rate in report.get("per_class_noise", {}).items():
            lines.append(f"| {cls} | {rate:.4f} ({rate * 100:.2f}%) |")

        lines.extend([
            "",
            "## Class Distribution",
            "",
            "| Class | Count |",
            "|-------|-------|",
        ])
        for cls, count in report.get("class_distribution", {}).items():
            lines.append(f"| {cls} | {count} |")

        content = "\n".join(lines)

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return content
