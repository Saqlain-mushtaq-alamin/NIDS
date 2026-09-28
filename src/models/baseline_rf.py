"""
src/models/baseline_rf.py — Random Forest baseline classifier for NIDS.
"""
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class RFBaseline:
    """
    Random Forest baseline classifier wrapping sklearn API.
    Supports automatic label encoding.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 20,
                 random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model_ = RandomForestClassifier(**params)
        self.label_encoder_ = LabelEncoder()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFBaseline":
        """Fit the model on training data."""
        y_encoded = self.label_encoder_.fit_transform(y)
        self.model_.fit(X, y_encoded)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        y_encoded = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> dict:
        """Get feature importance scores."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted.")
        importance = self.model_.feature_importances_
        return {
            "importances": importance.tolist(),
            "gain_importance": dict(zip(
                range(len(importance)),
                importance.tolist()
            )),
        }
