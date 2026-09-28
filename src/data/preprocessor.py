"""
src/data/preprocessor.py — Preprocessing pipeline for NIDS data.
Handles infinities, NaN, feature selection, scaling, and stratified splits.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Preprocessor:
    """
    NIDS preprocessing pipeline with fit/transform pattern.

    Steps:
    1. Replace inf/-inf with NaN
    2. Drop constant/near-constant features (variance < 1e-8)
    3. Drop IP/port/protocol features (optional, default True)
    4. Drop non-numeric columns
    5. Impute NaN with median
    6. Apply RobustScaler normalization
    7. Stratified train/val/test split
    """

    def __init__(self, drop_ip_port: bool = True, normalization: str = "robust"):
        self.drop_ip_port = drop_ip_port
        self.normalization = normalization
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: list[str] = []
        self.class_names_: list[str] = []
        self._fitted = False

    def _replace_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf and -inf with NaN."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def _drop_constant_features(self, df: pd.DataFrame, threshold: float = 1e-8) -> pd.DataFrame:
        """Drop features with variance below threshold."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        constant_cols = variances[variances < threshold].index.tolist()
        df.drop(columns=constant_cols, inplace=True, errors="ignore")
        return df

    def _drop_ip_port_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop IP address, port, and protocol features."""
        forbidden = {
            "src_ip", "dst_ip", "src_port", "dst_port",
            "source_ip", "destination_ip", "source_port", "destination_port",
            "source ip", "destination ip", "source port", "destination port",
            "srcip", "dstip", "sport", "dsport", "protocol", "proto",
        }
        drop_cols = [c for c in df.columns if c.lower() in forbidden]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        return df

    def _drop_non_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric columns (plus label)."""
        label_col = "label"
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in df.columns:
            non_numeric = [c for c in df.columns if c not in numeric_cols and c != label_col]
            df.drop(columns=non_numeric, inplace=True, errors="ignore")
        else:
            non_numeric = [c for c in df.columns if c not in numeric_cols]
            df.drop(columns=non_numeric, inplace=True, errors="ignore")
        return df

    def _impute_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute NaN values with column median."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit the preprocessor on the training data."""
        df = df.copy()
        df = self._replace_infinities(df)
        df = self._drop_constant_features(df)

        if self.drop_ip_port:
            df = self._drop_ip_port_features(df)

        df = self._drop_non_numeric(df)

        if "label" in df.columns:
            self.class_names_ = sorted(df["label"].unique().tolist())
            df = df.drop(columns=["label"])

        df = self._impute_nan(df)
        self.feature_names_ = df.columns.tolist()

        if self.normalization == "robust":
            self.scaler_ = RobustScaler()
            self.scaler_.fit(df.values)
        else:
            self.scaler_ = None

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using the fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = df.copy()
        df = self._replace_infinities(df)
        df = self._drop_constant_features(df)

        if self.drop_ip_port:
            df = self._drop_ip_port_features(df)

        df = self._drop_non_numeric(df)

        y = None
        if "label" in df.columns:
            y = df["label"].values
            df = df.drop(columns=["label"])

        df = self._impute_nan(df)

        # Align features
        missing = set(self.feature_names_) - set(df.columns)
        for col in missing:
            df[col] = 0
        extra = set(df.columns) - set(self.feature_names_)
        df.drop(columns=list(extra), inplace=True, errors="ignore")
        df = df[self.feature_names_]

        X = df.values.astype(np.float64)

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def split(self, df: pd.DataFrame, test_ratio: float = 0.2,
              val_ratio: float = 0.1, seed: int = 42) -> dict:
        """
        Perform stratified train/val/test split.

        Returns:
            dict with keys 'train', 'val', 'test', each containing
            {'X': np.ndarray, 'y': np.ndarray}.
        """
        X, y = self.fit_transform(df)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed, stratify=y
        )

        # Second split: train vs val
        val_fraction = val_ratio / (1 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=seed, stratify=y_temp
        )

        return {
            "train": {"X": X_train, "y": y_train},
            "val": {"X": X_val, "y": y_val},
            "test": {"X": X_test, "y": y_test},
        }

    def fit_and_split(self, df: pd.DataFrame, test_ratio: float = 0.2,
                      val_ratio: float = 0.1, seed: int = 42) -> dict:
        """Fit preprocessor and return stratified splits."""
        return self.split(df, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

    def save(self, path: Optional[str] = None) -> Path:
        """Serialize the fitted preprocessor to disk."""
        if path is None:
            path = _PROJECT_ROOT / "experiments" / "preprocessor.joblib"
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        return save_path

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Load a serialized preprocessor from disk."""
        return joblib.load(path)
