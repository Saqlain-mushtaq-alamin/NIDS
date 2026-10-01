"""
src/data/preprocessor.py — Preprocessing pipeline for NIDS data.

Handles infinities, NaN, feature selection, scaling, and stratified splits.

Phase 2 changes (cross-dataset generalization)
----------------------------------------------
* ``feature_mode`` selects the feature space:

  ``"full"``
      every numeric column except IP / port / protocol identity columns.
  ``"behavior_only"``
      the 16 canonical domain-invariant features of
      ``configs/feature_schema.yaml`` (PHASE_2 section 2.5).
  ``"behavior_extended"``
      the larger canonical behaviour set.

  The behaviour modes project the raw frame through
  :func:`src.data.feature_extractor.align_to_canonical`, so CICFlowMeter names,
  UNSW-NB15 names and home-lab names all resolve to the same semantic columns.
  A behaviour feature a dataset cannot supply becomes ``NaN`` and is reported in
  ``missing_features_`` rather than silently invented.

* :meth:`fit` learns every column decision from the training data only.
  :meth:`transform` no longer recomputes per-frame statistics (variance,
  medians, column subsets), so test frames cannot leak information back into the
  model.  Missing test columns are imputed with **training** medians.

* :meth:`split` fits the preprocessor on the *training rows only* and passes
  validation/test rows through :meth:`transform`, removing the train/test
  statistic leakage present in the Phase 1 version.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils.pandas_compat import apply_pandas_compat, as_object_array

apply_pandas_compat()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LABEL_COLUMN = "label"

FEATURE_MODES = ("full", "behavior_only", "behavior_extended")

BEHAVIOR_SCHEMA_KEYS = {
    "behavior_only": "behavior_core",
    "behavior_extended": "behavior_extended",
}

#: Denylist applied in every feature mode.  ``configs/feature_schema.yaml`` also
#: defines substring patterns, used to validate behaviour feature sets.
IDENTITY_COLUMNS: frozenset[str] = frozenset({
    "src_ip", "dst_ip", "src_port", "dst_port",
    "source_ip", "destination_ip", "source_port", "destination_port",
    "source ip", "destination ip", "source port", "destination port",
    "srcip", "dstip", "sport", "dsport", "protocol", "proto",
    "id", "uid", "flow id", "flow_id", "timestamp", "stime", "ltime",
})


def frame_rows(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Positional row selection that preserves dtypes and the original index."""
    return df.iloc[np.asarray(indices, dtype=int)]


class Preprocessor:
    """
    NIDS preprocessing pipeline with fit/transform semantics.

    Fit steps (training data only):

    1. Replace ``inf`` / ``-inf`` with ``NaN``
    2. Drop non-numeric and identity columns (IP / port / protocol)
    3. Project onto the selected feature mode
    4. Drop constant features (variance < ``variance_threshold``)
    5. Record per-feature medians for imputation
    6. Fit :class:`~sklearn.preprocessing.RobustScaler`

    Transform steps:

    1. Replace infinities with ``NaN``
    2. Align to the fitted feature list (missing columns -> ``NaN``)
    3. Impute with the **training** medians (0.0 as a last resort)
    4. Apply the fitted scaler
    """

    def __init__(self, drop_ip_port: bool = True, normalization: str = "robust",
                 feature_mode: str = "full", variance_threshold: float = 1e-8,
                 dataset: Optional[str] = None):
        """
        Args:
            drop_ip_port: Drop identity columns (IP / port / protocol).
            normalization: ``"robust"`` (default) or ``"none"``.
            feature_mode: ``"full"``, ``"behavior_only"`` or ``"behavior_extended"``.
            variance_threshold: Features whose training variance falls below this
                are treated as constant and dropped.
            dataset: Registry key used to resolve the canonical behaviour schema.
                Strongly recommended for the behaviour modes; without it the frame
                must already use canonical column names.
        """
        if feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"Unknown feature_mode '{feature_mode}'. Expected one of {FEATURE_MODES}."
            )
        if normalization not in ("robust", "none"):
            raise ValueError(
                f"Unknown normalization '{normalization}'. Expected 'robust' or 'none'."
            )

        self.drop_ip_port = drop_ip_port
        self.normalization = normalization
        self.feature_mode = feature_mode
        self.variance_threshold = variance_threshold
        self.dataset = dataset

        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: list[str] = []
        self.class_names_: list[str] = []
        self.medians_: Optional[np.ndarray] = None
        self.requested_features_: list[str] = []
        self.dropped_constant_: list[str] = []
        self.dropped_identity_: list[str] = []
        self.missing_features_: list[str] = []
        self._fitted = False

    # -- introspection ------------------------------------------------------
    def get_feature_names(self) -> list[str]:
        """Return the fitted feature names (empty before :meth:`fit`)."""
        return list(self.feature_names_)

    def get_class_names(self) -> list[str]:
        """Return the unified taxonomy labels observed during :meth:`fit`."""
        return list(self.class_names_)

    @property
    def n_features_(self) -> int:
        return len(self.feature_names_)

    @property
    def coverage_(self) -> float:
        """Fraction of requested features that carried usable signal at fit time."""
        if not self.requested_features_:
            return 0.0
        return 1.0 - len(self.missing_features_) / len(self.requested_features_)

    # -- helpers ------------------------------------------------------------
    def _labels(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if LABEL_COLUMN not in df.columns:
            return None
        return as_object_array(df[LABEL_COLUMN].astype(str).to_numpy())

    def _identity_columns(self, df: pd.DataFrame) -> list[str]:
        if not self.drop_ip_port:
            return []
        return [c for c in df.columns
                if c != LABEL_COLUMN and str(c).strip().lower() in IDENTITY_COLUMNS]

    def _numeric_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            c for c in df.columns
            if c != LABEL_COLUMN and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _behavior_schema(self) -> list[str]:
        """Resolve and validate the canonical behaviour feature list."""
        from src.data.feature_extractor import (
            assert_behavior_only,
            behavior_features,
            load_feature_schema,
        )

        names = behavior_features(BEHAVIOR_SCHEMA_KEYS[self.feature_mode])
        assert_behavior_only(names)

        declared = {str(n).strip().lower()
                    for n in load_feature_schema()["identity_columns"]}
        leaked = sorted(n for n in names if str(n).strip().lower() in declared)
        if leaked:
            raise ValueError(f"Behaviour schema declares identity columns: {leaked}")
        return names

    def _project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map ``df`` onto the model input space.

        ``full`` mode keeps the frame's own numeric columns (identity columns
        removed).  Behaviour modes project onto the canonical schema.
        """
        if self.feature_mode == "full":
            numeric = self._numeric_columns(df)
            keep = [c for c in numeric if c not in set(self.dropped_identity_)]
            return df[keep].copy()

        wanted = self._behavior_schema()

        if self.dataset is not None:
            from src.data.feature_extractor import align_to_canonical
            return align_to_canonical(df, self.dataset, wanted)

        lowered = {str(c).strip().lower(): c for c in df.columns}
        projected = pd.DataFrame(index=df.index)
        for name in wanted:
            source = lowered.get(name)
            if source is None:
                projected[name] = np.nan
            else:
                projected[name] = pd.to_numeric(df[source], errors="coerce")
        return projected

    def _finite(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric_columns(df)
        out = df.copy()
        if numeric:
            out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
        return out

    def _fillna(self, frame: pd.DataFrame) -> np.ndarray:
        """Impute with the fitted medians and return a float matrix."""
        if not self.feature_names_:
            return np.empty((len(frame), 0), dtype=np.float64)
        aligned = frame.reindex(columns=self.feature_names_)
        values = aligned.to_numpy(dtype=np.float64, na_value=np.nan)
        if self.medians_ is not None:
            bad = np.where(~np.isfinite(values))
            values[bad] = np.take(self.medians_, bad[1])
        return values

    # -- fit / transform ----------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit the preprocessor on **training** data only."""
        labels = self._labels(df)
        if labels is not None:
            self.class_names_ = sorted(pd.Series(labels).unique().tolist())

        self.dropped_identity_ = self._identity_columns(df)

        raw = self._finite(df.drop(columns=[LABEL_COLUMN], errors="ignore"))
        projected = self._project(raw)
        self.requested_features_ = list(projected.columns)

        self.missing_features_ = [
            name for name in self.requested_features_
            if not pd.api.types.is_numeric_dtype(projected[name])
            or projected[name].notna().sum() == 0
        ]

        variances = projected.var(numeric_only=True)
        constant = [
            name for name in projected.columns
            if name in variances.index and variances[name] < self.variance_threshold
        ]
        self.dropped_constant_ = constant
        working = projected.drop(columns=constant, errors="ignore")

        self.feature_names_ = working.columns.tolist()
        medians = working.median(numeric_only=True) if self.feature_names_ else pd.Series(dtype=float)
        self.medians_ = medians.reindex(self.feature_names_).fillna(0.0).to_numpy(dtype=np.float64)

        values = self._fillna(working)
        self.scaler_ = RobustScaler().fit(values) if self.normalization == "robust" else None
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using the fitted preprocessor.

        No statistic is recomputed from ``df``: only training-derived decisions are
        applied, so cross-dataset test sets cannot leak information backwards.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        labels = self._labels(df)
        raw = self._finite(df.drop(columns=[LABEL_COLUMN], errors="ignore"))
        projected = self._project(raw)

        values = self._fillna(projected)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)

        return values, labels

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    # -- splitting ----------------------------------------------------------
    def _stratify_or_none(self, labels: np.ndarray, n_outputs: int) -> Optional[np.ndarray]:
        """Return ``labels`` when stratification is feasible, else ``None``."""
        counts = pd.Series(labels).value_counts()
        if counts.empty:
            return None
        if n_outputs < len(counts) or int(counts.min()) < 2:
            return None
        return labels

    def split(self, df: pd.DataFrame, test_ratio: float = 0.2,
              val_ratio: float = 0.1, seed: int = 42,
              shuffle: bool = True) -> dict:
        """
        Stratified train/val/test split, fitted on the training rows only.

        Returns:
            dict with keys ``'train'``, ``'val'``, ``'test'``, each holding
            ``{'X': np.ndarray, 'y': np.ndarray}``.
        """
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"'{LABEL_COLUMN}' column required to split")

        labels = as_object_array(df[LABEL_COLUMN].astype(str).to_numpy())
        rows = np.arange(len(df))
        n_test = int(round(len(df) * test_ratio))
        n_val = int(round(len(df) * val_ratio))

        temp_idx, test_idx = train_test_split(
            rows, test_size=test_ratio, random_state=seed, shuffle=shuffle,
            stratify=self._stratify_or_none(labels, n_test),
        )

        if n_val <= 0:
            train_idx, val_idx = temp_idx, np.array([], dtype=int)
        else:
            temp_labels = labels[temp_idx]
            train_idx, val_idx = train_test_split(
                temp_idx, test_size=val_ratio / (1.0 - test_ratio),
                random_state=seed, shuffle=shuffle,
                stratify=self._stratify_or_none(temp_labels, n_val),
            )

        train_frame = frame_rows(df, train_idx)
        self.fit(train_frame)

        splits = {
            "train": self._materialise(train_frame),
            "test": self._materialise(frame_rows(df, test_idx)),
        }
        splits["val"] = (
            self._materialise(frame_rows(df, val_idx))
            if n_val > 0 else
            {"X": np.empty((0, self.n_features_)), "y": np.array([], dtype=object)}
        )
        return splits

    def _materialise(self, frame: pd.DataFrame) -> dict:
        X, y = self.transform(frame)
        return {"X": X, "y": y}

    def fit_and_split(self, df: pd.DataFrame, test_ratio: float = 0.2,
                      val_ratio: float = 0.1, seed: int = 42) -> dict:
        """Fit on the training split and return stratified splits."""
        return self.split(df, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

    # -- persistence --------------------------------------------------------
    def save(self, path: Optional[str] = None) -> Path:
        """Serialise the fitted preprocessor to disk."""
        if path is None:
            path = _PROJECT_ROOT / "experiments" / "preprocessor.joblib"
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        return save_path

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Load a serialised preprocessor from disk."""
        return joblib.load(path)
