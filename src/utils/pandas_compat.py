"""
src/utils/pandas_compat.py — pandas runtime compatibility shims.

pandas >= 3.0 enables a future string dtype that is backed by pyarrow.  On the
Windows builds used by this project that combination produced two problems:

1. ``ArrowStringArray`` cannot be sliced by ``sklearn.utils._safe_indexing``
   (``train_test_split``), raising
   ``TypeError: only integer scalar arrays can be converted to a scalar index``.
2. String column access could crash the interpreter outright when a corrupted
   ``pyarrow`` DLL was installed.

Importing this module pins pandas to the pure-python string representation for
the whole process.  It is imported by :mod:`src.data.loader` and
:mod:`src.data.preprocessor`, so it applies to both scripts and test runs.
"""
from __future__ import annotations

import os

_APPLIED = False


def apply_pandas_compat() -> None:
    """Pin pandas to pure-python string storage. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        _APPLIED = True
        return

    try:
        pd.options.mode.string_storage = "python"
    except (AttributeError, ValueError):  # pragma: no cover - very old pandas
        pass

    try:
        pd.options.future.infer_string = False
    except (AttributeError, ValueError):  # pragma: no cover - pandas < 3.0
        pass

    _APPLIED = True


def as_object_array(values) -> "object":
    """
    Return ``values`` as a plain NumPy ``object`` array of Python strings.

    Used wherever label arrays cross into scikit-learn, whose indexing helpers
    cannot handle pyarrow-backed string arrays.
    """
    import numpy as np

    if values is None:
        return None
    return np.asarray(values, dtype=object)


apply_pandas_compat()
