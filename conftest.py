"""
conftest.py — Global test configuration.
Sets environment variables before any imports to avoid pyarrow crashes on Windows.
"""
import os
import sys
from pathlib import Path

# Must be set before any pandas import to prevent pyarrow access violations
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Force pandas to not use pyarrow for string storage (prevents access violation)
os.environ["_PYARROWStringStorageType"] = ""

# Make the repository importable regardless of the invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# pandas >= 3.0 defaults to pyarrow-backed string columns (ArrowStringArray).
# sklearn's _safe_indexing cannot slice those arrays on Windows/pyarrow builds,
# which made train_test_split fail with
# "TypeError: only integer scalar arrays can be converted to a scalar index".
# Force the pure-python string representation for the whole test session.
import pandas as pd  # noqa: E402

pd.options.mode.string_storage = "python"
try:
    pd.options.future.infer_string = False
except AttributeError:  # pragma: no cover - older pandas
    pass
