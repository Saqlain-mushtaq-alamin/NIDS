"""
conftest.py — Global test configuration.
Sets environment variables before any imports to avoid pyarrow crashes on Windows.
"""
import os
import sys

# Must be set before any pandas import to prevent pyarrow access violations
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Force pandas to not use pyarrow for string storage (prevents access violation)
os.environ["_PYARROWStringStorageType"] = ""
