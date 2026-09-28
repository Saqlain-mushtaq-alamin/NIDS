"""
tests/test_environment.py — Environment and module import validation tests.
"""
import sys
import importlib
import pytest
import torch

REQUIRED_CORE = [
    "torch", "sklearn", "xgboost", "lightgbm", "pandas", "numpy",
    "matplotlib", "seaborn", "pytest", "yaml", "joblib"
]

OPTIONAL_STACK = [
    "shap", "captum", "torch_geometric", "river", "scapy", "pyshark", "streamlit", "plotly", "mlflow", "cleanlab"
]

def test_python_version():
    """Verify Python version is 3.11+."""
    assert sys.version_info >= (3, 11), f"Need Python 3.11+, got {sys.version}"

def test_core_imports():
    """Verify all core machine learning dependencies can be imported."""
    failed = []
    for mod in REQUIRED_CORE:
        try:
            importlib.import_module(mod)
        except ImportError:
            failed.append(mod)
    assert not failed, f"Missing core modules: {failed}"

def test_torch_cuda():
    """Verify PyTorch detects CUDA GPU acceleration."""
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
    assert torch.cuda.device_count() >= 1, "No CUDA devices found"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX" in device_name or "NVIDIA" in device_name, f"Unexpected GPU device: {device_name}"

def test_optional_imports_warning():
    """Check status of optional stack and log warnings for unavailable packages."""
    missing = []
    for mod in OPTIONAL_STACK:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        pytest.skip(f"Optional modules currently not installed: {missing}")
