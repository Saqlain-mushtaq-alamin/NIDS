"""
tests/test_config.py — Configuration and utility module validation tests.
"""
from pathlib import Path
import torch
import numpy as np
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.utils.logging_setup import setup_logger

def test_default_config_loads():
    """Verify default.yaml loads with expected structure and dot-notation."""
    cfg = load_config("configs/default.yaml")
    assert cfg.project.name == "NIDS-Cross-Domain-Generalization"
    assert cfg.reproducibility.global_seed == 42
    assert cfg.data.primary_dataset == "lycos-ids-2017"
    assert len(cfg.data.cross_eval_datasets) >= 2

def test_baseline_config_loads():
    """Verify baseline.yaml loads correctly."""
    cfg = load_config("configs/baseline.yaml")
    assert cfg.project.name == "NIDS-Phase1-Baseline"
    assert cfg.data.primary_dataset == "cic-ids-2017"

def test_cross_dataset_config_loads():
    """Verify cross_dataset.yaml loads correctly."""
    cfg = load_config("configs/cross_dataset.yaml")
    assert len(cfg.reproducibility.seeds) >= 5
    assert len(cfg.data.train_datasets) >= 3

def test_seed_reproducibility():
    """Verify set_global_seed enforces exact random reproducibility."""
    set_global_seed(42)
    r1 = np.random.randn(10)
    t1 = torch.randn(10)
    
    set_global_seed(42)
    r2 = np.random.randn(10)
    t2 = torch.randn(10)
    
    assert np.allclose(r1, r2), "NumPy random outputs should match for identical seeds"
    assert torch.allclose(t1, t2), "PyTorch random outputs should match for identical seeds"

def test_logger_initialization(tmp_path):
    """Verify logger creates log output file."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("TestLogger", log_file=str(log_file))
    logger.info("Test log message")
    
    assert log_file.is_file()
    assert "Test log message" in log_file.read_text()
