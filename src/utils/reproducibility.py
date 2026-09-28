"""
src/utils/reproducibility.py — Global random seed and deterministic execution utilities.
"""
import random
import os
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set all random seeds for Python, NumPy, PyTorch, and CUDA to ensure reproducible results.
    
    Args:
        seed: Integer random seed (default: 42).
        deterministic_cudnn: Whether to enforce deterministic CUDNN algorithms.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
