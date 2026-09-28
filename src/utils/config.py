"""
src/utils/config.py — YAML configuration loader utility with dot notation.
"""
from pathlib import Path
from typing import Any, Dict
import yaml

class ConfigDict(dict):
    """Dictionary subclass supporting attribute dot notation."""
    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            else:
                self[key] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

def load_config(config_path: str = "configs/default.yaml") -> ConfigDict:
    """
    Load a YAML configuration file and return as a ConfigDict object.
    
    Args:
        config_path: Path to the YAML configuration file (relative or absolute).
        
    Returns:
        ConfigDict object enabling dot notation access.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(r"d:\canvas\NIDS") / path
        
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        
    return ConfigDict(data)
