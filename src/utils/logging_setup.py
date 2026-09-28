"""
src/utils/logging_setup.py — Structured logger initialization utility.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "NIDS", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logger with console output and optional file logging.
    
    Args:
        name: Logger module name.
        log_file: Optional log file output path.
        level: Logging level (e.g. logging.INFO).
        
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
