# LexAgent v3.0 | logger.py
"""Centralized logging for LexAgent v3.0."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """Create a configured logger instance.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[LexAgent-V3] %(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
