# LexAgent v2.0 | Phase 1 | logger.py
"""Centralized logging for LexAgent v2.0."""

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
    
    # Avoid adding duplicate handlers if logger already exists
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler with formatted output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format: [LexAgent] 2024-01-01 12:00:00 | module_name | INFO | message
        formatter = logging.Formatter(
            "[LexAgent] %(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
