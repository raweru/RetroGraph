"""
Logging module for RetroGraph.

This module provides logging functionality for the RetroGraph package.
"""

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "retrograph",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up and configure the logger.
    
    Args:
        name: Name of the logger
        level: Logging level
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )  # Detailed format for file
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger() 