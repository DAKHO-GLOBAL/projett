"""
Logging configuration module.

This module provides functionality to set up and configure logging for the
trading system, ensuring consistent log formatting and handling across all
components.
"""

import os
import logging
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        log_level: Logging level (defaults to INFO)
        log_file: File to write logs to (optional)
        json_format: Whether to use JSON format for logs
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, assume it's configured
    if logger.handlers:
        return logger
    
    # Get log level from environment or parameter or default to INFO
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file is not None:
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    # Create formatter
    if json_format:
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add formatter to handlers and add handlers to logger
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger