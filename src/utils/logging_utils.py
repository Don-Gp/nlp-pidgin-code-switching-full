#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for the Nigerian Pidgin English Code-Switching detection project.
"""

import os
import logging
import sys
from datetime import datetime
import colorlog

# Default log directory
LOG_DIR = 'logs'


def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (int): Logging level
        console (bool): Whether to add console handler
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Return logger if handlers already exist
    if logger.handlers:
        return logger
    
    # Set format with colors for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Plain formatter for file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file specified
    if log_file:
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp_logger(name, logs_dir=LOG_DIR):
    """
    Get a logger that logs to a timestamped file.
    
    Args:
        name (str): Logger name
        logs_dir (str): Directory for log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if needed
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{name}_{timestamp}.log")
    
    return setup_logger(name, log_file=log_file)


def log_section(logger, section_name):
    """
    Log a section header for better organization.
    
    Args:
        logger (logging.Logger): Logger to use
        section_name (str): Section name
    """
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"{section_name}")
    logger.info(f"{separator}")


def log_dict(logger, data, title=None):
    """
    Log a dictionary in a readable format.
    
    Args:
        logger (logging.Logger): Logger to use
        data (dict): Dictionary to log
        title (str, optional): Title for the data
    """
    if title:
        logger.info(f"{title}:")
    
    for key, value in data.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Test the logging utilities
    logger = get_timestamp_logger("test")
    
    log_section(logger, "TEST SECTION")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    test_dict = {
        "key1": "value1",
        "key2": 42,
        "key3": {"nested": "value"}
    }
    
    log_dict(logger, test_dict, "Test Dictionary")