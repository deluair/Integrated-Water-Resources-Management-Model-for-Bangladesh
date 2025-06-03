"""Utility Package for Bangladesh Water Management Model.

This package contains various helper functions and utility classes used across
the water management model, such as logging setup, date manipulation,
file I/O helpers, and common calculations.
"""

__version__ = "0.1.0"
__author__ = "AI Model for Bangladesh Water Management"

# Expose key utilities directly at the package level if desired
# from .logging_config import setup_logging
# from .file_helpers import load_json, save_json, ensure_dir_exists
# from .date_utils import format_date, parse_date_string

# Or, just let them be imported from their respective modules

import logging

# Basic logging configuration for the package
# More advanced configuration can be in a dedicated logging_config.py
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level, can be overridden by application

logger.info(f"Bangladesh Water Management Utilities Package v{__version__} loaded.")