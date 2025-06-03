"""Logging Configuration for Bangladesh Water Management Model.

This module sets up centralized logging using the Loguru library,
allowing for configurable log levels, formats, and outputs (e.g., console, file).
"""

import sys
import os
from loguru import logger
from typing import Optional, Union, Dict, Any

# Default log directory (can be overridden by config)
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE_NAME = "bangladesh_water_model.log"

# Ensure the default log directory exists
if not os.path.exists(DEFAULT_LOG_DIR):
    try:
        os.makedirs(DEFAULT_LOG_DIR)
    except OSError as e:
        # Fallback if directory creation fails (e.g. permissions)
        sys.stderr.write(f"Warning: Could not create log directory {DEFAULT_LOG_DIR}: {e}\n")
        DEFAULT_LOG_DIR = "." # Log in current directory as fallback

DEFAULT_LOG_CONFIG = {
    "handlers": [
        {
            "sink": sys.stderr,
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "colorize": True,
            "enqueue": True, # For async safety
            "diagnose": False # Set to True for debugging Loguru itself
        },
        {
            "sink": os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILE_NAME),
            "level": "DEBUG",
            "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            "rotation": "10 MB", # Rotate log file when it reaches 10 MB
            "retention": "7 days", # Keep logs for 7 days
            "compression": "zip", # Compress rotated files
            "encoding": "utf-8",
            "enqueue": True,
            "serialize": False # Set to True to log in JSON format to file
        }
    ],
    "extra": {"user_id": None} # Example of global context data
}

# Variable to store the IDs of configured handlers to avoid duplication
_configured_handler_ids = set()

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configures Loguru logger based on provided or default settings.

    Args:
        config (Optional[Dict[str, Any]]): A dictionary with logging configuration.
                                            If None, uses DEFAULT_LOG_CONFIG.
    """
    global _configured_handler_ids
    
    # Remove default handler provided by Loguru if not already removed
    # This ensures we only have handlers we explicitly define.
    try:
        logger.remove(0)
    except ValueError:
        pass # No default handler to remove, or already removed

    log_config = config if config is not None else DEFAULT_LOG_CONFIG

    for handler_config in log_config.get("handlers", []):
        # Create a unique ID for the handler based on its sink and level
        # to prevent adding the same handler multiple times during re-configuration.
        sink_repr = str(handler_config.get("sink"))
        level_repr = str(handler_config.get("level"))
        handler_id = f"{sink_repr}_{level_repr}"

        if handler_id in _configured_handler_ids:
            # print(f"Skipping already configured handler: {handler_id}") # For debugging
            continue

        try:
            logger.add(**handler_config)
            _configured_handler_ids.add(handler_id)
        except Exception as e:
            # Fallback to basic stderr logging if advanced configuration fails
            sys.stderr.write(f"Failed to configure Loguru handler {handler_config.get('sink')}: {e}\n")
            # Attempt to add a very basic stderr handler if none are working
            if not _configured_handler_ids:
                try:
                    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
                    _configured_handler_ids.add(f"{str(sys.stderr)}_INFO") # Mark basic handler as added
                except Exception as basic_e:
                    sys.stderr.write(f"Failed to add basic Loguru stderr handler: {basic_e}\n")

    extra_context = log_config.get("extra", {})
    if extra_context:
        logger.configure(extra=extra_context)
    
    logger.info("Logging configured.")


def get_logger(name: Optional[str] = None) -> 'logger':
    """Returns a Loguru logger instance, optionally bound to a name.

    Args:
        name (Optional[str]): The name to bind to the logger (e.g., module name).

    Returns:
        A Loguru logger instance.
    """
    if not _configured_handler_ids: # Ensure setup_logging has been called at least once
        # print("Warning: Logging not explicitly configured. Applying default setup.") # For debugging
        setup_logging() # Apply default if not configured
        
    if name:
        return logger.bind(name=name)
    return logger

# Example of how to use it:
if __name__ == "__main__":
    # Default setup
    setup_logging()
    log = get_logger("my_module")

    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")

    # Example with custom configuration
    custom_config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "level": "DEBUG",
                "format": "{time:HH:mm:ss} | {level: <5} | {message}",
                "colorize": True
            }
        ]
    }
    # Re-setup logging (it will clear previous handlers and apply new ones)
    # Note: In a real app, setup_logging is usually called once at startup.
    # For this example, we clear the set to allow re-adding.
    _configured_handler_ids.clear() 
    setup_logging(custom_config)
    log_custom = get_logger("custom_module")
    log_custom.debug("Custom configured debug message.")
    log_custom.info("Custom configured info message.")

    # Test logging to file (check logs/bangladesh_water_model.log)
    _configured_handler_ids.clear()
    setup_logging() # Back to default to test file logging
    file_logger = get_logger("file_test")
    file_logger.info("This message should go to the log file and stderr.")
    file_logger.debug("This debug message should go only to the log file.")

    logger.info(f"Log file should be at: {os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILE_NAME)}")