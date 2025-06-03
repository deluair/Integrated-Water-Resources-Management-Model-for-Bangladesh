"""File Helper Utilities for Bangladesh Water Management Model.

This module provides common utility functions for file operations,
including reading/writing JSON, CSV, ensuring directory existence, etc.
"""

import json
import os
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger # Using Loguru from logging_config

# Attempt to use the centralized logger, fallback if not configured
try:
    from .logging_config import get_logger
    log = get_logger(__name__) # Bind to current module name
except ImportError:
    # Fallback basic logger if logging_config is not available (e.g. during isolated testing)
    import logging
    log = logging.getLogger(__name__)
    if not log.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        log.setLevel(logging.INFO)

def ensure_dir_exists(dir_path: Union[str, Path]) -> None:
    """Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (Union[str, Path]): The path to the directory.
    """
    path = Path(dir_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            log.info(f"Created directory: {path}")
        except OSError as e:
            log.error(f"Error creating directory {path}: {e}")
            raise
    elif not path.is_dir():
        log.error(f"Path {path} exists but is not a directory.")
        raise NotADirectoryError(f"Path {path} exists but is not a directory.")

def load_json(file_path: Union[str, Path]) -> Optional[Dict[Any, Any]]:
    """Loads a JSON file into a Python dictionary.

    Args:
        file_path (Union[str, Path]): The path to the JSON file.

    Returns:
        Optional[Dict[Any, Any]]: The loaded dictionary, or None if an error occurs.
    """
    path = Path(file_path)
    if not path.exists():
        log.error(f"JSON file not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            log.debug(f"Successfully loaded JSON from: {path}")
            return data
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from {path}: {e}")
        return None
    except IOError as e:
        log.error(f"IOError reading JSON file {path}: {e}")
        return None

def save_json(data: Dict[Any, Any], file_path: Union[str, Path], indent: int = 4) -> bool:
    """Saves a Python dictionary to a JSON file.

    Args:
        data (Dict[Any, Any]): The dictionary to save.
        file_path (Union[str, Path]): The path to save the JSON file.
        indent (int): JSON indentation level for pretty printing.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    path = Path(file_path)
    try:
        ensure_dir_exists(path.parent) # Ensure parent directory exists
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            log.debug(f"Successfully saved JSON to: {path}")
            return True
    except IOError as e:
        log.error(f"IOError writing JSON file {path}: {e}")
        return False
    except TypeError as e:
        log.error(f"TypeError during JSON serialization for {path}: {e}")
        return False

def load_csv(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """Loads a CSV file into a Pandas DataFrame.

    Args:
        file_path (Union[str, Path]): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
        Optional[pd.DataFrame]: The loaded DataFrame, or None if an error occurs.
    """
    path = Path(file_path)
    if not path.exists():
        log.error(f"CSV file not found: {path}")
        return None
    try:
        df = pd.read_csv(path, **kwargs)
        log.debug(f"Successfully loaded CSV from: {path}")
        return df
    except pd.errors.EmptyDataError:
        log.warning(f"CSV file is empty: {path}")
        return pd.DataFrame() # Return empty DataFrame for empty files
    except Exception as e: # Catch other pandas or general errors
        log.error(f"Error reading CSV file {path}: {e}")
        return None

def save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, **kwargs) -> bool:
    """Saves a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (Union[str, Path]): The path to save the CSV file.
        index (bool): Whether to write DataFrame index as a column.
        **kwargs: Additional keyword arguments to pass to df.to_csv().

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    path = Path(file_path)
    try:
        ensure_dir_exists(path.parent)
        df.to_csv(path, index=index, **kwargs)
        log.debug(f"Successfully saved CSV to: {path}")
        return True
    except Exception as e:
        log.error(f"Error writing CSV file {path}: {e}")
        return False

def get_project_root() -> Path:
    """Attempts to find the project root directory.
    Assumes the project root contains a specific marker file or directory (e.g., '.git', 'pyproject.toml', 'requirements.txt').
    Adjust markers as needed for your project structure.
    """
    current_path = Path(__file__).resolve()
    # Look for common project root markers
    markers = ['.git', 'pyproject.toml', 'requirements.txt', 'setup.py', 'bangladesh_water_management.md'] 
    for _ in range(5): # Limit search depth to 5 parent directories
        if any((current_path / marker).exists() for marker in markers):
            return current_path
        if current_path.parent == current_path: # Reached filesystem root
            break
        current_path = current_path.parent
    
    # Fallback if no marker found (might not be accurate)
    log.warning("Could not reliably determine project root. Falling back to script's grandparent directory.")
    return Path(__file__).resolve().parent.parent 

# Example Usage
if __name__ == "__main__":
    # This assumes logging_config.py is in the same directory or path is configured
    try:
        from .logging_config import setup_logging
        setup_logging() # Initialize our fancy logger
    except ImportError:
        log.info("Standalone execution: using basic logger for file_helpers example.")

    log.info("File Helpers Example")

    # Test ensure_dir_exists
    test_dir = Path("temp_test_dir/subdir")
    ensure_dir_exists(test_dir)
    log.info(f"Directory {test_dir} ensured.")

    # Test JSON save and load
    sample_data = {"name": "Bangladesh Water Model", "version": "1.0", "parameters": [1, 2, 3]}
    json_file_path = test_dir / "sample.json"
    if save_json(sample_data, json_file_path):
        loaded_data = load_json(json_file_path)
        if loaded_data:
            log.info(f"Loaded JSON data: {loaded_data}")
            assert loaded_data == sample_data

    # Test CSV save and load
    sample_df = pd.DataFrame({
        'colA': [10, 20, 30],
        'colB': ['apple', 'banana', 'cherry']
    })
    csv_file_path = test_dir / "sample.csv"
    if save_csv(sample_df, csv_file_path):
        loaded_df = load_csv(csv_file_path)
        if loaded_df is not None:
            log.info(f"Loaded CSV data:\n{loaded_df}")
            pd.testing.assert_frame_equal(loaded_df, sample_df)
    
    # Test get_project_root
    project_root = get_project_root()
    log.info(f"Determined project root: {project_root}")

    # Clean up test directory (optional)
    import shutil
    try:
        shutil.rmtree("temp_test_dir")
        log.info("Cleaned up temp_test_dir.")
    except OSError as e:
        log.error(f"Error removing temp_test_dir: {e}")