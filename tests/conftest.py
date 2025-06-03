"""Configuration and fixtures for pytest.

This file defines shared fixtures that can be used across multiple test files
in the 'tests' directory.

Fixtures defined here are automatically discovered by pytest.
"""

import pytest
import os
import tempfile
import shutil

from bangladesh_water_management.config import get_default_config, load_config, save_config

@pytest.fixture(scope="session")
def project_root_dir():
    """Returns the absolute path to the project's root directory."""
    # Assuming conftest.py is in the 'tests' directory, which is at the project root.
    # If tests is a subdirectory, adjust accordingly (e.g., os.path.dirname(os.path.dirname(__file__)))
    return os.path.dirname(os.path.dirname(__file__)) # Goes up two levels from tests/conftest.py to project root

@pytest.fixture(scope="session")
def default_config_data():
    """Provides the default configuration data for the application."""
    return get_default_config()

@pytest.fixture
def temp_test_dir():
    """Creates a temporary directory for tests that need to write files.
    
    This fixture has function scope, meaning a new temporary directory is created
    for each test function that uses it, and it's cleaned up afterwards.
    """
    with tempfile.TemporaryDirectory(prefix="bwm_test_") as tmpdir:
        yield tmpdir # The value yielded is what the test function receives
    # Directory is automatically cleaned up when the 'with' block exits

@pytest.fixture
def sample_config_file(temp_test_dir, default_config_data):
    """Creates a sample config file in a temporary directory and returns its path."""
    config_path = os.path.join(temp_test_dir, "sample_config.yaml")
    save_config(default_config_data, config_path)
    return config_path

@pytest.fixture
def outputs_test_dir(temp_test_dir):
    """Creates a dedicated 'outputs' subdirectory within the temp_test_dir."""
    outputs_dir = os.path.join(temp_test_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir

# Example of a more complex fixture that might set up a mock environment
# @pytest.fixture
# def mock_environment_variables(monkeypatch):
#     """Sets up mock environment variables for a test."""
#     monkeypatch.setenv("API_KEY", "test_api_key_123")
#     monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
#     yield
#     # Teardown (monkeypatch automatically undoes changes)

# You can add more shared fixtures here as needed, for example:
# - Mock database connections
# - Instantiated versions of common utility classes
# - Sample dataframes for testing data processing functions

print("conftest.py loaded: Shared fixtures are available.")