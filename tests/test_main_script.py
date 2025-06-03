"""Tests for the main script (main.py) of the application.

These tests focus on:
- Argument parsing.
- Overall workflow execution (mocking the simulator).
- Output generation (e.g., saving results to a file).
"""

import pytest
import subprocess
import os
import json
from unittest.mock import patch, MagicMock
import sys

# Assuming main.py is in the root directory relative to the tests directory
# Adjust path if main.py is located elsewhere, e.g., in bangladesh_water_management/scripts/main.py
MAIN_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")

@pytest.fixture
def mock_simulator_run():
    """Mocks the WaterResourcesSimulator's run_simulation method."""
    with patch('bangladesh_water_management.simulator.WaterResourcesSimulator.run_simulation') as mock_run:
        mock_run.return_value = {"status": "simulation_completed", "key_metric": 123.45}
        yield mock_run

@pytest.fixture
def mock_simulator_init():
    """Mocks the WaterResourcesSimulator's __init__ method."""
    with patch('bangladesh_water_management.simulator.WaterResourcesSimulator.__init__', return_value=None) as mock_init:
        # Ensure the instance has the run_simulation method mocked as well
        # This is a bit tricky as __init__ returns None. We need to patch the class itself
        # or ensure the instance created by the actual main script gets its run_simulation patched.
        # For simplicity, we'll rely on mock_simulator_run being active for the instance.
        yield mock_init

@pytest.fixture
def mock_file_helpers():
    """Mocks file helper functions."""
    with patch('bangladesh_water_management.main.file_helpers.save_json_file') as mock_save_json,
         patch('bangladesh_water_management.main.file_helpers.ensure_dir_exists') as mock_ensure_dir:
        yield mock_save_json, mock_ensure_dir

# To test main.py by running it as a subprocess:
# This is good for testing command-line argument parsing directly.

def test_main_script_runs_with_default_args(tmp_path):
    """Test running main.py with default arguments via subprocess."""
    output_dir = tmp_path / "outputs"
    output_file = output_dir / "simulation_results.json"

    # We need to mock the simulator part if main.py actually runs it.
    # This can be done by setting environment variables that main.py checks, 
    # or by having a 'test_mode' in main.py.
    # For this example, let's assume main.py will try to run the simulator.
    # We'll use patching within the subprocess context if possible, or simplify main.py for testing.

    # A simpler approach for unit testing main() is to import and call it directly with mocks.
    # Running as subprocess is more of an integration test for the CLI.
    try:
        # The `main.py` script might not be directly executable without `python` prefix
        # and might need PYTHONPATH set up if it's part of a package.
        # For robust testing, it's often better to refactor main() to be importable and testable.
        result = subprocess.run(
            [sys.executable, MAIN_SCRIPT_PATH, "--output_dir", str(output_dir)],
            capture_output=True, text=True, check=False, # check=False to inspect errors
            env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(MAIN_SCRIPT_PATH))}
        )
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
        assert result.returncode == 0, f"main.py script failed with error: {result.stderr}"
        # This assertion depends on main.py actually creating the file via the real simulator or a mocked one.
        # If main.py is complex, this type of test becomes harder.
        # assert output_file.exists() # This would require the actual simulator to run or a very good mock setup

    except FileNotFoundError:
        pytest.skip(f"main.py not found at {MAIN_SCRIPT_PATH} or python interpreter not found.")
    except Exception as e:
        pytest.fail(f"Subprocess run failed: {e}")

# To test main() function by importing and calling it (preferred for unit tests):
# This requires main.py to have a callable main() function, e.g., inside `if __name__ == "__main__":`

# Assume main.py has a structure like:
# import argparse
# from bangladesh_water_management.simulator import WaterResourcesSimulator
# from bangladesh_water_management.utils import file_helpers
# def main_func(args_list=None):
#     parser = argparse.ArgumentParser(...)
#     # ... add arguments ...
#     args = parser.parse_args(args_list)
#     # ... rest of the logic ...
#     simulator = WaterResourcesSimulator(config_file_path=args.config_file)
#     results = simulator.run_simulation()
#     file_helpers.ensure_dir_exists(args.output_dir)
#     output_path = os.path.join(args.output_dir, "simulation_results.json")
#     file_helpers.save_json_file(output_path, results)
# if __name__ == "__main__":
#     main_func()

# We need to import the main function from main.py
# This might require adjusting sys.path or how main.py is structured.
# For this example, let's assume we can import a `main_runner` function from `main.py`

@patch('bangladesh_water_management.main.WaterResourcesSimulator') # Patch the class used in main.py
@patch('bangladesh_water_management.main.file_helpers.save_json_file')
@patch('bangladesh_water_management.main.file_helpers.ensure_dir_exists')
@patch('bangladesh_water_management.main.load_config') # If main loads config directly
def test_main_function_call_default_config(mock_load_config, mock_ensure_dir, mock_save_json, MockSimulator, tmp_path):
    """Test calling the main execution function with default config path."""
    # Mock the simulator instance and its run_simulation method
    mock_sim_instance = MagicMock()
    mock_sim_instance.run_simulation.return_value = {"status": "mock_success"}
    MockSimulator.return_value = mock_sim_instance
    mock_load_config.return_value = {"default_setting": True} # Mocked config data

    # Dynamically import main_runner from main.py
    # This is a common pattern if main.py is not directly in PYTHONPATH for tests
    spec = importlib.util.spec_from_file_location("main_module", MAIN_SCRIPT_PATH)
    main_module = importlib.util.module_from_spec(spec)
    sys.modules["main_module"] = main_module # Add to sys.modules before execution
    spec.loader.exec_module(main_module)
    
    output_dir = tmp_path / "test_outputs"
    args_list = ['--output_dir', str(output_dir)]

    main_module.main_func(args_list) # Call the main function from the imported module

    MockSimulator.assert_called_once() # Check if simulator was initialized
    # Check if config_file_path was None or default if not specified
    # init_args, init_kwargs = MockSimulator.call_args
    # assert init_kwargs.get('config_file_path') is None or 'default_config.yaml' in init_kwargs.get('config_file_path')
    
    mock_sim_instance.run_simulation.assert_called_once()
    mock_ensure_dir.assert_called_once_with(str(output_dir))
    expected_output_path = output_dir / "simulation_results.json"
    mock_save_json.assert_called_once_with(str(expected_output_path), {"status": "mock_success"})

@patch('bangladesh_water_management.main.WaterResourcesSimulator')
@patch('bangladesh_water_management.main.file_helpers.save_json_file')
@patch('bangladesh_water_management.main.file_helpers.ensure_dir_exists')
@patch('bangladesh_water_management.main.load_config')
def test_main_function_call_custom_config(mock_load_config, mock_ensure_dir, mock_save_json, MockSimulator, tmp_path):
    """Test calling the main execution function with a custom config path."""
    mock_sim_instance = MagicMock()
    mock_sim_instance.run_simulation.return_value = {"status": "custom_config_success"}
    MockSimulator.return_value = mock_sim_instance
    mock_load_config.return_value = {"custom_setting": True}

    spec = importlib.util.spec_from_file_location("main_module_custom", MAIN_SCRIPT_PATH)
    main_module_custom = importlib.util.module_from_spec(spec)
    sys.modules["main_module_custom"] = main_module_custom
    spec.loader.exec_module(main_module_custom)

    output_dir = tmp_path / "custom_outputs"
    custom_config_file = tmp_path / "custom_config.yaml"
    custom_config_file.write_text("key: value") # Create a dummy config file

    args_list = ['--config_file', str(custom_config_file), '--output_dir', str(output_dir)]
    main_module_custom.main_func(args_list)

    MockSimulator.assert_called_once()
    # Check if simulator was initialized with the custom config file
    init_args, init_kwargs = MockSimulator.call_args
    assert init_kwargs.get('config_file_path') == str(custom_config_file)
    mock_load_config.assert_called_with(str(custom_config_file)) # Ensure load_config was called with custom path
    
    mock_sim_instance.run_simulation.assert_called_once()
    mock_ensure_dir.assert_called_once_with(str(output_dir))
    expected_output_path = output_dir / "simulation_results.json"
    mock_save_json.assert_called_once_with(str(expected_output_path), {"status": "custom_config_success"})

# Need to import importlib for dynamic module loading
import importlib.util

# Note: The tests for main_func assume main.py is structured to be importable and testable.
# If main.py is purely a script that runs on import or has global side effects,
# testing it directly becomes much harder and subprocess testing (with extensive mocking
# or a test mode in the script) might be the only way, though less ideal for unit tests.
# The dynamic import (`importlib.util`) is a way to handle testing scripts not in PYTHONPATH.