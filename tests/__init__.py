"""Test Suite for the Bangladesh Water Management Model.

This package contains all the unit tests, integration tests, and potentially
other test types for the various modules of the water management model.

To run tests, you would typically use a test runner like pytest from the project root:
  `pytest`
or
  `python -m pytest`

Make sure to install pytest and any other testing dependencies:
  `pip install pytest pytest-cov ...`
"""

__author__ = "AI Model for Bangladesh Water Management - Test Suite"
__version__ = "0.1.0"

# This __init__.py file makes the 'tests' directory a Python package.
# It can be left empty or used to define package-level fixtures or helpers
# for the test suite if needed.

import os
import sys

# You might want to add the project root to sys.path to ensure modules
# can be imported correctly by the test runner, especially if tests are run
# from within the tests directory itself or by certain IDE test runners.
# This is often handled by pytest's discovery mechanism or by how you structure your PYTHONPATH.

# Example: Add project root to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Assuming tests is a top-level directory
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

print(f"Bangladesh Water Management Model Test Suite v{__version__} initialized.")