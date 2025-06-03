"""Tests for utility functions in the utils directory.

This suite covers tests for:
- file_helpers.py
- date_utils.py
- calculations.py
- logging_config.py (basic import and setup check)
"""

import pytest
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import tempfile # For creating temporary files/dirs for testing file_helpers

from bangladesh_water_management.utils import file_helpers, date_utils, calculations, logging_config

# --- Tests for logging_config.py ---

def test_logging_config_setup():
    """Test that the logger can be imported and configured without errors."""
    try:
        logger = logging_config.get_logger("test_utils_logger")
        assert logger is not None
        logger.info("Test log message from test_utils.py") # Basic check
    except Exception as e:
        pytest.fail(f"Logger setup failed: {e}")

# --- Tests for file_helpers.py ---

@pytest.fixture
def temp_dir_fixture():
    """Create a temporary directory for file operation tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_ensure_dir_exists(temp_dir_fixture):
    """Test ensure_dir_exists function."""
    new_dir_path = os.path.join(temp_dir_fixture, "new_test_dir")
    assert not os.path.exists(new_dir_path)
    file_helpers.ensure_dir_exists(new_dir_path)
    assert os.path.exists(new_dir_path)
    assert os.path.isdir(new_dir_path)
    # Test with existing directory (should not raise error)
    file_helpers.ensure_dir_exists(new_dir_path)

def test_load_save_json(temp_dir_fixture):
    """Test load_json_file and save_json_file functions."""
    json_file_path = os.path.join(temp_dir_fixture, "test.json")
    test_data = {"key": "value", "number": 123, "nested": {"a": 1}}
    
    # Save JSON
    file_helpers.save_json_file(json_file_path, test_data)
    assert os.path.exists(json_file_path)
    
    # Load JSON
    loaded_data = file_helpers.load_json_file(json_file_path)
    assert loaded_data == test_data

    # Test load_json_file with non-existent file
    with pytest.raises(FileNotFoundError):
        file_helpers.load_json_file(os.path.join(temp_dir_fixture, "non_existent.json"))

def test_load_save_csv(temp_dir_fixture):
    """Test load_csv_file and save_csv_file functions."""
    csv_file_path = os.path.join(temp_dir_fixture, "test.csv")
    test_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    # Save CSV
    file_helpers.save_csv_file(csv_file_path, test_df)
    assert os.path.exists(csv_file_path)
    
    # Load CSV
    loaded_df = file_helpers.load_csv_file(csv_file_path)
    pd.testing.assert_frame_equal(loaded_df, test_df)

    # Test load_csv_file with non-existent file
    with pytest.raises(FileNotFoundError):
        file_helpers.load_csv_file(os.path.join(temp_dir_fixture, "non_existent.csv"))

# --- Tests for date_utils.py ---

def test_parse_date_string():
    """Test parse_date_string function."""
    assert date_utils.parse_date_string("2023-10-26") == datetime(2023, 10, 26)
    assert date_utils.parse_date_string("2023/10/26", "%Y/%m/%d") == datetime(2023, 10, 26)
    assert date_utils.parse_date_string("26-Oct-2023", "%d-%b-%Y") == datetime(2023, 10, 26)
    with pytest.raises(ValueError):
        date_utils.parse_date_string("invalid-date-string")

def test_format_date_object():
    """Test format_date_object function."""
    dt_obj = datetime(2023, 10, 26, 14, 30)
    assert date_utils.format_date_object(dt_obj) == "2023-10-26"
    assert date_utils.format_date_object(dt_obj, "%Y/%m/%d %H:%M") == "2023/10/26 14:30"

def test_get_current_datetime_utc():
    """Test get_current_datetime_utc function."""
    now_utc = date_utils.get_current_datetime_utc()
    assert now_utc.tzinfo == timezone.utc
    # Check if it's close to the actual current time
    assert abs((datetime.now(timezone.utc) - now_utc).total_seconds()) < 1

def test_add_time_delta():
    """Test add_time_delta function."""
    start_date = datetime(2023, 1, 1)
    assert date_utils.add_time_delta(start_date, days=5) == datetime(2023, 1, 6)
    assert date_utils.add_time_delta(start_date, weeks=-1) == datetime(2022, 12, 25)
    assert date_utils.add_time_delta(start_date, hours=24) == datetime(2023, 1, 2)

def test_generate_datetime_index():
    """Test generate_datetime_index function."""
    start_str = "2023-01-01"
    end_str = "2023-01-05"
    dt_index = date_utils.generate_datetime_index(start_str, end_str, freq='D')
    assert len(dt_index) == 5
    assert dt_index[0] == pd.Timestamp("2023-01-01")
    assert dt_index[-1] == pd.Timestamp("2023-01-05")

    dt_index_hourly = date_utils.generate_datetime_index("2023-01-01", "2023-01-01T02:00:00", freq='H')
    assert len(dt_index_hourly) == 3 # 00:00, 01:00, 02:00

# --- Tests for calculations.py ---

# Unit Conversions
def test_mm_to_inches():
    assert calculations.mm_to_inches(25.4) == pytest.approx(1.0)
    assert calculations.mm_to_inches(0) == 0.0

def test_cubic_meters_to_acre_feet():
    assert calculations.cubic_meters_to_acre_feet(1233.48) == pytest.approx(1.0)
    assert calculations.cubic_meters_to_acre_feet(0) == 0.0

def test_celsius_to_fahrenheit():
    assert calculations.celsius_to_fahrenheit(0) == 32.0
    assert calculations.celsius_to_fahrenheit(100) == 212.0
    assert calculations.celsius_to_fahrenheit(-40) == -40.0

# Statistical Functions
def test_safe_mean():
    assert calculations.safe_mean([1, 2, 3, 4, 5]) == 3.0
    assert np.isnan(calculations.safe_mean([])) # Default behavior for empty list
    assert calculations.safe_mean([1, 2, np.nan, 4, 5], ignore_nan=True) == 3.0
    assert np.isnan(calculations.safe_mean([np.nan, np.nan], ignore_nan=True))
    assert calculations.safe_mean([], default_value=0) == 0

def test_safe_std_dev():
    assert calculations.safe_std_dev([1, 1, 1, 1]) == 0.0
    assert calculations.safe_std_dev([1,2,3,4,5]) == pytest.approx(np.std([1,2,3,4,5]))
    assert np.isnan(calculations.safe_std_dev([]))
    assert calculations.safe_std_dev([], default_value=0) == 0

def test_weighted_average():
    values = pd.Series([10, 20, 30])
    weights = pd.Series([1, 1, 3]) # (10*1 + 20*1 + 30*3) / (1+1+3) = (10+20+90)/5 = 120/5 = 24
    assert calculations.weighted_average(values, weights) == 24.0
    with pytest.raises(ValueError): # Mismatched lengths
        calculations.weighted_average(pd.Series([1,2]), pd.Series([1,1,1]))
    assert np.isnan(calculations.weighted_average(pd.Series([]), pd.Series([]))) # Empty series

def test_normalize_data():
    data = pd.Series([10, 20, 30, 40, 50])
    normalized = calculations.normalize_data(data)
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert normalized.iloc[2] == 0.5 # (30-10)/(50-10) = 20/40 = 0.5

    data_const = pd.Series([5, 5, 5])
    normalized_const = calculations.normalize_data(data_const)
    assert all(normalized_const == 0) # or 0.5 depending on implementation for constant data

# Hydrological Calculations
def test_calculate_flow_volume():
    # 10 m3/s for 1 hour (3600 seconds)
    assert calculations.calculate_flow_volume(flow_rate_m3s=10, duration_seconds=3600) == 36000
    # 5 m3/s for 1 day (24*3600 seconds)
    assert calculations.calculate_flow_volume(flow_rate_m3s=5, duration_seconds=24*3600) == 5 * 24 * 3600

def test_calculate_manning_velocity():
    # Example values, ensure it runs and produces a float
    velocity = calculations.calculate_manning_velocity(n=0.03, R=1.5, S=0.001)
    assert isinstance(velocity, float)
    assert velocity > 0
    with pytest.raises(ValueError): # n=0
        calculations.calculate_manning_velocity(n=0, R=1, S=0.001)

# Interpolation
def test_linear_interpolate():
    x_points = np.array([0, 10])
    y_points = np.array([0, 100])
    assert calculations.linear_interpolate(5, x_points, y_points) == 50.0
    assert calculations.linear_interpolate(0, x_points, y_points) == 0.0
    assert calculations.linear_interpolate(10, x_points, y_points) == 100.0
    # Extrapolation (if allowed by numpy.interp, default is to clip or use fill_value)
    # By default, np.interp extrapolates by using the boundary values.
    assert calculations.linear_interpolate(-5, x_points, y_points) == 0.0 
    assert calculations.linear_interpolate(15, x_points, y_points) == 100.0

    x_points_multi = np.array([0, 5, 10, 15])
    y_points_multi = np.array([0, 50, 0, 50])
    assert calculations.linear_interpolate(2.5, x_points_multi, y_points_multi) == 25.0
    assert calculations.linear_interpolate(7.5, x_points_multi, y_points_multi) == 25.0
    assert calculations.linear_interpolate(12.5, x_points_multi, y_points_multi) == 25.0