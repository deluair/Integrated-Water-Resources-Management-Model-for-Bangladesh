"""Common Calculations Utilities for Bangladesh Water Management Model.

This module provides helper functions for common mathematical, statistical,
and scientific calculations that may be reused across various modules of the
water management model.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Callable
from scipy import stats
from loguru import logger

# Attempt to use the centralized logger, fallback if not configured
try:
    from .logging_config import get_logger
    log = get_logger(__name__) # Bind to current module name
except ImportError:
    # Fallback basic logger if logging_config is not available
    import logging
    log = logging.getLogger(__name__)
    if not log.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        log.setLevel(logging.INFO)

# --- Unit Conversion Constants and Functions ---
MM_PER_INCH = 25.4
INCH_PER_MM = 1 / MM_PER_INCH
CUBIC_METERS_PER_ACRE_FOOT = 1233.48
ACRE_FOOT_PER_CUBIC_METERS = 1 / CUBIC_METERS_PER_ACRE_FOOT
LITERS_PER_CUBIC_METER = 1000

def mm_to_inches(mm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts millimeters to inches."""
    return mm * INCH_PER_MM

def inches_to_mm(inches: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts inches to millimeters."""
    return inches * MM_PER_INCH

def cubic_meters_to_acre_feet(m3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts cubic meters to acre-feet."""
    return m3 * ACRE_FOOT_PER_CUBIC_METERS

def acre_feet_to_cubic_meters(af: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts acre-feet to cubic meters."""
    return af * CUBIC_METERS_PER_ACRE_FOOT

def celsius_to_fahrenheit(celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

# --- Statistical Functions ---

def safe_mean(data: Union[List[float], np.ndarray, pd.Series], default: float = np.nan) -> float:
    """Calculates mean, handling empty or all-NaN data safely."""
    if isinstance(data, pd.Series):
        data = data.dropna().values
    elif isinstance(data, list):
        data = np.array([x for x in data if x is not None and not np.isnan(x)])
    
    if data.size == 0:
        return default
    return np.mean(data)

def safe_std_dev(data: Union[List[float], np.ndarray, pd.Series], default: float = np.nan) -> float:
    """Calculates standard deviation, handling empty or all-NaN data safely."""
    if isinstance(data, pd.Series):
        data = data.dropna().values
    elif isinstance(data, list):
        data = np.array([x for x in data if x is not None and not np.isnan(x)])

    if data.size < 2: # Std dev undefined for less than 2 points
        return default
    return np.std(data, ddof=1) # ddof=1 for sample standard deviation

def weighted_average(values: Union[List[float], np.ndarray, pd.Series],
                     weights: Union[List[float], np.ndarray, pd.Series]) -> float:
    """Calculates the weighted average.
    
    Args:
        values: The values to average.
        weights: The corresponding weights.
        
    Returns:
        The weighted average. Returns np.nan if inputs are invalid or sum of weights is zero.
    """
    try:
        values_arr = np.asarray(values)
        weights_arr = np.asarray(weights)
        
        if values_arr.shape != weights_arr.shape:
            log.error("Values and weights must have the same shape for weighted average.")
            return np.nan
        
        # Handle NaNs: remove corresponding value-weight pairs
        valid_indices = ~np.isnan(values_arr) & ~np.isnan(weights_arr)
        values_arr = values_arr[valid_indices]
        weights_arr = weights_arr[valid_indices]

        if values_arr.size == 0 or weights_arr.sum() == 0:
            return np.nan
            
        return np.average(values_arr, weights=weights_arr)
    except Exception as e:
        log.error(f"Error calculating weighted average: {e}")
        return np.nan

def normalize_data(data: Union[List[float], np.ndarray, pd.Series],
                   method: str = 'min-max') -> Union[np.ndarray, pd.Series]:
    """Normalizes data using specified method.

    Args:
        data: Input data (list, numpy array, or pandas Series).
        method: 'min-max' or 'z-score'.

    Returns:
        Normalized data as numpy array or pandas Series.
    """
    if isinstance(data, list):
        data_arr = np.array(data, dtype=float)
    elif isinstance(data, pd.Series):
        data_arr = data.astype(float).values # Keep as Series for return if input is Series
    else:
        data_arr = np.asarray(data, dtype=float)

    if np.all(np.isnan(data_arr)) or data_arr.size == 0:
        log.warning("Data for normalization is all NaN or empty.")
        return data_arr # Return as is

    if method == 'min-max':
        min_val = np.nanmin(data_arr)
        max_val = np.nanmax(data_arr)
        if max_val == min_val:
            # Avoid division by zero if all non-NaN values are the same
            normalized = np.zeros_like(data_arr)
            normalized[~np.isnan(data_arr)] = 0.5 # Or 0, or 1, depending on preference
        else:
            normalized = (data_arr - min_val) / (max_val - min_val)
    elif method == 'z-score':
        mean_val = np.nanmean(data_arr)
        std_val = np.nanstd(data_arr)
        if std_val == 0:
            # Avoid division by zero if all non-NaN values are the same
            normalized = np.zeros_like(data_arr)
        else:
            normalized = (data_arr - mean_val) / std_val
    else:
        log.error(f"Unsupported normalization method: {method}")
        raise ValueError(f"Unsupported normalization method: {method}")

    if isinstance(data, pd.Series):
        return pd.Series(normalized, index=data.index, name=data.name)
    return normalized

# --- Simple Hydrological Calculations (Examples) ---

def calculate_flow_volume(discharge_rate: float, # m^3/s
                          duration_seconds: float) -> float: # seconds
    """Calculates flow volume given discharge rate and duration.

    Returns:
        Volume in cubic meters (m^3).
    """
    if discharge_rate < 0 or duration_seconds < 0:
        log.warning("Discharge rate and duration must be non-negative.")
        return np.nan
    return discharge_rate * duration_seconds

def mannings_velocity(hydraulic_radius: float, # meters
                      channel_slope: float,    # unitless (e.g., m/m)
                      manning_n: float         # Manning's roughness coefficient
                      ) -> float:
    """Calculates average velocity in an open channel using Manning's equation.

    Args:
        hydraulic_radius (R): Cross-sectional area / wetted perimeter.
        channel_slope (S): Slope of the energy grade line (often approximated by bed slope).
        manning_n (n): Manning's roughness coefficient.

    Returns:
        Velocity in m/s.
    """
    if hydraulic_radius <= 0 or channel_slope <= 0 or manning_n <= 0:
        log.warning("Manning's equation inputs (R, S, n) must be positive.")
        return np.nan
    # Manning's equation: V = (1/n) * R^(2/3) * S^(1/2)
    return (1.0 / manning_n) * (hydraulic_radius ** (2.0/3.0)) * (channel_slope ** 0.5)

# --- Interpolation ---
def linear_interpolate(x_points: List[float], y_points: List[float], x_target: float) -> Optional[float]:
    """Performs linear interpolation.

    Args:
        x_points: List of x-coordinates of known points (must be sorted).
        y_points: List of y-coordinates of known points.
        x_target: The x-coordinate at which to interpolate.

    Returns:
        The interpolated y-value, or None if interpolation is not possible.
    """
    if len(x_points) != len(y_points) or len(x_points) < 2:
        log.error("Interpolation requires at least two points and equal length x, y lists.")
        return None
    
    # Ensure x_points are sorted (required for np.interp)
    if not np.all(np.diff(x_points) >= 0):
        log.warning("x_points for interpolation should be sorted. Attempting to sort.")
        sorted_indices = np.argsort(x_points)
        x_points = np.array(x_points)[sorted_indices]
        y_points = np.array(y_points)[sorted_indices]
    
    try:
        return float(np.interp(x_target, x_points, y_points))
    except Exception as e:
        log.error(f"Error during linear interpolation for x={x_target}: {e}")
        return None

# Example Usage
if __name__ == "__main__":
    try:
        from .logging_config import setup_logging
        setup_logging() # Initialize our fancy logger
    except ImportError:
        log.info("Standalone execution: using basic logger for calculations example.")

    log.info("Calculations Utilities Example")

    # Unit Conversions
    log.info(f"100 mm to inches: {mm_to_inches(100)}")
    log.info(f"70 F to Celsius: {fahrenheit_to_celsius(70)}")
    log.info(f"1000 m^3 to acre-feet: {cubic_meters_to_acre_feet(1000)}")

    # Statistics
    data_list = [1, 2, 3, 4, 5, np.nan, 6]
    log.info(f"Safe mean of {data_list}: {safe_mean(data_list)}")
    log.info(f"Safe std dev of {data_list}: {safe_std_dev(data_list)}")
    
    values = [10, 20, 30]
    weights = [1, 2, 1]
    log.info(f"Weighted average of {values} with weights {weights}: {weighted_average(values, weights)}")

    norm_data = [100, 200, 300, 400, 500]
    log.info(f"Min-Max normalized {norm_data}: {normalize_data(norm_data, 'min-max')}")
    log.info(f"Z-score normalized {norm_data}: {normalize_data(norm_data, 'z-score')}")

    # Hydrology
    vol = calculate_flow_volume(discharge_rate=10, duration_seconds=3600) # 10 m3/s for 1 hour
    log.info(f"Volume for 10 m3/s over 1 hour: {vol} m^3")

    velocity = mannings_velocity(hydraulic_radius=2.0, channel_slope=0.001, manning_n=0.025)
    log.info(f"Manning's velocity (R=2, S=0.001, n=0.025): {velocity:.2f} m/s")

    # Interpolation
    x = [0, 1, 2, 3, 4]
    y = [0, 2, 1, 3, 2]
    target_x = 2.5
    interpolated_y = linear_interpolate(x, y, target_x)
    log.info(f"Interpolated value at x={target_x} for y={y} (x={x}): {interpolated_y}")

    # Test with unsorted data
    x_unsorted = [3,1,4,0,2]
    y_unsorted = [3,2,2,0,1]
    interpolated_y_unsorted = linear_interpolate(x_unsorted, y_unsorted, target_x)
    log.info(f"Interpolated value at x={target_x} for y={y_unsorted} (x={x_unsorted}): {interpolated_y_unsorted}")