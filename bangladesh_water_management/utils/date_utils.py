"""Date and Time Utilities for Bangladesh Water Management Model.

This module provides helper functions for common date and time operations,
including parsing, formatting, and calculations.
"""

from datetime import datetime, timedelta, date
from typing import Optional, Union, List
import pandas as pd
from loguru import logger # Using Loguru from logging_config

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

DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
\COMMON_DATE_FORMATS = [
    DEFAULT_DATE_FORMAT,
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%Y%m%d",
    DEFAULT_DATETIME_FORMAT,
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f", # With microseconds
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
]

def parse_date_string(date_str: Optional[str],
                      formats: Optional[List[str]] = None,
                      return_type: str = 'datetime') -> Optional[Union[datetime, date]]:
    """Parses a date string into a datetime or date object using a list of possible formats.

    Args:
        date_str (Optional[str]): The date string to parse.
        formats (Optional[List[str]]): A list of strptime format strings to try.
                                     If None, uses COMMON_DATE_FORMATS.
        return_type (str): 'datetime' or 'date'. Determines the type of the returned object.

    Returns:
        Optional[Union[datetime, date]]: The parsed date/datetime object, or None if parsing fails.
    """
    if not date_str:
        return None

    if isinstance(date_str, (datetime, date)): # Already a date/datetime object
        return date_str if return_type == 'datetime' or isinstance(date_str, datetime) else date_str.date()
        
    to_try_formats = formats if formats else COMMON_DATE_FORMATS

    for fmt in to_try_formats:
        try:
            dt_obj = datetime.strptime(str(date_str).strip(), fmt)
            log.debug(f"Parsed date string '{date_str}' using format '{fmt}' to {dt_obj}")
            return dt_obj if return_type == 'datetime' else dt_obj.date()
        except (ValueError, TypeError):
            continue
    
    # Try pandas to_datetime as a last resort, it's quite flexible
    try:
        pd_dt = pd.to_datetime(date_str)
        if pd.isna(pd_dt):
            log.warning(f"Could not parse date string '{date_str}' with any known format or pandas.")
            return None
        dt_obj = pd_dt.to_pydatetime()
        log.debug(f"Parsed date string '{date_str}' using pandas to {dt_obj}")
        return dt_obj if return_type == 'datetime' else dt_obj.date()
    except Exception as e:
        log.warning(f"Could not parse date string '{date_str}' with any known format or pandas: {e}")
        return None

def format_date(
    date_obj: Optional[Union[datetime, date]], 
    fmt: str = DEFAULT_DATE_FORMAT
) -> Optional[str]:
    """Formats a datetime or date object into a string.

    Args:
        date_obj (Optional[Union[datetime, date]]): The date/datetime object to format.
        fmt (str): The strftime format string.

    Returns:
        Optional[str]: The formatted date string, or None if input is None.
    """
    if date_obj is None:
        return None
    try:
        return date_obj.strftime(fmt)
    except AttributeError:
        log.warning(f"Input {date_obj} is not a valid date/datetime object for formatting.")
        return str(date_obj) # Fallback to string representation
    except Exception as e:
        log.error(f"Error formatting date {date_obj} with format {fmt}: {e}")
        return None

def get_current_year() -> int:
    """Returns the current year as an integer."""
    return datetime.now().year

def get_current_datetime_str(fmt: str = DEFAULT_DATETIME_FORMAT) -> str:
    """Returns the current datetime as a formatted string."""
    return datetime.now().strftime(fmt)

def add_time_delta(start_date: Union[datetime, date],
                   days: int = 0,
                   weeks: int = 0,
                   months: int = 0, # Approximate, use relativedelta for precision
                   years: int = 0   # Approximate, use relativedelta for precision
                   ) -> Union[datetime, date]:
    """Adds a time delta to a given date/datetime.
    For precise month/year arithmetic, consider using dateutil.relativedelta.
    This implementation for months/years is a simplification.

    Args:
        start_date (Union[datetime, date]): The starting date or datetime.
        days (int): Number of days to add.
        weeks (int): Number of weeks to add.
        months (int): Number of months to add (approximate).
        years (int): Number of years to add (approximate).

    Returns:
        Union[datetime, date]: The new date/datetime after adding the delta.
    """
    if not isinstance(start_date, (datetime, date)):
        parsed_date = parse_date_string(str(start_date))
        if not parsed_date:
            raise ValueError("Invalid start_date provided. Must be datetime, date, or parsable string.")
        start_date = parsed_date

    # Basic year/month addition (can be problematic around month ends)
    # For more robust month/year arithmetic, `dateutil.relativedelta` is recommended.
    new_year = start_date.year + years
    new_month = start_date.month + months
    
    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -=1

    # Handle day if it exceeds the number of days in the new month
    # e.g. Jan 31 + 1 month should not be Feb 31
    import calendar
    max_day_in_new_month = calendar.monthrange(new_year, new_month)[1]
    new_day = min(start_date.day, max_day_in_new_month)

    if isinstance(start_date, datetime):
        res_date = start_date.replace(year=new_year, month=new_month, day=new_day)
    else: # date object
        res_date = date(year=new_year, month=new_month, day=new_day)

    # Add days and weeks
    res_date += timedelta(days=days, weeks=weeks)
    return res_date

def generate_date_range(start_date_str: str,
                        end_date_str: str,
                        freq: str = 'D', # 'D' for day, 'M' for month end, 'MS' for month start
                        date_format: Optional[str] = None
                        ) -> Optional[pd.DatetimeIndex]:
    """Generates a Pandas DatetimeIndex between two dates.

    Args:
        start_date_str (str): The start date string.
        end_date_str (str): The end date string.
        freq (str): Frequency string (e.g., 'D', 'W', 'M', 'MS', 'H').
        date_format (Optional[str]): The format of the input date strings if not standard.

    Returns:
        Optional[pd.DatetimeIndex]: A DatetimeIndex, or None on error.
    """
    try:
        start = pd.to_datetime(start_date_str, format=date_format)
        end = pd.to_datetime(end_date_str, format=date_format)
        if pd.isna(start) or pd.isna(end):
            log.error(f"Could not parse start ('{start_date_str}') or end ('{end_date_str}') date for range generation.")
            return None
        return pd.date_range(start=start, end=end, freq=freq)
    except Exception as e:
        log.error(f"Error generating date range from '{start_date_str}' to '{end_date_str}': {e}")
        return None

# Example Usage
if __name__ == "__main__":
    try:
        from .logging_config import setup_logging
        setup_logging() # Initialize our fancy logger
    except ImportError:
        log.info("Standalone execution: using basic logger for date_utils example.")

    log.info("Date Utils Example")

    # Test parse_date_string
    date_strings = ["2023-10-26", "27-11-2024", "12/30/2025", "20261231", "2023-01-15T14:30:00", None, datetime.now()]
    for ds in date_strings:
        parsed_dt = parse_date_string(ds)
        parsed_d = parse_date_string(ds, return_type='date')
        log.info(f"Original: '{ds}' -> Parsed datetime: {parsed_dt}, Parsed date: {parsed_d}")

    # Test format_date
    now = datetime.now()
    log.info(f"Formatted default: {format_date(now)}")
    log.info(f"Formatted custom: {format_date(now, '%A, %B %d, %Y %I:%M %p')}")
    log.info(f"Formatted date part: {format_date(now.date(), '%d/%m/%y')}")

    # Test get_current_year and get_current_datetime_str
    log.info(f"Current year: {get_current_year()}")
    log.info(f"Current datetime string: {get_current_datetime_str()}")

    # Test add_time_delta
    start = datetime(2023, 1, 31)
    log.info(f"Start date: {start}")
    log.info(f"Add 10 days: {add_time_delta(start, days=10)}")
    log.info(f"Add 2 weeks: {add_time_delta(start, weeks=2)}")
    log.info(f"Add 1 month (approx): {add_time_delta(start, months=1)}") # Expects Feb 28, 2023
    log.info(f"Add 1 year (approx): {add_time_delta(start, years=1)}")
    log.info(f"Add 1 year, 2 months, 3 days: {add_time_delta(start, years=1, months=2, days=3)}")
    log.info(f"Add to date object: {add_time_delta(date(2023,3,15), months=2)}")

    # Test generate_date_range
    date_idx = generate_date_range("2023-01-01", "2023-01-05")
    if date_idx is not None:
        log.info(f"Daily date range:\n{date_idx}")
    
    monthly_idx = generate_date_range("2023-01-01", "2023-05-01", freq='MS') # Month Start
    if monthly_idx is not None:
        log.info(f"Monthly (start) date range:\n{monthly_idx}")

    hourly_idx = generate_date_range("2023-01-01 10:00", "2023-01-01 15:00", freq='H')
    if hourly_idx is not None:
        log.info(f"Hourly date range:\n{hourly_idx}")