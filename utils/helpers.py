"""
Utility functions for the Multi-Stock Sector Trend Analyzer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import streamlit as st


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that the date range is reasonable for analysis.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        bool: True if date range is valid
    """
    if start_date >= end_date:
        return False

    # Check if date range is at least 7 days (reduced from 30)
    if (end_date - start_date).days < 7:
        return False

    # Allow end date to be today (not strictly in the future)
    if end_date.date() > datetime.now().date():
        return False

    return True


def validate_date_range_with_message(start_date: datetime, end_date: datetime) -> tuple[bool, str]:
    """
    Validate date range and return detailed error message.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        tuple: (is_valid, error_message)
    """
    if start_date >= end_date:
        return False, "Start date must be before end date"

    days_diff = (end_date - start_date).days
    if days_diff < 7:
        return False, f"Date range too short ({days_diff} days). Minimum 7 days required for analysis."

    if end_date.date() > datetime.now().date():
        return False, "End date cannot be in the future"

    # Check if the date range is too far in the past (optional warning)
    if start_date < datetime.now() - timedelta(days=3650):  # 10 years
        return False, "Start date is too far in the past (maximum 10 years ago)"

    return True, "Date range is valid"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily returns from price series.
    
    Args:
        prices: Series of stock prices
        
    Returns:
        pd.Series: Daily returns
    """
    return prices.pct_change().dropna()


def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate approximate number of trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Approximate number of trading days
    """
    total_days = (end_date - start_date).days
    # Approximate: 5/7 of days are trading days
    return int(total_days * 5 / 7)


def safe_divide(numerator, denominator, default: float = 0.0):
    """
    Safely divide two numbers or arrays, returning default if denominator is zero.

    Args:
        numerator: Numerator value (can be scalar or array-like)
        denominator: Denominator value (can be scalar or array-like)
        default: Default value if division by zero

    Returns:
        Result of division or default value
    """
    # Handle pandas Series/arrays
    if hasattr(denominator, '__iter__') and not isinstance(denominator, str):
        # For arrays/Series, use numpy where for element-wise safe division
        import numpy as np
        return np.where((denominator == 0) | pd.isna(denominator), default, numerator / denominator)
    else:
        # For scalars
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator


def create_color_map(values: List[float]) -> Dict[str, str]:
    """
    Create a color mapping for trend visualization.
    
    Args:
        values: List of trend values
        
    Returns:
        Dict: Mapping of values to colors
    """
    color_map = {}
    for value in values:
        if value > 0:
            color_map[value] = '#00ff00'  # Green for positive
        elif value < 0:
            color_map[value] = '#ff0000'  # Red for negative
        else:
            color_map[value] = '#ffff00'  # Yellow for neutral
    
    return color_map


@st.cache_data
def load_cached_data(cache_key: str) -> Optional[pd.DataFrame]:
    """
    Load cached data if available.
    
    Args:
        cache_key: Unique key for cached data
        
    Returns:
        Optional[pd.DataFrame]: Cached data or None
    """
    # This is a placeholder for caching implementation
    # In a real application, you might use Redis, file system, or database
    return None


def display_metric_card(title: str, value: str, delta: Optional[str] = None):
    """
    Display a metric card in Streamlit.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
    """
    st.metric(label=title, value=value, delta=delta)
