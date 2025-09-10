"""
Data processing pipeline for stock market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.config import DEFAULT_ROLLING_WINDOW, MIN_DATA_POINTS
from utils.helpers import calculate_returns, safe_divide


class StockDataProcessor:
    """
    Handles data processing, grouping, and statistical computations
    """
    
    def __init__(self, rolling_window: int = DEFAULT_ROLLING_WINDOW):
        self.rolling_window = rolling_window
    
    def process_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw stock data and add basic calculated fields.
        
        Args:
            raw_data: Raw stock data from collector
            
        Returns:
            pd.DataFrame: Processed data with additional fields
        """
        df = raw_data.copy()
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Drop duplicate columns (keep first) to avoid DataFrame-from-duplicate-name issues
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Sort by symbol and date
        df = df.sort_values(['Symbol', 'Date'])

        # Ensure 'Close' is a 1D Series (not a DataFrame with duplicate 'Close' columns)
        close_obj = df['Close'] if 'Close' in df.columns else None
        if isinstance(close_obj, pd.DataFrame):
            close_series = close_obj.iloc[:, 0]
        else:
            close_series = close_obj

        # Calculate daily returns and price changes per stock
        df['Daily_Return'] = close_series.groupby(df['Symbol']).pct_change()
        df['Price_Change'] = close_series.groupby(df['Symbol']).diff()

        # Calculate percentage change
        df['Pct_Change'] = df['Daily_Return'] * 100

        # Calculate high-low spread using 1D Series for robustness
        high_obj = df['High'] if 'High' in df.columns else None
        high_series = high_obj.iloc[:, 0] if isinstance(high_obj, pd.DataFrame) else high_obj
        low_obj = df['Low'] if 'Low' in df.columns else None
        low_series = low_obj.iloc[:, 0] if isinstance(low_obj, pd.DataFrame) else low_obj
        vol_obj = df['Volume'] if 'Volume' in df.columns else None
        vol_series = vol_obj.iloc[:, 0] if isinstance(vol_obj, pd.DataFrame) else vol_obj

        df['HL_Spread'] = high_series - low_series
        df['HL_Spread_Pct'] = safe_divide(df['HL_Spread'], close_series) * 100

        # Calculate volume-weighted average price (VWAP) approximation
        df['VWAP'] = safe_divide((high_series + low_series + close_series) * vol_series, vol_series * 3)

        # Add rolling statistics
        df = self.calculate_rolling_statistics(df)

        return df
    
    def calculate_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling statistics for each stock.
        
        Args:
            df: Processed stock data
            
        Returns:
            pd.DataFrame: Data with rolling statistics
        """
        result_df = df.copy()

        # De-duplicate columns again just in case upstream concat introduced duplicates
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # Group by symbol for rolling calculations
        grouped = result_df.groupby('Symbol')

        # Use 1D Series for Close and Daily_Return if duplicates/multiindex occurred
        close_obj = result_df['Close'] if 'Close' in result_df.columns else None
        close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj
        dr_obj = result_df['Daily_Return'] if 'Daily_Return' in result_df.columns else None
        dr_series = dr_obj.iloc[:, 0] if isinstance(dr_obj, pd.DataFrame) else dr_obj

        # Rolling mean of close prices
        result_df['Rolling_Mean'] = close_series.groupby(result_df['Symbol']).rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Rolling standard deviation
        result_df['Rolling_Std'] = close_series.groupby(result_df['Symbol']).rolling(
            window=self.rolling_window, min_periods=1
        ).std().reset_index(0, drop=True)

        # Rolling mean of returns
        result_df['Rolling_Return_Mean'] = dr_series.groupby(result_df['Symbol']).rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Rolling volatility (std of returns)
        result_df['Rolling_Volatility'] = dr_series.groupby(result_df['Symbol']).rolling(
            window=self.rolling_window, min_periods=1
        ).std().reset_index(0, drop=True)

        # Rolling volume mean
        vol_obj = result_df['Volume'] if 'Volume' in result_df.columns else None
        vol_series = vol_obj.iloc[:, 0] if isinstance(vol_obj, pd.DataFrame) else vol_obj
        result_df['Rolling_Volume_Mean'] = vol_series.groupby(result_df['Symbol']).rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Bollinger Bands
        result_df['BB_Upper'] = result_df['Rolling_Mean'] + (2 * result_df['Rolling_Std'])
        result_df['BB_Lower'] = result_df['Rolling_Mean'] - (2 * result_df['Rolling_Std'])
        result_df['BB_Position'] = safe_divide(
            close_series - result_df['BB_Lower'],
            result_df['BB_Upper'] - result_df['BB_Lower']
        )
        
        return result_df
    
    def aggregate_by_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate stock data by sector for sector-level analysis.

        Args:
            df: Stock data with rolling statistics

        Returns:
            pd.DataFrame: Sector-level aggregated data
        """
        # Defensive: ensure required columns exist; compute if missing
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        required = ['Close', 'Volume', 'Daily_Return', 'Rolling_Return_Mean', 'Rolling_Volatility',
                    'Pct_Change', 'HL_Spread_Pct']
        missing = [c for c in required if c not in df.columns]

        # Try to compute missing columns from available primitives
        if missing:
            # Coerce primitives to Series
            close_obj = df['Close'] if 'Close' in df.columns else None
            close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj
            vol_obj = df['Volume'] if 'Volume' in df.columns else None
            vol_series = vol_obj.iloc[:, 0] if isinstance(vol_obj, pd.DataFrame) else vol_obj
            high_obj = df['High'] if 'High' in df.columns else None
            high_series = high_obj.iloc[:, 0] if isinstance(high_obj, pd.DataFrame) else high_obj
            low_obj = df['Low'] if 'Low' in df.columns else None
            low_series = low_obj.iloc[:, 0] if isinstance(low_obj, pd.DataFrame) else low_obj

            if 'Daily_Return' in missing and close_series is not None:
                df['Daily_Return'] = close_series.groupby(df['Symbol']).pct_change()
            if 'Pct_Change' in missing and 'Daily_Return' in df.columns:
                df['Pct_Change'] = df['Daily_Return'] * 100
            if 'HL_Spread_Pct' in missing and (high_series is not None) and (low_series is not None) and (close_series is not None):
                df['HL_Spread_Pct'] = safe_divide(high_series - low_series, close_series) * 100
            if 'Rolling_Return_Mean' in missing and 'Daily_Return' in df.columns:
                dr_obj = df['Daily_Return']
                dr_series = dr_obj.iloc[:, 0] if isinstance(dr_obj, pd.DataFrame) else dr_obj
                df['Rolling_Return_Mean'] = dr_series.groupby(df['Symbol']).rolling(
                    window=self.rolling_window, min_periods=1
                ).mean().reset_index(0, drop=True)
            if 'Rolling_Volatility' in missing and 'Daily_Return' in df.columns:
                dr_obj = df['Daily_Return']
                dr_series = dr_obj.iloc[:, 0] if isinstance(dr_obj, pd.DataFrame) else dr_obj
                df['Rolling_Volatility'] = dr_series.groupby(df['Symbol']).rolling(
                    window=self.rolling_window, min_periods=1
                ).std().reset_index(0, drop=True)
            if 'Volume' in missing:
                # If volume truly missing, fill with 0 to allow aggregation
                df['Volume'] = 0

        # Group by sector and date
        sector_groups = df.groupby(['Sector', 'Date'])

        # Ensure required columns exist as 1D Series and build a safe aggregation frame
        def to_series(df_in: pd.DataFrame, name: str):
            if name in df_in.columns:
                obj = df_in[name]
                return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            return None

        close_s = to_series(df, 'Close')
        vol_s = to_series(df, 'Volume')
        dr_s = to_series(df, 'Daily_Return')
        rrm_s = to_series(df, 'Rolling_Return_Mean')
        rv_s = to_series(df, 'Rolling_Volatility')
        pc_s = to_series(df, 'Pct_Change')
        hlp_s = to_series(df, 'HL_Spread_Pct')

        # Recompute missing basics if possible
        if dr_s is None and close_s is not None:
            dr_s = close_s.groupby(df['Symbol']).pct_change()
        if pc_s is None and dr_s is not None:
            pc_s = dr_s * 100
        if (hlp_s is None) and ('High' in df.columns) and ('Low' in df.columns) and (close_s is not None):
            high_s = to_series(df, 'High'); low_s = to_series(df, 'Low')
            if (high_s is not None) and (low_s is not None):
                hlp_s = safe_divide(high_s - low_s, close_s) * 100
        if (rrm_s is None) and (dr_s is not None):
            rrm_s = dr_s.groupby(df['Symbol']).rolling(window=self.rolling_window, min_periods=1).mean().reset_index(0, drop=True)
        if (rv_s is None) and (dr_s is not None):
            rv_s = dr_s.groupby(df['Symbol']).rolling(window=self.rolling_window, min_periods=1).std().reset_index(0, drop=True)
        if vol_s is None:
            # Fill missing volume with 0 to allow aggregation
            vol_s = pd.Series(0, index=df.index)

        # Build an aggregation DataFrame that always has the expected columns
        agg_df = pd.DataFrame({
            'Sector': df['Sector'],
            'Date': df['Date'],
            'Close': close_s if close_s is not None else pd.Series(np.nan, index=df.index),
            'Volume': vol_s,
            'Daily_Return': dr_s if dr_s is not None else pd.Series(np.nan, index=df.index),
            'Rolling_Return_Mean': rrm_s if rrm_s is not None else pd.Series(np.nan, index=df.index),
            'Rolling_Volatility': rv_s if rv_s is not None else pd.Series(np.nan, index=df.index),
            'Pct_Change': pc_s if pc_s is not None else pd.Series(np.nan, index=df.index),
            'HL_Spread_Pct': hlp_s if hlp_s is not None else pd.Series(np.nan, index=df.index),
        })

        # Calculate sector-level metrics with a fixed aggregation map
        sector_data = agg_df.groupby(['Sector', 'Date']).agg({
            'Close': ['mean', 'median', 'std'],
            'Volume': ['sum', 'mean'],
            'Daily_Return': ['mean', 'median', 'std'],
            'Rolling_Return_Mean': 'mean',
            'Rolling_Volatility': 'mean',
            'Pct_Change': ['mean', 'std'],
            'HL_Spread_Pct': 'mean'
        }).reset_index()

        # Flatten to the expected names
        sector_data.columns = [
            'Sector', 'Date', 'Avg_Close', 'Median_Close', 'Close_Std',
            'Total_Volume', 'Avg_Volume', 'Avg_Return', 'Median_Return', 'Return_Std',
            'Avg_Rolling_Return', 'Avg_Volatility', 'Avg_Pct_Change', 'Pct_Change_Std',
            'Avg_HL_Spread_Pct'
        ]

        # Sort by sector and date
        sector_data = sector_data.sort_values(['Sector', 'Date'])

        # Calculate sector-level rolling statistics
        sector_grouped = sector_data.groupby('Sector')

        # Sector momentum (rolling return)
        sector_data['Sector_Momentum'] = sector_grouped['Avg_Return'].rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Sector trend (price momentum)
        sector_data['Sector_Price_Momentum'] = sector_grouped['Avg_Close'].pct_change(
            periods=self.rolling_window
        ).reset_index(0, drop=True)

        # Sector volatility trend
        sector_data['Sector_Vol_Trend'] = sector_grouped['Avg_Volatility'].rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)

        return sector_data
    
    def calculate_sector_rankings(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector rankings based on various metrics.
        
        Args:
            sector_data: Sector-level aggregated data
            
        Returns:
            pd.DataFrame: Latest sector rankings
        """
        # Get the latest date for each sector
        latest_data = sector_data.groupby('Sector').last().reset_index()
        
        # Calculate composite scores
        latest_data['Performance_Score'] = (
            latest_data['Sector_Momentum'] * 0.4 +
            latest_data['Sector_Price_Momentum'] * 0.4 +
            (1 / (latest_data['Avg_Volatility'] + 0.001)) * 0.2  # Lower volatility is better
        )
        
        # Rank sectors
        latest_data['Performance_Rank'] = latest_data['Performance_Score'].rank(
            ascending=False, method='dense'
        )
        
        latest_data['Momentum_Rank'] = latest_data['Sector_Momentum'].rank(
            ascending=False, method='dense'
        )
        
        latest_data['Volatility_Rank'] = latest_data['Avg_Volatility'].rank(
            ascending=True, method='dense'  # Lower volatility gets better rank
        )
        
        # Sort by performance score
        latest_data = latest_data.sort_values('Performance_Score', ascending=False)
        
        return latest_data
    
    def filter_by_date_range(self, df: pd.DataFrame, 
                           start_date: pd.Timestamp, 
                           end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            df: Input dataframe
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            pd.DataFrame: Filtered data
        """
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        return df[mask].copy()
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality and completeness.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            Dict[str, bool]: Validation results
        """
        validation_results = {
            'has_data': len(df) > 0,
            'has_required_columns': all(col in df.columns for col in 
                                      ['Symbol', 'Date', 'Close', 'Sector']),
            'sufficient_data_points': len(df) >= MIN_DATA_POINTS,
            'no_missing_prices': df['Close'].notna().all(),
            'valid_date_range': df['Date'].notna().all()
        }
        
        return validation_results
