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
        
        # Sort by symbol and date
        df = df.sort_values(['Symbol', 'Date'])
        
        # Calculate daily returns for each stock
        df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
        
        # Calculate price change
        df['Price_Change'] = df.groupby('Symbol')['Close'].diff()
        
        # Calculate percentage change
        df['Pct_Change'] = df['Daily_Return'] * 100
        
        # Calculate high-low spread
        df['HL_Spread'] = df['High'] - df['Low']
        df['HL_Spread_Pct'] = safe_divide(df['HL_Spread'], df['Close']) * 100
        
        # Calculate volume-weighted average price (VWAP) approximation
        df['VWAP'] = safe_divide((df['High'] + df['Low'] + df['Close']) * df['Volume'],
                                df['Volume'] * 3)

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
        
        # Group by symbol for rolling calculations
        grouped = result_df.groupby('Symbol')
        
        # Rolling mean of close prices
        result_df['Rolling_Mean'] = grouped['Close'].rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Rolling standard deviation
        result_df['Rolling_Std'] = grouped['Close'].rolling(
            window=self.rolling_window, min_periods=1
        ).std().reset_index(0, drop=True)
        
        # Rolling mean of returns
        result_df['Rolling_Return_Mean'] = grouped['Daily_Return'].rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Rolling volatility (std of returns)
        result_df['Rolling_Volatility'] = grouped['Daily_Return'].rolling(
            window=self.rolling_window, min_periods=1
        ).std().reset_index(0, drop=True)
        
        # Rolling volume mean
        result_df['Rolling_Volume_Mean'] = grouped['Volume'].rolling(
            window=self.rolling_window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Bollinger Bands
        result_df['BB_Upper'] = result_df['Rolling_Mean'] + (2 * result_df['Rolling_Std'])
        result_df['BB_Lower'] = result_df['Rolling_Mean'] - (2 * result_df['Rolling_Std'])
        result_df['BB_Position'] = safe_divide(
            result_df['Close'] - result_df['BB_Lower'],
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
        # Group by sector and date
        sector_groups = df.groupby(['Sector', 'Date'])
        
        # Calculate sector-level metrics
        sector_data = sector_groups.agg({
            'Close': ['mean', 'median', 'std'],
            'Volume': ['sum', 'mean'],
            'Daily_Return': ['mean', 'median', 'std'],
            'Rolling_Return_Mean': 'mean',
            'Rolling_Volatility': 'mean',
            'Pct_Change': ['mean', 'std'],
            'HL_Spread_Pct': 'mean'
        }).reset_index()
        
        # Flatten column names
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
