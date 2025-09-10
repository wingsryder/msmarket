"""
Feature engineering module for extracting technical indicators and signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.config import MOMENTUM_PERIODS, VOLATILITY_WINDOW, MA_PERIODS
from utils.helpers import safe_divide


class FeatureEngineer:
    """
    Extracts technical indicators and features for ML model training
    """
    
    def __init__(self):
        self.momentum_periods = MOMENTUM_PERIODS
        self.volatility_window = VOLATILITY_WINDOW
        self.ma_periods = MA_PERIODS
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multiple moving averages for each stock.
        
        Args:
            df: Stock data with Close prices
            
        Returns:
            pd.DataFrame: Data with moving average features
        """
        result_df = df.copy()

        # De-duplicate columns and coerce 'Close' to a 1D Series
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        close_obj = result_df['Close'] if 'Close' in result_df.columns else None
        close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj

        for period in self.ma_periods:
            # Simple Moving Average
            result_df[f'SMA_{period}'] = close_series.groupby(result_df['Symbol']).rolling(
                window=period, min_periods=1
            ).mean().reset_index(0, drop=True)

            # Exponential Moving Average
            result_df[f'EMA_{period}'] = close_series.groupby(result_df['Symbol']).ewm(
                span=period, adjust=False
            ).mean().reset_index(0, drop=True)

            # Price relative to moving average
            result_df[f'Price_to_SMA_{period}'] = safe_divide(
                close_series, result_df[f'SMA_{period}']
            )

            result_df[f'Price_to_EMA_{period}'] = safe_divide(
                close_series, result_df[f'EMA_{period}']
            )

        return result_df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators for trend analysis.
        
        Args:
            df: Stock data with price information
            
        Returns:
            pd.DataFrame: Data with momentum features
        """
        result_df = df.copy()

        # De-duplicate columns and coerce 'Close' to a 1D Series
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        close_obj = result_df['Close'] if 'Close' in result_df.columns else None
        close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj

        for period in self.momentum_periods:
            # Rate of Change (ROC)
            result_df[f'ROC_{period}'] = close_series.groupby(result_df['Symbol']).pct_change(
                periods=period
            ).reset_index(0, drop=True) * 100

            # Momentum (price difference)
            result_df[f'Momentum_{period}'] = close_series.groupby(result_df['Symbol']).diff(
                periods=period
            ).reset_index(0, drop=True)

            # Relative Strength Index (RSI) approximation
            delta = close_series.groupby(result_df['Symbol']).diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.groupby(result_df['Symbol']).rolling(
                window=period, min_periods=1
            ).mean().reset_index(0, drop=True)

            avg_loss = loss.groupby(result_df['Symbol']).rolling(
                window=period, min_periods=1
            ).mean().reset_index(0, drop=True)

            rs = safe_divide(avg_gain, avg_loss, default=0)
            result_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        return result_df
    
    def calculate_volatility_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various volatility measures.
        
        Args:
            df: Stock data with returns
            
        Returns:
            pd.DataFrame: Data with volatility features
        """
        result_df = df.copy()

        # De-duplicate columns and coerce Series inputs
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        dr_obj = result_df['Daily_Return'] if 'Daily_Return' in result_df.columns else None
        dr_series = dr_obj.iloc[:, 0] if isinstance(dr_obj, pd.DataFrame) else dr_obj
        high_obj = result_df['High'] if 'High' in result_df.columns else None
        high_series = high_obj.iloc[:, 0] if isinstance(high_obj, pd.DataFrame) else high_obj
        low_obj = result_df['Low'] if 'Low' in result_df.columns else None
        low_series = low_obj.iloc[:, 0] if isinstance(low_obj, pd.DataFrame) else low_obj
        close_obj = result_df['Close'] if 'Close' in result_df.columns else None
        close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj

        # Historical volatility (annualized)
        result_df['Historical_Volatility'] = dr_series.groupby(result_df['Symbol']).rolling(
            window=self.volatility_window, min_periods=1
        ).std().reset_index(0, drop=True) * np.sqrt(252) * 100

        # Average True Range (ATR)
        result_df['True_Range'] = np.maximum(
            high_series - low_series,
            np.maximum(
                abs(high_series - close_series.shift(1)),
                abs(low_series - close_series.shift(1))
            )
        )

        result_df['ATR'] = result_df['True_Range'].groupby(result_df['Symbol']).rolling(
            window=14, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Volatility ratio (current vs historical)
        short_vol = dr_series.groupby(result_df['Symbol']).rolling(
            window=5, min_periods=1
        ).std().reset_index(0, drop=True)

        long_vol = dr_series.groupby(result_df['Symbol']).rolling(
            window=20, min_periods=1
        ).std().reset_index(0, drop=True)

        result_df['Volatility_Ratio'] = safe_divide(short_vol, long_vol)

        return result_df
    
    def calculate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend signals and technical indicators.
        
        Args:
            df: Stock data with moving averages
            
        Returns:
            pd.DataFrame: Data with trend signals
        """
        result_df = df.copy()

        # De-duplicate columns and coerce 'Close' to Series
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        close_obj = result_df['Close'] if 'Close' in result_df.columns else None
        close_series = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj

        # MACD (Moving Average Convergence Divergence)
        ema_12 = close_series.groupby(result_df['Symbol']).ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
        ema_26 = close_series.groupby(result_df['Symbol']).ewm(span=26, adjust=False).mean().reset_index(0, drop=True)

        result_df['MACD'] = ema_12 - ema_26
        result_df['MACD_Signal'] = result_df['MACD'].groupby(result_df['Symbol']).ewm(span=9, adjust=False).mean().reset_index(0, drop=True)
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']

        # Bollinger Band signals
        if 'BB_Upper' in result_df.columns and 'BB_Lower' in result_df.columns:
            result_df['BB_Signal'] = np.where(
                close_series > result_df['BB_Upper'], 1,  # Overbought
                np.where(close_series < result_df['BB_Lower'], -1, 0)  # Oversold
            )

        # Moving average crossover signals
        if 'SMA_5' in result_df.columns and 'SMA_20' in result_df.columns:
            result_df['MA_Crossover'] = np.where(
                result_df['SMA_5'] > result_df['SMA_20'], 1, -1
            )

        # Price momentum signal
        result_df['Price_Momentum_Signal'] = np.where(
            result_df['ROC_10'] > 2, 1,  # Strong positive momentum
            np.where(result_df['ROC_10'] < -2, -1, 0)  # Strong negative momentum
        )

        return result_df
    
    def create_sector_features(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for sector-level analysis.
        
        Args:
            sector_data: Aggregated sector data
            
        Returns:
            pd.DataFrame: Sector data with engineered features
        """
        result_df = sector_data.copy()
        
        # Sector relative strength
        result_df['Sector_Relative_Strength'] = result_df.groupby('Date')['Avg_Return'].rank(
            pct=True
        )
        
        # Sector momentum score
        result_df['Momentum_Score'] = (
            result_df['Sector_Momentum'] * 0.5 +
            result_df['Sector_Price_Momentum'] * 0.3 +
            result_df['Avg_Rolling_Return'] * 0.2
        )
        
        # Volatility-adjusted returns
        result_df['Risk_Adjusted_Return'] = safe_divide(
            result_df['Avg_Return'], result_df['Avg_Volatility']
        )
        
        # Trend consistency (lower volatility of returns indicates consistent trend)
        result_df['Trend_Consistency'] = 1 / (result_df['Return_Std'] + 0.001)
        
        # Volume momentum
        result_df['Volume_Momentum'] = result_df.groupby('Sector')['Total_Volume'].pct_change(
            periods=5
        )
        
        return result_df
    
    def extract_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and prepare features for machine learning model.
        
        Args:
            df: Processed data with all indicators
            
        Returns:
            pd.DataFrame: ML-ready features
        """
        feature_columns = [
            'Sector_Momentum', 'Sector_Price_Momentum', 'Avg_Volatility',
            'Risk_Adjusted_Return', 'Trend_Consistency', 'Volume_Momentum',
            'Momentum_Score', 'Sector_Relative_Strength'
        ]
        
        # Select available feature columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No ML features available in the data")
        
        ml_features = df[['Sector', 'Date'] + available_features].copy()
        
        # Handle missing values
        ml_features = ml_features.ffill()  # Forward fill
        ml_features = ml_features.fillna(0)  # Fill remaining NaNs with 0
        
        return ml_features
