"""
API providers for Indian stock market data (NSE/BSE)
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import streamlit as st
from abc import ABC, abstractmethod


class BaseAPIProvider(ABC):
    """Base class for all API providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    @abstractmethod
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical stock data"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API is available and properly configured"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        pass


class YFinanceProvider(BaseAPIProvider):
    """Yahoo Finance provider using yfinance library"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.name = "Yahoo Finance"
    
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch Indian stock data from Yahoo Finance (NSE/BSE)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Reset index to get Date as a column
            data = data.reset_index()
            
            # Standardize column names
            data = data.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Select only required columns
            required_columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[required_columns]
            
            return data
            
        except Exception as e:
            st.warning(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
            return None
    
    def is_available(self) -> bool:
        """Yahoo Finance doesn't require API key"""
        return True
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Yahoo Finance rate limit info"""
        return {
            'requests_per_day': 'Unlimited',
            'requests_per_minute': 'Soft limit ~60',
            'current_usage': 'Unknown'
        }


def get_api_provider(provider_name: str, api_key: Optional[str] = None) -> BaseAPIProvider:
    """Factory function to get API provider instance"""
    providers = {
        'yfinance': YFinanceProvider
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {list(providers.keys())}")

    return providers[provider_name](api_key)
