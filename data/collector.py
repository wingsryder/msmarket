"""
Data collection module with multi-API and multi-market support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import streamlit as st
from utils.config import MARKETS, API_PROVIDERS, DEFAULT_MARKET, DEFAULT_API, DEFAULT_LOOKBACK_DAYS
from utils.helpers import validate_date_range, validate_date_range_with_message
from data.api_providers import get_api_provider


class StockDataCollector:
    """
    Handles data collection with multi-API and multi-market support
    """

    def __init__(self, market: str = DEFAULT_MARKET, api_provider: str = DEFAULT_API, api_key: Optional[str] = None):
        """
        Initialize the collector with market and API provider settings

        Args:
            market: Market to use ('US', 'INDIA')
            api_provider: API provider to use ('yfinance' for Indian markets)
            api_key: API key for providers that require it
        """
        self.market = market
        self.api_provider_name = api_provider
        self.api_key = api_key

        # Get market configuration
        if market not in MARKETS:
            raise ValueError(f"Unsupported market: {market}. Available: {list(MARKETS.keys())}")

        self.market_config = MARKETS[market]
        self.sector_stocks = self.market_config['sectors']

        # Initialize API provider
        self.api_provider = get_api_provider(api_provider, api_key)

        # Check if API provider supports this market
        if market not in API_PROVIDERS[api_provider]['supported_markets']:
            st.warning(f"{API_PROVIDERS[api_provider]['name']} may have limited support for {market} market")

    def get_market_info(self) -> Dict:
        """Get information about the current market"""
        return {
            'market': self.market,
            'market_name': self.market_config['name'],
            'api_provider': self.api_provider_name,
            'api_name': API_PROVIDERS[self.api_provider_name]['name'],
            'sectors': list(self.sector_stocks.keys()),
            'total_stocks': sum(len(stocks) for stocks in self.sector_stocks.values())
        }
        
    def fetch_stock_data(self,
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a single symbol using the configured API provider.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TCS.NS')
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            Optional[pd.DataFrame]: Stock data with OHLCV columns
        """
        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use the configured API provider
                data = self.api_provider.fetch_stock_data(symbol, start_date, end_date)

                if data is not None and not data.empty:
                    return data

                # If no data but no exception, try shorter date range on retry
                if attempt < max_retries - 1:
                    # Try with a shorter date range (last 30 days)
                    shorter_start = end_date - timedelta(days=30)
                    if shorter_start > start_date:
                        st.info(f"Retrying {symbol} with shorter date range...")
                        data = self.api_provider.fetch_stock_data(symbol, shorter_start, end_date)
                        if data is not None and not data.empty:
                            return data

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    st.info(f"Retry {attempt + 1}/{max_retries} for {symbol} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Error fetching data for {symbol} after {max_retries} attempts: {str(e)}")

        return None
    
    def fetch_sector_data(self, 
                         sectors: List[str], 
                         start_date: datetime, 
                         end_date: datetime,
                         custom_stocks: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Fetch stock data for multiple sectors.
        
        Args:
            sectors: List of sector names to fetch
            start_date: Start date for data collection
            end_date: End date for data collection
            custom_stocks: Optional custom stock symbols per sector
            
        Returns:
            pd.DataFrame: Combined data for all sectors with sector column
        """
        # Validate date range with detailed error message
        is_valid, error_message = validate_date_range_with_message(start_date, end_date)
        if not is_valid:
            raise ValueError(f"Invalid date range: {error_message}")
        
        all_data = []
        stocks_to_use = custom_stocks if custom_stocks else self.sector_stocks
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_sectors = len(sectors)
        
        successful_sectors = []
        failed_sectors = []

        for i, sector in enumerate(sectors):
            if sector not in stocks_to_use:
                st.warning(f"Sector '{sector}' not found in configuration. Available sectors: {list(self.sector_stocks.keys())}")
                failed_sectors.append(sector)
                continue

            status_text.text(f"Fetching data for {sector} sector...")

            sector_stocks = stocks_to_use[sector]
            sector_data = []
            successful_stocks = 0

            for stock in sector_stocks:
                stock_data = self.fetch_stock_data(stock, start_date, end_date)
                if stock_data is not None:
                    stock_data['Sector'] = sector
                    sector_data.append(stock_data)
                    successful_stocks += 1

            if sector_data:
                sector_df = pd.concat(sector_data, ignore_index=True)
                all_data.append(sector_df)
                successful_sectors.append(sector)
                st.success(f"✅ {sector}: {successful_stocks}/{len(sector_stocks)} stocks fetched")
            else:
                failed_sectors.append(sector)
                st.error(f"❌ {sector}: No data fetched for any stocks")

            # Update progress
            progress_bar.progress((i + 1) / total_sectors)
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            # Provide detailed error information with success/failure breakdown
            error_details = []

            if successful_sectors:
                error_details.append(f"✅ Successful sectors: {successful_sectors}")

            if failed_sectors:
                error_details.append(f"❌ Failed sectors: {failed_sectors}")

            if not sectors:
                error_details.append("No sectors were selected")
            else:
                error_details.append("This might be due to:")
                error_details.append("- Yahoo Finance API rate limiting (wait 5-10 minutes)")
                error_details.append("- Date range too long (try shorter period: 1-4 weeks)")

            raise ValueError("No data could be fetched for any sector. " + " | ".join(error_details))
        
        # Combine all sector data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Sort by date and symbol
        combined_data = combined_data.sort_values(['Date', 'Symbol'])
        combined_data = combined_data.reset_index(drop=True)

        # Show final success message
        total_stocks = combined_data['Symbol'].nunique()
        total_rows = len(combined_data)
        date_range = f"{combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}"

        st.success(f"🎉 Data collection complete! {len(successful_sectors)} sectors, {total_stocks} stocks, {total_rows} data points ({date_range})")

        return combined_data
    
    def get_available_sectors(self) -> List[str]:
        """
        Get list of available sectors.
        
        Returns:
            List[str]: Available sector names
        """
        return list(self.sector_stocks.keys())
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get stock symbols for a specific sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List[str]: Stock symbols in the sector
        """
        return self.sector_stocks.get(sector, [])
    
    def validate_custom_stocks(self, custom_stocks: Dict[str, List[str]]) -> bool:
        """
        Validate custom stock symbols format.
        
        Args:
            custom_stocks: Dictionary of sector -> stock symbols
            
        Returns:
            bool: True if format is valid
        """
        if not isinstance(custom_stocks, dict):
            return False
        
        for sector, stocks in custom_stocks.items():
            if not isinstance(stocks, list) or not stocks:
                return False
            
            for stock in stocks:
                if not isinstance(stock, str) or len(stock) < 1:
                    return False
        
        return True
