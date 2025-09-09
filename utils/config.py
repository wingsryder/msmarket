"""
Configuration file for the Multi-Stock Sector Trend Analyzer
"""

# Market configurations - Indian Stock Market Only
MARKETS = {
    'INDIA': {
        'name': 'India (NSE/BSE)',
        'suffix': '.NS',  # NSE suffix for yfinance
        'sectors': {
            'Technology': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS', 'OFSS.NS'],
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS'],
            'Finance': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'LICI.NS', 'HDFCAMC.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'PFC.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'ADANIGREEN.NS', 'TATAPOWER.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'UBL.NS', 'EMAMILTD.NS'],
            'Pharmaceuticals': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'GLENMARK.NS', 'ALKEM.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 'APOLLOTYRE.NS'],
            'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'SAIL.NS', 'NMDC.NS', 'MOIL.NS', 'JINDALSTEL.NS', 'WELCORP.NS', 'RATNAMANI.NS'],
            'Telecom': ['BHARTIARTL.NS', 'IDEA.NS', 'RCOM.NS', 'MTNL.NS', 'GTPL.NS', 'TTML.NS', 'HFCL.NS', 'RAILTEL.NS', 'ROUTE.NS', 'INDUSINDBK.NS'],
            'Cement': ['ULTRACEMCO.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'ACC.NS', 'JKCEMENT.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS', 'STARCEMENT.NS', 'INDIACEM.NS', 'ORIENTCEM.NS']
        }
    }
}

# API configurations
API_PROVIDERS = {
    'yfinance': {
        'name': 'Yahoo Finance (yfinance)',
        'description': 'Free, reliable historical data for Indian stock markets (NSE/BSE).',
        'rate_limit': 'No strict limits',
        'requires_key': False,
        'supported_markets': ['INDIA'],
        'data_delay': 'Real-time to 15 minutes',
        'features': ['Historical prices', 'Volume', 'Dividends', 'Splits']
    }
}

# Default configurations
DEFAULT_MARKET = 'INDIA'
DEFAULT_API = 'yfinance'

# Backward compatibility - use default market sectors
SECTOR_STOCKS = MARKETS[DEFAULT_MARKET]['sectors']

# Default analysis parameters
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_LOOKBACK_DAYS = 252  # ~1 year of trading days
MIN_DATA_POINTS = 30

# Feature engineering parameters
MOMENTUM_PERIODS = [5, 10, 20]
VOLATILITY_WINDOW = 20
MA_PERIODS = [5, 10, 20, 50]

# ML model parameters
TREND_LABELS = {
    'Bullish': 1,
    'Neutral': 0,
    'Bearish': -1
}

# Trend classification thresholds (percentage change)
BULLISH_THRESHOLD = 0.02  # 2% positive change
BEARISH_THRESHOLD = -0.02  # 2% negative change

# Streamlit configuration
PAGE_TITLE = "Multi-Stock Sector Trend Analyzer"
PAGE_ICON = "📈"
LAYOUT = "wide"
