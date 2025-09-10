# Multi-Stock Sector Trend Analyzer

A comprehensive web application that analyzes Indian stock market trends across different sectors to identify which sectors are currently trending bullish or bearish using machine learning and technical analysis.

## 🚀 Key Features

### 📊 Data & Market Coverage
- **Indian Market Focus**: NSE/BSE stocks with .NS suffix symbols
- **100 Pre-configured Stocks**: Across 10 major sectors (Technology, Banking, Finance, Energy, FMCG, etc.)
- **Smart Data Fetching**: Cache-first approach with offline backup and live API fallback
- **Resilient Data Collection**: Handles Yahoo Finance rate limits with retry logic and local caching

### 🤖 Machine Learning & Analytics
- **Trend Classification**: ML-powered Bullish/Neutral/Bearish predictions using scikit-learn
- **Advanced Feature Engineering**: 40+ technical indicators including momentum, volatility, and trend signals
- **Comprehensive ML Dashboard**: Learning curves, confusion matrix, ROC analysis, feature importance with error bars
- **Risk-Adjusted Analysis**: Sector performance normalized by volatility and risk metrics

### 🎨 Interactive Visualizations
- **Real-time Charts**: Plotly-powered interactive visualizations
- **Multiple Chart Types**: Bar charts, line plots, scatter plots, heatmaps, pie charts
- **Risk vs Return Analysis**: Bubble charts showing volatility vs performance
- **Sector Momentum Tracking**: Time series analysis of sector trends

### 💾 Offline Capabilities
- **Local Cache System**: Gzipped CSV storage for 1-year historical data
- **Preload Functionality**: Download all NSE data for offline analysis
- **Cache-First Strategy**: Prioritizes local data over API calls for faster performance
- **Fallback Logic**: Seamless switching between cache and live data

## 📋 Requirements

- **Python**: 3.8 or higher
- **Internet**: For initial data download (optional for cached analysis)
- **Storage**: ~50MB for full NSE cache
- **Dependencies**: Listed in requirements.txt

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**: Open `http://localhost:8501` in your browser

## 🎯 Quick Start Guide

### First Time Setup
1. **Preload Data** (Recommended): Click "Preload NSE cache (1 year)" in the sidebar
2. **Select Sectors**: Choose from Technology, Banking, Finance, Energy, FMCG
3. **Set Date Range**: Pick your analysis period (1 week to 1 year)
4. **Run Analysis**: Click "Run Analysis" to start

### Understanding Results
- **Sector Rankings**: Performance scores with color-coded visualization
- **Trend Predictions**: ML-powered bullish/bearish forecasts
- **Risk Analysis**: Volatility vs return scatter plots
- **ML Insights**: Model performance and feature importance

## 📁 Project Architecture

```
stock/
├── app.py                    # Main Streamlit application
├── data/
│   ├── collector.py          # Smart data collection with caching
│   ├── processor.py          # Data processing and aggregation
│   └── yf_cache/            # Local cache directory (auto-created)
├── models/
│   ├── trend_classifier.py   # ML trend classification
│   └── features.py           # Technical indicator calculations
├── utils/
│   ├── config.py             # Market configuration and stock symbols
│   └── helpers.py            # Utility functions and validation
│   └── yf_cache_test.py     # Standalone cache testing utility
├── visualizations/
│   └── charts.py             # Interactive chart components
├── requirements.txt          # Project dependencies
└── README.md                 # This documentation
```

## 🏢 Supported Sectors & Stocks

### Technology (10 stocks)
TCS.NS, INFY.NS, HCLTECH.NS, WIPRO.NS, TECHM.NS, MPHASIS.NS, COFORGE.NS, PERSISTENT.NS, LTTS.NS, OFSS.NS

### Banking (10 stocks)
HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS, AXISBANK.NS, INDUSINDBK.NS, FEDERALBNK.NS, BANDHANBNK.NS, IDFCFIRSTB.NS, PNB.NS

### Finance (10 stocks)
BAJFINANCE.NS, BAJAJFINSV.NS, HDFCLIFE.NS, SBILIFE.NS, ICICIPRULI.NS, LICI.NS, HDFCAMC.NS, MUTHOOTFIN.NS, CHOLAFIN.NS, PFC.NS

### Energy (10 stocks)
RELIANCE.NS, ONGC.NS, COALINDIA.NS, NTPC.NS, POWERGRID.NS, IOC.NS, BPCL.NS, GAIL.NS, TATAPOWER.NS, ADANIGREEN.NS

### FMCG (10 stocks)
HINDUNILVR.NS, NESTLEIND.NS, ITC.NS, DABUR.NS, MARICO.NS, GODREJCP.NS, BRITANNIA.NS, COLPAL.NS, TATACONSUM.NS, UBL.NS

*Complete sector definitions in `utils/config.py`*

## 🧠 Technical Indicators & Features

### Price-Based Indicators
- **Moving Averages**: SMA/EMA (5, 10, 20, 50 periods)
- **Price Momentum**: ROC, momentum, percentage changes
- **Trend Signals**: MACD, signal line crossovers

### Volatility & Risk Measures
- **Historical Volatility**: Rolling standard deviation
- **Bollinger Bands**: Upper/lower bands with signals
- **Average True Range (ATR)**: Volatility measurement

### Volume & Market Indicators
- **Volume Momentum**: Trading volume changes
- **VWAP**: Volume-weighted average price
- **Volume-Price Trend**: Combined volume-price analysis

### Sector-Level Aggregations
- **Sector Momentum**: Weighted average returns
- **Risk-Adjusted Returns**: Sharpe-like ratios
- **Trend Consistency**: Stability measurements
- **Performance Rankings**: Multi-factor scoring

## 📊 Visualization Gallery

### Core Analytics
- **Sector Performance Rankings**: Horizontal/vertical bar charts with color coding
- **Sector Momentum Over Time**: Multi-line time series with interactive legends
- **Risk vs Return Analysis**: Bubble scatter plots with performance-based sizing

### ML Model Insights
- **Confusion Matrix**: Interactive heatmap with percentage annotations
- **ROC Curves**: Multi-class one-vs-rest with AUC scores
- **Learning Curves**: Training vs validation performance over dataset size
- **Feature Importance**: Bar charts with error bars (Random Forest)

### Advanced Views
- **Sector Heatmaps**: Performance over time with color gradients
- **Trend Distribution**: Pie charts of bullish/neutral/bearish predictions
- **Prediction Confidence**: Model certainty by sector

## 💾 Cache Management

### Cache Structure
```
data/yf_cache/
├── TCS.NS.csv.gz           # Individual stock files
├── INFY.NS.csv.gz
├── HDFCBANK.NS.csv.gz
└── ...                     # One file per symbol
```

### Cache Features
- **Automatic TTL**: Data freshness validation
- **Compression**: Gzipped CSV for space efficiency
- **Date Range Filtering**: Smart slicing for requested periods
- **Fallback Logic**: Cache → API → Partial cache hierarchy

### Cache Commands
- **Preload**: Download 1-year data for all 100 stocks
- **Auto-refresh**: Fetch missing data on-demand
- **Manual cleanup**: Remove cache files if needed

## ⚙️ Configuration & Customization

### Analysis Parameters
- **Rolling Window**: 3-30 days (default: 7)
- **Date Range**: Up to 1 year of historical data
- **Trend Thresholds**: ±2% for bullish/bearish classification
- **ML Model**: Random Forest with 100 estimators

### Adding Custom Stocks
Edit `utils/config.py` to add new sectors or stocks:
```python
MARKETS = {
    'INDIA': {
        'sectors': {
            'YourSector': ['SYMBOL1.NS', 'SYMBOL2.NS', ...]
        }
    }
}
```

### Performance Tuning
- **Batch Size**: Adjust concurrent API requests
- **Cache TTL**: Modify data freshness requirements
- **ML Parameters**: Tune model hyperparameters
- **Visualization**: Customize chart themes and colors

## 🔧 Technology Stack

- **Backend**: Python 3.8+, Pandas, NumPy
- **ML/Analytics**: scikit-learn, technical indicators
- **Frontend**: Streamlit 1.50+
- **Visualization**: Plotly, interactive charts
- **Data Source**: Yahoo Finance (yfinance)
- **Storage**: Local filesystem cache (CSV.gz)

## 🚀 Performance Features

- **Cache-First Architecture**: Minimizes API calls
- **Batch Processing**: Efficient multi-stock downloads
- **Lazy Loading**: Load data only when needed
- **Session State**: Preserve analysis results
- **Error Recovery**: Graceful handling of API failures

## 📈 Use Cases

### Investment Analysis
- **Sector Rotation**: Identify trending sectors for portfolio allocation
- **Risk Assessment**: Compare volatility across sectors
- **Timing Analysis**: Entry/exit points based on momentum

### Research & Education
- **Market Behavior**: Study sector correlations and trends
- **ML Learning**: Understand feature importance in financial prediction
- **Technical Analysis**: Explore various indicators and their effectiveness

### Professional Trading
- **Screening**: Quick sector-wide performance overview
- **Risk Management**: Volatility-adjusted position sizing
- **Trend Following**: ML-powered trend identification

## 🛠️ Troubleshooting

### Common Issues
1. **No Data Fetched**: Run preload first, check internet connection
2. **Rate Limiting**: Wait 10-15 minutes, use shorter date ranges
3. **Cache Issues**: Clear `data/yf_cache/` directory and re-preload
4. **ML Errors**: Ensure sufficient data points (>30 per sector)

### Performance Tips
- Use preloaded cache for faster analysis
- Select fewer sectors for quicker processing
- Choose shorter date ranges for real-time analysis
- Clear browser cache if UI becomes unresponsive

---

**Built with ❤️ for Indian stock market sector analysis**
