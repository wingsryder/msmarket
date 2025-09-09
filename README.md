# Multi-Stock Sector Trend Analyzer

A comprehensive web application that analyzes stock market trends across different sectors to identify which sectors are currently trending bullish or bearish using machine learning and technical analysis.

## 🚀 Features

- Real-time data from Yahoo Finance (NSE/BSE) with resilient fetching
- India-only market focus with NSE symbols (suffix .NS)
- Feature engineering: momentum, volatility, risk-adjusted returns, trend consistency
- Machine learning trend classification (Bullish/Neutral/Bearish) using scikit-learn
- ML Performance Dashboard — learning curves, confusion matrix, CV analysis, classification report, ROC, and enhanced feature importance
- Interactive Streamlit UI with Plotly visuals
- Customizable analysis: date range, sectors, and optional custom symbols
- Multiple output formats: bar/line charts, tables, ranked lists, heatmaps

## 📋 Requirements

- Python 3.8 or higher
- Internet connection for fetching stock data
- Streamlit 1.50+ 

## 🛠️ Installation

1. Clone this repository or download the files
2. Navigate to the project directory:
   ```bash
   cd <root_directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Set Date Range**: Choose start and end dates for analysis
2. **Select Sectors**: Choose which sectors to analyze from the dropdown
3. **Custom Stocks** (Optional): Add custom stock symbols for specific sectors
4. **Analysis Parameters**: Set rolling window size for calculations
5. **Output Format**: Choose visualization type
6. **Run Analysis**: Click the "Run Analysis" button to start
7. After a successful run with sufficient data, open the "Machine Learning Model Analysis" tabs to explore ML performance (learning curves, confusion matrix, CV analysis, classification report, ROC, feature importance).


## 📁 Project Structure

```
stock/
├── app.py                    # Main Streamlit application
├── data/
│   ├── __init__.py
│   ├── api_providers.py      # API providers for Indian stock market data
│   ├── collector.py          # Data collection from Yahoo Finance
│   └── processor.py          # Data processing and aggregation
├── models/
│   ├── __init__.py
│   ├── trend_classifier.py   # ML model for trend classification
│   └── features.py           # Feature engineering functions
├── utils/
│   ├── __init__.py
│   ├── config.py             # Configuration and constants
│   └── helpers.py            # Utility functions
├── visualizations/
│   ├── __init__.py
│   └── charts.py             # Visualization components
├── requirements.txt          # fProject dependencies
└── README.md                 # This file
```

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Finance Data**: yfinance (Yahoo Finance API; NSE/BSE .NS suffix)
- **Visualization**: Plotly, Matplotlib

## 📊 Available Sectors

The application includes pre-configured stock symbols for the following sectors:

- Technology: TCS.NS, INFY.NS, HCLTECH.NS, WIPRO.NS, TECHM.NS, MPHASIS.NS, COFORGE.NS, PERSISTENT.NS, LTTS.NS, OFSS.NS
- Banking: HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS, AXISBANK.NS, INDUSINDBK.NS, FEDERALBNK.NS, BANDHANBNK.NS, IDFCFIRSTB.NS, PNB.NS
- Finance: BAJFINANCE.NS, BAJAJFINSV.NS, HDFCLIFE.NS, SBILIFE.NS, ICICIPRULI.NS, LICI.NS, HDFCAMC.NS, MUTHOOTFIN.NS, CHOLAFIN.NS, PFC.NS
- Energy: RELIANCE.NS, ONGC.NS, COALINDIA.NS, NTPC.NS, POWERGRID.NS, IOC.NS, BPCL.NS, GAIL.NS, TATAPOWER.NS, ADANIGREEN.NS
- FMCG: HINDUNILVR.NS, NESTLEIND.NS, ITC.NS, DABUR.NS, MARICO.NS, GODREJCP.NS, BRITANNIA.NS, COLPAL.NS, TATACONSUM.NS, UBL.NS

Full sector lists are defined in utils/config.py. Replace or extend as needed.

## 🧠 Machine Learning Features

The application uses the following features for trend classification:

### Technical Indicators
- **Moving Averages**: Simple and Exponential Moving Averages (5, 10, 20, 50 periods)
- **Momentum Indicators**: Rate of Change (ROC), Momentum, RSI
- **Volatility Measures**: Historical Volatility, Average True Range (ATR)
- **Trend Signals**: MACD, Bollinger Bands, Moving Average Crossovers

### Sector-Level Features
- **Sector Momentum**: Rolling average returns
- **Price Momentum**: Percentage change over time periods
- **Risk-Adjusted Returns**: Returns normalized by volatility
- **Trend Consistency**: Measure of trend stability
- **Volume Momentum**: Changes in trading volume

## 📈 Output Formats

### 1. Bar Charts
- Sector performance rankings
- Prediction confidence levels

### 2. Line Charts
- Sector momentum over time
- Historical trend analysis

### 3. Data Tables
- Detailed sector rankings with metrics
- Sortable and filterable data

### 4. Ranked Lists
- Top performing sectors
- ML trend predictions with confidence scores

### 5. Advanced Visualizations
- Risk vs Return scatter plots
- Trend distribution pie charts
- Feature importance charts
- Sector performance heatmaps

## 🤖 ML Model Performance Visualizations

The app now includes a dedicated ML analysis area (visible after a successful run with enough data):

- Performance Overview: Train vs Test vs Cross‑Validation accuracy (mean/std)
- Cross‑Validation Analysis: Distribution and box plot of CV scores
- Learning Curves: Training and validation scores vs training set size
- Confusion Matrix & Classification Report: Per‑class precision/recall/F1
- ROC Curves (multi‑class): One‑vs‑rest ROC with AUC per class
- Enhanced Feature Importance: Importance with variability/error bars (RF)

Where to find it: In the Streamlit UI under the tabs titled “Machine Learning Model Analysis”. Some charts appear only if the model provides the required outputs (e.g., probabilities for ROC) and when there are sufficient samples (≥ 50).

## ⚙️ Configuration

### Customizing Sectors and Stocks

You can modify the `utils/config.py` file to:
- Add new sectors
- Update stock symbols for existing sectors
- Adjust analysis parameters
- Modify ML model thresholds

### Analysis Parameters

- **Rolling Window**: Default 7 days (adjustable 3-30 days)
- **Lookback Period**: Default 252 days (~1 year)
- **Trend Thresholds**: ±2% for bullish/bearish classification
- **Minimum Data Points**: 30 for reliable analysis
