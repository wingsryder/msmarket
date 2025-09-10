"""
Multi-Stock Sector Trend Analyzer - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.collector import StockDataCollector
from data.processor import StockDataProcessor
from models.features import FeatureEngineer
from models.trend_classifier import SectorTrendClassifier
from visualizations.charts import ChartGenerator
from utils.config import MARKETS, PAGE_TITLE, PAGE_ICON, LAYOUT, DEFAULT_ROLLING_WINDOW
from pathlib import Path
from utils import yf_cache_build as yfcache

from utils.helpers import validate_date_range_with_message, display_metric_card

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    """Main application function"""

    # Title and description
    st.title(" Indian Stock Market Sector Analyzer")
    st.markdown("""
    Analyze **Indian stock market trends** across different sectors (NSE/BSE) to identify which sectors are currently
    trending bullish or bearish using machine learning and technical analysis.

    ** Coverage**: 100 Indian stocks across 10 major sectors including Technology, Banking, Energy, FMCG, and more.
    """)

    # Sidebar for input controls
    st.sidebar.header("Analysis Parameters")


    # Market and API Selection
    st.sidebar.subheader(" Indian Stock Market")

    # Fixed market selection - India only
    selected_market = 'INDIA'

    # Show market info
    st.sidebar.info(f" **Market**: {MARKETS[selected_market]['name']}")

    # Fixed API provider - Yahoo Finance only
    selected_api = 'yfinance'

    # No API key required for Yahoo Finance
    api_key = None

    # Initialize components with selected market and API
    try:
        collector = StockDataCollector(market=selected_market, api_provider=selected_api, api_key=api_key)
        processor = StockDataProcessor()
        feature_engineer = FeatureEngineer()
        chart_generator = ChartGenerator()
    except Exception as e:
        st.sidebar.error(f"Error initializing data collector: {e}")
        return

    # Date inputs
    st.sidebar.subheader("Date Range")
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )

    start_date = st.sidebar.date_input(
        "Start Date",
        value=end_date - timedelta(days=30),
        max_value=end_date
    )

    # Sector selection
    st.sidebar.subheader("Sector Selection")
    available_sectors = list(collector.sector_stocks.keys())
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors to Analyze",
        options=available_sectors,
        default=available_sectors[:2] if len(available_sectors) >= 4 else available_sectors,  # Default to first 4 sectors
        help=f"Available sectors for {MARKETS[selected_market]['name']}"
    )

    # Custom stocks input (optional)
    st.sidebar.subheader("Custom Stock Symbols (Optional)")
    use_custom_stocks = st.sidebar.checkbox("Use custom stock symbols")
    custom_stocks = {}

    if use_custom_stocks:
        for sector in selected_sectors:
            stocks_input = st.sidebar.text_input(
                f"{sector} stocks (comma-separated)",
                placeholder="AAPL,MSFT,GOOGL"
            )
            if stocks_input:
                custom_stocks[sector] = [s.strip().upper() for s in stocks_input.split(',')]

    # Offline backup cache controls
    st.sidebar.subheader("Backup Cache (Offline)")
    if st.sidebar.button(" Preload NSE cache (1 year)"):
        with st.spinner("Downloading 1 year of data for all NSE symbols and saving to data/yf_cache ..."):
            try:
                output_dir = Path("data/yf_cache")
                symbols = yfcache.get_all_symbols_from_config()
                batch_df = yfcache.batch_download(symbols, period="1y")
                pb = st.sidebar.progress(0.0)
                downloaded = fallback = used_cache = failed = 0
                for i, sym in enumerate(symbols):
                    sym_df = yfcache.extract_symbol_from_batch(batch_df, sym) if batch_df is not None else pd.DataFrame(columns=yfcache.REQUIRED_COLS)
                    if sym_df.empty:
                        per_df = yfcache.fetch_per_symbol(sym, period="1y")
                        if not per_df.empty:
                            sym_df = per_df
                            fallback += 1
                    if sym_df.empty:
                        cached = yfcache.load_symbol_cache(sym, output_dir)
                        if cached is not None and not cached.empty:
                            sym_df = yfcache.standardize_df(cached, sym)
                            used_cache += 1
                    if sym_df.empty:
                        failed += 1
                    else:
                        yfcache.save_symbol_cache(sym_df, sym, output_dir)
                        downloaded += 1
                    pb.progress((i + 1) / len(symbols))
                st.success(f"Preload finished. Saved/updated {downloaded} symbols. Fallback per-symbol: {fallback}. Used existing cache: {used_cache}. Failed: {failed}.")
            except Exception as e:
                st.error(f"Preload failed: {e}")

    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    rolling_window = st.sidebar.number_input(
        "Rolling Window Size (days)",
        min_value=3,
        max_value=30,
        value=DEFAULT_ROLLING_WINDOW
    )

    # Output format selection
    st.sidebar.subheader("Output Format")
    output_format = st.sidebar.radio(
        "Select visualization type",
        options=["Bar Charts", "Line Charts", "Data Tables", "Ranked Lists", "All Formats"]
    )

    # Validate inputs
    if not selected_sectors:
        st.warning("Please select at least one sector to analyze.")
        return

    # Validate date range with detailed feedback
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.min.time())
    is_valid, error_message = validate_date_range_with_message(start_datetime, end_datetime)

    if not is_valid:
        st.error(f"Date range error: {error_message}")
        return

    # Data collection and processing
    if st.sidebar.button("Run Analysis", type="primary"):

        # Clear any previous cache
        try:
            st.cache_data.clear()
        except:
            pass

        with st.spinner("Collecting and processing data..."):
            try:
                # Collect data with better error handling
                st.info(" Fetching market data...")
                raw_data = collector.fetch_sector_data(
                    sectors=selected_sectors,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    custom_stocks=custom_stocks if use_custom_stocks else None
                )

                # Check if data was collected
                if raw_data is None or raw_data.empty:
                    st.error(" No data could be fetched. Please try:")
                    st.error("• Selecting fewer sectors")
                    st.error("• Refreshing the page")
                    return

                st.success(f" Data collected: {raw_data.shape[0]} records for {raw_data['Symbol'].nunique()} stocks")

                # Process data (includes rolling statistics internally)
                st.info(" Processing data...")
                processor.rolling_window = rolling_window
                processed_data = processor.process_raw_data(raw_data)

                # Feature engineering
                st.info(" Calculating technical indicators...")
                processed_data = feature_engineer.calculate_moving_averages(processed_data)
                processed_data = feature_engineer.calculate_momentum_indicators(processed_data)
                processed_data = feature_engineer.calculate_volatility_measures(processed_data)
                processed_data = feature_engineer.calculate_trend_signals(processed_data)

                # Sector-level aggregation
                st.info(" Aggregating sector data...")
                sector_data = processor.aggregate_by_sector(processed_data)
                sector_data = feature_engineer.create_sector_features(sector_data)

                # Store in session state
                st.session_state.processed_data = processed_data
                st.session_state.sector_data = sector_data
                st.session_state.data_loaded = True

                st.success(" Data collection and processing completed!")

            except Exception as e:
                error_msg = str(e)
                st.error(f" Error during data processing: {error_msg}")

                # Provide specific guidance based on error type
                if "No data could be fetched" in error_msg:
                    st.error("**Troubleshooting Steps:**")
                    st.error("1. Try a shorter date range (1-2 weeks)")
                    st.error("2. Refresh the page and try again")
                elif "rate limit" in error_msg.lower():
                    st.error("**API Rate Limit Detected:**")
                    st.error("• Wait 10-15 minutes before trying again")
                    st.error("• Use shorter date ranges")
                else:
                    st.error("**General Troubleshooting:**")
                    st.error("• Try refreshing the page")
                    st.error("• Use different date ranges")

                # Show technical details in expander
                with st.expander("🔍 Technical Details"):
                    st.code(error_msg)

                return

    # Display results if data is loaded
    if st.session_state.data_loaded:

        # Get data from session state
        sector_data = st.session_state.sector_data

        # Calculate sector rankings
        sector_rankings = processor.calculate_sector_rankings(sector_data)

        # Train ML model
        with st.spinner("Training machine learning model..."):
            try:
                classifier = SectorTrendClassifier(model_type='random_forest')

                # Prepare ML features
                ml_features = feature_engineer.extract_ml_features(sector_data)

                # Train model
                metrics = classifier.train(ml_features)

                # Make predictions
                predictions = classifier.predict(ml_features.groupby('Sector').last().reset_index())

                st.session_state.classifier = classifier
                st.session_state.predictions = predictions
                st.session_state.metrics = metrics
                st.session_state.model_trained = True

            except Exception as e:
                st.warning(f"ML model training failed: {str(e)}. Continuing with basic analysis.")
                st.session_state.model_trained = False

        # Display summary metrics
        st.header(" Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_metric_card(
                "Total Sectors",
                str(len(sector_rankings))
            )

        with col2:
            best_sector = sector_rankings.iloc[0]['Sector']
            display_metric_card(
                "Best Performing",
                best_sector
            )

        with col3:
            avg_volatility = sector_rankings['Avg_Volatility'].mean()
            display_metric_card(
                "Average Volatility",
                f"{avg_volatility:.4f}"
            )

        with col4:
            if st.session_state.model_trained:
                test_accuracy = st.session_state.metrics['test_accuracy']
                display_metric_card(
                    "Model Accuracy",
                    f"{test_accuracy:.2%}"
                )

        # Display visualizations based on selected format
        if output_format in ["Bar Charts", "All Formats"]:
            st.header(" Sector Performance Rankings")
            fig_bar = chart_generator.create_sector_performance_chart(sector_rankings, 'bar')
            st.plotly_chart(fig_bar, width='stretch')

        if output_format in ["Line Charts", "All Formats"]:
            st.header(" Sector Momentum Over Time")
            fig_momentum = chart_generator.create_sector_momentum_chart(sector_data, selected_sectors)
            st.plotly_chart(fig_momentum, width='stretch')

        if output_format in ["Data Tables", "All Formats"]:
            st.header(" Sector Rankings Table")
            chart_generator.display_sector_table(sector_rankings)

        if output_format in ["Ranked Lists", "All Formats"]:
            st.header(" Top Performing Sectors")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Top 3 Sectors")
                for i, (_, row) in enumerate(sector_rankings.head(3).iterrows()):
                    st.write(f"{i+1}. **{row['Sector']}** - Score: {row['Performance_Score']:.4f}")

            with col2:
                if st.session_state.model_trained:
                    st.subheader(" ML Predictions")
                    predictions = st.session_state.predictions
                    for _, row in predictions.iterrows():
                        trend_emoji = "🟢" if row['Predicted_Trend'] == 'Bullish' else "🔴" if row['Predicted_Trend'] == 'Bearish' else "🟡"
                        st.write(f"{trend_emoji} **{row['Sector']}**: {row['Predicted_Trend']} ({row['Prediction_Confidence']:.2f})")

        # Additional visualizations for "All Formats"
        if output_format == "All Formats":

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk vs Return Analysis")
                fig_scatter = chart_generator.create_volatility_vs_return_scatter(sector_rankings)
                st.plotly_chart(fig_scatter, width='stretch')

            with col2:
                if st.session_state.model_trained:
                    st.subheader("Trend Predictions")
                    predictions = st.session_state.predictions
                    fig_pie = chart_generator.create_trend_prediction_chart(predictions)
                    st.plotly_chart(fig_pie, width='stretch')

            # Feature importance
            if st.session_state.model_trained:
                st.subheader(" Model Feature Importance")
                try:
                    importance_df = st.session_state.classifier.get_feature_importance()
                    fig_importance = chart_generator.create_feature_importance_chart(importance_df)
                    st.plotly_chart(fig_importance, width='stretch')
                except Exception as e:
                    st.info("Feature importance not available for this model type.")

        # ML Model Performance Analysis
        if st.session_state.model_trained:
            st.header(" Machine Learning Model Analysis")

            # Create tabs for different ML visualizations
            ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
                " Performance Overview",
                " Training Analysis",
                " Detailed Metrics",
                " Model Insights"
            ])

            with ml_tab1:
                # Performance comparison
                perf_fig = chart_generator.create_train_test_comparison_chart(
                    st.session_state.metrics
                )
                st.plotly_chart(perf_fig, width='stretch')

                # CV analysis
                cv_fig = chart_generator.create_cv_scores_visualization(
                    st.session_state.metrics['cv_scores']
                )
                st.plotly_chart(cv_fig, width='stretch')

            with ml_tab2:
                # Learning curves (if available)
                if st.session_state.metrics.get('learning_curves'):
                    lc_data = st.session_state.metrics['learning_curves']
                    lc_fig = chart_generator.create_learning_curves(
                        lc_data['train_sizes'],
                        lc_data['train_scores'],
                        lc_data['val_scores']
                    )
                    st.plotly_chart(lc_fig, width='stretch')
                else:
                    st.info("Learning curves not available - enable in model training for detailed analysis")

            with ml_tab3:
                # Confusion matrix
                cm_fig = chart_generator.create_confusion_matrix_heatmap(
                    st.session_state.metrics['y_test'],
                    st.session_state.metrics['y_pred'],
                    st.session_state.metrics['class_names']
                )
                st.plotly_chart(cm_fig, width='stretch')

                # Classification report
                cr_fig = chart_generator.create_classification_report_chart(
                    st.session_state.metrics['y_test'],
                    st.session_state.metrics['y_pred'],
                    st.session_state.metrics['class_names']
                )
                st.plotly_chart(cr_fig, width='stretch')

            with ml_tab4:
                # Performance dashboard
                dashboard_fig = chart_generator.create_ml_performance_dashboard(
                    st.session_state.metrics
                )
                st.plotly_chart(dashboard_fig, width='stretch')

if __name__ == "__main__":
    main()
