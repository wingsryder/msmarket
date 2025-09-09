"""
Visualization components for the Multi-Stock Sector Trend Analyzer
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional
from utils.helpers import format_percentage


class ChartGenerator:
    """
    Generates various charts and visualizations for sector analysis
    """
    
    def __init__(self):
        self.color_scheme = {
            'Bullish': '#00ff00',
            'Neutral': '#ffff00', 
            'Bearish': '#ff0000',
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#4682B4'
        }
    
    def create_sector_performance_chart(self, sector_rankings: pd.DataFrame, 
                                      chart_type: str = 'bar') -> go.Figure:
        """
        Create sector performance comparison chart.
        
        Args:
            sector_rankings: Dataframe with sector rankings and scores
            chart_type: Type of chart ('bar', 'horizontal_bar')
            
        Returns:
            go.Figure: Plotly figure object
        """
        if chart_type == 'bar':
            fig = px.bar(
                sector_rankings,
                x='Sector',
                y='Performance_Score',
                title='Sector Performance Rankings',
                color='Performance_Score',
                color_continuous_scale='RdYlGn',
                text='Performance_Score'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            
        elif chart_type == 'horizontal_bar':
            fig = px.bar(
                sector_rankings,
                x='Performance_Score',
                y='Sector',
                orientation='h',
                title='Sector Performance Rankings',
                color='Performance_Score',
                color_continuous_scale='RdYlGn',
                text='Performance_Score'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_x=0.5
        )
        
        return fig
    
    def create_trend_prediction_chart(self, predictions_df: pd.DataFrame) -> go.Figure:
        """
        Create trend prediction visualization.
        
        Args:
            predictions_df: Dataframe with trend predictions
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Count predictions by trend
        trend_counts = predictions_df['Predicted_Trend'].value_counts()
        
        fig = px.pie(
            values=trend_counts.values,
            names=trend_counts.index,
            title='Sector Trend Distribution',
            color=trend_counts.index,
            color_discrete_map=self.color_scheme
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            title_x=0.5
        )
        
        return fig
    
    def create_sector_momentum_chart(self, sector_data: pd.DataFrame, 
                                   selected_sectors: List[str]) -> go.Figure:
        """
        Create sector momentum time series chart.
        
        Args:
            sector_data: Time series data for sectors
            selected_sectors: List of sectors to display
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        for sector in selected_sectors:
            sector_subset = sector_data[sector_data['Sector'] == sector]
            
            if not sector_subset.empty:
                fig.add_trace(go.Scatter(
                    x=sector_subset['Date'],
                    y=sector_subset['Sector_Momentum'],
                    mode='lines+markers',
                    name=sector,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title='Sector Momentum Over Time',
            xaxis_title='Date',
            yaxis_title='Momentum Score',
            height=500,
            hovermode='x unified',
            title_x=0.5
        )
        
        return fig
    
    def create_volatility_vs_return_scatter(self, sector_rankings: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot of volatility vs returns.
        
        Args:
            sector_rankings: Sector data with volatility and return metrics
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.scatter(
            sector_rankings,
            x='Avg_Volatility',
            y='Sector_Momentum',
            size='Performance_Score',
            color='Performance_Score',
            hover_name='Sector',
            title='Risk vs Return Analysis',
            labels={
                'Avg_Volatility': 'Volatility (Risk)',
                'Sector_Momentum': 'Average Return'
            },
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_df: pd.DataFrame) -> go.Figure:
        """
        Create feature importance chart for ML model.
        
        Args:
            importance_df: Dataframe with feature importance scores
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.bar(
            importance_df.head(10),  # Top 10 features
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance (ML Model)',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            title_x=0.5
        )
        
        return fig
    
    def create_sector_heatmap(self, sector_data: pd.DataFrame, 
                            metric: str = 'Performance_Score') -> go.Figure:
        """
        Create heatmap of sector performance over time.
        
        Args:
            sector_data: Time series sector data
            metric: Metric to display in heatmap
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Pivot data for heatmap
        pivot_data = sector_data.pivot(index='Sector', columns='Date', values=metric)
        
        fig = px.imshow(
            pivot_data,
            title=f'Sector {metric} Heatmap Over Time',
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        
        fig.update_layout(
            height=400,
            title_x=0.5,
            xaxis_title='Date',
            yaxis_title='Sector'
        )
        
        return fig
    
    def create_prediction_confidence_chart(self, predictions_df: pd.DataFrame) -> go.Figure:
        """
        Create chart showing prediction confidence levels.
        
        Args:
            predictions_df: Dataframe with predictions and confidence scores
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.bar(
            predictions_df,
            x='Sector',
            y='Prediction_Confidence',
            color='Predicted_Trend',
            title='Prediction Confidence by Sector',
            color_discrete_map=self.color_scheme,
            text='Prediction_Confidence'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            title_x=0.5,
            yaxis_title='Confidence Score'
        )
        
        return fig
    
    def create_summary_metrics_display(self, sector_rankings: pd.DataFrame) -> Dict:
        """
        Create summary metrics for display.
        
        Args:
            sector_rankings: Sector rankings data
            
        Returns:
            Dict: Summary metrics
        """
        metrics = {
            'total_sectors': len(sector_rankings),
            'best_performing_sector': sector_rankings.iloc[0]['Sector'],
            'best_performance_score': sector_rankings.iloc[0]['Performance_Score'],
            'avg_volatility': sector_rankings['Avg_Volatility'].mean(),
            'avg_momentum': sector_rankings['Sector_Momentum'].mean()
        }
        
        return metrics
    
    def display_sector_table(self, sector_rankings: pd.DataFrame):
        """
        Display formatted sector rankings table.
        
        Args:
            sector_rankings: Sector rankings data
        """
        # Format the dataframe for display
        display_df = sector_rankings.copy()
        
        # Select and rename columns for display
        columns_to_show = {
            'Sector': 'Sector',
            'Performance_Score': 'Performance Score',
            'Sector_Momentum': 'Momentum',
            'Avg_Volatility': 'Volatility',
            'Performance_Rank': 'Rank'
        }
        
        display_df = display_df[list(columns_to_show.keys())].rename(columns=columns_to_show)
        
        # Format numeric columns
        display_df['Performance Score'] = display_df['Performance Score'].round(4)
        display_df['Momentum'] = display_df['Momentum'].apply(lambda x: format_percentage(x))
        display_df['Volatility'] = display_df['Volatility'].round(4)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
