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
            width='stretch',
            hide_index=True
        )


    def create_train_test_comparison_chart(self, metrics):
        """Compare training and testing performance"""
        fig = go.Figure()

        categories = ['Training Accuracy', 'Test Accuracy', 'CV Mean Accuracy']
        values = [
            metrics['train_accuracy'],
            metrics['test_accuracy'],
            metrics['cv_mean_accuracy']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.2%}' for v in values],
            textposition='auto'
        ))

        fig.update_layout(
            title='Model Performance Comparison',
            yaxis_title='Accuracy Score',
            height=400,
            showlegend=False
        )

        return fig


    def create_cv_scores_visualization(self, cv_scores):
        """Show cross-validation score distribution"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['CV Scores Distribution', 'CV Scores Box Plot']
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=cv_scores, nbinsx=10, name='CV Scores'),
            row=1, col=1
        )

        # Box plot
        fig.add_trace(
            go.Box(y=cv_scores, name='CV Scores', boxpoints='all'),
            row=1, col=2
        )

        fig.update_layout(
            title='Cross-Validation Performance Analysis',
            height=400
        )

        return fig


    def create_learning_curves(self, train_sizes, train_scores, val_scores):
        """Show how model performance improves with more data"""
        fig = go.Figure()

        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores.mean(axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(
                type='data',
                array=train_scores.std(axis=1),
                visible=True
            )
        ))

        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores.mean(axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(
                type='data',
                array=val_scores.std(axis=1),
                visible=True
            )
        ))

        fig.update_layout(
            title='Learning Curves - Performance vs Training Set Size',
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            height=500
        )

        return fig


    def create_confusion_matrix_heatmap(self, y_true, y_pred, class_names):
        """Interactive confusion matrix"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=[[f'{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)'
                   for j in range(len(class_names))]
                  for i in range(len(class_names))],
            texttemplate='%{text}',
            textfont={"size": 12}
        ))

        fig.update_layout(
            title='Confusion Matrix - Predictions vs Actual',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500
        )

        return fig


    def create_classification_report_chart(self, y_true, y_pred, class_names):
        """Visual classification report"""
        from sklearn.metrics import classification_report

        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # Extract metrics for each class
        classes = class_names
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Precision',
            x=classes,
            y=precision,
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            name='Recall',
            x=classes,
            y=recall,
            marker_color='lightgreen'
        ))

        fig.add_trace(go.Bar(
            name='F1-Score',
            x=classes,
            y=f1_score,
            marker_color='lightcoral'
        ))

        fig.update_layout(
            title='Classification Report - Per Class Performance',
            xaxis_title='Classes',
            yaxis_title='Score',
            barmode='group',
            height=500
        )

        return fig


    def create_ml_performance_dashboard(self, metrics):
        """Comprehensive ML metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Accuracy Metrics',
                'Cross-Validation Stability',
                'Overfitting Check',
                'Model Confidence'
            ],
            specs=[
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )

        # Accuracy comparison
        accuracies = ['Train', 'Test', 'CV Mean']
        values = [
            metrics['train_accuracy'],
            metrics['test_accuracy'],
            metrics['cv_mean_accuracy']
        ]

        fig.add_trace(
            go.Bar(x=accuracies, y=values, name='Accuracy'),
            row=1, col=1
        )

        # CV Standard deviation gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics['cv_std_accuracy'],
                title={'text': "CV Std Dev"},
                gauge={
                    'axis': {'range': [None, 0.1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.02], 'color': "lightgreen"},
                        {'range': [0.02, 0.05], 'color': "yellow"},
                        {'range': [0.05, 0.1], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )

        # Overfitting check (train vs test)
        overfitting_gap = metrics['train_accuracy'] - metrics['test_accuracy']
        fig.add_trace(
            go.Scatter(
                x=['Overfitting Gap'],
                y=[overfitting_gap],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red' if overfitting_gap > 0.1 else 'green'
                ),
                name='Gap'
            ),
            row=2, col=1
        )

        fig.update_layout(height=600, title_text="ML Model Performance Dashboard")
        return fig
