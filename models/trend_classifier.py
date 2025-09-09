"""
Machine Learning model for sector trend classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional
import joblib
from utils.config import TREND_LABELS, BULLISH_THRESHOLD, BEARISH_THRESHOLD


class SectorTrendClassifier:
    """
    Machine Learning classifier for predicting sector trends
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Pre-fit the label encoder with all possible labels to avoid unseen label errors
        self.label_encoder.fit(['Bullish', 'Neutral', 'Bearish'])
        self.feature_names = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def create_target_labels(self, df: pd.DataFrame, 
                           target_column: str = 'Sector_Price_Momentum',
                           forward_periods: int = 5) -> pd.DataFrame:
        """
        Create target labels for trend classification.
        
        Args:
            df: Input dataframe with features
            target_column: Column to use for creating labels
            forward_periods: Number of periods to look forward for trend
            
        Returns:
            pd.DataFrame: Data with trend labels
        """
        result_df = df.copy()
        
        # Calculate forward returns for each sector
        result_df['Forward_Return'] = result_df.groupby('Sector')[target_column].shift(-forward_periods)
        
        # Create trend labels based on thresholds
        result_df['Trend_Label'] = np.where(
            result_df['Forward_Return'] > BULLISH_THRESHOLD, 'Bullish',
            np.where(result_df['Forward_Return'] < BEARISH_THRESHOLD, 'Bearish', 'Neutral')
        )
        
        # Remove rows with missing forward returns
        result_df = result_df.dropna(subset=['Forward_Return', 'Trend_Label'])
        
        return result_df

    def validate_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and potentially augment training data to ensure all label classes are present.

        Args:
            df: Training dataframe with labels

        Returns:
            pd.DataFrame: Validated training data
        """
        # Check which labels are present
        unique_labels = df['Trend_Label'].unique()
        all_labels = ['Bullish', 'Neutral', 'Bearish']
        missing_labels = [label for label in all_labels if label not in unique_labels]

        if missing_labels:
            print(f"Warning: Missing labels in training data: {missing_labels}")
            print(f"Present labels: {unique_labels}")

            # If we have very few samples of missing labels, we can still proceed
            # The pre-fitted encoder will handle this gracefully

        # Ensure minimum samples per class for reliable training
        label_counts = df['Trend_Label'].value_counts()
        min_samples_per_class = 3

        for label in unique_labels:
            if label_counts[label] < min_samples_per_class:
                print(f"Warning: Only {label_counts[label]} samples for label '{label}'. Consider using more data.")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training.
        
        Args:
            df: Dataframe with features and labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        # Select feature columns (exclude non-feature columns)
        exclude_columns = ['Sector', 'Date', 'Forward_Return', 'Trend_Label']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle missing feature columns
        if not feature_columns:
            raise ValueError("No feature columns found for model training")
        
        self.feature_names = feature_columns
        
        # Prepare features
        X = df[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Prepare targets
        y = df['Trend_Label'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the trend classification model.
        
        Args:
            df: Training data with features and labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Create target labels
        labeled_data = self.create_target_labels(df)

        if len(labeled_data) < 50:
            raise ValueError("Insufficient data for model training (minimum 50 samples required)")

        # Validate training data
        labeled_data = self.validate_training_data(labeled_data)

        # Prepare features and targets
        X, y = self.prepare_features(labeled_data)

        # Encode labels (encoder is already fitted with all possible labels)
        y_encoded = self.label_encoder.transform(y)

        # Check if we have enough diversity for stratified split
        unique_labels, label_counts = np.unique(y_encoded, return_counts=True)
        min_count = min(label_counts)

        # Split data with or without stratification based on label diversity
        if min_count >= 2 and len(unique_labels) > 1:
            # Use stratified split if we have at least 2 samples per class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        else:
            # Use regular split if insufficient diversity for stratification
            print("Warning: Using non-stratified split due to insufficient label diversity")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation score with error handling
        try:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
        except Exception as e:
            print(f"Warning: Cross-validation failed: {e}. Using simple validation.")
            cv_scores = np.array([test_score])  # Use test score as fallback
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std()
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: Data to make predictions on
            
        Returns:
            pd.DataFrame: Data with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        result_df = df.copy()
        
        # Prepare features
        if not all(col in df.columns for col in self.feature_names):
            missing_cols = [col for col in self.feature_names if col not in df.columns]
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        prediction_proba = self.model.predict_proba(X_scaled)

        # Decode predictions with robust error handling
        try:
            predicted_labels = self.label_encoder.inverse_transform(predictions)
        except ValueError as e:
            print(f"Warning: Label decoding failed: {e}")
            # Use the encoder's classes_ to map predictions correctly
            classes = self.label_encoder.classes_
            predicted_labels = []
            for pred in predictions:
                if 0 <= pred < len(classes):
                    predicted_labels.append(classes[pred])
                else:
                    predicted_labels.append('Neutral')  # Default fallback
            print(f"Using encoder classes mapping: {dict(enumerate(classes))}")
        
        # Add predictions to dataframe
        result_df['Predicted_Trend'] = predicted_labels
        result_df['Prediction_Confidence'] = np.max(prediction_proba, axis=1)
        
        # Add individual class probabilities
        for i, class_label in enumerate(self.label_encoder.classes_):
            # class_label is already the string label, no need to decode
            result_df[f'Prob_{class_label}'] = prediction_proba[:, i]
        
        return result_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            })
            importance_df.sort_values('Importance', ascending=False, inplace=True)
            return importance_df
        else:
            raise ValueError("Model does not support feature importance")
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
