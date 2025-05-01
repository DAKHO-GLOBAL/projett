"""
Data preprocessing module for preparing market data for training and inference.

This module provides functionality to preprocess raw market data, including
cleaning, normalization, and feature engineering, to make it suitable for use
in the reinforcement learning trading system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """Class for preprocessing raw market data."""
    
    def __init__(self, config: Dict):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters.
        """
        self.config = config
        
        # Initialize preprocessing parameters
        self.feature_window = config['data'].get('feature_window', 20)
        
        # Initialize scalers
        self.price_scaler = None
        self.volume_scaler = None
        self.indicator_scalers = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw market data by handling missing values, duplicates, and outliers.
        
        Args:
            df: Raw market data DataFrame with OHLCV columns
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check for and handle missing values
        if df.isnull().any().any():
            logger.info(f"Found {df.isnull().sum().sum()} missing values, filling...")
            
            # Forward fill most missing values
            df = df.ffill()
            
            # For any remaining NaNs at the beginning, backward fill
            df = df.bfill()
        
        # Remove duplicate indices
        if df.index.duplicated().any():
            logger.info(f"Found {df.index.duplicated().sum()} duplicate timestamps, removing...")
            df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index to ensure chronological order
        df = df.sort_index()
        
        # Handle potential outliers in price data using Median Absolute Deviation (MAD)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                
                # Define outlier threshold (3 times the MAD)
                threshold = 3 * mad
                
                # Identify outliers
                outliers = np.abs(df[col] - median) > threshold
                
                if outliers.any():
                    logger.info(f"Found {outliers.sum()} outliers in {col}, replacing with median...")
                    # Replace outliers with the median value
                    df.loc[outliers, col] = median
        
        return df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        fit_scalers: bool = True,
        columns_to_scale: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Normalize numerical features to improve model training.
        
        Args:
            df: DataFrame with features to normalize
            fit_scalers: Whether to fit new scalers (True) or use existing ones (False)
            columns_to_scale: Dictionary mapping scaler types to lists of column names
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        df = df.copy()
        
        # Default columns to scale if not provided
        if columns_to_scale is None:
            columns_to_scale = {
                'price': ['open', 'high', 'low', 'close'],
                'volume': ['volume']
            }
        
        # Normalize price columns
        if 'price' in columns_to_scale and len(columns_to_scale['price']) > 0:
            price_cols = [col for col in columns_to_scale['price'] if col in df.columns]
            
            if price_cols:
                if fit_scalers or self.price_scaler is None:
                    self.price_scaler = MinMaxScaler()
                    scaled_prices = self.price_scaler.fit_transform(df[price_cols])
                else:
                    scaled_prices = self.price_scaler.transform(df[price_cols])
                
                # Replace original values with scaled values
                for i, col in enumerate(price_cols):
                    df[f"{col}_norm"] = scaled_prices[:, i]
        
        # Normalize volume columns
        if 'volume' in columns_to_scale and len(columns_to_scale['volume']) > 0:
            volume_cols = [col for col in columns_to_scale['volume'] if col in df.columns]
            
            if volume_cols:
                if fit_scalers or self.volume_scaler is None:
                    self.volume_scaler = StandardScaler()
                    scaled_volumes = self.volume_scaler.fit_transform(df[volume_cols].values.reshape(-1, len(volume_cols)))
                else:
                    scaled_volumes = self.volume_scaler.transform(df[volume_cols].values.reshape(-1, len(volume_cols)))
                
                # Replace original values with scaled values
                for i, col in enumerate(volume_cols):
                    df[f"{col}_norm"] = scaled_volumes[:, i]
        
        # Normalize other indicator columns if present
        if 'indicators' in columns_to_scale and len(columns_to_scale['indicators']) > 0:
            indicator_cols = [col for col in columns_to_scale['indicators'] if col in df.columns]
            
            for col in indicator_cols:
                if fit_scalers or col not in self.indicator_scalers:
                    # Use StandardScaler for most indicators
                    self.indicator_scalers[col] = StandardScaler()
                    df[f"{col}_norm"] = self.indicator_scalers[col].fit_transform(
                        df[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    df[f"{col}_norm"] = self.indicator_scalers[col].transform(
                        df[col].values.reshape(-1, 1)
                    ).flatten()
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate price returns over various time periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for return calculation
            
        Returns:
            pd.DataFrame: DataFrame with added return columns
        """
        df = df.copy()
        
        # Calculate simple returns
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Calculate log returns
        for period in periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        return df
    
    def create_rolling_windows(
        self,
        df: pd.DataFrame,
        window_size: int = None,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Create rolling window features for sequential data processing.
        
        Args:
            df: DataFrame with features
            window_size: Size of the rolling window
            feature_columns: List of feature columns to include in windows
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: 
                - 3D array of shape (n_samples, window_size, n_features)
                - DataFrame with timestamps for each window
        """
        if window_size is None:
            window_size = self.feature_window
        
        if feature_columns is None:
            # Default to using normalized columns
            feature_columns = [col for col in df.columns if col.endswith('_norm')]
        
        # Filter dataframe to only include feature columns
        feature_df = df[feature_columns]
        
        # Create empty array for the windows
        n_samples = len(df) - window_size + 1
        n_features = len(feature_columns)
        windows = np.zeros((n_samples, window_size, n_features))
        
        # Fill the windows array
        for i in range(n_samples):
            windows[i] = feature_df.iloc[i:i+window_size].values
        
        # Create a DataFrame with timestamps for each window
        # Each window is identified by its last timestamp
        window_times = df.index[window_size-1:].to_frame()
        
        return windows, window_times
    
    def prepare_data_for_training(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        training_ratio: float = 0.8
    ) -> Dict:
        """
        Prepare data for training, including cleaning, normalization, and splitting.
        
        Args:
            df: Raw market data DataFrame
            window_size: Size of the rolling window for features
            training_ratio: Ratio of data to use for training (vs. validation)
            
        Returns:
            Dict: Dictionary containing processed training and validation data
        """
        # Set default window size if not provided
        if window_size is None:
            window_size = self.feature_window
        
        # Clean the data
        cleaned_df = self.clean_data(df)
        
        # Calculate returns
        returns_df = self.calculate_returns(cleaned_df)
        
        # Normalize the data
        normalized_df = self.normalize_data(returns_df)
        
        # Drop rows with NaNs that might have been introduced by calculations
        normalized_df = normalized_df.dropna()
        
        # Create rolling windows
        windows, window_times = self.create_rolling_windows(normalized_df, window_size)
        
        # Calculate split index
        split_idx = int(len(windows) * training_ratio)
        
        # Split data into training and validation sets
        X_train = windows[:split_idx]
        X_val = windows[split_idx:]
        
        # Get the corresponding times
        train_times = window_times.iloc[:split_idx]
        val_times = window_times.iloc[split_idx:]
        
        # Combine everything into a dictionary
        data_dict = {
            'X_train': X_train,
            'X_val': X_val,
            'train_times': train_times,
            'val_times': val_times,
            'feature_columns': normalized_df.columns.tolist(),
            'original_df': df,
            'processed_df': normalized_df
        }
        
        logger.info(f"Prepared data for training with {len(X_train)} training samples "
                    f"and {len(X_val)} validation samples")
        
        return data_dict