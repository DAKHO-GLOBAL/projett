"""
Observation space module for the trading environment.

This module provides functionality to define and preprocess the observation space
for the trading environment, which is the state representation that the agent sees.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ObservationSpace:
    """Class for defining and processing the observation space for the trading environment."""
    
    def __init__(
        self,
        config: Dict,
        window_size: int = 20,
        features: Optional[List[str]] = None,
        include_account_info: bool = True,
        include_indicators: bool = True,
        normalize: bool = True
    ):
        """
        Initialize the observation space.
        
        Args:
            config: Configuration dictionary
            window_size: Number of time steps to include in market data
            features: List of features to include in the observation
            include_account_info: Whether to include account information
            include_indicators: Whether to include technical indicators
            normalize: Whether to normalize features
        """
        self.config = config
        self.window_size = window_size
        self.features = features
        self.include_account_info = include_account_info
        self.include_indicators = include_indicators
        self.normalize = normalize
        
        # Define the observation space structure
        self._define_spaces()
    
    def _define_spaces(self):
        """Define the structure of the observation space."""
        # Number of features in the state
        if self.features is not None:
            n_features = len(self.features)
        else:
            # Default to OHLCV if features not specified
            n_features = 5
        
        # Market data shape (time window x features)
        market_shape = (self.window_size, n_features)
        
        # Account information: [balance, equity, position, unrealized_pnl, realized_pnl]
        account_shape = (5,) if self.include_account_info else (0,)
        
        # Define the observation space as a Dict space
        spaces_dict = {
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=market_shape)
        }
        
        if self.include_account_info:
            spaces_dict['account'] = spaces.Box(low=-np.inf, high=np.inf, shape=account_shape)
        
        self.observation_space = spaces.Dict(spaces_dict)
    
    def preprocess_observation(
        self,
        market_data: pd.DataFrame,
        account_info: Optional[Dict] = None
    ) -> Dict:
        """
        Preprocess raw data into the observation format expected by the agent.
        
        Args:
            market_data: DataFrame containing market data
            account_info: Dictionary containing account information
            
        Returns:
            Dict: Processed observation
        """
        # Extract feature columns if specified
        if self.features is not None:
            # Check if all requested features exist in the data
            missing_features = [f for f in self.features if f not in market_data.columns]
            if missing_features:
                logger.warning(f"Missing features in data: {missing_features}")
                # Use available features
                available_features = [f for f in self.features if f in market_data.columns]
                market_window = market_data[available_features].values
            else:
                market_window = market_data[self.features].values
        else:
            # Default to standard OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in ohlcv_cols if col in market_data.columns]
            market_window = market_data[available_cols].values
        
        # Ensure we have the right window size
        if len(market_window) > self.window_size:
            # Take the most recent window
            market_window = market_window[-self.window_size:]
        elif len(market_window) < self.window_size:
            # Pad with zeros if not enough data
            padding = np.zeros((self.window_size - len(market_window), market_window.shape[1]))
            market_window = np.vstack((padding, market_window))
        
        # Normalize market data if requested
        if self.normalize:
            market_window = self._normalize_market_data(market_window)
        
        # Create the observation dictionary
        observation = {
            'market': market_window
        }
        
        # Add account information if requested and available
        if self.include_account_info and account_info is not None:
            account_array = np.array([
                account_info.get('balance', 0) / account_info.get('initial_balance', 1),
                account_info.get('equity', 0) / account_info.get('initial_balance', 1),
                account_info.get('position', 0),
                account_info.get('unrealized_pnl', 0) / max(account_info.get('balance', 1), 1),
                account_info.get('realized_pnl', 0) / max(account_info.get('initial_balance', 1), 1)
            ])
            observation['account'] = account_array
        
        return observation
    
    def _normalize_market_data(self, market_window: np.ndarray) -> np.ndarray:
        """
        Normalize market data to improve model training.
        
        Args:
            market_window: Market data array
            
        Returns:
            np.ndarray: Normalized market data
        """
        # Avoid division by zero
        epsilon = 1e-8
        
        # Create a copy to avoid modifying the original
        normalized = market_window.copy()
        
        # Normalize each feature column independently
        for i in range(normalized.shape[1]):
            # Get the feature column
            feature_col = normalized[:, i]
            
            # Skip if column is all zeros
            if np.all(feature_col == 0):
                continue
            
            # Compute min and max for the column
            col_min = np.min(feature_col)
            col_max = np.max(feature_col)
            
            # Check if min and max are different
            if abs(col_max - col_min) > epsilon:
                # Min-max normalization to [0, 1]
                normalized[:, i] = (feature_col - col_min) / (col_max - col_min)
            else:
                # If min and max are the same, set to 0.5
                normalized[:, i] = 0.5
        
        return normalized
    
    def get_observation_shape(self) -> Dict[str, Tuple]:
        """
        Get the shape of the observation space.
        
        Returns:
            Dict[str, Tuple]: Dictionary mapping space keys to their shapes
        """
        return {
            key: space.shape
            for key, space in self.observation_space.spaces.items()
        }
    
    def get_observation_size(self) -> int:
        """
        Get the total size (number of elements) in the observation space.
        
        Returns:
            int: Total number of elements
        """
        total_size = 0
        for space in self.observation_space.spaces.values():
            total_size += np.prod(space.shape)
        return total_size
    
    def get_flattened_observation(self, observation: Dict) -> np.ndarray:
        """
        Flatten a structured observation into a 1D array.
        
        This is useful for certain RL algorithms that expect flat observations.
        
        Args:
            observation: Structured observation dictionary
            
        Returns:
            np.ndarray: Flattened observation
        """
        # Flatten each part of the observation
        flattened_parts = []
        
        if 'market' in observation:
            flattened_parts.append(observation['market'].flatten())
        
        if 'account' in observation:
            flattened_parts.append(observation['account'].flatten())
        
        # Concatenate all parts
        return np.concatenate(flattened_parts)
    
    def unflatten_observation(self, flat_observation: np.ndarray) -> Dict:
        """
        Convert a flat observation back to the structured format.
        
        Args:
            flat_observation: Flattened observation array
            
        Returns:
            Dict: Structured observation
        """
        observation = {}
        start_idx = 0
        
        # Restore market data
        if 'market' in self.observation_space.spaces:
            market_shape = self.observation_space.spaces['market'].shape
            market_size = np.prod(market_shape)
            market_data = flat_observation[start_idx:start_idx + market_size].reshape(market_shape)
            observation['market'] = market_data
            start_idx += market_size
        
        # Restore account information
        if 'account' in self.observation_space.spaces:
            account_shape = self.observation_space.spaces['account'].shape
            account_size = np.prod(account_shape)
            account_data = flat_observation[start_idx:start_idx + account_size].reshape(account_shape)
            observation['account'] = account_data
        
        return observation