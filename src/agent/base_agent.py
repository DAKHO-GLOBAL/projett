"""
Base agent module for reinforcement learning trading.

This module defines the abstract base class for all reinforcement learning agents
used in the trading system. It provides a common interface and shared functionality
that all specific agent implementations should follow.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all reinforcement learning agents."""
    
    def __init__(
        self,
        config: Dict,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu',
        model_name: str = 'default_model'
    ):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to run the model on ('cpu' or 'cuda')
            model_name: Name for saving/loading the model
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.model_name = model_name
        
        # Create model save directory if it doesn't exist
        self.model_dir = os.path.join(config['paths']['models'], 'saved')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize training metrics
        self.metrics = {
            'train_rewards': [],
            'eval_rewards': [],
            'train_losses': [],
            'learning_steps': 0
        }
        
        logger.info(f"Initialized base agent with state_dim={state_dim}, action_dim={action_dim}, device={device}")
    
    @abstractmethod
    def select_action(self, state: Any, evaluate: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            evaluate: Whether to use exploration or deterministic action
            
        Returns:
            np.ndarray: Selected action
        """
        pass
    
    @abstractmethod
    def train(self, replay_buffer: Any, batch_size: int = 64) -> Dict:
        """
        Train the agent using experiences from the replay buffer.
        
        Args:
            replay_buffer: Buffer containing experiences
            batch_size: Number of experiences to sample for training
            
        Returns:
            Dict: Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Optional path to save the model to
            
        Returns:
            str: Path where the model was saved
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def get_default_save_path(self) -> str:
        """
        Get the default path for saving the model.
        
        Returns:
            str: Default save path
        """
        return os.path.join(self.model_dir, f"{self.model_name}.pt")
    
    def update_metrics(self, new_metrics: Dict) -> None:
        """
        Update the agent's training metrics.
        
        Args:
            new_metrics: Dictionary with new metric values to add
        """
        for key, value in new_metrics.items():
            if key == 'learning_steps':
                # Handle learning_steps as a scalar value, not a list
                self.metrics['learning_steps'] = value
            elif key in self.metrics:
                if isinstance(self.metrics[key], list):
                    # If the metric is already a list, append or extend
                    if isinstance(value, list):
                        self.metrics[key].extend(value)
                    else:
                        self.metrics[key].append(value)
                else:
                    # Convert to list if current value is not a list
                    if isinstance(value, list):
                        self.metrics[key] = [self.metrics[key]] + value
                    else:
                        self.metrics[key] = [self.metrics[key], value]
            else:
                # Initialize new metric
                if isinstance(value, list):
                    self.metrics[key] = value
                else:
                    self.metrics[key] = [value]
    
    def get_metrics(self) -> Dict:
        """
        Get the agent's training metrics.
        
        Returns:
            Dict: Dictionary with all training metrics
        """
        return self.metrics