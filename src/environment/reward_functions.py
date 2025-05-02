"""
Reward functions for the trading environment.

This module provides various reward functions to be used in the trading
environment. The reward function is one of the most critical components
of a reinforcement learning system as it guides the agent's learning.
"""

import logging
from typing import Dict, List, Optional, Union, Callable

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.metrics import calculate_sharpe_ratio, calculate_sortino_ratio

logger = setup_logger(__name__)

class RewardCalculator:
    """Class for calculating rewards based on trading actions and outcomes."""
    
    def __init__(self, reward_type: str = 'sharpe', window_size: int = 20):
        """
        Initialize the reward calculator.
        
        Args:
            reward_type: Type of reward function to use
            window_size: Window size for rolling calculations
        """
        self.reward_type = reward_type
        self.window_size = window_size
        self.previous_balance = None
        self.previous_position = 0  # Pour suivre la position précédente
        
        # Map reward type to the corresponding calculation function
        self.reward_functions = {
            'pnl': self._calculate_pnl_reward,
            'sharpe': self._calculate_sharpe_reward,
            'sortino': self._calculate_sortino_reward,
            'differential_sharpe': self._calculate_differential_sharpe_reward,
            'risk_adjusted': self._calculate_risk_adjusted_reward,
            'profit_factor': self._calculate_profit_factor_reward,
            'asymmetric': self._calculate_asymmetric_reward,
            'directional': self._calculate_directional_reward,
            'position_duration': self._calculate_position_duration_reward,
            'gold_trading': self._calculate_trading_focused_reward  # Ajoutez cette ligne
        }
    
    def calculate_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate the reward based on the current state and action.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Initialize previous balance if not set
        if self.previous_balance is None:
            self.previous_balance = balance
        
        # Get the appropriate reward function
        reward_func = self.reward_functions.get(
            self.reward_type, self._calculate_pnl_reward
        )
        
        # Calculate reward
        reward = reward_func(
            action=action,
            position=position,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            balance=balance,
            equity=equity,
            prices=prices
        )
        
        # Update previous balance for next calculation
        self.previous_balance = balance
        
        return reward
    
    def _calculate_pnl_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on profit/loss.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the change in balance
        balance_change = balance - self.previous_balance
        
        # Basic reward is the change in balance (realized P&L)
        # plus the change in unrealized P&L
        reward = balance_change
        
        # Normalize the reward to be in a reasonable range
        if balance > 0:
            reward = reward / (balance * 0.01)  # Normalize to percentage of balance
        
        return reward
    


    def _calculate_trading_focused_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward focusing on trading actions and profitability.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate basic P&L component
        balance_change = balance - self.previous_balance
        pnl_reward = balance_change / (self.previous_balance * 0.01) if self.previous_balance > 0 else 0
        
        # Calculate action incentive component
        action_reward = 0.0
        
        # Incentivize taking positions
        if action != 0:
            if position == 0:  # New position
                action_reward += 0.2
            elif action == -position:  # Position reversal (more aggressive)
                action_reward += 0.3
        else:  # Hold action
            if position == 0:  # Staying flat
                action_reward -= 0.1  # Small penalty for staying out of market
        
        # Reward for holding profitable positions
        if position != 0 and unrealized_pnl > 0:
            action_reward += 0.2
        
        # Reward for closing profitable positions
        if realized_pnl > 0 and action == 0 and position == 0 and hasattr(self, 'previous_position') and self.previous_position != 0:
            action_reward += 0.5
        
        # Track previous position for next calculation
        self.previous_position = position
        
        # Market condition awareness for gold
        if len(prices) >= 5:
            # Price momentum awareness
            short_trend = prices[-1] - prices[-3]
            if (action == 1 and short_trend > 0) or (action == -1 and short_trend < 0):
                # Reward trend-following behavior
                action_reward += 0.15
            
            # Volatility awareness - gold is volatile, reward more cautious actions during high volatility
            recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            if recent_volatility > 0.002:  # High volatility threshold for gold
                # Reduce risk during high volatility
                if action == 0 and position != 0:  # Closing positions in high volatility
                    action_reward += 0.2
        
        # Final reward is weighted combination
        reward = 0.6 * pnl_reward + 0.4 * action_reward
        
        return reward
    def _calculate_sharpe_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on Sharpe ratio of recent returns.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the return for this step
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate returns based on position
        position_returns = []
        for i in range(len(returns)):
            # If we were in a position during this period
            if position != 0:
                # Long position: we gain when price goes up
                if position == 1:
                    position_returns.append(returns[i])
                # Short position: we gain when price goes down
                else:
                    position_returns.append(-returns[i])
            else:
                # No position: no return
                position_returns.append(0)
        
        # Convert to numpy array
        position_returns = np.array(position_returns)
        
        # Calculate Sharpe ratio if we have enough data
        if len(position_returns) >= self.window_size:
            # Use recent window for Sharpe calculation
            recent_returns = position_returns[-self.window_size:]
            
            # Calculate Sharpe ratio (annualized)
            sharpe = calculate_sharpe_ratio(recent_returns)
            
            # Scale the reward
            reward = max(min(sharpe, 10), -10)  # Clipping to prevent extreme values
        else:
            # Not enough data, use simple P&L reward
            reward = (equity - self.previous_balance) / (self.previous_balance * 0.01)
        
        return reward
    
    def _calculate_sortino_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on Sortino ratio of recent returns.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the return for this step
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate returns based on position
        position_returns = []
        for i in range(len(returns)):
            # If we were in a position during this period
            if position != 0:
                # Long position: we gain when price goes up
                if position == 1:
                    position_returns.append(returns[i])
                # Short position: we gain when price goes down
                else:
                    position_returns.append(-returns[i])
            else:
                # No position: no return
                position_returns.append(0)
        
        # Convert to numpy array
        position_returns = np.array(position_returns)
        
        # Calculate Sortino ratio if we have enough data
        if len(position_returns) >= self.window_size:
            # Use recent window for Sortino calculation
            recent_returns = position_returns[-self.window_size:]
            
            # Calculate Sortino ratio (annualized)
            sortino = calculate_sortino_ratio(recent_returns)
            
            # Scale the reward
            reward = max(min(sortino, 10), -10)  # Clipping to prevent extreme values
        else:
            # Not enough data, use simple P&L reward
            reward = (equity - self.previous_balance) / (self.previous_balance * 0.01)
        
        return reward
    
    def _calculate_differential_sharpe_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on differential Sharpe ratio.
        
        This reward is designed to be more stable for online learning.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate returns from prices
        price_returns = np.diff(prices) / prices[:-1]
        
        # Convert price returns to strategy returns based on position
        strategy_returns = position * price_returns[-1] if len(price_returns) > 0 else 0
        
        # Calculate immediate reward based on current return
        reward = strategy_returns
        
        # Scale the reward to be in a reasonable range
        reward = max(min(reward * 100, 5), -5)  # Limit to +/-5
        
        return reward
    
    def _calculate_risk_adjusted_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on risk-adjusted returns.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate balance change
        balance_change = balance - self.previous_balance
        
        # Calculate the risk (using a simple measure of price volatility)
        if len(prices) >= self.window_size:
            # Calculate historical volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Risk-adjusted reward
            if volatility > 0:
                reward = balance_change / (balance * volatility)
            else:
                reward = balance_change / balance if balance > 0 else 0
        else:
            # Simple normalized return if not enough data
            reward = balance_change / balance if balance > 0 else 0
        
        # Scale the reward
        reward = max(min(reward * 100, 5), -5)  # Limit to +/-5
        
        return reward
    
    def _calculate_profit_factor_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on profit factor (ratio of gains to losses).
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the balance change
        balance_change = balance - self.previous_balance
        
        # Calculate the basic reward: a positive reward for profit, negative for loss
        if balance_change > 0:
            # For profits, scale by the balance
            reward = balance_change / (balance * 0.01)  # As percentage of balance
        elif balance_change < 0:
            # For losses, apply a higher penalty to discourage large losses
            reward = 1.5 * balance_change / (balance * 0.01)  # 50% higher penalty for losses
        else:
            # No change, small negative reward to encourage action
            reward = -0.1
        
        return reward
    
    def _calculate_asymmetric_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate asymmetric reward (logarithmic utility for gains, linear for losses).
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the balance change
        balance_change = balance - self.previous_balance
        
        # Calculate return as percentage of previous balance
        pct_return = balance_change / self.previous_balance if self.previous_balance > 0 else 0
        
        # Apply asymmetric utility function
        if pct_return > 0:
            # Logarithmic utility for gains (diminishing returns)
            reward = np.log1p(100 * pct_return)  # log(1 + x) for small x is approximately x
        else:
            # Linear penalty for losses
            reward = 100 * pct_return  # Direct percentage as penalty
        
        return reward
    
    def _calculate_directional_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward based on correctness of direction prediction.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Need at least 2 prices to determine direction
        if len(prices) < 2:
            return 0
        
        # Calculate the price change
        price_change = prices[-1] - prices[-2]
        
        # Determine market direction
        market_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0
        
        # Calculate directional correctness
        if action != 0:  # If we took a directional action
            # +1 if our action matched the market direction, -1 if opposite
            direction_reward = 1 if action == market_direction else -1
        else:
            # Small negative reward for holding (opportunity cost)
            direction_reward = -0.1
        
        # Combine with a scaled P&L component
        pnl_reward = (equity - self.previous_balance) / (self.previous_balance * 0.01) if self.previous_balance > 0 else 0
        
        # Mix directional and P&L rewards (70% directional, 30% P&L)
        reward = 0.7 * direction_reward + 0.3 * pnl_reward
        
        return reward
    
    def _calculate_position_duration_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calculate reward that encourages shorter profitable trades.
        
        Args:
            action: The action taken (-1: sell, 0: hold, 1: buy)
            position: Current position (-1: short, 0: flat, 1: long)
            unrealized_pnl: Current unrealized profit/loss
            realized_pnl: Realized profit/loss
            balance: Account balance
            equity: Account equity
            prices: Array of recent prices
            
        Returns:
            float: The calculated reward
        """
        # Calculate the balance change
        balance_change = balance - self.previous_balance
        
        # Basic reward component: balance change
        if balance_change != 0:
            # Profit/loss as percentage of balance
            pnl_reward = balance_change / (self.previous_balance * 0.01) if self.previous_balance > 0 else 0
            
            # Add a small penalty for maintaining a position (encouraging efficiency)
            if position != 0 and action == 0:  # If holding a position
                duration_penalty = -0.05  # Small penalty per step
            else:
                duration_penalty = 0
            
            reward = pnl_reward + duration_penalty
        else:
            # No change in balance
            if position != 0:
                # Small penalty for holding a position without profit
                reward = -0.1
            else:
                # Neutral reward for being flat
                reward = 0
        
        return reward