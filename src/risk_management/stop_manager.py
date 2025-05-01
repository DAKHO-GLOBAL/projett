"""
Stop loss and take profit management module.

This module provides functionality to calculate and manage stop loss and
take profit levels based on various strategies, such as fixed pips, percentage,
ATR-based stops, or support/resistance levels.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class StopManager:
    """Class for managing stop loss and take profit levels."""
    
    def __init__(self, config: Dict):
        """
        Initialize the StopManager.
        
        Args:
            config: Configuration dictionary with stop loss and take profit parameters
        """
        self.config = config
        
        # Get stop loss parameters from config
        self.stop_loss_config = config.get('trading_params', {}).get('stop_loss', {})
        
        # Set default values if not specified
        self.stop_loss_type = self.stop_loss_config.get('type', 'atr')
        self.atr_multiplier = self.stop_loss_config.get('atr_multiplier', 2.0)
        self.fixed_pips = self.stop_loss_config.get('fixed_pips', 50)
        self.percent = self.stop_loss_config.get('percent', 0.01)  # 1% stop loss
        
        # Get take profit parameters from config
        self.take_profit_config = config.get('trading_params', {}).get('take_profit', {})
        
        # Set default values if not specified
        self.take_profit_type = self.take_profit_config.get('type', 'risk_reward')
        self.risk_reward_ratio = self.take_profit_config.get('risk_reward_ratio', 2.0)
        self.tp_fixed_pips = self.take_profit_config.get('fixed_pips', 100)
        self.tp_percent = self.take_profit_config.get('percent', 0.02)  # 2% take profit
        
        # Get trade management parameters
        self.trade_management = config.get('trading_params', {}).get('trade_management', {})
        self.use_trailing_stop = self.trade_management.get('use_trailing_stop', False)
        self.trailing_activation = self.trade_management.get('trailing_activation', 0.5)
        self.partial_exits = self.trade_management.get('partial_exits', False)
        self.partial_exit_levels = self.trade_management.get('partial_exit_levels', [0.3, 0.6])
        self.partial_exit_amounts = self.trade_management.get('partial_exit_amounts', [0.3, 0.3])
        
        logger.info(f"StopManager initialized with stop loss type: {self.stop_loss_type}, take profit type: {self.take_profit_type}")
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,  # 1 for long, -1 for short
        atr: Optional[float] = None,
        symbol: str = "",
        recent_prices: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate stop loss price based on strategy.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            atr: Average True Range (optional)
            symbol: Trading symbol
            recent_prices: Recent price data (optional, for support/resistance)
            
        Returns:
            float: Stop loss price
        """
        if self.stop_loss_type == 'fixed_pips':
            return self._calculate_fixed_pips_stop(entry_price, direction)
        elif self.stop_loss_type == 'percent':
            return self._calculate_percent_stop(entry_price, direction)
        elif self.stop_loss_type == 'atr':
            return self._calculate_atr_stop(entry_price, direction, atr)
        elif self.stop_loss_type == 'support_resistance':
            return self._calculate_support_resistance_stop(entry_price, direction, recent_prices)
        else:
            logger.warning(f"Unknown stop loss type: {self.stop_loss_type}, using fixed pips")
            return self._calculate_fixed_pips_stop(entry_price, direction)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: int,  # 1 for long, -1 for short
        symbol: str = ""
    ) -> float:
        """
        Calculate take profit price based on strategy.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction (1 for long, -1 for short)
            symbol: Trading symbol
            
        Returns:
            float: Take profit price
        """
        if self.take_profit_type == 'fixed_pips':
            return self._calculate_fixed_pips_tp(entry_price, direction)
        elif self.take_profit_type == 'percent':
            return self._calculate_percent_tp(entry_price, direction)
        elif self.take_profit_type == 'risk_reward':
            return self._calculate_risk_reward_tp(entry_price, stop_loss, direction)
        else:
            logger.warning(f"Unknown take profit type: {self.take_profit_type}, using risk reward")
            return self._calculate_risk_reward_tp(entry_price, stop_loss, direction)
    
    def _calculate_fixed_pips_stop(self, entry_price: float, direction: int) -> float:
        """
        Calculate stop loss based on fixed pips.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Stop loss price
        """
        # Convert pips to price (assuming 4 decimal places for forex)
        pip_value = self.fixed_pips / 10000
        
        # For long positions, stop is below entry; for short positions, stop is above entry
        if direction == 1:  # Long
            stop_loss = entry_price - pip_value
        else:  # Short
            stop_loss = entry_price + pip_value
        
        return stop_loss
    
    def _calculate_percent_stop(self, entry_price: float, direction: int) -> float:
        """
        Calculate stop loss based on percentage.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Stop loss price
        """
        # For long positions, stop is below entry; for short positions, stop is above entry
        if direction == 1:  # Long
            stop_loss = entry_price * (1 - self.percent)
        else:  # Short
            stop_loss = entry_price * (1 + self.percent)
        
        return stop_loss
    
    def _calculate_atr_stop(
        self,
        entry_price: float,
        direction: int,
        atr: Optional[float]
    ) -> float:
        """
        Calculate stop loss based on Average True Range (ATR).
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            atr: Average True Range
            
        Returns:
            float: Stop loss price
        """
        if atr is None or atr <= 0:
            logger.warning("Invalid ATR value, using fixed pips stop")
            return self._calculate_fixed_pips_stop(entry_price, direction)
        
        # Calculate stop distance as ATR multiplier * ATR
        stop_distance = self.atr_multiplier * atr
        
        # For long positions, stop is below entry; for short positions, stop is above entry
        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def _calculate_support_resistance_stop(
        self,
        entry_price: float,
        direction: int,
        recent_prices: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate stop loss based on support and resistance levels.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            recent_prices: Recent price data
            
        Returns:
            float: Stop loss price
        """
        if recent_prices is None or recent_prices.empty:
            logger.warning("No price data for support/resistance, using fixed pips stop")
            return self._calculate_fixed_pips_stop(entry_price, direction)
        
        # Simple implementation: use recent swing high/low
        if 'high' in recent_prices.columns and 'low' in recent_prices.columns:
            # For long positions, use recent low; for short positions, use recent high
            if direction == 1:  # Long
                # Find the lowest low in the recent window
                stop_loss = recent_prices['low'].min()
                
                # Make sure stop loss is below entry price
                if stop_loss >= entry_price:
                    # If not, use fixed pip stop
                    logger.warning("Support level not valid for long position, using fixed pips")
                    return self._calculate_fixed_pips_stop(entry_price, direction)
            else:  # Short
                # Find the highest high in the recent window
                stop_loss = recent_prices['high'].max()
                
                # Make sure stop loss is above entry price
                if stop_loss <= entry_price:
                    # If not, use fixed pip stop
                    logger.warning("Resistance level not valid for short position, using fixed pips")
                    return self._calculate_fixed_pips_stop(entry_price, direction)
            
            return stop_loss
        else:
            logger.warning("Price data doesn't have high/low columns, using fixed pips stop")
            return self._calculate_fixed_pips_stop(entry_price, direction)
    
    def _calculate_fixed_pips_tp(self, entry_price: float, direction: int) -> float:
        """
        Calculate take profit based on fixed pips.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Take profit price
        """
        # Convert pips to price (assuming 4 decimal places for forex)
        pip_value = self.tp_fixed_pips / 10000
        
        # For long positions, TP is above entry; for short positions, TP is below entry
        if direction == 1:  # Long
            take_profit = entry_price + pip_value
        else:  # Short
            take_profit = entry_price - pip_value
        
        return take_profit
    
    def _calculate_percent_tp(self, entry_price: float, direction: int) -> float:
        """
        Calculate take profit based on percentage.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Take profit price
        """
        # For long positions, TP is above entry; for short positions, TP is below entry
        if direction == 1:  # Long
            take_profit = entry_price * (1 + self.tp_percent)
        else:  # Short
            take_profit = entry_price * (1 - self.tp_percent)
        
        return take_profit
    
    def _calculate_risk_reward_tp(
        self,
        entry_price: float,
        stop_loss: float,
        direction: int
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Take profit price
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        
        # Calculate take profit distance as risk-reward ratio * stop distance
        tp_distance = self.risk_reward_ratio * stop_distance
        
        # For long positions, TP is above entry; for short positions, TP is below entry
        if direction == 1:  # Long
            take_profit = entry_price + tp_distance
        else:  # Short
            take_profit = entry_price - tp_distance
        
        return take_profit
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: int,
        initial_stop: float,
        take_profit: float
    ) -> float:
        """
        Calculate trailing stop level.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            direction: Trade direction (1 for long, -1 for short)
            initial_stop: Initial stop loss
            take_profit: Take profit level
            
        Returns:
            float: New stop loss level
        """
        if not self.use_trailing_stop:
            return initial_stop
        
        # Calculate the profit distance
        profit_target = abs(take_profit - entry_price)
        current_profit = abs(current_price - entry_price)
        
        # Check if we have reached the activation threshold
        if current_profit >= profit_target * self.trailing_activation:
            # Trailing stop logic
            if direction == 1:  # Long
                # For long positions, trailing stop should move up
                # Calculate how far we are into the profit range
                profit_percent = (current_price - entry_price) / profit_target
                
                # Move stop up proportionally, but never below entry once in profit
                if current_price > entry_price:
                    # New stop is max of: initial stop, entry, or trailing level
                    trailing_level = entry_price + (current_price - entry_price) * 0.5
                    return max(initial_stop, entry_price, trailing_level)
                else:
                    return initial_stop
            else:  # Short
                # For short positions, trailing stop should move down
                # Calculate how far we are into the profit range
                profit_percent = (entry_price - current_price) / profit_target
                
                # Move stop down proportionally, but never above entry once in profit
                if current_price < entry_price:
                    # New stop is min of: initial stop, entry, or trailing level
                    trailing_level = entry_price - (entry_price - current_price) * 0.5
                    return min(initial_stop, entry_price, trailing_level)
                else:
                    return initial_stop
        else:
            # Not enough profit yet, keep original stop
            return initial_stop
    
    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        direction: int,
        initial_stop: float,
        take_profit: float
    ) -> bool:
        """
        Determine if stop loss should be moved to breakeven.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            direction: Trade direction (1 for long, -1 for short)
            initial_stop: Initial stop loss
            take_profit: Take profit level
            
        Returns:
            bool: True if stop should be moved to breakeven
        """
        # Calculate the profit distance
        profit_target = abs(take_profit - entry_price)
        current_profit = abs(current_price - entry_price)
        
        # Check if we have reached halfway to the target
        breakeven_threshold = 0.5  # Move to breakeven at 50% of target
        
        if current_profit >= profit_target * breakeven_threshold:
            # Check direction to ensure we're in profit
            if (direction == 1 and current_price > entry_price) or (direction == -1 and current_price < entry_price):
                return True
        
        return False
    
    def calculate_partial_exit_levels(
        self,
        entry_price: float,
        take_profit: float,
        direction: int
    ) -> List[Dict]:
        """
        Calculate levels for partial position exits.
        
        Args:
            entry_price: Entry price
            take_profit: Take profit level
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            List[Dict]: List of partial exit levels
        """
        if not self.partial_exits:
            return []
        
        # Calculate the profit distance
        profit_distance = abs(take_profit - entry_price)
        
        # Calculate exit levels
        exit_levels = []
        
        for i, level_pct in enumerate(self.partial_exit_levels):
            # Calculate the price level
            if direction == 1:  # Long
                price_level = entry_price + profit_distance * level_pct
            else:  # Short
                price_level = entry_price - profit_distance * level_pct
            
            # Add to exit levels
            exit_levels.append({
                'level': i + 1,
                'price': price_level,
                'percent': self.partial_exit_amounts[i] if i < len(self.partial_exit_amounts) else 0.5
            })
        
        return exit_levels