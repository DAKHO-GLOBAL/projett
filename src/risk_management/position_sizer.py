"""
Position sizing module for determining trade position sizes.

This module provides functionality to calculate position sizes based on various
risk management methods, such as fixed lot size, percentage of equity, or 
percentage of risk.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PositionSizer:
    """Class for calculating position sizes based on risk management rules."""
    
    def __init__(self, config: Dict):
        """
        Initialize the PositionSizer.
        
        Args:
            config: Configuration dictionary with position sizing parameters
        """
        self.config = config
        
        # Get position sizing parameters from config
        self.position_config = config.get('trading_params', {}).get('position_sizing', {})
        
        # Set default values if not specified
        self.sizing_type = self.position_config.get('type', 'percent_risk')
        self.risk_per_trade = self.position_config.get('risk_per_trade', 0.01)  # 1% risk per trade
        self.max_risk_per_day = self.position_config.get('max_risk_per_day', 0.05)  # 5% max risk per day
        self.fixed_lot = self.position_config.get('fixed_lot', 0.01)  # 0.01 lot (micro lot)
        self.percent_equity = self.position_config.get('percent_equity', 0.02)  # 2% of equity
        self.max_positions = self.position_config.get('max_positions', 5)
        
        logger.info(f"PositionSizer initialized with type: {self.sizing_type}")
    
    def calculate_position_size(
        self,
        action: int,
        balance: float,
        price: float,
        symbol: str,
        atr: Optional[float] = None,
        stop_loss_pips: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            action: Action (-1 for sell, 1 for buy)
            balance: Account balance
            price: Current market price
            symbol: Trading symbol
            atr: Average True Range (optional)
            stop_loss_pips: Stop loss distance in pips (optional)
            
        Returns:
            float: Position size in lots
        """
        # Get the contract size and pip value for the symbol
        contract_size, pip_value = self._get_symbol_info(symbol)
        
        # Calculate position size based on the selected method
        if self.sizing_type == 'fixed_lot':
            position_size = self._calculate_fixed_lot_size()
        elif self.sizing_type == 'percent_equity':
            position_size = self._calculate_percent_equity_size(balance, price, contract_size)
        elif self.sizing_type == 'percent_risk':
            position_size = self._calculate_percent_risk_size(
                balance=balance,
                price=price,
                contract_size=contract_size,
                pip_value=pip_value,
                stop_loss_pips=stop_loss_pips,
                atr=atr,
                symbol=symbol
            )
        else:
            logger.warning(f"Unknown position sizing type: {self.sizing_type}, using fixed lot")
            position_size = self._calculate_fixed_lot_size()
        
        # Round to the nearest lot step (usually 0.01 for mini lots)
        lot_step = 0.01
        position_size = round(position_size / lot_step) * lot_step
        
        # Ensure minimum position size (usually 0.01 lot)
        min_lot = 0.01
        if position_size < min_lot:
            position_size = min_lot
        
        logger.info(f"Calculated position size: {position_size} lots for {symbol} at {price}")
        
        return position_size
    
    def _get_symbol_info(self, symbol: str) -> Tuple[float, float]:
        """
        Get the contract size and pip value for the symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple[float, float]: (contract_size, pip_value)
        """
        # Default values for common forex pairs
        contract_size = 100000  # Standard lot
        pip_value = 10  # $10 per pip for standard lot
        
        # Try to get more accurate values from config if available
        symbol_info = self.config.get('symbol_info', {}).get(symbol, {})
        
        if symbol_info:
            contract_size = symbol_info.get('contract_size', contract_size)
            pip_value = symbol_info.get('pip_value', pip_value)
        
        return contract_size, pip_value
    
    def _calculate_fixed_lot_size(self) -> float:
        """
        Calculate position size using fixed lot method.
        
        Returns:
            float: Position size in lots
        """
        return self.fixed_lot
    
    def _calculate_percent_equity_size(
        self,
        balance: float,
        price: float,
        contract_size: float
    ) -> float:
        """
        Calculate position size as a percentage of equity.
        
        Args:
            balance: Account balance
            price: Current market price
            contract_size: Contract size for the symbol
            
        Returns:
            float: Position size in lots
        """
        # Calculate the position value we want to risk
        position_value = balance * self.percent_equity
        
        # Calculate the number of lots
        position_size = position_value / (price * contract_size)
        
        return position_size
    
    def _calculate_percent_risk_size(
        self,
        balance: float,
        price: float,
        contract_size: float,
        pip_value: float,
        stop_loss_pips: Optional[float] = None,
        atr: Optional[float] = None,
        symbol: str = ""
    ) -> float:
        """
        Calculate position size based on percentage risk.
        
        Args:
            balance: Account balance
            price: Current market price
            contract_size: Contract size for the symbol
            pip_value: Value of one pip in account currency
            stop_loss_pips: Stop loss distance in pips
            atr: Average True Range (optional)
            symbol: Trading symbol
            
        Returns:
            float: Position size in lots
        """
        # Calculate the amount to risk
        risk_amount = balance * self.risk_per_trade
        
        # Determine stop loss distance
        if stop_loss_pips is None:
            # If ATR is provided, use it to determine stop loss
            if atr is not None:
                # Get ATR multiplier from config
                atr_multiplier = self.config.get('trading_params', {}).get('stop_loss', {}).get('atr_multiplier', 2.0)
                stop_loss_pips = atr * atr_multiplier * 10000  # Convert to pips
            else:
                # Use default stop loss from config
                stop_loss_config = self.config.get('trading_params', {}).get('stop_loss', {})
                stop_loss_type = stop_loss_config.get('type', 'fixed_pips')
                
                if stop_loss_type == 'fixed_pips':
                    stop_loss_pips = stop_loss_config.get('fixed_pips', 50)
                elif stop_loss_type == 'percent':
                    # Convert percent to pips
                    percent = stop_loss_config.get('percent', 0.01)
                    stop_loss_pips = price * percent * 10000
                else:
                    # Default to 50 pips if no valid stop loss type
                    stop_loss_pips = 50
        
        # Calculate the risk per pip
        if stop_loss_pips > 0:
            risk_per_pip = risk_amount / stop_loss_pips
        else:
            # Default to a safe value if stop loss is not valid
            logger.warning(f"Invalid stop loss distance for {symbol}: {stop_loss_pips}")
            risk_per_pip = risk_amount / 50  # Assume 50 pips
        
        # Calculate the number of lots
        standard_lot_size = 100000
        lots_per_pip = risk_per_pip / pip_value
        position_size = lots_per_pip / (standard_lot_size / contract_size)
        
        return position_size
    
    def get_max_positions_allowed(self, current_positions: int) -> int:
        """
        Get the maximum number of additional positions allowed.
        
        Args:
            current_positions: Current number of open positions
            
        Returns:
            int: Maximum number of additional positions allowed
        """
        return max(0, self.max_positions - current_positions)
    
    def calculate_position_exposure(
        self,
        position_size: float,
        price: float,
        balance: float,
        symbol: str
    ) -> Dict:
        """
        Calculate the exposure and risk of a position.
        
        Args:
            position_size: Position size in lots
            price: Current market price
            balance: Account balance
            symbol: Trading symbol
            
        Returns:
            Dict: Dictionary with exposure metrics
        """
        contract_size, pip_value = self._get_symbol_info(symbol)
        
        # Calculate position value
        position_value = position_size * contract_size * price
        
        # Calculate exposure as percentage of balance
        exposure_percent = (position_value / balance) * 100 if balance > 0 else 0
        
        # Calculate pip value for this position
        position_pip_value = (position_size / 1.0) * pip_value
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'exposure_percent': exposure_percent,
            'pip_value': position_pip_value
        }