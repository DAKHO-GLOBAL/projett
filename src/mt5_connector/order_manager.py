"""
Order management module for handling trading orders through MT5.

This module provides functionality for managing orders, positions, and trade
execution through the MetaTrader 5 API.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from src.mt5_connector.connector import MT5Connector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class OrderManager:
    """Class for managing orders and positions through MT5."""
    
    def __init__(self, config: Dict, mt5_connector: Optional[MT5Connector] = None):
        """
        Initialize the OrderManager.
        
        Args:
            config: Configuration dictionary
            mt5_connector: Optional MT5Connector instance
        """
        self.config = config
        
        # Initialize MT5Connector if not provided
        if mt5_connector is None:
            self.mt5_connector = MT5Connector(config)
            self.mt5_connector.initialize()
        else:
            self.mt5_connector = mt5_connector
        
        # Default magic number for identifying our orders
        self.magic_number = config.get('trading', {}).get('magic_number', 12345)
        
        logger.info("OrderManager initialized")
    
    def open_market_position(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Open a market position.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            order_type: Order type ("BUY" or "SELL")
            volume: Order volume in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment
            magic: Magic number for identifying orders
            
        Returns:
            Optional[Dict]: Dictionary with order result or None if error
        """
        # Use default magic number if not provided
        if magic is None:
            magic = self.magic_number
        
        # Validate order type
        if order_type not in ["BUY", "SELL"]:
            logger.error(f"Invalid order type: {order_type}. Must be 'BUY' or 'SELL'")
            return None
        
        # Place market order
        result = self.mt5_connector.place_market_order(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            magic=magic
        )
        
        return result
    
    def open_pending_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expiration: Optional[datetime] = None,
        comment: str = "",
        magic: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Open a pending order.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            order_type: Order type (BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP)
            volume: Order volume in lots
            price: Order price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            expiration: Order expiration datetime (optional)
            comment: Order comment
            magic: Magic number for identifying orders
            
        Returns:
            Optional[Dict]: Dictionary with order result or None if error
        """
        # Use default magic number if not provided
        if magic is None:
            magic = self.magic_number
        
        # Validate order type
        if order_type not in ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]:
            logger.error(f"Invalid pending order type: {order_type}")
            return None
        
        # Place pending order
        result = self.mt5_connector.place_pending_order(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiration=expiration,
            comment=comment,
            magic=magic
        )
        
        return result
    
    def close_position(
        self,
        position_id: int,
        comment: str = ""
    ) -> Optional[Dict]:
        """
        Close a specific position.
        
        Args:
            position_id: Position ticket ID
            comment: Order comment
            
        Returns:
            Optional[Dict]: Dictionary with close result or None if error
        """
        return self.mt5_connector.close_position(
            position_id=position_id,
            comment=comment
        )
    
    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        comment: str = ""
    ) -> List[Dict]:
        """
        Close all positions, optionally filtered by symbol and type.
        
        Args:
            symbol: Symbol to filter by (or None for all positions)
            order_type: Order type to filter by ("BUY", "SELL", or None for all)
            comment: Order comment
            
        Returns:
            List[Dict]: List of close results
        """
        # Get open positions
        positions = self.mt5_connector.get_positions(symbol=symbol)
        
        # Filter by order type if specified
        if order_type is not None:
            if order_type == "BUY":
                positions = [p for p in positions if p['type_str'] == "BUY"]
            elif order_type == "SELL":
                positions = [p for p in positions if p['type_str'] == "SELL"]
        
        # Close each position
        results = []
        for position in positions:
            result = self.close_position(
                position_id=position['ticket'],
                comment=comment
            )
            if result:
                results.append(result)
        
        logger.info(f"Closed {len(results)} positions out of {len(positions)} attempted")
        
        return results
    
    def modify_position(
        self,
        position_id: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = ""
    ) -> Optional[Dict]:
        """
        Modify an existing position (change SL/TP).
        
        Args:
            position_id: Position ticket ID
            stop_loss: New stop loss price (or None to keep current)
            take_profit: New take profit price (or None to keep current)
            comment: Order comment
            
        Returns:
            Optional[Dict]: Dictionary with modify result or None if error
        """
        return self.mt5_connector.modify_position(
            position_id=position_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )
    
    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        include_details: bool = True
    ) -> List[Dict]:
        """
        Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol: Symbol to filter by (or None for all positions)
            include_details: Whether to include detailed position info
            
        Returns:
            List[Dict]: List of position dictionaries
        """
        positions = self.mt5_connector.get_positions(symbol=symbol)
        
        if include_details:
            # Add additional calculated fields
            for position in positions:
                # Calculate profit percentage
                if 'profit' in position and 'volume' in position:
                    # Get symbol info for proper calculation
                    symbol_info = self.mt5_connector.get_symbol_info(position['symbol'])
                    
                    if symbol_info:
                        # Contract size (e.g., 100000 for standard lot)
                        contract_size = symbol_info.get('trade_contract_size', 100000)
                        
                        # Calculate profit percentage based on position size and entry price
                        position_value = position['volume'] * contract_size * position['price_open']
                        
                        if position_value > 0:
                            position['profit_percent'] = (position['profit'] / position_value) * 100
                        else:
                            position['profit_percent'] = 0
                
                # Calculate duration
                if 'time_open' in position:
                    position['duration'] = datetime.now() - position['time_open']
                    position['duration_seconds'] = position['duration'].total_seconds()
                    position['duration_hours'] = position['duration_seconds'] / 3600
        
        return positions
    
    def get_position_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get a summary of all open positions.
        
        Args:
            symbol: Symbol to filter by (or None for all positions)
            
        Returns:
            Dict: Summary of open positions
        """
        positions = self.get_open_positions(symbol=symbol)
        
        # Calculate summary statistics
        total_positions = len(positions)
        total_volume = sum(p['volume'] for p in positions)
        total_profit = sum(p['profit'] for p in positions)
        
        buy_positions = [p for p in positions if p['type_str'] == "BUY"]
        sell_positions = [p for p in positions if p['type_str'] == "SELL"]
        
        total_buy_volume = sum(p['volume'] for p in buy_positions)
        total_sell_volume = sum(p['volume'] for p in sell_positions)
        
        total_buy_profit = sum(p['profit'] for p in buy_positions)
        total_sell_profit = sum(p['profit'] for p in sell_positions)
        
        # Calculate average profit per position
        avg_profit = total_profit / total_positions if total_positions > 0 else 0
        
        # Count positions by symbol
        symbols = {}
        for p in positions:
            symbol = p['symbol']
            if symbol not in symbols:
                symbols[symbol] = {
                    'count': 0,
                    'volume': 0,
                    'profit': 0
                }
            symbols[symbol]['count'] += 1
            symbols[symbol]['volume'] += p['volume']
            symbols[symbol]['profit'] += p['profit']
        
        return {
            'total_positions': total_positions,
            'total_volume': total_volume,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'buy_positions': len(buy_positions),
            'sell_positions': len(sell_positions),
            'buy_volume': total_buy_volume,
            'sell_volume': total_sell_volume,
            'buy_profit': total_buy_profit,
            'sell_profit': total_sell_profit,
            'symbols': symbols
        }
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all pending orders, optionally filtered by symbol.
        
        Args:
            symbol: Symbol to filter by (or None for all orders)
            
        Returns:
            List[Dict]: List of order dictionaries
        """
        return self.mt5_connector.get_orders(symbol=symbol)
    
    def cancel_pending_order(self, order_id: int) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ticket ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.mt5_connector.initialized and not self.mt5_connector.initialize():
            return False
        
        # Prepare cancel request
        import MetaTrader5 as mt5
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order_id,
        }
        
        # Send the cancel request
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Failed to cancel order {order_id}: {mt5.last_error()}")
            return False
        
        # Check if the cancel was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Cancel order failed with retcode {result.retcode}: {result.comment}")
            return False
        
        logger.info(f"Order {order_id} cancelled successfully")
        return True
    
    def cancel_all_pending_orders(
        self,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> List[int]:
        """
        Cancel all pending orders, optionally filtered by symbol and type.
        
        Args:
            symbol: Symbol to filter by (or None for all orders)
            order_type: Order type to filter by (or None for all)
            
        Returns:
            List[int]: List of cancelled order IDs
        """
        # Get pending orders
        orders = self.get_pending_orders(symbol=symbol)
        
        # Filter by order type if specified
        if order_type is not None:
            orders = [o for o in orders if o['type_str'] == order_type]
        
        # Cancel each order
        cancelled_ids = []
        for order in orders:
            if self.cancel_pending_order(order['ticket']):
                cancelled_ids.append(order['ticket'])
        
        logger.info(f"Cancelled {len(cancelled_ids)} orders out of {len(orders)} attempted")
        
        return cancelled_ids
    
    def get_order_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get historical orders within date range, optionally filtered by symbol.
        
        Args:
            from_date: Start date for history
            to_date: End date for history (defaults to now)
            symbol: Symbol to filter by (or None for all symbols)
            
        Returns:
            List[Dict]: List of historical order dictionaries
        """
        return self.mt5_connector.get_order_history(
            from_date=from_date,
            to_date=to_date,
            symbol=symbol
        )
    
    def get_deal_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get historical deals within date range, optionally filtered by symbol.
        
        Args:
            from_date: Start date for history
            to_date: End date for history (defaults to now)
            symbol: Symbol to filter by (or None for all symbols)
            
        Returns:
            List[Dict]: List of historical deal dictionaries
        """
        return self.mt5_connector.get_deal_history(
            from_date=from_date,
            to_date=to_date,
            symbol=symbol
        )
    
    def calculate_trade_statistics(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Calculate trading performance statistics.
        
        Args:
            from_date: Start date for statistics
            to_date: End date for statistics (defaults to now)
            symbol: Symbol to filter by (or None for all symbols)
            
        Returns:
            Dict: Dictionary with trading statistics
        """
        # Get deal history
        deals = self.get_deal_history(
            from_date=from_date,
            to_date=to_date,
            symbol=symbol
        )
        
        if not deals:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_profit_trades': 0,
                'total_loss_trades': 0,
                'average_profit': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'max_profit_trade': 0,
                'max_loss_trade': 0,
                'average_trade': 0,
                'average_win': 0,
                'average_loss': 0,
                'risk_reward_ratio': 0
            }
        
        # Extract relevant information
        profits = [d['profit'] for d in deals]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        # Calculate statistics
        total_trades = len(profits)
        profitable_trades = len(winning_trades)
        losing_trades = len(losing_trades)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(profits)
        total_profit_trades = sum(winning_trades)
        total_loss_trades = sum(losing_trades)
        
        average_profit = total_profit / total_trades if total_trades > 0 else 0
        average_win = total_profit_trades / profitable_trades if profitable_trades > 0 else 0
        average_loss = total_loss_trades / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(total_profit_trades / total_loss_trades) if total_loss_trades != 0 else 0
        
        max_profit_trade = max(profits) if profits else 0
        max_loss_trade = min(profits) if profits else 0
        
        risk_reward_ratio = abs(average_win / average_loss) if average_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_profit_trades': total_profit_trades,
            'total_loss_trades': total_loss_trades,
            'average_profit': average_profit,
            'average_win': average_win,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'max_profit_trade': max_profit_trade,
            'max_loss_trade': max_loss_trade,
            'risk_reward_ratio': risk_reward_ratio
        }