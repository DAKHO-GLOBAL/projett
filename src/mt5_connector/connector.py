"""
MetaTrader 5 connector module for trading and market data access.

This module provides functionality to connect to MetaTrader 5 and execute
various trading operations, such as placing orders, retrieving historical
data, and monitoring account information.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MT5Connector:
    """Class for interacting with the MetaTrader 5 terminal."""
    
    # Mapping of timeframe strings to MT5 timeframe constants
    TIMEFRAME_DICT = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    
    # Mapping of order types
    ORDER_TYPE_DICT = {
        "BUY": mt5.ORDER_TYPE_BUY,
        "SELL": mt5.ORDER_TYPE_SELL,
        "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
        "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
        "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
        "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
    }
    
    def __init__(self, config: Dict):
        """
        Initialize the MT5Connector.
        
        Args:
            config: Configuration dictionary containing MT5 connection parameters
        """
        self.config = config
        self.mt5_config = config['mt5']
        self.initialized = False
        
        logger.info("MT5Connector initialized with configuration")
    
    def initialize(self) -> bool:
        """
        Initialize connection to MetaTrader 5.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.initialized:
            logger.info("MT5 connection already initialized")
            return True
        
        # Initialize MT5 connection
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
            return False
        
        # Log in to MT5 account
        if not mt5.login(
            login=self.mt5_config['login'],
            password=self.mt5_config['password'],
            server=self.mt5_config['server']
        ):
            logger.error(f"MT5 login failed, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        self.initialized = True
        logger.info("MT5 connection initialized successfully")
        
        # Print account info for verification
        account_info = mt5.account_info()
        if account_info is not None:
            logger.info(f"Connected to account: {account_info.login}, "
                       f"Balance: {account_info.balance}, "
                       f"Server: {self.mt5_config['server']}")
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown the MT5 connection."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 connection closed")
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.
        
        Returns:
            Optional[Dict]: Dictionary with account information or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
        
        # Convert named tuple to dictionary
        return {prop: getattr(account_info, prop) for prop in dir(account_info) 
                if not prop.startswith('_')}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get information about a trading symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Optional[Dict]: Dictionary with symbol information or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None
        
        # Convert named tuple to dictionary
        return {prop: getattr(symbol_info, prop) for prop in dir(symbol_info) 
                if not prop.startswith('_')}
    
    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest price tick for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Optional[Dict]: Dictionary with latest tick data or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get current tick for {symbol}: {mt5.last_error()}")
            return None
        
        # Convert named tuple to dictionary
        tick_dict = {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'flags': tick.flags,
        }
        
        return tick_dict
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_bars: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve historical price data from MT5.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "M15", "H1")
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to now)
            num_bars: Number of bars to retrieve (alternative to start_date)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV data or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        # Validate timeframe
        if timeframe not in self.TIMEFRAME_DICT:
            logger.error(f"Invalid timeframe: {timeframe}. " 
                       f"Available timeframes: {list(self.TIMEFRAME_DICT.keys())}")
            return None
        
        mt5_timeframe = self.TIMEFRAME_DICT[timeframe]
        
        # Set default end date to now if not provided
        if end_date is None:
            end_date = datetime.now()
        
        # Set start date based on parameters
        if start_date is None and num_bars is None:
            # Default to 30 days if neither start_date nor num_bars provided
            days_back = self.config.get('data', {}).get('history_days', 30)
            start_date = end_date - timedelta(days=days_back)
        
        # Convert datetime to MT5 format (seconds since epoch)
        end_date_timestamp = int(end_date.timestamp())
        
        # Fetch data based on either date range or number of bars
        if num_bars is not None:
            # Fetch by number of bars
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_bars)
        else:
            # Fetch by date range
            start_date_timestamp = int(start_date.timestamp())
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, 
                                        start_date_timestamp, end_date_timestamp)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {symbol} {timeframe}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        # Convert time column from seconds since epoch to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns for clarity
        df = df.rename(columns={
            'time': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        })
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe} "
                    f"from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def place_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> Optional[Dict]:
        """
        Place a market order.
        
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
        if not self.initialized and not self.initialize():
            return None
        
        # Validate order type
        if order_type not in ["BUY", "SELL"]:
            logger.error(f"Invalid order type: {order_type}. Must be 'BUY' or 'SELL'")
            return None
        
        # Get current tick
        tick = self.get_current_tick(symbol)
        if tick is None:
            return None
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.ORDER_TYPE_DICT[order_type],
            "price": tick['ask'] if order_type == "BUY" else tick['bid'],
            "deviation": 10,  # Maximum price deviation in points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or kill
        }
        
        # Add stop loss and take profit if provided
        if stop_loss is not None:
            request["sl"] = stop_loss
        
        if take_profit is not None:
            request["tp"] = take_profit
        
        # Send the order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Failed to place order: {mt5.last_error()}")
            return None
        
        # Check if the order was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed with retcode {result.retcode}: {result.comment}")
            return None
        
        # Convert order result to dictionary
        result_dict = {
            'order_id': result.order,
            'symbol': symbol,
            'volume': volume,
            'price': result.price,
            'type': order_type,
            'comment': comment,
            'retcode': result.retcode,
            'request': request
        }
        
        logger.info(f"Order placed successfully: {result_dict}")
        
        return result_dict
    
    def place_pending_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expiration: Optional[datetime] = None,
        comment: str = "",
        magic: int = 0
    ) -> Optional[Dict]:
        """
        Place a pending order.
        
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
        if not self.initialized and not self.initialize():
            return None
        
        # Validate order type
        if order_type not in ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]:
            logger.error(f"Invalid pending order type: {order_type}")
            return None
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": self.ORDER_TYPE_DICT[order_type],
            "price": price,
            "deviation": 10,  # Maximum price deviation in points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or kill
        }
        
        # Add stop loss and take profit if provided
        if stop_loss is not None:
            request["sl"] = stop_loss
        
        if take_profit is not None:
            request["tp"] = take_profit
        
        # Add expiration if provided
        if expiration is not None:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration.timestamp())
        
        # Send the order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Failed to place pending order: {mt5.last_error()}")
            return None
        
        # Check if the order was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Pending order failed with retcode {result.retcode}: {result.comment}")
            return None
        
        # Convert order result to dictionary
        result_dict = {
            'order_id': result.order,
            'symbol': symbol,
            'volume': volume,
            'price': price,
            'type': order_type,
            'comment': comment,
            'retcode': result.retcode,
            'request': request
        }
        
        logger.info(f"Pending order placed successfully: {result_dict}")
        
        return result_dict
    
    def close_position(
        self,
        position_id: int,
        comment: str = ""
    ) -> Optional[Dict]:
        """
        Close an existing position.
        
        Args:
            position_id: Position ticket ID
            comment: Order comment
            
        Returns:
            Optional[Dict]: Dictionary with close result or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        # Get position info
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            logger.error(f"Position {position_id} not found: {mt5.last_error()}")
            return None
        
        position = position[0]
        
        # Get symbol info
        symbol = position.symbol
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Get current tick
        tick = self.get_current_tick(symbol)
        if tick is None:
            return None
        
        # Determine order type for closing (opposite of position type)
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position_id,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "price": tick['bid'] if close_type == mt5.ORDER_TYPE_SELL else tick['ask'],
            "deviation": 10,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send the close request
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Failed to close position {position_id}: {mt5.last_error()}")
            return None
        
        # Check if the close was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close position failed with retcode {result.retcode}: {result.comment}")
            return None
        
        # Convert result to dictionary
        result_dict = {
            'position_id': position_id,
            'symbol': symbol,
            'volume': position.volume,
            'price': result.price,
            'profit': result.profit,
            'comment': comment,
            'retcode': result.retcode,
            'request': request
        }
        
        logger.info(f"Position {position_id} closed successfully: {result_dict}")
        
        return result_dict
    
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
        if not self.initialized and not self.initialize():
            return None
        
        # Get position info
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            logger.error(f"Position {position_id} not found: {mt5.last_error()}")
            return None
        
        position = position[0]
        
        # If both stop_loss and take_profit are None, nothing to change
        if stop_loss is None and take_profit is None:
            logger.warning("No changes specified for position modification")
            return {
                'position_id': position_id,
                'result': 'no_change',
                'message': 'No changes specified'
            }
        
        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position_id,
            "symbol": position.symbol,
            "comment": comment,
        }
        
        # Add stop loss if provided, otherwise keep existing
        if stop_loss is not None:
            request["sl"] = stop_loss
        else:
            request["sl"] = position.sl
        
        # Add take profit if provided, otherwise keep existing
        if take_profit is not None:
            request["tp"] = take_profit
        else:
            request["tp"] = position.tp
        
        # Send the modification request
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Failed to modify position {position_id}: {mt5.last_error()}")
            return None
        
        # Check if the modification was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify position failed with retcode {result.retcode}: {result.comment}")
            return None
        
        # Convert result to dictionary
        result_dict = {
            'position_id': position_id,
            'symbol': position.symbol,
            'stop_loss': request['sl'],
            'take_profit': request['tp'],
            'comment': comment,
            'retcode': result.retcode,
            'request': request
        }
        
        logger.info(f"Position {position_id} modified successfully: {result_dict}")
        
        return result_dict
    
    def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol: Symbol to filter by (or None for all positions)
            
        Returns:
            List[Dict]: List of position dictionaries
        """
        if not self.initialized and not self.initialize():
            return []
        
        # Get positions
        if symbol is not None:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return []
        
        # Convert to list of dictionaries
        position_list = []
        for position in positions:
            # Convert named tuple to dictionary
            position_dict = {prop: getattr(position, prop) for prop in dir(position) 
                           if not prop.startswith('_')}
            
            # Add some additional derived fields
            position_dict['time_open'] = datetime.fromtimestamp(position.time)
            position_dict['type_str'] = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            
            position_list.append(position_dict)
        
        return position_list
    
    def get_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all pending orders, optionally filtered by symbol.
        
        Args:
            symbol: Symbol to filter by (or None for all orders)
            
        Returns:
            List[Dict]: List of order dictionaries
        """
        if not self.initialized and not self.initialize():
            return []
        
        # Get orders
        if symbol is not None:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()
        
        if orders is None:
            logger.error(f"Failed to get orders: {mt5.last_error()}")
            return []
        
        # Convert to list of dictionaries
        order_list = []
        for order in orders:
            # Convert named tuple to dictionary
            order_dict = {prop: getattr(order, prop) for prop in dir(order) 
                        if not prop.startswith('_')}
            
            # Add some additional derived fields
            order_dict['time_setup'] = datetime.fromtimestamp(order.time_setup)
            
            # Get type as string
            if order.type == mt5.ORDER_TYPE_BUY_LIMIT:
                order_dict['type_str'] = "BUY_LIMIT"
            elif order.type == mt5.ORDER_TYPE_SELL_LIMIT:
                order_dict['type_str'] = "SELL_LIMIT"
            elif order.type == mt5.ORDER_TYPE_BUY_STOP:
                order_dict['type_str'] = "BUY_STOP"
            elif order.type == mt5.ORDER_TYPE_SELL_STOP:
                order_dict['type_str'] = "SELL_STOP"
            else:
                order_dict['type_str'] = str(order.type)
            
            order_list.append(order_dict)
        
        return order_list
    
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
        if not self.initialized and not self.initialize():
            return []
        
        # Set default dates if not provided
        if to_date is None:
            to_date = datetime.now()
        
        if from_date is None:
            from_date = to_date - timedelta(days=30)  # Default to last 30 days
        
        # Convert to MT5 format (seconds since epoch)
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        # Get history
        if symbol is not None:
            history = mt5.history_orders_get(from_date=from_timestamp, to_date=to_timestamp, symbol=symbol)
        else:
            history = mt5.history_orders_get(from_date=from_timestamp, to_date=to_timestamp)
        
        if history is None:
            logger.error(f"Failed to get order history: {mt5.last_error()}")
            return []
        
        # Convert to list of dictionaries
        history_list = []
        for order in history:
            # Convert named tuple to dictionary
            order_dict = {prop: getattr(order, prop) for prop in dir(order) 
                        if not prop.startswith('_')}
            
            # Add some additional derived fields
            order_dict['time_setup'] = datetime.fromtimestamp(order.time_setup)
            if order.time_done > 0:
                order_dict['time_done'] = datetime.fromtimestamp(order.time_done)
            
            history_list.append(order_dict)
        
        return history_list
    
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
        if not self.initialized and not self.initialize():
            return []
        
        # Set default dates if not provided
        if to_date is None:
            to_date = datetime.now()
        
        if from_date is None:
            from_date = to_date - timedelta(days=30)  # Default to last 30 days
        
        # Convert to MT5 format (seconds since epoch)
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        # Get history
        if symbol is not None:
            history = mt5.history_deals_get(from_date=from_timestamp, to_date=to_timestamp, symbol=symbol)
        else:
            history = mt5.history_deals_get(from_date=from_timestamp, to_date=to_timestamp)
        
        if history is None:
            logger.error(f"Failed to get deal history: {mt5.last_error()}")
            return []
        
        # Convert to list of dictionaries
        history_list = []
        for deal in history:
            # Convert named tuple to dictionary
            deal_dict = {prop: getattr(deal, prop) for prop in dir(deal) 
                       if not prop.startswith('_')}
            
            # Add some additional derived fields
            deal_dict['time'] = datetime.fromtimestamp(deal.time)
            
            # Add type as string
            if deal.type == mt5.DEAL_TYPE_BUY:
                deal_dict['type_str'] = "BUY"
            elif deal.type == mt5.DEAL_TYPE_SELL:
                deal_dict['type_str'] = "SELL"
            else:
                deal_dict['type_str'] = str(deal.type)
            
            # Add entry as string
            if deal.entry == mt5.DEAL_ENTRY_IN:
                deal_dict['entry_str'] = "IN"
            elif deal.entry == mt5.DEAL_ENTRY_OUT:
                deal_dict['entry_str'] = "OUT"
            elif deal.entry == mt5.DEAL_ENTRY_INOUT:
                deal_dict['entry_str'] = "INOUT"
            else:
                deal_dict['entry_str'] = str(deal.entry)
            
            history_list.append(deal_dict)
        
        return history_list