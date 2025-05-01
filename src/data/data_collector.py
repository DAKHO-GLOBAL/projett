"""
Data collection module for retrieving market data from MetaTrader 5.

This module provides functionality to connect to MetaTrader 5 and collect
historical and real-time market data for various symbols and timeframes.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MT5DataCollector:
    """Class for collecting and managing market data from MetaTrader 5."""
    
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
    
    def __init__(self, config: Dict):
        """
        Initialize the MT5DataCollector.
        
        Args:
            config: Configuration dictionary containing MT5 connection parameters
                   and data collection settings.
        """
        self.config = config
        self.mt5_config = config['mt5']
        self.data_config = config['data']
        self.initialized = False
        
        # Create data directories if they don't exist
        self.raw_data_path = Path(config['paths']['data']) / 'raw'
        self.processed_data_path = Path(config['paths']['data']) / 'processed'
        
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize connection to MetaTrader 5.
        
        Returns:
            bool: True if initialization successful, False otherwise.
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
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols in MT5.
        
        Returns:
            List[str]: List of symbol names
        """
        if not self.initialized and not self.initialize():
            return []
        
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get detailed information about a specific symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Optional[dict]: Dictionary with symbol information or None if error
        """
        if not self.initialized and not self.initialize():
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        # Convert named tuple to dictionary
        return {prop: getattr(symbol_info, prop) for prop in dir(symbol_info) 
                if not prop.startswith('_')}
    
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
            # Default to config history_days if neither start_date nor num_bars provided
            days_back = self.data_config.get('history_days', 365)
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
            logger.error(f"No data received for {symbol} {timeframe}, error: {mt5.last_error()}")
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
            logger.error(f"Failed to get current tick for {symbol}")
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
    
    def save_historical_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        format: str = 'csv'
    ) -> str:
        """
        Save historical data to file.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe string
            data: DataFrame containing the data
            format: File format ('csv' or 'parquet')
            
        Returns:
            str: Path to the saved file
        """
        # Create filename with symbol, timeframe and date range
        start_date = data.index.min().strftime('%Y%m%d')
        end_date = data.index.max().strftime('%Y%m%d')
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if format.lower() == 'csv':
            filepath = self.raw_data_path / f"{filename}.csv"
            data.to_csv(filepath)
        elif format.lower() == 'parquet':
            filepath = self.raw_data_path / f"{filename}.parquet"
            data.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")
        
        logger.info(f"Saved {len(data)} records to {filepath}")
        return str(filepath)
    
    def load_historical_data(
        self,
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical data from file.
        
        Args:
            filepath: Path to the data file
            format: File format ('csv' or 'parquet'), inferred from extension if None
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded data
        """
        filepath = Path(filepath)
        
        # Infer format from file extension if not provided
        if format is None:
            if filepath.suffix.lower() == '.csv':
                format = 'csv'
            elif filepath.suffix.lower() == '.parquet':
                format = 'parquet'
            else:
                raise ValueError(f"Could not infer format from file extension: {filepath.suffix}")
        
        # Load data based on format
        if format.lower() == 'csv':
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format.lower() == 'parquet':
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")
        
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    
    def collect_data_for_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        days: int = 365,
        save: bool = True
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Collect historical data for multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols to collect data for, defaults to config symbols
            timeframes: List of timeframes to collect, defaults to config timeframes
            days: Number of days of history to collect
            save: Whether to save the collected data to files
            
        Returns:
            Dict[Tuple[str, str], pd.DataFrame]: Dictionary mapping (symbol, timeframe) 
                                               to the corresponding DataFrame
        """
        if not self.initialized and not self.initialize():
            return {}
        
        # Use default symbols and timeframes from config if not provided
        if symbols is None:
            symbols = self.config['trading']['symbols']
        
        if timeframes is None:
            timeframes = self.config['trading']['timeframes']
        
        # Dictionary to hold all collected data
        all_data = {}
        
        # End date is now
        end_date = datetime.now()
        # Start date is end_date minus specified days
        start_date = end_date - timedelta(days=days)
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Collecting data for {symbol} {timeframe} "
                            f"from {start_date} to {end_date}")
                
                # Get historical data
                df = self.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    all_data[(symbol, timeframe)] = df
                    
                    # Save data if requested
                    if save:
                        self.save_historical_data(symbol, timeframe, df)
                
                # Sleep briefly to avoid overwhelming the API
                time.sleep(0.5)
        
        return all_data