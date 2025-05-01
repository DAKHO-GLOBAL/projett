"""
Feature generator module for creating technical indicators and derived features.

This module provides functionality to calculate technical indicators and other
derived features from price data, which can be used as inputs for the 
reinforcement learning models.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ta  # Technical Analysis library

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureGenerator:
    """Class for generating technical indicators and derived features from price data."""
    
    def __init__(self, config: Dict):
        """
        Initialize the FeatureGenerator.
        
        Args:
            config: Configuration dictionary containing feature parameters.
        """
        self.config = config
        
        # Define feature groups with default parameters
        self.feature_groups = {
            'trend': True,
            'momentum': True,
            'volatility': True,
            'volume': True,
            'custom': True,
        }
        
        # Update feature groups from config if provided
        if 'features' in config['data']:
            self.feature_groups.update(config['data']['features'])
    
    def generate_all_features(
        self, 
        df: pd.DataFrame, 
        include_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV price data
            include_groups: List of feature groups to include, defaults to all enabled groups
            
        Returns:
            pd.DataFrame: DataFrame with original data and added features
        """
        df = df.copy()
        
        # Determine which feature groups to include
        if include_groups is None:
            include_groups = [group for group, enabled in self.feature_groups.items() if enabled]
        
        # Generate features for each requested group
        for group in include_groups:
            if group == 'trend':
                df = self.add_trend_indicators(df)
            elif group == 'momentum':
                df = self.add_momentum_indicators(df)
            elif group == 'volatility':
                df = self.add_volatility_indicators(df)
            elif group == 'volume':
                df = self.add_volume_indicators(df)
            elif group == 'custom':
                df = self.add_custom_features(df)
        
        # Remove rows with NaN values created by indicators that need historical data
        df_cleaned = df.dropna()
        
        # Log how many rows were removed
        rows_removed = len(df) - len(df_cleaned)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with NaN values from features")
        
        return df_cleaned
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added trend indicators
        """
        # Moving Averages
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Exponential Moving Averages
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            close=df['close'], 
            window_slow=26, 
            window_fast=12, 
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Add Directional Movement Index (DMI)
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).psar()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['high'],
            low=df['low']
        )
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        # Moving Average crossovers (binary indicators)
        df['sma_10_20_cross'] = np.where(
            df['sma_10'] > df['sma_20'], 1, -1
        )
        df['ema_10_20_cross'] = np.where(
            df['ema_10'] > df['ema_20'], 1, -1
        )
        
        # Price relative to moving averages
        df['close_over_sma_50'] = df['close'] / df['sma_50'] - 1
        df['close_over_ema_50'] = df['close'] / df['ema_50'] - 1
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added momentum indicators
        """
        # Relative Strength Index (RSI)
        df['rsi_14'] = ta.momentum.RSIIndicator(
            close=df['close'], window=14
        ).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change (ROC)
        df['roc_10'] = ta.momentum.ROCIndicator(
            close=df['close'], window=10
        ).roc()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        ).williams_r()
        
        # Awesome Oscillator
        df['ao'] = ta.momentum.AwesomeOscillatorIndicator(
            high=df['high'],
            low=df['low'],
            window1=5,
            window2=34
        ).awesome_oscillator()
        
        # Fisher Transform of RSI
        # Custom calculation since it's not in the ta library
        rsi = df['rsi_14'] / 100  # Scale to 0-1
        # Apply Fisher Transform: 0.5 * ln((1 + x) / (1 - x))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-9
        df['fisher_rsi'] = 0.5 * np.log(
            (1 + rsi - epsilon) / (1 - rsi + epsilon)
        )
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added volatility indicators
        """
        # Average True Range (ATR)
        df['atr_14'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Donchian Channels
        df['donchian_high_20'] = df['high'].rolling(window=20).max()
        df['donchian_low_20'] = df['low'].rolling(window=20).min()
        df['donchian_mid_20'] = (df['donchian_high_20'] + df['donchian_low_20']) / 2
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20,
            window_atr=10
        )
        df['kc_high'] = keltner.keltner_channel_hband()
        df['kc_mid'] = keltner.keltner_channel_mband()
        df['kc_low'] = keltner.keltner_channel_lband()
        
        # Historical Volatility (using std dev of log returns)
        df['hist_vol_10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        df['hist_vol_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Normalized ATR (ATR / Close price)
        df['norm_atr_14'] = df['atr_14'] / df['close']
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added volume indicators
        """
        # Check if volume data is available
        if 'volume' in df.columns and df['volume'].sum() > 0:
            volume_column = 'volume'
        # Ensuite vérifier tick_volume si volume n'est pas disponible
        elif 'tick_volume' in df.columns and df['tick_volume'].sum() > 0:
            # Create a copy of tick_volume as volume for indicators
            df['volume_for_indicators'] = df['tick_volume']
            volume_column = 'volume_for_indicators'
        else:
            logger.warning("Volume data not available or all zeros, skipping volume indicators")
            return df
        
        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df[volume_column]
        ).on_balance_volume()
        
        # Volume Weighted Average Price (VWAP)
        # This is typically calculated intraday, so we'll reset it each day
        # For simplicity, we'll use a rolling window approach here
        df['vwap_20'] = (df[volume_column] * (df['high'] + df['low'] + df['close']) / 3).rolling(20).sum() / df[volume_column].rolling(20).sum()
        
        # Accumulation/Distribution Line
        df['adl'] = ta.volume.AccDistIndexIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df[volume_column]
        ).acc_dist_index()
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df[volume_column],
            window=20
        ).chaikin_money_flow()
        
        # Force Index
        df['fi_13'] = ta.volume.ForceIndexIndicator(
            close=df['close'],
            volume=df[volume_column],
            window=13
        ).force_index()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df[volume_column],
            window=14
        ).money_flow_index()
        
        # Volume Oscillator (VO) - Difference between fast and slow EMA of volume
        df['volume_ema_5'] = ta.trend.ema_indicator(df[volume_column], window=5)
        df['volume_ema_10'] = ta.trend.ema_indicator(df[volume_column], window=10)
        df['volume_oscillator'] = (df['volume_ema_5'] - df['volume_ema_10']) / df['volume_ema_10'] * 100
        
        return df
    
    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom features and indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added custom features
        """
        # Candlestick pattern features
        
        # Doji pattern (open and close are nearly equal)
        body_size = abs(df['close'] - df['open'])
        shadow_size = df['high'] - df['low']
        df['doji'] = (body_size / (shadow_size + 1e-9) < 0.1).astype(int)
        
        # Hammer pattern (small body, long lower shadow, small upper shadow)
        body_center = (df['open'] + df['close']) / 2
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        df['hammer'] = (
            (body_size / (shadow_size + 1e-9) < 0.3) &  # Small body
            (lower_shadow > 2 * body_size) &  # Long lower shadow
            (upper_shadow < 0.3 * body_size)  # Short upper shadow
        ).astype(int)
        
        # Price patterns and ranges
        
        # Daily range as percentage
        df['day_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Z-score of close price (how many standard deviations from the mean)
        df['close_zscore_20'] = (
            df['close'] - df['close'].rolling(20).mean()
        ) / df['close'].rolling(20).std()
        
        # Price distance from n-day high/low
        df['dist_from_20d_high'] = df['close'] / df['high'].rolling(20).max() - 1
        df['dist_from_20d_low'] = df['close'] / df['low'].rolling(20).min() - 1
        
        # Heikin-Ashi candles (smooth candlestick representation)
        df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Price velocity and acceleration
        df['price_velocity'] = df['close'].diff(1)
        df['price_acceleration'] = df['price_velocity'].diff(1)
        
        # Directional indicators (simple versions)
        df['trend_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consec_up'] = df['trend_up'].rolling(5).sum()
        df['consec_down'] = (1 - df['trend_up']).rolling(5).sum()
        
        # Range contraction/expansion
        df['range_5d_avg'] = (df['high'] - df['low']).rolling(5).mean()
        df['range_20d_avg'] = (df['high'] - df['low']).rolling(20).mean()
        df['range_expansion'] = df['range_5d_avg'] / df['range_20d_avg']
        
        # Time-based features (can be useful for capturing market session patterns)
        # Converting index to pandas datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping time-based features")
        else:
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            
            # Market session indicators (assuming standard forex sessions)
            # 0=Monday, 4=Friday
            df['is_weekend'] = ((df['day_of_week'] >= 5) | 
                               ((df['day_of_week'] == 4) & (df['hour'] >= 21))).astype(int)
            
            # Asian, European, and US sessions (simplified)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            
            # Session overlap periods (typically higher volatility)
            df['asia_europe_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
            df['europe_us_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df