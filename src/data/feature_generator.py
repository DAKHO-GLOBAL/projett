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
            'trend': True,      # Garde les indicateurs de tendance (essentiels)
            'momentum': True,   # Garde les indicateurs de momentum (très importants pour l'or)
            'volatility': True, # Garde les indicateurs de volatilité (cruciaux pour l'or)
            'volume': False,    # Désactive - souvent bruité pour l'or
            'custom': False,    # Désactive - peut ajouter du bruit
            'custom_gold': False, # Active les caractéristiques spécifiques à l'or
            'gold_best': True   # Active uniquement les meilleures caractéristiques pour l'or
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
            elif group == 'custum_gold':
                df = self.add_gold_specific_features(df)
            elif group =='gold_best':
                df=self.add_enhanced_gold_features(df)
        
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
    
    # Dans src/data/feature_generator.py, ajoutez cette méthode

    def add_gold_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add gold-specific features to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added gold-specific features
        """
        # Daily ranges - gold tends to respect daily ranges
        df['day_high'] = df['high'].resample('D').transform('max')
        df['day_low'] = df['low'].resample('D').transform('min')
        df['day_range'] = df['day_high'] - df['day_low']
        df['price_in_range'] = (df['close'] - df['day_low']) / df['day_range']
        
        # Gold volatility features
        df['gold_volatility'] = df['high'] - df['low']
        df['gold_volatility_ma'] = df['gold_volatility'].rolling(20).mean()
        df['volatility_ratio'] = df['gold_volatility'] / df['gold_volatility_ma']
        
        # Gold tends to be sensitive to round numbers
        round_levels = np.floor(df['close'] / 50) * 50  # Round to nearest $50
        df['distance_to_round'] = df['close'] - round_levels
        df['near_round_level'] = (abs(df['distance_to_round']) < 2.5).astype(int)
        
        # Gold often shows momentum behavior
        df['momentum_1h'] = df['close'].pct_change(12)  # For 5-min data, 12 bars = 1 hour
        df['momentum_4h'] = df['close'].pct_change(48)  # 4 hours
        
        # Trading session features (gold is sensitive to session changes)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            
            # Create session indicators (gold reacts differently to different sessions)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['sydney_session'] = ((df['hour'] >= 21) | (df['hour'] < 2)).astype(int)
            
            # Transition periods are often most volatile for gold
            df['session_transition'] = (
                ((df['hour'] >= 7) & (df['hour'] <= 9)) |  # Asian to London
                ((df['hour'] >= 13) & (df['hour'] <= 15)) |  # London/NY overlap
                ((df['hour'] >= 20) & (df['hour'] <= 22))    # NY to Sydney
            ).astype(int)
        
        return df
    

    def add_enhanced_gold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des caractéristiques avancées spécifiques à l'or au DataFrame.
        
        Args:
            df: DataFrame avec données OHLCV
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques avancées pour l'or
        """
        # 1. Caractéristiques de support et résistance psychologiques
        # L'or est particulièrement sensible aux niveaux psychologiques (1800, 1900, 2000)
        if 'close' in df.columns:
            # Niveaux psychologiques (multiples de 50 et 100 $)
            round_level_100 = np.round(df['close'] / 100) * 100
            round_level_50 = np.round(df['close'] / 50) * 50
            
            # Distance aux niveaux psychologiques
            df['dist_to_psych_100'] = (df['close'] - round_level_100) / df['close']
            df['dist_to_psych_50'] = (df['close'] - round_level_50) / df['close']
            
            # Indicateur de proximité à un niveau psychologique important
            df['near_psych_level'] = (
                (abs(df['dist_to_psych_100']) < 0.005) | 
                (abs(df['dist_to_psych_50']) < 0.0025)
            ).astype(int)
            
            # Franchissement d'un niveau psychologique
            df['cross_psych_100'] = (
                (df['close'] > round_level_100) & 
                (df['close'].shift(1) < round_level_100.shift(1))
            ).astype(int) - (
                (df['close'] < round_level_100) & 
                (df['close'].shift(1) > round_level_100.shift(1))
            ).astype(int)
        
        # 2. Caractéristiques de volume et de session
        if isinstance(df.index, pd.DatetimeIndex):
            # L'or a des comportements différents selon les sessions
            df['hour'] = df.index.hour
            
            # Sessions de marché importantes pour l'or
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            
            # Périodes de transition - particulièrement importantes pour l'or
            df['session_transition'] = (
                ((df['hour'] == 8) | (df['hour'] == 9)) |  # Asie-Londres
                ((df['hour'] == 13) | (df['hour'] == 14)) |  # Londres-NY
                ((df['hour'] == 21) | (df['hour'] == 22))  # NY-Asie
            ).astype(int)
            
            # Jours de la semaine - l'or a des comportements différents
            df['day_of_week'] = df.index.dayofweek
            
            # Vendredi après-midi - souvent des mouvements de clôture de position
            df['friday_close'] = (
                (df['day_of_week'] == 4) & 
                (df['hour'] >= 15) & 
                (df['hour'] <= 21)
            ).astype(int)
            
            # Premier jour du mois - souvent des rapports économiques importants
            df['first_day_of_month'] = (df.index.day == 1).astype(int)
            
            # Fenêtre NFP (Non-Farm Payroll) - vendredi de la première semaine du mois
            df['nfp_window'] = (
                (df.index.day <= 7) & 
                (df.index.day >= 1) & 
                (df['day_of_week'] == 4)
            ).astype(int)
        
        # 3. Caractéristiques de momentum et de tendance
        if 'close' in df.columns:
            # L'or présente souvent un momentum fort
            df['gold_momentum_1h'] = df['close'].pct_change(12)  # Pour 5-min
            df['gold_momentum_4h'] = df['close'].pct_change(48)
            df['gold_momentum_1d'] = df['close'].pct_change(288)
            
            # Indicateurs de force de momentum
            df['momentum_acceleration'] = df['gold_momentum_1h'] - df['gold_momentum_1h'].shift(12)
            
            # Divergence de momentum (prix vs. RSI)
            if 'rsi_14' in df.columns:
                df['price_higher'] = (df['close'] > df['close'].shift(24)).astype(int)
                df['rsi_higher'] = (df['rsi_14'] > df['rsi_14'].shift(24)).astype(int)
                df['rsi_divergence'] = ((df['price_higher'] == 1) & (df['rsi_higher'] == 0)).astype(int) - \
                                    ((df['price_higher'] == 0) & (df['rsi_higher'] == 1)).astype(int)
        
        # 4. Caractéristiques de volatilité spécifiques à l'or
        # L'or est connu pour sa volatilité variable selon les périodes
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Calculer la volatilité à différentes périodes
            df['daily_range'] = (df['high'] - df['low']) / df['close'] * 100  # en pourcentage
            df['daily_range_ma10'] = df['daily_range'].rolling(window=10).mean()
            df['daily_range_ma30'] = df['daily_range'].rolling(window=30).mean()
            
            # Ratio de volatilité (comparaison court terme vs moyen terme)
            df['volatility_ratio'] = df['daily_range_ma10'] / df['daily_range_ma30']
            
            # Contraction de range (signal potentiel de breakout imminent)
            df['range_contraction'] = (
                (df['daily_range'] < df['daily_range'].shift(1)) & 
                (df['daily_range'].shift(1) < df['daily_range'].shift(2)) & 
                (df['daily_range'].shift(2) < df['daily_range'].shift(3))
            ).astype(int)
            
            # Expansion de range (souvent après une contraction)
            df['range_expansion'] = (
                (df['daily_range'] > df['daily_range'].shift(1) * 1.5)
            ).astype(int)
            
            # Indicateur de "quiet gold" - périodes de faible volatilité
            df['quiet_gold'] = (df['daily_range'] < df['daily_range_ma30'] * 0.6).astype(int)
        
        # 5. Indicateurs de Sentiment et Flux
        if 'volume' in df.columns or 'tick_volume' in df.columns:
            # Utiliser le volume disponible
            vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
            
            # Différence de volume (variation par rapport à la moyenne)
            df['volume_ratio'] = df[vol_col] / df[vol_col].rolling(20).mean()
            
            # Volume anormal (pics de volume - souvent associés à des mouvements importants de l'or)
            df['abnormal_volume'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # Volume cumulatif directionnel (accumulation ou distribution)
            df['directional_volume'] = df[vol_col] * np.sign(df['close'] - df['close'].shift(1))
            df['cumulative_volume'] = df['directional_volume'].rolling(window=20).sum()
            
            # Intensité d'achat/vente
            if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
                # Intensité d'achat: (close - low) / (high - low)
                # Intensité de vente: (high - close) / (high - low)
                range_hl = df['high'] - df['low']
                df['buying_intensity'] = (df['close'] - df['low']) / range_hl
                df['selling_intensity'] = (df['high'] - df['close']) / range_hl
                
                # Pression d'achat/vente sur plusieurs périodes
                df['buying_pressure'] = df['buying_intensity'].rolling(10).mean()
                df['selling_pressure'] = df['selling_intensity'].rolling(10).mean()
        
        # 6. Caractéristiques de cassure (breakout) spécifiques à l'or
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Canal de prix (high et low sur 20 périodes)
            df['channel_high_20'] = df['high'].rolling(20).max()
            df['channel_low_20'] = df['low'].rolling(20).min()
            df['channel_width'] = (df['channel_high_20'] - df['channel_low_20']) / df['close'] * 100
            
            # Détection de cassure de canal
            df['breakout_up'] = (df['close'] > df['channel_high_20'].shift(1)).astype(int)
            df['breakout_down'] = (df['close'] < df['channel_low_20'].shift(1)).astype(int)
            
            # Force de la cassure
            df['breakout_strength'] = (
                (df['close'] - df['channel_high_20'].shift(1)) / df['channel_high_20'].shift(1) * 100
            ) * df['breakout_up'] + (
                (df['channel_low_20'].shift(1) - df['close']) / df['channel_low_20'].shift(1) * 100
            ) * df['breakout_down']
        
        # 7. Caractéristiques spécifiques aux réactions aux nouvelles économiques
        # Les réactions de l'or aux nouvelles sont souvent différentes des autres actifs
        if 'close' in df.columns:
            # Volatilité diurne vs nocturne
            if isinstance(df.index, pd.DatetimeIndex):
                # Période active du marché (8h-20h UTC)
                df['active_market_hours'] = ((df['hour'] >= 8) & (df['hour'] < 20)).astype(int)
                
                # Calculer la volatilité en périodes actives vs inactives
                df['active_volatility'] = df['close'].pct_change().abs() * df['active_market_hours']
                df['inactive_volatility'] = df['close'].pct_change().abs() * (1 - df['active_market_hours'])
                
                # Moyennes mobiles de volatilité
                df['active_vol_ma'] = df['active_volatility'].rolling(48).mean() * 100  # Convertir en pourcentage
                df['inactive_vol_ma'] = df['inactive_volatility'].rolling(48).mean() * 100
                
                # Ratio de volatilité (actif/inactif)
                df['vol_ratio_active_inactive'] = df['active_vol_ma'] / df['inactive_vol_ma'].replace(0, np.nan)
            
            # Réactions de retour à la moyenne (caractéristique de l'or)
            # L'or a tendance à revenir à la moyenne après des mouvements extrêmes
            df['deviation_from_ma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean() * 100
            df['extreme_deviation'] = (abs(df['deviation_from_ma20']) > 2.0).astype(int)
            
            # Indicateur de retour probable à la moyenne
            df['mean_reversion_signal'] = (
                (df['deviation_from_ma20'] > 2.0) |  # Fortement suracheté
                (df['deviation_from_ma20'] < -2.0)   # Fortement survendu
            ).astype(int)
        
        # 8. Indicateurs de corrélation
        # L'or a des corrélations importantes avec d'autres actifs (USD, taux, etc.)
        # Ici, nous utilisons des approximations puisque nous n'avons pas les données externes
        if isinstance(df.index, pd.DatetimeIndex) and 'close' in df.columns:
            # Proxy d'inversion de tendance du dollar (l'or monte souvent quand le dollar baisse)
            # Simuler par l'inversion de tendance de l'or lui-même
            df['usd_proxy_trend'] = -1 * np.sign(df['close'].pct_change(24).rolling(12).mean())
            
            # Proxy d'incertitude du marché (l'or monte souvent en période d'incertitude)
            # Simuler par la volatilité de l'or
            if 'atr_14' in df.columns:
                df['market_uncertainty_proxy'] = df['atr_14'] / df['atr_14'].rolling(60).mean()
                df['high_uncertainty'] = (df['market_uncertainty_proxy'] > 1.3).astype(int)
        
        # Supprimer les lignes avec des NaN introduits par les calculs
        # df = df.dropna()
        # Ne pas supprimer les NaN ici, laissez cette responsabilité à l'appelant
        
        return df