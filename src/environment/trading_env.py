"""
Trading environment module for reinforcement learning.

This module implements a custom OpenAI Gym (gymnasium) environment for trading
financial instruments. It handles the state representation, action space, 
reward calculation, and interaction with the MT5 connector.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.mt5_connector.connector import MT5Connector
from src.risk_management.position_sizer import PositionSizer
from src.utils.logger import setup_logger
from src.environment.reward_functions import RewardCalculator

logger = setup_logger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning that follows gym interface.
    """
    
    metadata = {'render.modes': ['human']}


# Mise à jour de la classe TradingEnvironment dans src/environment/trading_env.py

    def __init__(
        self,
        config: Dict,
        data: Optional[pd.DataFrame] = None,
        mode: str = 'backtest',
        window_size: int = 20,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        reward_type: str = 'sharpe',
        initial_balance: float = 10000.0,
        commission: float = 0.0001,  # 1 pip commission
        mt5_connector: Optional[MT5Connector] = None,
        position_sizer: Optional[PositionSizer] = None,
        features: Optional[List[str]] = None,
        reward_calculator: Optional[Any] = None  # Ajout du paramètre reward_calculator
    ):
        """
        Initialize the trading environment.
        
        Args:
            config: Configuration dictionary
            data: Historical price data for backtesting
            mode: 'backtest', 'paper', or 'live'
            window_size: Number of time steps to include in state
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Trading timeframe (e.g., 'M15')
            reward_type: Type of reward function to use
            initial_balance: Initial account balance
            commission: Trading commission as a fraction of trade value
            mt5_connector: Optional MT5 connector instance
            position_sizer: Optional position sizer instance
            features: List of feature columns to include in state
            reward_calculator: Optional custom reward calculator instance
        """
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.mode = mode
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Set trading parameters
        self.symbol = symbol or config['trading']['default_symbol']
        self.timeframe = timeframe or config['trading']['default_timeframe']
        
        # Initialize MT5 connector if we're in paper or live mode and one wasn't provided
        if mode in ['paper', 'live'] and mt5_connector is None:
            from src.mt5_connector.connector import MT5Connector
            self.mt5_connector = MT5Connector(config)
            self.mt5_connector.initialize()
        else:
            self.mt5_connector = mt5_connector
        
        # Initialize position sizer if one wasn't provided
        if position_sizer is None:
            from src.risk_management.position_sizer import PositionSizer
            self.position_sizer = PositionSizer(config)
        else:
            self.position_sizer = position_sizer
        
        # Initialize reward calculator
        if reward_calculator is not None:
            self.reward_calculator = reward_calculator
        else:
            self.reward_calculator = RewardCalculator(reward_type, window_size)
        
        # Set up data for backtesting
        self.data = data
        if mode == 'backtest' and data is None:
            raise ValueError("Data must be provided for backtest mode")
        
        # If features is None, use all columns in data except specific excluded ones
        if features is None and data is not None:
            excluded = ['open', 'high', 'low', 'close', 'volume']
            self.features = [col for col in data.columns if col not in excluded]
        else:
            self.features = features or []
        
        # Initialize state
        self._reset_state()
        
        # Define action and observation spaces
        self._define_spaces()

        
    def _define_spaces(self):
        """Define the action and observation spaces for the environment."""
        # Action space: [-1, 0, 1] for sell, hold, buy
        # We use discrete actions to simplify the problem
        self.action_space = spaces.Discrete(3)
        
        # Observation space includes:
        # 1. Price data window (normalized)
        # 2. Technical indicators
        # 3. Account information (normalized)
        
        # Number of features in the state
        n_features = len(self.features) if self.features else 5  # Default to OHLCV
        
        # Price window size (e.g., last 20 candles)
        price_shape = (self.window_size, n_features)
        
        # Account information: [balance, equity, current_position, unrealized_pnl, realized_pnl]
        account_shape = (5,)
        
        # Total observation space
        self.observation_space = spaces.Dict({
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=price_shape),
            'account': spaces.Box(low=-np.inf, high=np.inf, shape=account_shape)
        })
    
    def _reset_state(self):
        """Reset the internal state variables."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_position = 0  # -1 for short, 0 for flat, 1 for long
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        
        # Performance metrics
        self.returns = []
        self.daily_returns = []
        self.holding_period = 0
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        self._reset_state()
        
        if self.mode == 'backtest':
            # For backtesting, make sure we have at least window_size data points
            if len(self.data) <= self.window_size:
                raise ValueError(f"Not enough data points. Need more than {self.window_size}")
            
            self.current_step = self.window_size
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self):
        """
        Construct the current observation (state).
        
        Returns:
            Dict: Observation state containing market and account data
        """
        if self.mode == 'backtest':
            # For backtesting, get a window of historical data
            market_data = self._get_market_window()
        else:
            # For paper/live trading, get the latest data from MT5
            market_data = self._get_live_market_window()
        
        # Get account state
        account_data = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.equity / self.initial_balance,   # Normalized equity
            self.current_position,                # Current position (-1, 0, 1)
            self.unrealized_pnl / self.balance,   # Normalized unrealized PnL
            self.realized_pnl / self.balance      # Normalized realized PnL
        ])
        
        return {
            'market': market_data,
            'account': account_data
        }
    
    def _get_market_window(self):
        """
        Get a window of market data for backtesting.
        
        Returns:
            np.ndarray: Market data window
        """
        # Get the data window ending at the current step
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Extract the feature columns we want to include
        if self.features:
            market_window = self.data.iloc[start_idx:end_idx][self.features].values
        else:
            # Default to normalized OHLCV if features not specified
            norm_cols = [col for col in self.data.columns if col.endswith('_norm')]
            market_window = self.data.iloc[start_idx:end_idx][norm_cols[:5]].values
        
        return market_window
    
    def _get_live_market_window(self):
        """
        Get a window of market data for paper/live trading.
        
        Returns:
            np.ndarray: Market data window
        """
        if self.mt5_connector is None:
            raise ValueError("MT5 connector is required for paper/live trading")
        
        # Get the latest data window from MT5
        df = self.mt5_connector.get_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            num_bars=self.window_size
        )
        
        if df is None or len(df) < self.window_size:
            raise ValueError(f"Could not fetch enough data from MT5: {len(df) if df is not None else 0}/{self.window_size}")
        
        # Extract the feature columns or default to OHLCV
        if self.features:
            # If features require calculated indicators, we need to generate them
            from src.data.feature_generator import FeatureGenerator
            feature_gen = FeatureGenerator(self.config)
            df = feature_gen.generate_all_features(df)
            
            # Extract specified features
            market_window = df[self.features].values
        else:
            # Normalize the OHLCV data for a basic state
            from src.data.preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(self.config)
            df = preprocessor.normalize_data(df)
            
            # Extract normalized columns
            norm_cols = [col for col in df.columns if col.endswith('_norm')]
            market_window = df[norm_cols[:5]].values
        
        return market_window
    
    def _get_current_price(self):
        """
        Get the current market price.
        
        Returns:
            Tuple[float, float, float]: (bid, ask, spread) prices
        """
        if self.mode == 'backtest':
            # For backtesting, use the close price from the data
            close_price = self.data.iloc[self.current_step]['close']
            # Simulate bid-ask spread
            bid = close_price * (1 - self.commission / 2)
            ask = close_price * (1 + self.commission / 2)
            spread = ask - bid
            return bid, ask, spread
        
        else:
            # For paper/live trading, get the latest price from MT5
            if self.mt5_connector is None:
                raise ValueError("MT5 connector is required for paper/live trading")
            
            tick = self.mt5_connector.get_current_tick(self.symbol)
            
            if tick is None:
                raise ValueError(f"Could not fetch current price for {self.symbol}")
            
            return tick['bid'], tick['ask'], tick['ask'] - tick['bid']
        
    def step(self, action):
        """
        Execute one time step within the environment.
    
        Args:
            action: The action to take (0: sell, 1: hold, 2: buy)
        
        Returns:
            observation: The new state
            reward: The reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert action from [0, 1, 2] to [-1, 0, 1]
        action_map = [-1, 0, 1]  # sell, hold, buy
        
        # IMPORTANT: Forcer des actions périodiquement pendant le backtest
        if self.mode == 'backtest':
            # Forcer des actions tous les X pas (environ une fois par jour en M5)
            if self.current_step % 150 == 0:
                # 40% chances d'acheter, 40% chances de vendre, 20% de garder l'action originale
                random_val = np.random.random()
                if random_val < 0.4:
                    action = 0  # Vendre
                elif random_val < 0.8:
                    action = 2  # Acheter
                # Sinon, garder l'action originale
            # Si une longue période sans trade, forcer un changement de position
            elif len(self.trade_history) == 0 and self.current_step > 500:
                # Forcer une action si aucun trade après 500 pas
                action = 0 if np.random.random() < 0.5 else 2
        
        action = action_map[action]
    
        # Get current prices
        bid, ask, spread = self._get_current_price()
    
        # Execute the trade
        self._execute_trade(action, bid, ask)
    
        # Move to the next time step
        if self.mode == 'backtest':
            self.current_step += 1
            done = self.current_step >= len(self.data) - 1
        else:
            # For paper/live trading, we're never done (continuous trading)
            done = False
    
        # Calculate reward
        if self.mode == 'backtest':
            # Get price data for reward calculation
            prices = self.data.iloc[self.current_step-self.window_size:self.current_step]['close'].values
            
            # Ajouter une récompense supplémentaire pour avoir exécuté des trades
            base_reward = self.reward_calculator.calculate_reward(
                action=action,
                position=self.current_position,
                unrealized_pnl=self.unrealized_pnl,
                realized_pnl=self.realized_pnl,
                balance=self.balance,
                equity=self.equity,
                prices=prices
            )
            
            # Ajouter un bonus pour les actions de trading
            trading_bonus = 0
            if action != 0:  # Si action de trading (pas "hold")
                trading_bonus = 0.2
            
            # Pénaliser davantage l'inaction (surtout en l'absence de trades)
            if action == 0 and self.current_position == 0:
                # Pénalité plus forte si aucun trade n'a encore été exécuté
                if len(self.trade_history) == 0:
                    trading_bonus = -0.3
            
            # Combiner les récompenses
            reward = base_reward + trading_bonus
        else:
            # Code pour le paper/live trading (inchangé)
            df = self.mt5_connector.get_historical_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                num_bars=self.window_size
            )
            prices = df['close'].values
            reward = self.reward_calculator.calculate_reward(
                action=action,
                position=self.current_position,
                unrealized_pnl=self.unrealized_pnl,
                realized_pnl=self.realized_pnl,
                balance=self.balance,
                equity=self.equity,
                prices=prices
            )
    
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
    
        return observation, reward, done, False, info

    def _execute_trade(self, action, bid, ask):
        """
        Execute a trade based on the given action.
        
        Args:
            action: The action to take (-1: sell, 0: hold, 1: buy)
            bid: Current bid price
            ask: Current ask price
        """
        # Calculate mid price for PnL calculations
        mid_price = (bid + ask) / 2
        
        # Update the holding period if we have a position
        if self.current_position != 0:
            self.holding_period += 1
        
        # MODIFICATION: Forcer l'exécution des trades pour l'or
        # Surtout en backtest où nous avons observé que le modèle ne tradait pas
        if self.mode == 'backtest':
            # Si action est d'ouvrir une position et aucun trade exécuté jusqu'à présent
            force_trade = (action != 0 and len(self.trade_history) == 0)
            
            # Ou si action est d'ouvrir une position et pas de trade depuis longtemps
            if action != 0 and len(self.trade_history) > 0:
                last_trade_time = self.trade_history[-1]['exit_time']
                time_since_last_trade = self.current_step - last_trade_time
                force_trade = force_trade or (time_since_last_trade > 300)  # ~25 heures en M5
        else:
            force_trade = False
        
        # Calculate position size with additional logic for forced trades
        if (action != 0 and action != self.current_position) or force_trade:
            # Calculer la taille normale de position
            position_size = self.position_sizer.calculate_position_size(
                action=action,
                balance=self.balance,
                price=mid_price,
                symbol=self.symbol
            )
            
            # Force a minimum position size for gold to ensure trades are executed
            min_position_size = 0.01  # Micro lot minimum
            position_size = max(position_size, min_position_size)
            
            # Pour les trades forcés, utiliser une taille plus petite mais garantie
            if force_trade:
                position_size = min_position_size
                # Assurons-nous que l'action n'est pas "hold" pour un trade forcé
                if action == 0:
                    # Choisir action d'achat ou de vente aléatoirement
                    action = 1 if np.random.random() < 0.5 else -1
        else:
            position_size = 0
        
        # Handle closing an existing position
        if self.current_position != 0 and (action == 0 or action == -self.current_position):
            # Calculate PnL
            if self.current_position == 1:  # Closing a long position
                exit_price = bid  # We sell at the bid price
                pnl = (exit_price - self.entry_price) * abs(self.current_position)
            else:  # Closing a short position
                exit_price = ask  # We buy at the ask price
                pnl = (self.entry_price - exit_price) * abs(self.current_position)
            
            # Deduct commission
            pnl -= self.commission * abs(self.current_position) * mid_price
            
            # Update balance and pnl
            self.realized_pnl += pnl
            self.balance += pnl
            self.equity = self.balance  # When closing, equity equals balance
            self.unrealized_pnl = 0
            
            # Record the trade
            self.trade_history.append({
                'entry_time': self.current_step - self.holding_period,
                'exit_time': self.current_step,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'position': self.current_position,
                'pnl': pnl,
                'holding_period': self.holding_period,
                'forced': force_trade  # Marquer si c'était un trade forcé
            })
            
            # Reset position and holding period
            self.current_position = 0
            self.entry_price = 0
            self.holding_period = 0
        
        # Handle opening a new position - MODIFICATION pour assurer l'exécution
        if (action != 0 and self.current_position == 0) or (self.current_position == 0 and force_trade):
            if action == 1:  # Buy
                self.entry_price = ask  # We buy at the ask price
            else:  # Sell
                self.entry_price = bid  # We sell at the bid price
            
            # Update position
            self.current_position = action
            
            # Log information about this trade
            if self.mode == 'backtest' and force_trade:
                logger.info(f"Forced trade executed at step {self.current_step}: {'BUY' if action == 1 else 'SELL'} at {self.entry_price}")
        
        # Handle reversing a position (e.g., from long to short)
        if self.current_position != 0 and action == -self.current_position:
            if action == 1:  # Buy (was short)
                self.entry_price = ask  # We buy at the ask price
            else:  # Sell (was long)
                self.entry_price = bid  # We sell at the bid price
            
            # Update position
            self.current_position = action
        
        # Calculate unrealized PnL for open positions
        if self.current_position != 0:
            if self.current_position == 1:  # Long position
                self.unrealized_pnl = (bid - self.entry_price) * abs(self.current_position)
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - ask) * abs(self.current_position)
            
            # Deduct estimated commission
            self.unrealized_pnl -= self.commission * abs(self.current_position) * mid_price
            
            # Update equity
            self.equity = self.balance + self.unrealized_pnl

    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            Dict: Dictionary containing additional information
        """
        return {
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'holding_period': self.holding_period,
            'trade_count': len(self.trade_history),
            'position_size': abs(self.current_position)
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Render mode (human, rgb_array, etc.)
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not supported")
        
        # Get current step info
        if self.mode == 'backtest':
            current_time = self.data.index[self.current_step]
            current_price = self.data.iloc[self.current_step]['close']
        else:
            # For paper/live trading, get the latest info from MT5
            tick = self.mt5_connector.get_current_tick(self.symbol)
            current_time = tick['time']
            current_price = (tick['bid'] + tick['ask']) / 2
        
        # Print current state
        position_str = "FLAT" if self.current_position == 0 else "LONG" if self.current_position == 1 else "SHORT"
        
        print(f"Time: {current_time}")
        print(f"Symbol: {self.symbol}")
        print(f"Price: {current_price:.5f}")
        print(f"Position: {position_str}")
        print(f"Entry Price: {self.entry_price:.5f}")
        print(f"Unrealized PnL: ${self.unrealized_pnl:.2f}")
        print(f"Realized PnL: ${self.realized_pnl:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Equity: ${self.equity:.2f}")
        print(f"Trade Count: {len(self.trade_history)}")
        print("-" * 40)
    
    def close(self):
        """Close the environment and release resources."""
        if self.mt5_connector is not None and self.mode in ['paper', 'live']:
            self.mt5_connector.shutdown()
        
        super().close()