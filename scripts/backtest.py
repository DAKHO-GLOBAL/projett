"""
Script for backtesting the trained reinforcement learning trading agent.

This script loads a trained agent model and evaluates its performance on
historical data, generating trading signals, performance metrics, and
visualization of the results.
"""

import os
import sys
import time
import logging
import yaml
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_collector import MT5DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_generator import FeatureGenerator
from src.environment.trading_env import TradingEnvironment
from src.agent.ppo_agent import PPOAgent
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer
from src.utils.metrics import calculate_comprehensive_metrics


# Importer la nouvelle classe de récompense
from src.environment.custom_rewards import GoldTradingReward

# Créer une instance
gold_reward = GoldTradingReward()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest RL trading agent')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--symbol', type=str, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe')
    parser.add_argument('--start_date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
    
    config = {}
    
    # Load main config
    config_path = os.path.join(config_dir, 'config.yml')
    main_config = load_config(config_path)
    config.update(main_config)
    
    # Load model config
    model_config_path = os.path.join(config_dir, 'model_config.yml')
    model_config = load_config(model_config_path)
    config['model_config'] = model_config
    
    # Load trading parameters
    trading_params_path = os.path.join(config_dir, 'trading_params.yml')
    trading_params = load_config(trading_params_path)
    config['trading_params'] = trading_params
    
    # Setup logging
    log_path = os.path.join(config['paths']['logs'], 'backtest.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger('backtest', log_file=log_path)
    
    # Log start of backtesting
    logger.info(f"Starting backtesting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Get parameters from arguments or config
    symbol = args.symbol if args.symbol else config['trading']['default_symbol']
    timeframe = args.timeframe if args.timeframe else config['trading']['default_timeframe']
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = None
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = None
    
    model_path = args.model if args.model else os.path.join(
        config['paths']['models'], 
        'saved', 
        f"{symbol}_{timeframe}_ppo.pt"
    )
    
    # Validate model path
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        available_models = [f for f in os.listdir(os.path.dirname(model_path)) if f.endswith('.pt')]
        if available_models:
            logger.info(f"Available models: {available_models}")
        return
    
    try:
        # Data collection and preprocessing
        logger.info("Collecting and preprocessing data...")
        
        # Initialize data collector
        data_collector = MT5DataCollector(config)
        data_collector.initialize()
        
        # Collect data
        df = data_collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or len(df) < 100:
            logger.error("Failed to collect sufficient data!")
            return
        
        logger.info(f"Collected {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        # Generate features
        logger.info("Generating technical indicators and features...")
        feature_generator = FeatureGenerator(config)
        df = feature_generator.generate_all_features(df)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(config)
        df = preprocessor.clean_data(df)
        df = preprocessor.calculate_returns(df)
        df = preprocessor.normalize_data(df)
        
        # Define feature set for the environment
        feature_columns = [col for col in df.columns if col.endswith('_norm')]
        logger.info(f"Using {len(feature_columns)} features")
        
        # Create environment
        logger.info("Setting up trading environment for backtesting...")
        window_size = config['data'].get('feature_window', 20)
        
        env = TradingEnvironment(
            config=config,
            data=df,
            mode='backtest',
            window_size=window_size,
            symbol=symbol,
            timeframe=timeframe,
            reward_type='gold_trading',  # Utiliser cette chaîne comme identifiant
            reward_calculator=gold_reward,  # Passer directement l'objet 
            features=feature_columns
        )
        
        # Initialize the observation to get state dimensions
        obs, _ = env.reset()
        
        # Calculate state and action dimensions
        if isinstance(obs, dict):
            # For dict observation space, flatten each observation component
            state_dim = 0
            if 'market' in obs:
                state_dim += np.prod(obs['market'].shape)
            if 'account' in obs:
                state_dim += np.prod(obs['account'].shape)
        else:
            state_dim = np.prod(obs.shape)
        
        action_dim = env.action_space.n
        
        # Initialize the agent
        logger.info("Loading trained PPO agent...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = PPOAgent(
            config=config,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            model_name=f"{symbol}_{timeframe}_ppo"
        )
        
        # Load the trained model
        logger.info(f"Loading model from {model_path}")
        agent.load_model(model_path)
        
        # Run backtest
        logger.info("Starting backtest...")
        start_time = time.time()
        
        # Reset environment
        obs, _ = env.reset()
        
        done = False
        
        # Track trade history
        trade_history = []
        
        # Track equity curve
        equity_curve = [env.balance]
        
        # Track returns
        returns = []
        
        # Track actions
        actions = []
        prices = []
        timestamps = []
        positions = []
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(obs, evaluate=True)
            
            # Record the state before the action
            timestamps.append(df.index[env.current_step])
            prices.append(df.iloc[env.current_step]['close'])
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Record the action
            actions.append(action)
            positions.append(env.current_position)
            
            # Record equity
            equity_curve.append(info['equity'])
            
            # Calculate return
            if len(equity_curve) > 1:
                ret = (equity_curve[-1] / equity_curve[-2]) - 1
                returns.append(ret)
            
            # Update observation
            obs = next_obs
            
            # Record trade if position closed
            if len(env.trade_history) > len(trade_history):
                trade = env.trade_history[-1]
                trade_history.append(trade)
                
                logger.info(f"Trade: Entry: {trade['entry_time']}, Exit: {trade['exit_time']}, "
                           f"Position: {trade['position']}, P&L: {trade['pnl']:.2f}")
            
            if truncated:
                break
        
        # Calculate backtest duration
        duration = time.time() - start_time
        logger.info(f"Backtest completed in {duration:.2f} seconds")
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'action': actions,
            'position': positions
        })
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df.set_index('timestamp', inplace=True)
        
        # Add buy/sell signals for visualization
        signals_df['buy'] = (signals_df['action'] == 2).astype(int)  # Buy action (2)
        signals_df['sell'] = (signals_df['action'] == 0).astype(int)  # Sell action (0)
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        
        # Extract trade profits/losses
        trade_pnls = np.array([trade['pnl'] for trade in trade_history])
        
        # Convert equity curve to numpy array
        equity_array = np.array(equity_curve)
        
        # Calculate returns array
        returns_array = np.array(returns)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            returns=returns_array,
            equity_curve=equity_array,
            trades=trade_pnls
        )
        
        # Log key metrics
        logger.info(f"Total Return: {metrics['total_return']:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.4f}")
        logger.info(f"Total Trades: {len(trade_history)}")
        
        # Visualize backtest results
        logger.info("Generating backtest visualizations...")
        visualizer = Visualizer(config)
        
        # Plot price with signals
        visualizer.plot_price_with_signals(
            df=df,
            signals=signals_df,
            title=f"Backtest Results - {symbol} {timeframe}",
            filename=f"backtest_signals_{symbol}_{timeframe}.png"
        )
        
        # Plot equity curve
        equity_series = pd.Series(equity_curve, index=[df.index[0]] + list(df.index[window_size:]))
        
        visualizer.plot_equity_curve(
            equity=equity_series,
            title=f"Equity Curve - {symbol} {timeframe}",
            filename=f"backtest_equity_{symbol}_{timeframe}.png"
        )
        
        # Save trade history to CSV
        trades_df = pd.DataFrame(trade_history)
        trades_df.to_csv(os.path.join(config['paths']['data'], f"backtest_trades_{symbol}_{timeframe}.csv"))
        
        # Save signals to CSV
        signals_df.to_csv(os.path.join(config['paths']['data'], f"backtest_signals_{symbol}_{timeframe}.csv"))
        
        # Plot trade analysis if we have enough trades
        if len(trade_history) > 5:
            # Add more details to trades DataFrame
            if 'entry_time' in trades_df.columns:
                # Convert timestamps to datetime
                trades_df['entry_time'] = pd.to_datetime(
                    [df.index[t] for t in trades_df['entry_time']]
                )
                trades_df['exit_time'] = pd.to_datetime(
                    [df.index[t] for t in trades_df['exit_time']]
                )
                
                # Calculate duration in hours
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                
                # Add symbol column
                trades_df['symbol'] = symbol
                
                # Add trade type
                trades_df['type'] = trades_df['position'].map({1: 'BUY', -1: 'SELL'})
                
                visualizer.plot_trade_analysis(
                    trades=trades_df,
                    title=f"Trade Analysis - {symbol} {timeframe}",
                    filename=f"backtest_trades_{symbol}_{timeframe}.png"
                )
        
        # Plot performance metrics
        visualizer.plot_performance_metrics(
            metrics=metrics,
            title=f"Performance Metrics - {symbol} {timeframe}",
            filename=f"backtest_metrics_{symbol}_{timeframe}.png"
        )
        
        logger.info("Backtesting completed successfully!")
    
    except Exception as e:
        logger.exception(f"Error during backtesting: {e}")
    
    finally:
        # Clean up
        if 'data_collector' in locals() and data_collector.initialized:
            data_collector.shutdown()
        
        logger.info(f"Backtesting script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()