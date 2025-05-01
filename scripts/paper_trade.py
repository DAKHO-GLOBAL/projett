"""
Script for paper trading with the trained reinforcement learning agent.

This script loads a trained agent model and runs it in paper trading mode,
connecting to MetaTrader 5 to receive real-time market data but executing
trades in a simulated environment without real money.
"""

import os
import sys
import time
import logging
import yaml
import argparse
import signal
from datetime import datetime, timedelta
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
from src.mt5_connector.connector import MT5Connector
from src.mt5_connector.order_manager import OrderManager
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.stop_manager import StopManager

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Paper trade with RL trading agent')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--symbol', type=str, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe')
    parser.add_argument('--duration', type=int, default=60, help='Trading duration in minutes')
    parser.add_argument('--interval', type=int, default=5, help='Trading interval in seconds')
    
    return parser.parse_args()

def signal_handler(signal, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal. Shutting down...")
    global running
    running = False

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
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
    log_path = os.path.join(config['paths']['logs'], 'paper_trade.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    global logger
    logger = setup_logger('paper_trade', log_file=log_path)
    
    # Log start of paper trading
    logger.info(f"Starting paper trading at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Get parameters from arguments or config
    symbol = args.symbol if args.symbol else config['trading']['default_symbol']
    timeframe = args.timeframe if args.timeframe else config['trading']['default_timeframe']
    duration = args.duration
    interval = args.interval
    
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
        # Initialize MT5 connector
        logger.info("Initializing MT5 connector...")
        mt5_connector = MT5Connector(config)
        if not mt5_connector.initialize():
            logger.error("Failed to initialize MT5 connector!")
            return
        
        # Get account info
        account_info = mt5_connector.get_account_info()
        if account_info is None:
            logger.error("Failed to get account information!")
            return
        
        logger.info(f"Connected to account: {account_info['login']}, Balance: {account_info['balance']}, Server: {account_info['server']}")
        
        # Initialize components
        logger.info("Initializing trading components...")
        
        # Feature generator for real-time data preprocessing
        feature_generator = FeatureGenerator(config)
        
        # Data preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Position sizer
        position_sizer = PositionSizer(config)
        
        # Stop manager
        stop_manager = StopManager(config)
        
        # Order manager
        order_manager = OrderManager(config, mt5_connector)
        
        # Create environment
        logger.info("Setting up trading environment for paper trading...")
        window_size = config['data'].get('feature_window', 20)
        
        # Get initial historical data for setting up the environment
        historical_data = mt5_connector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            num_bars=window_size + 100  # Get extra data for feature generation
        )
        
        if historical_data is None or len(historical_data) < window_size:
            logger.error("Failed to get sufficient historical data!")
            return
        
        # Generate features
        historical_data = feature_generator.generate_all_features(historical_data)
        
        # Preprocess data
        historical_data = preprocessor.clean_data(historical_data)
        historical_data = preprocessor.calculate_returns(historical_data)
        historical_data = preprocessor.normalize_data(historical_data)
        
        # Define feature set for the environment
        feature_columns = [col for col in historical_data.columns if col.endswith('_norm')]
        logger.info(f"Using {len(feature_columns)} features")
        
        # Create paper trading environment
        env = TradingEnvironment(
            config=config,
            data=None,  # No static data for paper trading
            mode='paper',
            window_size=window_size,
            symbol=symbol,
            timeframe=timeframe,
            reward_type='sharpe',
            mt5_connector=mt5_connector,
            position_sizer=position_sizer,
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
        
        # Start paper trading
        logger.info(f"Starting paper trading for {duration} minutes with {interval} second intervals...")
        
        # Calculate end time
        end_time = datetime.now() + timedelta(minutes=duration)
        
        # Trading loop
        global running
        running = True
        
        # Track trade performance
        trade_history = []
        equity_history = []
        action_history = []
        
        # Main trading loop
        while running and datetime.now() < end_time:
            try:
                # Get current state
                tick = mt5_connector.get_current_tick(symbol)
                if tick is None:
                    logger.warning("Failed to get current tick, retrying...")
                    time.sleep(1)
                    continue
                
                # Get historical data for feature generation
                current_data = mt5_connector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_bars=window_size + 100  # Get extra data for feature generation
                )
                
                if current_data is None or len(current_data) < window_size:
                    logger.warning("Failed to get sufficient historical data, retrying...")
                    time.sleep(1)
                    continue
                
                # Generate features
                current_data = feature_generator.generate_all_features(current_data)
                
                # Preprocess data
                current_data = preprocessor.clean_data(current_data)
                current_data = preprocessor.calculate_returns(current_data)
                current_data = preprocessor.normalize_data(current_data)
                
                # Update environment
                obs, _ = env.reset()  # Reset to get the latest state
                
                # Select action
                action, _, _ = agent.select_action(obs, evaluate=True)
                
                # Map action to trading action (0=sell, 1=hold, 2=buy)
                if action == 0:
                    trade_action = "SELL"
                elif action == 2:
                    trade_action = "BUY"
                else:
                    trade_action = "HOLD"
                
                # Log the action
                logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                           f"Symbol: {symbol}, Price: {tick['bid']}/{tick['ask']}, "
                           f"Action: {trade_action}")
                
                # Execute the action in the environment
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Record state
                equity_history.append({
                    'timestamp': datetime.now(),
                    'balance': info['balance'],
                    'equity': info['equity'],
                    'unrealized_pnl': info['unrealized_pnl'],
                    'position': info['current_position']
                })
                
                # Record action
                action_history.append({
                    'timestamp': datetime.now(),
                    'action': action,
                    'trade_action': trade_action,
                    'price_bid': tick['bid'],
                    'price_ask': tick['ask'],
                    'reward': reward
                })
                
                # Record trade if position closed
                if len(env.trade_history) > len(trade_history):
                    trade = env.trade_history[-1]
                    trade_history.append(trade)
                    
                    logger.info(f"Trade closed: Entry: {trade['entry_time']}, Exit: {trade['exit_time']}, "
                               f"Position: {trade['position']}, P&L: {trade['pnl']:.2f}")
                
                # Wait for next interval
                time.sleep(interval)
            
            except Exception as e:
                logger.exception(f"Error during paper trading loop: {e}")
                time.sleep(interval)
        
        # Trading completed
        logger.info("Paper trading completed!")
        
        # Save results
        if equity_history:
            equity_df = pd.DataFrame(equity_history)
            equity_df.to_csv(os.path.join(config['paths']['data'], f"paper_trade_equity_{symbol}_{timeframe}.csv"))
        
        if action_history:
            action_df = pd.DataFrame(action_history)
            action_df.to_csv(os.path.join(config['paths']['data'], f"paper_trade_actions_{symbol}_{timeframe}.csv"))
        
        if trade_history:
            trade_df = pd.DataFrame(trade_history)
            trade_df.to_csv(os.path.join(config['paths']['data'], f"paper_trade_trades_{symbol}_{timeframe}.csv"))
            
            # Calculate performance
            total_trades = len(trade_history)
            profitable_trades = sum(1 for trade in trade_history if trade['pnl'] > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            total_profit = sum(trade['pnl'] for trade in trade_history)
            
            logger.info(f"Performance Summary:")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Profitable Trades: {profitable_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Total Profit: {total_profit:.2f}")
        else:
            logger.info("No trades were executed during the paper trading period.")
    
    except Exception as e:
        logger.exception(f"Error during paper trading: {e}")
    
    finally:
        # Clean up
        if 'mt5_connector' in locals() and mt5_connector.initialized:
            mt5_connector.shutdown()
        
        logger.info(f"Paper trading script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()