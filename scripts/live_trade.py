"""
Script for live trading with the trained reinforcement learning agent.

This script loads a trained agent model and runs it in live trading mode,
connecting to MetaTrader 5 to receive real-time market data and execute
actual trades with real money.

WARNING: This script performs real trading with real money. Use at your own risk.
"""

import os
import sys
import time
import logging
import yaml
import argparse
import signal
import json
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
from src.mt5_connector.account_manager import AccountManager
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.stop_manager import StopManager

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Live trade with RL trading agent')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--symbol', type=str, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe')
    parser.add_argument('--confirm', action='store_true', help='Confirm live trading with real money')
    parser.add_argument('--max_risk', type=float, default=0.01, help='Maximum risk per trade (fraction of account)')
    
    return parser.parse_args()

def signal_handler(signal, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal. Shutting down...")
    global running
    running = False

def get_confirmation():
    """Get user confirmation for live trading."""
    print("\n" + "!" * 80)
    print("WARNING: LIVE TRADING WITH REAL MONEY".center(80))
    print("!" * 80)
    print("\nThis script will execute REAL trades with REAL money.")
    print("Make sure you understand the risks involved.\n")
    
    confirmation = input("Do you want to proceed with LIVE trading? (yes/no): ")
    
    return confirmation.lower() == 'yes'

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check for confirmation
    if not args.confirm and not get_confirmation():
        print("Live trading canceled by user.")
        return
    
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
    
    # Override risk parameter from command line
    if args.max_risk:
        config['trading_params']['position_sizing']['risk_per_trade'] = args.max_risk
    
    # Setup logging
    log_path = os.path.join(config['paths']['logs'], 'live_trade.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    global logger
    logger = setup_logger('live_trade', log_file=log_path)
    
    # Log start of live trading
    logger.info(f"Starting LIVE trading at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Get parameters from arguments or config
    symbol = args.symbol if args.symbol else config['trading']['default_symbol']
    timeframe = args.timeframe if args.timeframe else config['trading']['default_timeframe']
    
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
        
        # Account manager
        account_manager = AccountManager(config, mt5_connector)
        
        # Create environment
        logger.info("Setting up trading environment for live trading...")
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
        
        # Create live trading environment
        env = TradingEnvironment(
            config=config,
            data=None,  # No static data for live trading
            mode='live',
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
        
        # Check trading hours
        trading_hours = config['trading_params'].get('trading_hours', {})
        trading_enabled = trading_hours.get('enabled', False)
        
        # Save trading session info
        session_info = {
            'start_time': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'model': model_path,
            'account_id': account_info['login'],
            'initial_balance': account_info['balance'],
            'risk_per_trade': config['trading_params']['position_sizing']['risk_per_trade'],
            'max_positions': config['trading_params']['position_sizing']['max_positions']
        }
        
        session_file = os.path.join(config['paths']['logs'], f"live_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(session_file, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        logger.info(f"Session info saved to {session_file}")
        
        # Start live trading
        logger.info("Starting LIVE trading session...")
        
        # Trading loop
        global running
        running = True
        
        # Track trade performance
        trade_history = []
        equity_history = []
        action_history = []
        
        # Calculate appropriate time interval based on timeframe
        if timeframe == "M1":
            interval = 10  # seconds
        elif timeframe == "M5":
            interval = 30
        elif timeframe == "M15":
            interval = 60
        elif timeframe == "M30":
            interval = 120
        elif timeframe == "H1":
            interval = 300
        else:
            interval = 60  # Default
        
        logger.info(f"Trading interval set to {interval} seconds for {timeframe} timeframe")
        
        # Last candle time to track new candles
        last_candle_time = None
        
        # Main trading loop
        while running:
            try:
                # Check if we're within trading hours if enabled
                if trading_enabled:
                    now = datetime.now().replace(microsecond=0)
                    current_day = now.strftime('%A')
                    
                    start_time_str = trading_hours.get('start_time', '08:00')
                    end_time_str = trading_hours.get('end_time', '16:00')
                    trade_days = trading_hours.get('trade_days', ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                    
                    start_time = datetime.strptime(start_time_str, '%H:%M').time()
                    end_time = datetime.strptime(end_time_str, '%H:%M').time()
                    
                    if current_day not in trade_days or not (start_time <= now.time() <= end_time):
                        logger.info(f"Outside trading hours ({start_time_str}-{end_time_str} on {trade_days}). Waiting...")
                        time.sleep(60)  # Check again in a minute
                        continue
                
                # Get current time
                current_time = datetime.now()
                
                # Get current tick
                tick = mt5_connector.get_current_tick(symbol)
                if tick is None:
                    logger.warning("Failed to get current tick, retrying...")
                    time.sleep(1)
                    continue
                
                # Get latest candle
                latest_data = mt5_connector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_bars=1
                )
                
                if latest_data is None or latest_data.empty:
                    logger.warning("Failed to get latest candle, retrying...")
                    time.sleep(1)
                    continue
                
                # Check if we have a new candle
                latest_candle_time = latest_data.index[0]
                
                if last_candle_time is not None and latest_candle_time <= last_candle_time:
                    # No new candle yet, wait
                    time.sleep(interval)
                    continue
                
                # Update last candle time
                last_candle_time = latest_candle_time
                
                logger.info(f"New candle detected at {latest_candle_time}")
                
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
                logger.info(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                           f"Symbol: {symbol}, Price: {tick['bid']}/{tick['ask']}, "
                           f"Action: {trade_action}")
                
                # Check risk limits before executing
                risk_check = account_manager.check_risk_limits()
                
                if risk_check['max_positions_exceeded']:
                    logger.warning(f"Maximum positions ({risk_check['total_positions']}/{config['trading_params']['position_sizing']['max_positions']}) reached, skipping trade")
                    continue
                
                if risk_check['max_risk_exceeded']:
                    logger.warning("Maximum risk per trade exceeded, skipping trade")
                    continue
                
                if risk_check['max_daily_risk_exceeded']:
                    logger.warning("Maximum daily risk exceeded, skipping trade")
                    continue
                
                # Execute the action in the environment (which handles the actual trade execution)
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Record equity state
                account_info = mt5_connector.get_account_info()
                if account_info:
                    equity_history.append({
                        'timestamp': current_time,
                        'balance': account_info['balance'],
                        'equity': account_info['equity'],
                        'margin': account_info['margin'],
                        'free_margin': account_info['margin_free'],
                        'margin_level': account_info['margin_level']
                    })
                
                # Record action
                action_history.append({
                    'timestamp': current_time,
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
                
                # Check for trailing stops and breakeven for existing positions
                positions = order_manager.get_open_positions(symbol=symbol)
                
                for position in positions:
                    # Get position details
                    position_id = position['ticket']
                    entry_price = position['price_open']
                    current_price = tick['bid'] if position['type_str'] == 'BUY' else tick['ask']
                    direction = 1 if position['type_str'] == 'BUY' else -1
                    stop_loss = position['sl']
                    take_profit = position['tp']
                    
                    # Calculate trailing stop if enabled
                    if stop_manager.use_trailing_stop:
                        new_stop = stop_manager.calculate_trailing_stop(
                            entry_price=entry_price,
                            current_price=current_price,
                            direction=direction,
                            initial_stop=stop_loss,
                            take_profit=take_profit
                        )
                        
                        # If trailing stop moved, update the position
                        if new_stop != stop_loss:
                            logger.info(f"Updating trailing stop for position {position_id} from {stop_loss} to {new_stop}")
                            order_manager.modify_position(
                                position_id=position_id,
                                stop_loss=new_stop,
                                comment="Trailing stop update"
                            )
                    
                    # Check if we should move to breakeven
                    elif stop_manager.should_move_to_breakeven(
                        entry_price=entry_price,
                        current_price=current_price,
                        direction=direction,
                        initial_stop=stop_loss,
                        take_profit=take_profit
                    ):
                        # Add a small buffer to breakeven (e.g., 1 pip)
                        buffer = 0.0001 if symbol.endswith('JPY') else 0.00001
                        breakeven_stop = entry_price + buffer * direction
                        
                        logger.info(f"Moving position {position_id} to breakeven at {breakeven_stop}")
                        order_manager.modify_position(
                            position_id=position_id,
                            stop_loss=breakeven_stop,
                            comment="Move to breakeven"
                        )
                
                # Save results periodically
                if len(equity_history) % 10 == 0:
                    # Save equity history
                    equity_df = pd.DataFrame(equity_history)
                    equity_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_equity_{symbol}_{timeframe}.csv"))
                    
                    # Save action history
                    action_df = pd.DataFrame(action_history)
                    action_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_actions_{symbol}_{timeframe}.csv"))
                    
                    # Save trade history
                    if trade_history:
                        trade_df = pd.DataFrame(trade_history)
                        trade_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_trades_{symbol}_{timeframe}.csv"))
                
                # Wait for next interval - adjust based on timeframe
                time.sleep(interval)
            
            except Exception as e:
                logger.exception(f"Error during live trading loop: {e}")
                time.sleep(interval)
        
        # Trading completed
        logger.info("Live trading stopped!")
        
        # Save final results
        if equity_history:
            equity_df = pd.DataFrame(equity_history)
            equity_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_equity_{symbol}_{timeframe}_final.csv"))
        
        if action_history:
            action_df = pd.DataFrame(action_history)
            action_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_actions_{symbol}_{timeframe}_final.csv"))
        
        if trade_history:
            trade_df = pd.DataFrame(trade_history)
            trade_df.to_csv(os.path.join(config['paths']['data'], f"live_trade_trades_{symbol}_{timeframe}_final.csv"))
            
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
            
            # Update session info
            session_info['end_time'] = datetime.now().isoformat()
            session_info['total_trades'] = total_trades
            session_info['profitable_trades'] = profitable_trades
            session_info['win_rate'] = win_rate
            session_info['total_profit'] = total_profit
            
            if account_info:
                session_info['final_balance'] = account_info['balance']
                session_info['final_equity'] = account_info['equity']
            
            with open(session_file, 'w') as f:
                json.dump(session_info, f, indent=2)
            
            logger.info(f"Updated session info saved to {session_file}")
        
    except Exception as e:
        logger.exception(f"Error during live trading: {e}")
    
    finally:
        # Clean up
        if 'mt5_connector' in locals() and mt5_connector.initialized:
            mt5_connector.shutdown()
        
        logger.info(f"Live trading script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()