"""
Script for training the reinforcement learning trading agent.

This script configures and runs the training process for the reinforcement
learning agent, including data loading, preprocessing, environment setup,
and model training.
"""

import os
import sys
import time
import logging
import yaml
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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_training_directories(config):
    """Create necessary directories for training."""
    for path_name, path in config['paths'].items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")

def main():
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
    log_path = os.path.join(config['paths']['logs'], 'training.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger('train', log_file=log_path)
    
    # Create directories
    setup_training_directories(config)
    
    # Log start of training
    logger.info(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Data collection and preprocessing
        logger.info("Collecting and preprocessing data...")
        
        # Initialize data collector
        data_collector = MT5DataCollector(config)
        data_collector.initialize()
        
        # Get symbol and timeframe from config
        symbol = config['trading']['default_symbol']
        timeframe = config['trading']['default_timeframe']
        
        # Collect data
        df = data_collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            num_bars=10000  # Get a good amount of historical data
        )
        
        if df is None or len(df) < 100:
            logger.error("Failed to collect sufficient data!")
            return
        
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
        
        # Split data for training and validation
        train_ratio = 0.8
        split_idx = int(len(df) * train_ratio)
        train_data = df.iloc[:split_idx]
        val_data = df.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_data)} samples")
        logger.info(f"Validation data: {len(val_data)} samples")
        
        # Calculate ATR for stop loss
        if 'atr_14' in df.columns:
            atr = df['atr_14'].mean()
            logger.info(f"Average ATR: {atr:.5f}")
        else:
            atr = None
            logger.warning("ATR not available in features")
        
        # Create trading environment
        logger.info("Setting up trading environment...")
        
        # Define feature set for the environment
        feature_columns = [col for col in df.columns if col.endswith('_norm')]
        logger.info(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")
        
        # Create environment
        window_size = config['data'].get('feature_window', 20)
        env = TradingEnvironment(
            config=config,
            data=train_data,
            mode='backtest',
            window_size=window_size,
            symbol=symbol,
            timeframe=timeframe,
            reward_type='trading_focused',  # Utilisez la nouvelle fonction de récompense
            features=feature_columns
        )
        
        # Create validation environment
        val_env = TradingEnvironment(
            config=config,
            data=val_data,
            mode='backtest',
            window_size=window_size,
            symbol=symbol,
            timeframe=timeframe,
            reward_type='trading_focused',
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
        
        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Action dimension: {action_dim}")
        
        # Initialize the agent
        logger.info("Initializing PPO agent...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = PPOAgent(
            config=config,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            model_name=f"{symbol}_{timeframe}_ppo"
        )
        
        # Training parameters
        total_timesteps = config['model_config']['ppo'].get('total_timesteps', 100000)
        eval_freq = config['model_config']['ppo'].get('eval_freq', 10000)
        n_eval_episodes = config['model_config']['ppo'].get('n_eval_episodes', 5)
        save_freq = config['model_config']['ppo'].get('save_freq', 10000)
        log_interval = config['model_config']['ppo'].get('log_interval', 1)
        
        # Training loop
        logger.info(f"Starting training for {total_timesteps} steps...")
        
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        eval_rewards = []
        
        # Initialize variables
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(1, total_timesteps + 1):
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, log_prob, value, reward, done)
            
            # Update variables
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Check if episode ended
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset environment
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Log progress
                if len(episode_rewards) % log_interval == 0:
                    mean_reward = np.mean(episode_rewards[-log_interval:])
                    mean_length = np.mean(episode_lengths[-log_interval:])
                    logger.info(f"Step: {step}, Episodes: {len(episode_rewards)}, "
                               f"Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.2f}")
            
            # Train the agent
            if step % env.window_size == 0 and len(agent.memory) >= agent.batch_size:
                train_metrics = agent.train()
                logger.info(f"Training at step {step}: Actor loss: {train_metrics['actor_loss']:.5f}, "
                           f"Critic loss: {train_metrics['critic_loss']:.5f}")
            
            # Evaluate the agent
            if step % eval_freq == 0:
                logger.info(f"Evaluating agent at step {step}...")
                eval_rewards_episode = []
                
                for _ in range(n_eval_episodes):
                    eval_obs, _ = val_env.reset()
                    eval_done = False
                    eval_episode_reward = 0
                    
                    while not eval_done:
                        eval_action, _, _ = agent.select_action(eval_obs, evaluate=True)
                        eval_obs, eval_reward, eval_done, eval_truncated, _ = val_env.step(eval_action)
                        eval_episode_reward += eval_reward
                        
                        if eval_truncated:
                            break
                    
                    eval_rewards_episode.append(eval_episode_reward)
                
                mean_eval_reward = np.mean(eval_rewards_episode)
                eval_rewards.append(mean_eval_reward)
                
                logger.info(f"Evaluation at step {step}: Mean reward: {mean_eval_reward:.2f}")
                
                # Update metrics
                agent.update_metrics({
                    'eval_rewards': [mean_eval_reward],
                    'train_rewards': episode_rewards[-log_interval:] if len(episode_rewards) >= log_interval else []
                })
            
            # Save the model
            if step % save_freq == 0:
                save_path = agent.save_model()
                logger.info(f"Model saved to {save_path} at step {step}")
        
        # Training finished, save final model
        save_path = agent.save_model()
        logger.info(f"Final model saved to {save_path}")
        
        # Calculate training duration
        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds")
        
        # Visualize training results
        logger.info("Generating training visualizations...")
        visualizer = Visualizer(config)
        
        # Plot learning curve
        visualizer.plot_learning_curve(
            agent.get_metrics(),
            title=f"Learning Curve - {symbol} {timeframe}",
            filename=f"learning_curve_{symbol}_{timeframe}.png"
        )
        
        logger.info("Training completed successfully!")
    
    except Exception as e:
        logger.exception(f"Error during training: {e}")
    
    finally:
        # Clean up
        if 'data_collector' in locals() and data_collector.initialized:
            data_collector.shutdown()
        
        logger.info(f"Training script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()