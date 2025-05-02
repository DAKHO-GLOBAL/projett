# """
# Script for backtesting the trained reinforcement learning trading agent.

# This script loads a trained agent model and evaluates its performance on
# historical data, generating trading signals, performance metrics, and
# visualization of the results.
# """

# import os
# import sys
# import time
# import logging
# import yaml
# import argparse
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch

# # Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.data.data_collector import MT5DataCollector
# from src.data.preprocessor import DataPreprocessor
# from src.data.feature_generator import FeatureGenerator
# from src.environment.trading_env import TradingEnvironment
# from src.agent.ppo_agent import PPOAgent
# from src.utils.logger import setup_logger
# from src.utils.visualizer import Visualizer
# from src.utils.metrics import calculate_comprehensive_metrics


# # Importer la nouvelle classe de récompense
# from src.environment.custom_rewards import GoldTradingReward



# def load_config(config_path):
#     """Load configuration from YAML file."""
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

# def parse_arguments():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Backtest RL trading agent')
#     parser.add_argument('--model', type=str, help='Path to the model file')
#     parser.add_argument('--symbol', type=str, help='Trading symbol')
#     parser.add_argument('--timeframe', type=str, help='Trading timeframe')
#     parser.add_argument('--start_date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
#     parser.add_argument('--end_date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    
#     return parser.parse_args()

# def main():
#     # Parse arguments
#     args = parse_arguments()
    
#     # Load configuration
#     config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
    
#     config = {}
    
#     # Load main config
#     config_path = os.path.join(config_dir, 'config.yml')
#     main_config = load_config(config_path)
#     config.update(main_config)
    
#     # Load model config
#     model_config_path = os.path.join(config_dir, 'model_config.yml')
#     model_config = load_config(model_config_path)
#     config['model_config'] = model_config
    
#     # Load trading parameters
#     trading_params_path = os.path.join(config_dir, 'trading_params.yml')
#     trading_params = load_config(trading_params_path)
#     config['trading_params'] = trading_params
    
#     # Setup logging
#     log_path = os.path.join(config['paths']['logs'], 'backtest.log')
#     os.makedirs(os.path.dirname(log_path), exist_ok=True)
#     logger = setup_logger('backtest', log_file=log_path)
    
#     # Log start of backtesting
#     logger.info(f"Starting backtesting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
#     # Get parameters from arguments or config
#     symbol = args.symbol if args.symbol else config['trading']['default_symbol']
#     timeframe = args.timeframe if args.timeframe else config['trading']['default_timeframe']
    
#     if args.start_date:
#         start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
#     else:
#         start_date = None
    
#     if args.end_date:
#         end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
#     else:
#         end_date = None
    
#     model_path = args.model if args.model else os.path.join(
#         config['paths']['models'], 
#         'saved', 
#         f"{symbol}_{timeframe}_ppo.pt"
#     )
    
#     # Validate model path
#     if not os.path.exists(model_path):
#         logger.error(f"Model not found at {model_path}")
#         available_models = [f for f in os.listdir(os.path.dirname(model_path)) if f.endswith('.pt')]
#         if available_models:
#             logger.info(f"Available models: {available_models}")
#         return
    
#     try:
#         # Data collection and preprocessing
#         logger.info("Collecting and preprocessing data...")
        
#         # Initialize data collector
#         data_collector = MT5DataCollector(config)
#         data_collector.initialize()
        
#         # Collect data
#         df = data_collector.get_historical_data(
#             symbol=symbol,
#             timeframe=timeframe,
#             start_date=start_date,
#             end_date=end_date
#         )
        
#         if df is None or len(df) < 100:
#             logger.error("Failed to collect sufficient data!")
#             return
        
#         logger.info(f"Collected {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
#         # Generate features
#         logger.info("Generating technical indicators and features...")
#         feature_generator = FeatureGenerator(config)
#         df = feature_generator.generate_all_features(df)
        
#         # Preprocess data
#         logger.info("Preprocessing data...")
#         preprocessor = DataPreprocessor(config)
#         df = preprocessor.clean_data(df)
#         df = preprocessor.calculate_returns(df)
#         df = preprocessor.normalize_data(df)
        
#         # Define feature set for the environment
#         feature_columns = [col for col in df.columns if col.endswith('_norm')]
#         logger.info(f"Using {len(feature_columns)} features")
        
#         # Create environment
#         logger.info("Setting up trading environment for backtesting...")
#         window_size = config['data'].get('feature_window', 20)

#         # Créer une instance
#         gold_reward = GoldTradingReward()
        
#         env = TradingEnvironment(
#             config=config,
#             data=df,
#             mode='backtest',
#             window_size=window_size,
#             symbol=symbol,
#             timeframe=timeframe,
#             reward_type='gold_trading',  # Identifiant de la récompense
#             features=feature_columns,
#             reward_calculator=gold_reward  # Passer l'instance
#         )
        
#         # Initialize the observation to get state dimensions
#         obs, _ = env.reset()
        
#         # Calculate state and action dimensions
#         if isinstance(obs, dict):
#             # For dict observation space, flatten each observation component
#             state_dim = 0
#             if 'market' in obs:
#                 state_dim += np.prod(obs['market'].shape)
#             if 'account' in obs:
#                 state_dim += np.prod(obs['account'].shape)
#         else:
#             state_dim = np.prod(obs.shape)
        
#         action_dim = env.action_space.n
        
#         # Initialize the agent
#         logger.info("Loading trained PPO agent...")
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         agent = PPOAgent(
#             config=config,
#             state_dim=state_dim,
#             action_dim=action_dim,
#             device=device,
#             model_name=f"{symbol}_{timeframe}_ppo"
#         )
        
#         # Load the trained model
#         logger.info(f"Loading model from {model_path}")
#         agent.load_model(model_path)
        
#         # Run backtest
#         logger.info("Starting backtest...")
#         start_time = time.time()
        
#         # Reset environment
#         obs, _ = env.reset()
        
#         done = False
        
#         # Track trade history
#         trade_history = []
        
#         # Track equity curve
#         equity_curve = [env.balance]
        
#         # Track returns
#         returns = []
        
#         # Track actions
#         actions = []
#         prices = []
#         timestamps = []
#         positions = []
        
#         while not done:
#             # Select action
#             action, _, _ = agent.select_action(obs, evaluate=True)
            
#             # Record the state before the action
#             timestamps.append(df.index[env.current_step])
#             prices.append(df.iloc[env.current_step]['close'])
            
#             # Execute action
#             next_obs, reward, done, truncated, info = env.step(action)
            
#             # Record the action
#             actions.append(action)
#             positions.append(env.current_position)
            
#             # Record equity
#             equity_curve.append(info['equity'])
            
#             # Calculate return
#             if len(equity_curve) > 1:
#                 ret = (equity_curve[-1] / equity_curve[-2]) - 1
#                 returns.append(ret)
            
#             # Update observation
#             obs = next_obs
            
#             # Record trade if position closed
#             if len(env.trade_history) > len(trade_history):
#                 trade = env.trade_history[-1]
#                 trade_history.append(trade)
                
#                 logger.info(f"Trade: Entry: {trade['entry_time']}, Exit: {trade['exit_time']}, "
#                            f"Position: {trade['position']}, P&L: {trade['pnl']:.2f}")
            
#             if truncated:
#                 break
        
#         # Calculate backtest duration
#         duration = time.time() - start_time
#         logger.info(f"Backtest completed in {duration:.2f} seconds")
        
#         # Create signals DataFrame
#         signals_df = pd.DataFrame({
#             'timestamp': timestamps,
#             'price': prices,
#             'action': actions,
#             'position': positions
#         })
#         signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
#         signals_df.set_index('timestamp', inplace=True)
        
#         # Add buy/sell signals for visualization
#         signals_df['buy'] = (signals_df['action'] == 2).astype(int)  # Buy action (2)
#         signals_df['sell'] = (signals_df['action'] == 0).astype(int)  # Sell action (0)
        
#         # Calculate performance metrics
#         logger.info("Calculating performance metrics...")
        
#         # Extract trade profits/losses
#         trade_pnls = np.array([trade['pnl'] for trade in trade_history])
        
#         # Convert equity curve to numpy array
#         equity_array = np.array(equity_curve)
        
#         # Calculate returns array
#         returns_array = np.array(returns)
        
#         # Calculate comprehensive metrics
#         metrics = calculate_comprehensive_metrics(
#             returns=returns_array,
#             equity_curve=equity_array,
#             trades=trade_pnls
#         )
        
#         # Log key metrics
#         logger.info(f"Total Return: {metrics['total_return']:.2f}%")
#         logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
#         logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
#         logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
#         logger.info(f"Profit Factor: {metrics['profit_factor']:.4f}")
#         logger.info(f"Total Trades: {len(trade_history)}")
        
#         # Visualize backtest results
#         logger.info("Generating backtest visualizations...")
#         visualizer = Visualizer(config)
        
#         # Plot price with signals
#         visualizer.plot_price_with_signals(
#             df=df,
#             signals=signals_df,
#             title=f"Backtest Results - {symbol} {timeframe}",
#             filename=f"backtest_signals_{symbol}_{timeframe}.png"
#         )
        
#         # Plot equity curve
#         equity_series = pd.Series(equity_curve, index=[df.index[0]] + list(df.index[window_size:]))
        
#         visualizer.plot_equity_curve(
#             equity=equity_series,
#             title=f"Equity Curve - {symbol} {timeframe}",
#             filename=f"backtest_equity_{symbol}_{timeframe}.png"
#         )
        
#         # Save trade history to CSV
#         trades_df = pd.DataFrame(trade_history)
#         trades_df.to_csv(os.path.join(config['paths']['data'], f"backtest_trades_{symbol}_{timeframe}.csv"))
        
#         # Save signals to CSV
#         signals_df.to_csv(os.path.join(config['paths']['data'], f"backtest_signals_{symbol}_{timeframe}.csv"))
        
#         # Plot trade analysis if we have enough trades
#         if len(trade_history) > 5:
#             # Add more details to trades DataFrame
#             if 'entry_time' in trades_df.columns:
#                 # Convert timestamps to datetime
#                 trades_df['entry_time'] = pd.to_datetime(
#                     [df.index[t] for t in trades_df['entry_time']]
#                 )
#                 trades_df['exit_time'] = pd.to_datetime(
#                     [df.index[t] for t in trades_df['exit_time']]
#                 )
                
#                 # Calculate duration in hours
#                 trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                
#                 # Add symbol column
#                 trades_df['symbol'] = symbol
                
#                 # Add trade type
#                 trades_df['type'] = trades_df['position'].map({1: 'BUY', -1: 'SELL'})
                
#                 visualizer.plot_trade_analysis(
#                     trades=trades_df,
#                     title=f"Trade Analysis - {symbol} {timeframe}",
#                     filename=f"backtest_trades_{symbol}_{timeframe}.png"
#                 )
        
#         # Plot performance metrics
#         visualizer.plot_performance_metrics(
#             metrics=metrics,
#             title=f"Performance Metrics - {symbol} {timeframe}",
#             filename=f"backtest_metrics_{symbol}_{timeframe}.png"
#         )
        
#         logger.info("Backtesting completed successfully!")
    
#     except Exception as e:
#         logger.exception(f"Error during backtesting: {e}")
    
#     finally:
#         # Clean up
#         if 'data_collector' in locals() and data_collector.initialized:
#             data_collector.shutdown()
        
#         logger.info(f"Backtesting script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python
"""
Script de backtest optimisé pour l'apprentissage par renforcement sur l'or.

Ce script charge un modèle entraîné et évalue ses performances en
simulation sur des données historiques, avec des optimisations 
spécifiques pour le trading de l'or.
"""

import os
import sys
import time
import logging
from matplotlib import pyplot as plt
import yaml
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ajouter le répertoire racine au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_collector import MT5DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_generator import FeatureGenerator
from src.environment.trading_env import TradingEnvironment
from src.agent.ppo_agent import PPOAgent
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer
from src.utils.metrics import calculate_comprehensive_metrics

# Importer notre classe de récompense personnalisée pour l'or
from src.environment.custom_rewards import GoldTradingReward

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Backtest avancé pour le trading de l\'or')
    parser.add_argument('--model', type=str, help='Chemin vers le fichier modèle')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbole de trading')
    parser.add_argument('--timeframe', type=str, help='Timeframe de trading')
    parser.add_argument('--start_date', type=str, help='Date de début pour le backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Date de fin pour le backtest (YYYY-MM-DD)')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='Balance initiale')
    parser.add_argument('--enhanced_features', action='store_true', help='Utiliser des caractéristiques améliorées pour l\'or')
    parser.add_argument('--detailed_analysis', action='store_true', help='Générer une analyse détaillée')
    
    return parser.parse_args()

def main():
    # Analyser les arguments
    args = parse_arguments()
    
    # Charger la configuration
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
    
    config = {}
    
    # Charger la config principale
    config_path = os.path.join(config_dir, 'config.yml')
    main_config = load_config(config_path)
    config.update(main_config)
    
    # Charger la config du modèle
    model_config_path = os.path.join(config_dir, 'model_config.yml')
    model_config = load_config(model_config_path)
    config['model_config'] = model_config
    
    # Charger les paramètres de trading
    trading_params_path = os.path.join(config_dir, 'trading_params.yml')
    trading_params = load_config(trading_params_path)
    config['trading_params'] = trading_params
    
    # Configurer le logging
    log_path = os.path.join(config['paths']['logs'], 'backtest_gold.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger('backtest_gold', log_file=log_path)
    
    # Démarrer le backtest
    logger.info(f"Démarrage du backtest d'or à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Utilisation du device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Obtenir les paramètres
    symbol = args.symbol if args.symbol else config['trading']['default_symbol']
    timeframe = args.timeframe if args.timeframe else config['trading']['default_timeframe']
    initial_balance = args.initial_balance
    
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
    
    # Valider le chemin du modèle
    if not os.path.exists(model_path):
        logger.error(f"Modèle non trouvé à {model_path}")
        available_models = [f for f in os.listdir(os.path.dirname(model_path)) if f.endswith('.pt')]
        if available_models:
            logger.info(f"Modèles disponibles: {available_models}")
        return
    
    try:
        # Collection et prétraitement des données
        logger.info("Collecte et prétraitement des données...")
        
        # Initialiser le collecteur de données
        data_collector = MT5DataCollector(config)
        data_collector.initialize()
        
        # Récupérer les données historiques
        df = data_collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or len(df) < 100:
            logger.error("Impossible de récupérer suffisamment de données!")
            return
        
        logger.info(f"Collecté {len(df)} points de données de {df.index[0]} à {df.index[-1]}")
        
        # Calculer les indicateurs techniques et caractéristiques
        logger.info("Génération des indicateurs techniques...")
        feature_generator = FeatureGenerator(config)
        
        # Caractéristiques standard ou améliorées
        if args.enhanced_features:
            df = feature_generator.generate_all_features(df)
            # Ajouter des caractéristiques spécifiques à l'or
            df = feature_generator.add_gold_specific_features(df)
            logger.info("Caractéristiques améliorées pour l'or ajoutées")
        else:
            df = feature_generator.generate_all_features(df)
        
        # Prétraiter les données
        logger.info("Prétraitement des données...")
        preprocessor = DataPreprocessor(config)
        df = preprocessor.clean_data(df)
        df = preprocessor.calculate_returns(df)
        df = preprocessor.normalize_data(df)
        
        # Définir les caractéristiques pour l'environnement
        if args.enhanced_features:
            # Utiliser des caractéristiques spécifiques à l'or
            feature_columns = [
                # Prix normalisés
                'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
                # Indicateurs techniques importants pour l'or
                'rsi_14_norm', 'atr_14_norm', 'bb_width_norm', 'macd_norm', 'ema_20_norm',
                # Caractéristiques spécifiques à l'or
                'dist_to_psych_100_norm', 'near_psych_level', 'gold_momentum_1h_norm',
                'gold_momentum_4h_norm', 'session_transition'
            ]
            
            # Vérifier si toutes les caractéristiques sont disponibles
            available_cols = [col for col in feature_columns if col in df.columns]
            missing_cols = [col for col in feature_columns if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Caractéristiques manquantes: {missing_cols}")
                feature_columns = available_cols
        else:
            feature_columns = [col for col in df.columns if col.endswith('_norm')]
        
        logger.info(f"Utilisation de {len(feature_columns)} caractéristiques")
        
        # Créer une instance de la classe de récompense personnalisée pour l'or
        gold_reward = GoldTradingReward()
        
        # Créer l'environnement
        logger.info("Configuration de l'environnement de trading pour le backtest d'or...")
        window_size = config['data'].get('feature_window', 20)
        
        env = TradingEnvironment(
            config=config,
            data=df,
            mode='backtest',
            window_size=window_size,
            symbol=symbol,
            timeframe=timeframe,
            reward_type='gold_trading',  # Identifiant pour la fonction de récompense
            reward_calculator=gold_reward,  # Passer l'instance de notre calculateur
            features=feature_columns,
            initial_balance=initial_balance
        )
        
        # Initialiser l'observation pour obtenir les dimensions d'état
        obs, _ = env.reset()
        
        # Calculer les dimensions d'état et d'action
        if isinstance(obs, dict):
            # Pour les espaces d'observation en dictionnaire, aplatir chaque composant
            state_dim = 0
            if 'market' in obs:
                state_dim += np.prod(obs['market'].shape)
            if 'account' in obs:
                state_dim += np.prod(obs['account'].shape)
        else:
            state_dim = np.prod(obs.shape)
        
        action_dim = env.action_space.n
        
        # Initialiser l'agent
        logger.info("Chargement de l'agent PPO...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = PPOAgent(
            config=config,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            model_name=f"{symbol}_{timeframe}_ppo"
        )
        
        # Charger le modèle entraîné
        logger.info(f"Chargement du modèle depuis {model_path}")
        agent.load_model(model_path)
        
        # Lancer le backtest
        logger.info("Démarrage du backtest...")
        start_time = time.time()
        
        # Réinitialiser l'environnement
        obs, _ = env.reset()
        
        done = False
        
        # Suivi de l'historique des trades
        trade_history = []
        
        # Suivi de la courbe d'équité
        equity_curve = [env.balance]
        
        # Suivi des rendements
        returns = []
        
        # Suivi des actions, prix et positions
        actions = []
        prices = []
        timestamps = []
        positions = []
        
        # Données pour l'analyse
        drawdowns = []
        unrealized_pnls = []
        current_drawdown = 0
        max_drawdown = 0
        peak_equity = env.balance
        
        # Boucle principale de backtest
        while not done:
            # Sélectionner une action
            action, _, _ = agent.select_action(obs, evaluate=True)
            
            # Enregistrer l'état avant l'action
            timestamps.append(df.index[env.current_step])
            prices.append(df.iloc[env.current_step]['close'])
            
            # Exécuter l'action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Enregistrer l'action et la position
            actions.append(action)
            positions.append(env.current_position)
            
            # Enregistrer l'équité
            current_equity = info['equity']
            equity_curve.append(current_equity)
            
            # Calculer le rendement
            if len(equity_curve) > 1:
                ret = (equity_curve[-1] / equity_curve[-2]) - 1
                returns.append(ret)
            
            # Calculer le drawdown
            # Calculer le drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
                current_drawdown = 0
            else:
                current_drawdown = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, current_drawdown)
            
            # Enregistrer le drawdown et le PnL non réalisé
            drawdowns.append(current_drawdown)
            unrealized_pnls.append(info['unrealized_pnl'])
            
            # Mettre à jour l'observation
            obs = next_obs
            
            # Enregistrer le trade si une position a été fermée
            if len(env.trade_history) > len(trade_history):
                trade = env.trade_history[-1]
                trade_history.append(trade)
                
                logger.info(f"Trade: Entrée: {trade['entry_time']}, Sortie: {trade['exit_time']}, "
                           f"Position: {trade['position']}, P&L: {trade['pnl']:.2f}")
            
            if truncated:
                break
        
        # Calculer la durée du backtest
        duration = time.time() - start_time
        logger.info(f"Backtest terminé en {duration:.2f} secondes")
        
        # Créer un DataFrame des signaux
        signals_df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'action': actions,
            'position': positions
        })
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df.set_index('timestamp', inplace=True)
        
        # Ajouter les signaux d'achat/vente pour la visualisation
        signals_df['buy'] = (signals_df['action'] == 2).astype(int)  # Action d'achat (2)
        signals_df['sell'] = (signals_df['action'] == 0).astype(int)  # Action de vente (0)
        
        # Calculer les métriques de performance
        logger.info("Calcul des métriques de performance...")
        
        # Extraire les profits/pertes des trades
        trade_pnls = np.array([trade['pnl'] for trade in trade_history])
        
        # Convertir la courbe d'équité en tableau numpy
        equity_array = np.array(equity_curve)
        
        # Calculer le tableau de rendements
        returns_array = np.array(returns)
        
        # Calculer les métriques complètes
        metrics = calculate_comprehensive_metrics(
            returns=returns_array,
            equity_curve=equity_array,
            trades=trade_pnls
        )
        
        # Métriques spécifiques à l'or
        gold_metrics = {
            'avg_trade_duration': np.mean([trade['holding_period'] for trade in trade_history]) if trade_history else 0,
            'max_trade_duration': max([trade['holding_period'] for trade in trade_history]) if trade_history else 0,
            'win_rate_long': sum(1 for t in trade_history if t['position'] == 1 and t['pnl'] > 0) / 
                            sum(1 for t in trade_history if t['position'] == 1) if sum(1 for t in trade_history if t['position'] == 1) else 0,
            'win_rate_short': sum(1 for t in trade_history if t['position'] == -1 and t['pnl'] > 0) / 
                             sum(1 for t in trade_history if t['position'] == -1) if sum(1 for t in trade_history if t['position'] == -1) else 0,
            'trades_per_day': len(trade_history) / (len(df) / (288 if timeframe == 'M5' else 24)),  # Approximation selon timeframe
            'max_consecutive_losses': max_consecutive_elements([1 if t['pnl'] <= 0 else 0 for t in trade_history]) if trade_history else 0,
            'max_drawdown_percent': max_drawdown
        }
        
        # Ajouter les métriques spécifiques à l'or
        metrics.update(gold_metrics)
        
        # Journaliser les métriques clés
        logger.info(f"Rendement total: {metrics['total_return']:.2f}%")
        logger.info(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Drawdown maximum: {metrics['max_drawdown']:.2f}%")
        logger.info(f"Taux de réussite: {metrics['win_rate']:.2f}%")
        logger.info(f"Factor de profit: {metrics['profit_factor']:.4f}")
        logger.info(f"Trades totaux: {len(trade_history)}")
        logger.info(f"Taux de réussite long: {metrics['win_rate_long']:.2f}")
        logger.info(f"Taux de réussite short: {metrics['win_rate_short']:.2f}")
        logger.info(f"Pertes consécutives max: {metrics['max_consecutive_losses']}")
        
        # Visualiser les résultats du backtest
        logger.info("Génération des visualisations de backtest...")
        visualizer = Visualizer(config)
        
        # Graphique des prix avec signaux
        visualizer.plot_price_with_signals(
            df=df,
            signals=signals_df,
            title=f"Résultats du Backtest - {symbol} {timeframe}",
            filename=f"backtest_gold_signals_{symbol}_{timeframe}.png"
        )
        
        # Graphique de la courbe d'équité
        equity_series = pd.Series(equity_curve, index=[df.index[0]] + list(df.index[window_size:]))
        
        # Créer une série pour le drawdown
        drawdown_series = pd.Series(drawdowns, index=list(df.index[window_size:]))
        
        visualizer.plot_equity_curve(
            equity=equity_series,
            drawdown=drawdown_series,
            title=f"Courbe d'Équité - {symbol} {timeframe}",
            filename=f"backtest_gold_equity_{symbol}_{timeframe}.png"
        )
        
        # Enregistrer l'historique des trades au format CSV
        trades_df = pd.DataFrame(trade_history)
        trades_df.to_csv(os.path.join(config['paths']['data'], f"backtest_gold_trades_{symbol}_{timeframe}.csv"))
        
        # Enregistrer les signaux au format CSV
        signals_df.to_csv(os.path.join(config['paths']['data'], f"backtest_gold_signals_{symbol}_{timeframe}.csv"))
        
        # Générer une analyse détaillée des trades
        if len(trade_history) > 5 and args.detailed_analysis:
            # Ajouter plus de détails au DataFrame des trades
            if 'entry_time' in trades_df.columns:
                # Convertir les horodatages en datetime
                trades_df['entry_time'] = pd.to_datetime(
                    [df.index[t] for t in trades_df['entry_time']]
                )
                trades_df['exit_time'] = pd.to_datetime(
                    [df.index[t] for t in trades_df['exit_time']]
                )
                
                # Calculer la durée en heures
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                
                # Ajouter la colonne de symbole
                trades_df['symbol'] = symbol
                
                # Ajouter le type de trade
                trades_df['type'] = trades_df['position'].map({1: 'BUY', -1: 'SELL'})
                
                # Générer l'analyse des trades
                visualizer.plot_trade_analysis(
                    trades=trades_df,
                    title=f"Analyse des Trades - {symbol} {timeframe}",
                    filename=f"backtest_gold_trades_{symbol}_{timeframe}.png"
                )
                
                # Analyse par session de trading
                if isinstance(trades_df['entry_time'].iloc[0], pd.Timestamp):
                    trades_df['hour'] = trades_df['entry_time'].dt.hour
                    trades_df['day_of_week'] = trades_df['entry_time'].dt.dayofweek
                    
                    # Classifier les sessions
                    trades_df['session'] = 'other'
                    trades_df.loc[(trades_df['hour'] >= 0) & (trades_df['hour'] < 8), 'session'] = 'asian'
                    trades_df.loc[(trades_df['hour'] >= 8) & (trades_df['hour'] < 16), 'session'] = 'european'
                    trades_df.loc[(trades_df['hour'] >= 13) & (trades_df['hour'] < 21), 'session'] = 'us'
                    
                    # Analyser les performances par session
                    session_perf = trades_df.groupby('session').agg({
                        'pnl': ['sum', 'mean', 'count'],
                        'position': 'count'
                    })
                    
                    logger.info(f"Performance par session:\n{session_perf}")
                    
                    # Analyser les performances par jour de la semaine
                    day_perf = trades_df.groupby('day_of_week').agg({
                        'pnl': ['sum', 'mean', 'count'],
                        'position': 'count'
                    })
                    
                    logger.info(f"Performance par jour de la semaine:\n{day_perf}")
        
        # Graphique des métriques de performance
        visualizer.plot_performance_metrics(
            metrics=metrics,
            title=f"Métriques de Performance - {symbol} {timeframe}",
            filename=f"backtest_gold_metrics_{symbol}_{timeframe}.png"
        )
        
        # Calculer les statistiques par mois
        if isinstance(equity_series.index[0], pd.Timestamp):
            # Regrouper par mois
            monthly_returns = pd.Series(returns, index=df.index[window_size+1:]).resample('M').sum() * 100
            
            # Créer un graphique des rendements mensuels
            plt.figure(figsize=(12, 6))
            monthly_returns.plot(kind='bar', color=['green' if x > 0 else 'red' for x in monthly_returns])
            plt.title(f"Rendements Mensuels - {symbol} {timeframe}")
            plt.xlabel("Mois")
            plt.ylabel("Rendement (%)")
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.savefig(os.path.join(config['paths']['data'], 'plots', f"backtest_gold_monthly_{symbol}_{timeframe}.png"))
        
        logger.info("Backtest d'or terminé avec succès!")
    
    except Exception as e:
        logger.exception(f"Erreur pendant le backtest: {e}")
    
    finally:
        # Nettoyage
        if 'data_collector' in locals() and data_collector.initialized:
            data_collector.shutdown()
        
        logger.info(f"Script de backtest d'or terminé à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def max_consecutive_elements(lst):
    """Calcule le nombre maximum d'éléments consécutifs ayant la valeur 1 dans une liste."""
    if not lst:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for item in lst:
        if item == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

if __name__ == "__main__":
    main()
