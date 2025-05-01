"""
Performance metrics calculation module.

This module provides functionality to calculate various financial and 
trading performance metrics, such as Sharpe ratio, Sortino ratio, drawdown,
and other risk/return metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to match returns frequency
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate mean and standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)  # ddof=1 for sample standard deviation
    
    # Avoid division by zero
    if std_dev == 0:
        return 0.0
    
    # Calculate Sharpe ratio and annualize
    sharpe = mean_excess_return / std_dev
    sharpe_annualized = sharpe * np.sqrt(annualization_factor)
    
    return sharpe_annualized

def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        target_return: Minimum acceptable return
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to match returns frequency
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate mean excess return
    mean_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < target_return]
    
    if len(downside_returns) == 0:
        # No downside returns, avoid division by zero
        return np.inf if mean_excess_return > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    # Calculate Sortino ratio and annualize
    sortino = mean_excess_return / downside_deviation if downside_deviation > 0 else 0.0
    sortino_annualized = sortino * np.sqrt(annualization_factor)
    
    return sortino_annualized

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        float: Maximum drawdown (as a positive percentage)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Maximum drawdown is the most negative value
    max_drawdown = np.min(drawdown)
    
    # Return as positive percentage
    return -max_drawdown

def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate the drawdown series.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        np.ndarray: Array of drawdown values (as positive percentages)
    """
    if len(equity_curve) == 0:
        return np.array([])
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Return as positive percentages
    return -drawdown

def calculate_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Array of returns
        equity_curve: Array of equity values
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
    
    # Calculate annualized return
    annualized_return = np.mean(returns) * annualization_factor
    
    # Calculate maximum drawdown
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Calculate Calmar ratio
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    calmar = annualized_return / max_dd
    
    return calmar

def calculate_win_rate(trades: np.ndarray) -> float:
    """
    Calculate the win rate.
    
    Args:
        trades: Array of trade profits/losses
        
    Returns:
        float: Win rate (as a proportion)
    """
    if len(trades) == 0:
        return 0.0
    
    # Calculate number of winning trades
    winning_trades = np.sum(trades > 0)
    
    # Calculate win rate
    win_rate = winning_trades / len(trades)
    
    return win_rate

def calculate_profit_factor(trades: np.ndarray) -> float:
    """
    Calculate the profit factor.
    
    Args:
        trades: Array of trade profits/losses
        
    Returns:
        float: Profit factor
    """
    if len(trades) == 0:
        return 0.0
    
    # Separate winning and losing trades
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    # Calculate profit factor
    if len(losing_trades) == 0 or np.sum(np.abs(losing_trades)) == 0:
        return np.inf if len(winning_trades) > 0 else 0.0
    
    profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))
    
    return profit_factor

def calculate_average_trade(trades: np.ndarray) -> Dict[str, float]:
    """
    Calculate average trade metrics.
    
    Args:
        trades: Array of trade profits/losses
        
    Returns:
        Dict: Dictionary with average trade metrics
    """
    if len(trades) == 0:
        return {
            'average_trade': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_loss_ratio': 0.0
        }
    
    # Calculate average trade
    average_trade = np.mean(trades)
    
    # Separate winning and losing trades
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    # Calculate average win and loss
    average_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    average_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    
    # Calculate win/loss ratio
    win_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else np.inf
    
    return {
        'average_trade': average_trade,
        'average_win': average_win,
        'average_loss': average_loss,
        'win_loss_ratio': win_loss_ratio
    }

def calculate_expectancy(trades: np.ndarray) -> float:
    """
    Calculate the expectancy (expected value per trade).
    
    Args:
        trades: Array of trade profits/losses
        
    Returns:
        float: Expectancy
    """
    if len(trades) == 0:
        return 0.0
    
    # Calculate win rate
    win_rate = calculate_win_rate(trades)
    
    # Separate winning and losing trades
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    # Calculate average win and loss
    average_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    average_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    
    # Calculate expectancy
    expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
    
    return expectancy

def calculate_comprehensive_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    trades: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        returns: Array of returns
        equity_curve: Array of equity values
        trades: Array of trade profits/losses
        
    Returns:
        Dict: Dictionary with comprehensive metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Return and risk metrics
    metrics['total_return'] = (equity_curve[-1] / equity_curve[0] - 1) * 100 if len(equity_curve) > 0 else 0.0
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    metrics['max_drawdown'] = calculate_max_drawdown(equity_curve) * 100  # as percentage
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, equity_curve)
    
    # Annualized return (assuming daily returns)
    if len(returns) > 0:
        annualization_factor = 252  # trading days in a year
        metrics['annualized_return'] = np.mean(returns) * annualization_factor * 100  # as percentage
    else:
        metrics['annualized_return'] = 0.0
    
    # Trade metrics
    metrics['win_rate'] = calculate_win_rate(trades) * 100  # as percentage
    metrics['profit_factor'] = calculate_profit_factor(trades)
    
    # Average trade metrics
    avg_trade_metrics = calculate_average_trade(trades)
    metrics.update(avg_trade_metrics)
    
    # Expectancy
    metrics['expectancy'] = calculate_expectancy(trades)
    
    # Recovery factor
    if metrics['max_drawdown'] > 0:
        metrics['recovery_factor'] = metrics['total_return'] / metrics['max_drawdown']
    else:
        metrics['recovery_factor'] = np.inf if metrics['total_return'] > 0 else 0.0
    
    return metrics