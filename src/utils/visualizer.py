"""
Visualization utilities for trading system results.

This module provides functionality to visualize trading system results,
including price charts, trading signals, equity curves, and performance metrics.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class Visualizer:
    """Class for visualizing trading system results."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set matplotlib style
        plt.style.use('seaborn-darkgrid')
        
        # Create plots directory if it doesn't exist
        self.plots_dir = os.path.join(config['paths']['data'], 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set color scheme
        self.colors = {
            'buy': '#2ecc71',      # Green
            'sell': '#e74c3c',     # Red
            'price': '#3498db',    # Blue
            'profit': '#2ecc71',   # Green
            'loss': '#e74c3c',     # Red
            'equity': '#3498db',   # Blue
            'balance': '#f39c12',  # Orange
            'drawdown': '#e74c3c'  # Red
        }
    
    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
        title: str = "Price Chart with Trading Signals",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot price chart with trading signals.
        
        Args:
            df: DataFrame with price data
            signals: DataFrame with trading signals
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        if 'close' in df.columns:
            ax.plot(df.index, df['close'], label='Close Price', color=self.colors['price'], linewidth=1.5)
        
        # Add signals if provided
        if signals is not None and not signals.empty:
            # Buy signals
            if 'buy' in signals.columns:
                buy_signals = signals[signals['buy'] == 1]
                if not buy_signals.empty:
                    ax.scatter(buy_signals.index, buy_signals['price'], 
                              color=self.colors['buy'], s=100, marker='^', label='Buy Signal')
            
            # Sell signals
            if 'sell' in signals.columns:
                sell_signals = signals[signals['sell'] == 1]
                if not sell_signals.empty:
                    ax.scatter(sell_signals.index, sell_signals['price'], 
                              color=self.colors['sell'], s=100, marker='v', label='Sell Signal')
        
        # Style the plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format the date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved price chart to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_equity_curve(
        self,
        equity: pd.Series,
        balance: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot equity curve with optional balance and drawdown.
        
        Args:
            equity: Series with equity values
            balance: Series with balance values (optional)
            drawdown: Series with drawdown values (optional)
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot equity
        ax1.plot(equity.index, equity, label='Equity', color=self.colors['equity'], linewidth=2)
        
        # Plot balance if provided
        if balance is not None:
            ax1.plot(balance.index, balance, label='Balance', color=self.colors['balance'], linewidth=1.5, linestyle='--')
        
        # Style the primary axis
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Account Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown if provided
        if drawdown is not None:
            # Create a secondary y-axis for drawdown
            ax2 = ax1.twinx()
            ax2.fill_between(drawdown.index, 0, drawdown, color=self.colors['drawdown'], alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.invert_yaxis()  # Invert y-axis for better drawdown visualization
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend(loc='best')
        
        # Format the date axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_trade_analysis(
        self,
        trades: pd.DataFrame,
        title: str = "Trade Analysis",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (15, 12)
    ) -> plt.Figure:
        """
        Plot comprehensive trade analysis with multiple subplots.
        
        Args:
            trades: DataFrame with trade data
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if trades.empty:
            logger.warning("No trades to plot")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Define grid for subplots
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. Profit Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        profit_data = trades['profit']
        sns.histplot(profit_data, kde=True, ax=ax1, color=self.colors['price'])
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Profit Distribution', fontsize=14)
        ax1.set_xlabel('Profit', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        
        # 2. Cumulative Profit
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_profit = trades['profit'].cumsum()
        ax2.plot(cumulative_profit.index, cumulative_profit, color=self.colors['equity'], linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Cumulative Profit', fontsize=14)
        ax2.set_xlabel('Trade #', fontsize=10)
        ax2.set_ylabel('Cumulative Profit', fontsize=10)
        
        # 3. Profit by Trade Duration
        ax3 = fig.add_subplot(gs[1, 0])
        if 'duration' in trades.columns:
            sns.scatterplot(x='duration', y='profit', data=trades, ax=ax3, 
                           hue='profit', palette=['red' if x < 0 else 'green' for x in trades['profit']])
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax3.set_title('Profit vs Duration', fontsize=14)
            ax3.set_xlabel('Trade Duration', fontsize=10)
            ax3.set_ylabel('Profit', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Duration data not available', ha='center', va='center')
            ax3.set_title('Profit vs Duration', fontsize=14)
        
        # 4. Win/Loss Ratio by Symbol
        ax4 = fig.add_subplot(gs[1, 1])
        if 'symbol' in trades.columns:
            # Calculate win/loss ratio by symbol
            win_loss_by_symbol = trades.groupby('symbol').apply(
                lambda x: pd.Series({
                    'win_count': (x['profit'] > 0).sum(),
                    'loss_count': (x['profit'] <= 0).sum(),
                    'win_rate': (x['profit'] > 0).mean() * 100
                })
            )
            
            win_loss_by_symbol['total'] = win_loss_by_symbol['win_count'] + win_loss_by_symbol['loss_count']
            win_loss_by_symbol = win_loss_by_symbol.sort_values('total', ascending=False)
            
            # Plot win rate by symbol
            sns.barplot(x=win_loss_by_symbol.index, y='win_rate', data=win_loss_by_symbol, ax=ax4, palette='Blues_d')
            ax4.axhline(y=50, color='red', linestyle='--', linewidth=1)
            ax4.set_title('Win Rate by Symbol', fontsize=14)
            ax4.set_xlabel('Symbol', fontsize=10)
            ax4.set_ylabel('Win Rate (%)', fontsize=10)
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Symbol data not available', ha='center', va='center')
            ax4.set_title('Win Rate by Symbol', fontsize=14)
        
        # 5. Monthly Performance
        ax5 = fig.add_subplot(gs[2, 0])
        if 'exit_time' in trades.columns:
            # Extract month from exit time
            trades['month'] = trades['exit_time'].dt.strftime('%Y-%m')
            
            # Calculate monthly performance
            monthly_performance = trades.groupby('month')['profit'].sum()
            
            # Plot monthly performance
            sns.barplot(x=monthly_performance.index, y=monthly_performance.values, ax=ax5, 
                       palette=['red' if x < 0 else 'green' for x in monthly_performance.values])
            ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax5.set_title('Monthly Performance', fontsize=14)
            ax5.set_xlabel('Month', fontsize=10)
            ax5.set_ylabel('Profit', fontsize=10)
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'Exit time data not available', ha='center', va='center')
            ax5.set_title('Monthly Performance', fontsize=14)
        
        # 6. Profit by Trade Type (Buy/Sell)
        ax6 = fig.add_subplot(gs[2, 1])
        if 'type' in trades.columns:
            # Calculate average profit by trade type
            avg_profit_by_type = trades.groupby('type')['profit'].mean()
            
            # Plot average profit by trade type
            sns.barplot(x=avg_profit_by_type.index, y=avg_profit_by_type.values, ax=ax6, 
                       palette=['green', 'red'])
            ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax6.set_title('Average Profit by Trade Type', fontsize=14)
            ax6.set_xlabel('Trade Type', fontsize=10)
            ax6.set_ylabel('Average Profit', fontsize=10)
        else:
            ax6.text(0.5, 0.5, 'Trade type data not available', ha='center', va='center')
            ax6.set_title('Average Profit by Trade Type', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade analysis to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_performance_metrics(
        self,
        metrics: Dict,
        title: str = "Performance Metrics",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot key performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        # Select key metrics to display
        key_metrics = [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
            'win_rate', 'profit_factor', 'annualized_return'
        ]
        
        # Filter metrics to include only those available
        available_metrics = {k: v for k, v in metrics.items() if k in key_metrics}
        
        if not available_metrics:
            logger.warning("No metrics to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        metric_names = list(available_metrics.keys())
        metric_values = list(available_metrics.values())
        
        # Determine colors based on metric values
        colors = []
        for name, value in zip(metric_names, metric_values):
            if name in ['max_drawdown']:
                # For metrics where negative is good
                colors.append(self.colors['profit'] if value < 0 else self.colors['loss'])
            else:
                # For metrics where positive is good
                colors.append(self.colors['profit'] if value > 0 else self.colors['loss'])
        
        # Create bar chart
        bars = ax.bar(metric_names, metric_values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Style the plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance metrics to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot feature importance for models.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if not feature_importance:
            logger.warning("No feature importance data to plot")
            return None
        
        # Create DataFrame from feature importance dict
        df = pd.DataFrame({'Feature': list(feature_importance.keys()),
                          'Importance': list(feature_importance.values())})
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=df, ax=ax, palette='Blues_d')
        
        # Style the plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_learning_curve(
        self,
        learning_data: Dict,
        title: str = "Learning Curve",
        filename: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot learning curves for RL agent training.
        
        Args:
            learning_data: Dictionary with training metrics
            title: Plot title
            filename: Filename to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if not learning_data:
            logger.warning("No learning data to plot")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot training rewards
        if 'train_rewards' in learning_data and learning_data['train_rewards']:
            ax = axes[0, 0]
            rewards = learning_data['train_rewards']
            ax.plot(range(len(rewards)), rewards, color=self.colors['equity'], linewidth=1.5)
            
            # Add smoothed line
            window_size = min(25, len(rewards) // 5) if len(rewards) > 25 else 1
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size-1, len(rewards)), smoothed, color=self.colors['equity'], linewidth=2.5)
            
            ax.set_title('Training Rewards', fontsize=14)
            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Reward', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Plot evaluation rewards
        if 'eval_rewards' in learning_data and learning_data['eval_rewards']:
            ax = axes[0, 1]
            rewards = learning_data['eval_rewards']
            ax.plot(range(len(rewards)), rewards, color=self.colors['price'], linewidth=2)
            ax.set_title('Evaluation Rewards', fontsize=14)
            ax.set_xlabel('Evaluation', fontsize=10)
            ax.set_ylabel('Reward', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Plot actor loss
        if 'actor_loss' in learning_data and learning_data['actor_loss']:
            ax = axes[1, 0]
            losses = learning_data['actor_loss']
            ax.plot(range(len(losses)), losses, color=self.colors['sell'], linewidth=1.5)
            ax.set_title('Actor Loss', fontsize=14)
            ax.set_xlabel('Update', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Plot critic loss
        if 'critic_loss' in learning_data and learning_data['critic_loss']:
            ax = axes[1, 1]
            losses = learning_data['critic_loss']
            ax.plot(range(len(losses)), losses, color=self.colors['sell'], linewidth=1.5)
            ax.set_title('Critic Loss', fontsize=14)
            ax.set_xlabel('Update', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save if filename provided
        if filename is not None:
            filename = os.path.join(self.plots_dir, filename)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curve to {filename}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig