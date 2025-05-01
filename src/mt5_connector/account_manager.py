"""
Account management module for MT5 trading accounts.

This module provides functionality to manage MT5 account information,
track balance and equity, and manage account-level settings.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from src.mt5_connector.connector import MT5Connector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AccountManager:
    """Class for managing MT5 trading accounts."""
    
    def __init__(self, config: Dict, mt5_connector: Optional[MT5Connector] = None):
        """
        Initialize the AccountManager.
        
        Args:
            config: Configuration dictionary
            mt5_connector: Optional MT5Connector instance
        """
        self.config = config
        
        # Initialize MT5Connector if not provided
        if mt5_connector is None:
            self.mt5_connector = MT5Connector(config)
            self.mt5_connector.initialize()
        else:
            self.mt5_connector = mt5_connector
        
        # Store initial account information
        self.initial_info = self.get_account_info()
        
        logger.info("AccountManager initialized")
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get current account information.
        
        Returns:
            Optional[Dict]: Dictionary with account information or None if error
        """
        return self.mt5_connector.get_account_info()
    
    def calculate_drawdown(self) -> Dict:
        """
        Calculate current and maximum drawdown.
        
        Returns:
            Dict: Dictionary with drawdown metrics
        """
        # Get account info
        account_info = self.get_account_info()
        if account_info is None:
            return {
                'current_dd_percent': 0,
                'max_dd_percent': 0,
                'current_dd_value': 0,
                'max_dd_value': 0
            }
        
        # Get current balance and equity
        balance = account_info['balance']
        equity = account_info['equity']
        
        # If this is the first check, initialize tracking
        if not hasattr(self, 'max_equity'):
            self.max_equity = equity
            self.max_balance = balance
        else:
            # Update max values if needed
            if equity > self.max_equity:
                self.max_equity = equity
            if balance > self.max_balance:
                self.max_balance = balance
        
        # Calculate drawdowns
        equity_dd_value = self.max_equity - equity
        equity_dd_percent = (equity_dd_value / self.max_equity) * 100 if self.max_equity > 0 else 0
        
        balance_dd_value = self.max_balance - balance
        balance_dd_percent = (balance_dd_value / self.max_balance) * 100 if self.max_balance > 0 else 0
        
        # Track max drawdown if needed
        if not hasattr(self, 'max_equity_dd'):
            self.max_equity_dd = equity_dd_percent
            self.max_equity_dd_value = equity_dd_value
        else:
            if equity_dd_percent > self.max_equity_dd:
                self.max_equity_dd = equity_dd_percent
                self.max_equity_dd_value = equity_dd_value
        
        return {
            'current_equity_dd_percent': equity_dd_percent,
            'max_equity_dd_percent': self.max_equity_dd,
            'current_equity_dd_value': equity_dd_value,
            'max_equity_dd_value': self.max_equity_dd_value,
            'current_balance_dd_percent': balance_dd_percent,
            'max_equity': self.max_equity,
            'max_balance': self.max_balance
        }
    
    def calculate_margin_level(self) -> Optional[Dict]:
        """
        Calculate current margin level and related metrics.
        
        Returns:
            Optional[Dict]: Dictionary with margin metrics or None if error
        """
        # Get account info
        account_info = self.get_account_info()
        if account_info is None:
            return None
        
        # Extract margin information
        equity = account_info['equity']
        margin = account_info['margin']
        free_margin = account_info['margin_free']
        
        # Calculate margin level
        margin_level = (equity / margin) * 100 if margin > 0 else float('inf')
        
        return {
            'equity': equity,
            'margin': margin,
            'free_margin': free_margin,
            'margin_level': margin_level,
            'margin_used_percent': (margin / equity) * 100 if equity > 0 else 0,
            'free_margin_percent': (free_margin / equity) * 100 if equity > 0 else 0
        }
    
    def check_risk_limits(self) -> Dict:
        """
        Check if current risk levels exceed defined limits.
        
        Returns:
            Dict: Dictionary with risk check results
        """
        # Get risk limits from config
        risk_limits = self.config.get('trading_params', {}).get('position_sizing', {})
        max_risk_per_trade = risk_limits.get('risk_per_trade', 0.02)  # Default 2%
        max_risk_per_day = risk_limits.get('max_risk_per_day', 0.06)  # Default 6%
        max_positions = risk_limits.get('max_positions', 5)
        
        # Get current positions
        positions = self.mt5_connector.get_positions()
        
        # Get account info
        account_info = self.get_account_info()
        if account_info is None or not positions:
            return {
                'total_positions': 0,
                'max_positions_exceeded': False,
                'max_risk_exceeded': False,
                'max_daily_risk_exceeded': False,
                'current_open_risk': 0,
                'risk_per_position': []
            }
        
        equity = account_info['equity']
        
        # Calculate risk per position
        risk_per_position = []
        total_risk = 0
        
        for position in positions:
            # Get symbol info for proper stop loss calculation
            symbol_info = self.mt5_connector.get_symbol_info(position['symbol'])
            
            if symbol_info and position.get('sl', 0) != 0:
                # Calculate potential loss if stop loss is hit
                if position['type'] == 0:  # BUY
                    risk_amount = (position['price_open'] - position['sl']) * position['volume'] * symbol_info['trade_contract_size']
                else:  # SELL
                    risk_amount = (position['sl'] - position['price_open']) * position['volume'] * symbol_info['trade_contract_size']
                
                # Convert to account currency if needed
                if symbol_info['currency_profit'] != account_info['currency']:
                    # This is simplified and should be replaced with proper conversion
                    risk_amount = risk_amount  # Convert currency
                
                # Calculate risk as percentage of equity
                risk_percent = (risk_amount / equity) * 100
                
                risk_per_position.append({
                    'ticket': position['ticket'],
                    'symbol': position['symbol'],
                    'risk_amount': risk_amount,
                    'risk_percent': risk_percent,
                    'exceeds_limit': risk_percent > max_risk_per_trade * 100
                })
                
                total_risk += risk_percent
        
        # Check against limits
        max_positions_exceeded = len(positions) > max_positions
        max_risk_exceeded = any(r['exceeds_limit'] for r in risk_per_position)
        max_daily_risk_exceeded = total_risk > max_risk_per_day * 100
        
        return {
            'total_positions': len(positions),
            'max_positions_exceeded': max_positions_exceeded,
            'max_risk_exceeded': max_risk_exceeded,
            'max_daily_risk_exceeded': max_daily_risk_exceeded,
            'current_open_risk': total_risk,
            'risk_per_position': risk_per_position
        }
    
    def track_account_growth(
        self,
        interval: str = 'daily',
        lookback: int = 30
    ) -> pd.DataFrame:
        """
        Track account balance and equity growth over time.
        
        Args:
            interval: Time interval ('daily', 'weekly', or 'monthly')
            lookback: Number of intervals to look back
            
        Returns:
            pd.DataFrame: DataFrame with account metrics over time
        """
        # Get account info
        current_info = self.get_account_info()
        if current_info is None:
            return pd.DataFrame()
        
        # Get current time
        now = datetime.now()
        
        # Calculate start date based on interval and lookback
        if interval == 'daily':
            start_date = now - timedelta(days=lookback)
        elif interval == 'weekly':
            start_date = now - timedelta(weeks=lookback)
        elif interval == 'monthly':
            start_date = now - timedelta(days=30 * lookback)
        else:
            logger.error(f"Invalid interval: {interval}. Must be 'daily', 'weekly', or 'monthly'")
            return pd.DataFrame()
        
        # Get trade history
        deals = self.mt5_connector.get_deal_history(
            from_date=start_date,
            to_date=now
        )
        
        if not deals:
            # No history data, create simple DataFrame with current info
            return pd.DataFrame({
                'date': [now],
                'balance': [current_info['balance']],
                'equity': [current_info['equity']],
                'profit': [0],
                'deposits': [0],
                'withdrawals': [0]
            })
        
        # Initialize data structure
        if interval == 'daily':
            # Create a date range for each day
            date_range = pd.date_range(start=start_date, end=now, freq='D')
        elif interval == 'weekly':
            # Create a date range for each week
            date_range = pd.date_range(start=start_date, end=now, freq='W')
        else:  # monthly
            # Create a date range for each month
            date_range = pd.date_range(start=start_date, end=now, freq='M')
            # Add current month if not included
            if date_range[-1].month != now.month or date_range[-1].year != now.year:
                date_range = date_range.append(pd.DatetimeIndex([now]))
        
        # Convert deals to DataFrame for easier processing
        deals_df = pd.DataFrame(deals)
        
        # Add datetime column
        if 'time' not in deals_df.columns:
            deals_df['time'] = pd.to_datetime([d.get('time_open', now) for d in deals])
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=date_range)
        result['date'] = result.index
        
        # Compute balance and equity for each period
        initial_balance = self.initial_info.get('balance', 0) if self.initial_info else 0
        balance = initial_balance
        
        result['balance'] = 0
        result['equity'] = 0
        result['profit'] = 0
        result['deposits'] = 0
        result['withdrawals'] = 0
        
        # Process deals chronologically
        for i, date in enumerate(date_range):
            # Filter deals for this period
            if i == 0:
                period_deals = deals_df[deals_df['time'] <= date]
            else:
                period_deals = deals_df[(deals_df['time'] > date_range[i-1]) & 
                                       (deals_df['time'] <= date)]
            
            # Calculate profit, deposits, and withdrawals for this period
            profit = period_deals[period_deals['type'] == 0]['profit'].sum() if 'profit' in period_deals.columns else 0
            deposits = period_deals[period_deals['type'] == 1]['profit'].sum() if 'profit' in period_deals.columns else 0
            withdrawals = period_deals[period_deals['type'] == 2]['profit'].sum() if 'profit' in period_deals.columns else 0
            
            # Update balance
            balance += profit + deposits + withdrawals
            
            # Store in result
            result.loc[date, 'balance'] = balance
            result.loc[date, 'profit'] = profit
            result.loc[date, 'deposits'] = deposits
            result.loc[date, 'withdrawals'] = withdrawals
        
        # Update with current equity
        result.loc[result.index[-1], 'equity'] = current_info['equity']
        
        # For older entries, estimate equity based on balance (simplified)
        for i in range(len(result) - 1):
            result.loc[result.index[i], 'equity'] = result.loc[result.index[i], 'balance']
        
        return result.reset_index(drop=True)
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate various account performance metrics.
        
        Returns:
            Dict: Dictionary with performance metrics
        """
        # Get account growth data
        growth_data = self.track_account_growth(interval='daily', lookback=90)
        
        if growth_data.empty:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'win_rate': 0
            }
        
        # Calculate returns
        if len(growth_data) > 1:
            initial_balance = growth_data['balance'].iloc[0]
            final_balance = growth_data['balance'].iloc[-1]
            
            # Total return
            total_return = (final_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
            
            # Daily returns
            daily_returns = growth_data['balance'].pct_change().dropna()
            
            # Annualized return (assuming 252 trading days per year)
            annualized_return = ((1 + total_return/100) ** (252 / len(growth_data)) - 1) * 100
            
            # Annualized volatility
            annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Sortino ratio (using only negative returns for downside risk)
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) * 100
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            max_drawdown = drawdown.min()
        else:
            # Not enough data points
            total_return = 0
            annualized_return = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
        
        # Get trade statistics
        from_date = datetime.now() - timedelta(days=90)
        trade_stats = self.calculate_trade_statistics(from_date=from_date)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': trade_stats.get('profit_factor', 0),
            'win_rate': trade_stats.get('win_rate', 0) * 100
        }
    
    def calculate_trade_statistics(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Calculate trading performance statistics.
        
        Args:
            from_date: Start date for statistics
            to_date: End date for statistics (defaults to now)
            symbol: Symbol to filter by (or None for all symbols)
            
        Returns:
            Dict: Dictionary with trading statistics
        """
        # Import OrderManager to use its statistics calculation
        from src.mt5_connector.order_manager import OrderManager
        
        # Create OrderManager with our connector
        order_manager = OrderManager(self.config, self.mt5_connector)
        
        # Use OrderManager's statistics calculation
        return order_manager.calculate_trade_statistics(
            from_date=from_date,
            to_date=to_date,
            symbol=symbol
        )