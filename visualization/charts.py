# visualization/charts.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List


class TradingCharts:
    """
    Create various charts for trading analysis
    """

    def __init__(self, style='darkgrid'):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_portfolio_value(self, portfolio_values: np.ndarray, save_path: str = None):
        """
        Plot portfolio value over time
        """
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values, linewidth=2, color='#2ecc71')
        plt.axhline(y=portfolio_values[0], color='gray', linestyle='--', label='Initial Balance')
        plt.title('Portfolio Value Over Time', fontsize=16)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_returns_distribution(self, returns: np.ndarray, save_path: str = None):
        """
        Plot distribution of returns
        """
        plt.figure(figsize=(12, 6))
        plt.hist(returns * 100, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribution of Returns', fontsize=16)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_drawdown(self, portfolio_values: np.ndarray, save_path: str = None):
        """
        Plot drawdown over time
        """
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        plt.figure(figsize=(14, 7))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='#e74c3c')
        plt.plot(drawdown, linewidth=2, color='#c0392b')
        plt.title('Drawdown Over Time', fontsize=16)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_asset_comparison(self, data: pd.DataFrame, assets: List[str], save_path: str = None):
        """
        Plot normalized price comparison of multiple assets
        """
        plt.figure(figsize=(14, 7))

        for asset in assets:
            close_cols = [col for col in data.columns if asset in col and 'Close' in col]
            if close_cols:
                prices = data[close_cols[0]]
                normalized = (prices / prices.iloc[0] - 1) * 100
                plt.plot(data.index, normalized, label=asset, linewidth=2)

        plt.title('Asset Price Comparison (Normalized)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price Change (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_trade_analysis(self, trade_history: List[Dict], save_path: str = None):
        """
        Plot trade analysis
        """
        df = pd.DataFrame(trade_history)

        if df.empty:
            print("No trades to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Trades over time
        axes[0, 0].scatter(df['step'], df['value'], c=['green' if a == 'BUY' else 'red' for a in df['action']], alpha=0.6)
        axes[0, 0].set_title('Trades Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Trade Value ($)')
        axes[0, 0].grid(True, alpha=0.3)

        # Trade distribution by asset
        asset_counts = df['asset'].value_counts()
        axes[0, 1].bar(range(len(asset_counts)), asset_counts.values)
        axes[0, 1].set_title('Trades by Asset')
        axes[0, 1].set_xlabel('Asset')
        axes[0, 1].set_ylabel('Number of Trades')
        axes[0, 1].set_xticks(range(len(asset_counts)))
        axes[0, 1].set_xticklabels(asset_counts.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Buy vs Sell
        action_counts = df['action'].value_counts()
        axes[1, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Buy vs Sell Distribution')

        # Trade value distribution
        axes[1, 1].hist(df['value'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        axes[1, 1].set_title('Trade Value Distribution')
        axes[1, 1].set_xlabel('Trade Value ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, data: pd.DataFrame, save_path: str = None):
        """
        Plot correlation matrix of asset returns
        """
        # Extract close prices
        close_cols = [col for col in data.columns if 'Close' in col]
        close_data = data[close_cols]

        # Calculate returns
        returns = close_data.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Asset Correlation Matrix', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
