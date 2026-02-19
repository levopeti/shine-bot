# backtesting/metrics.py

import numpy as np
import pandas as pd
from typing import List


class PerformanceMetrics:
    """
    Calculate various trading performance metrics
    """

    @staticmethod
    def total_return(portfolio_values: np.ndarray) -> float:
        """
        Calculate total return
        """
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    @staticmethod
    def annualized_return(portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return
        """
        total_return = PerformanceMetrics.total_return(portfolio_values)
        n_periods = len(portfolio_values)
        years = n_periods / periods_per_year

        if years > 0:
            return (1 + total_return) ** (1 / years) - 1
        return 0

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year

        if len(excess_returns) > 0 and np.std(excess_returns) > 0:
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        return 0

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility)
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
        return 0

    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        """
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def calmar_ratio(portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        """
        ann_return = PerformanceMetrics.annualized_return(portfolio_values, periods_per_year)
        max_dd = abs(PerformanceMetrics.max_drawdown(portfolio_values))

        if max_dd > 0:
            return ann_return / max_dd
        return 0

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive returns)
        """
        if len(returns) > 0:
            return np.sum(returns > 0) / len(returns)
        return 0

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Calculate profit factor (total gains / total losses)
        """
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))

        if losses > 0:
            return gains / losses
        return 0

    @staticmethod
    def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility
        """
        return np.std(returns) * np.sqrt(periods_per_year)
