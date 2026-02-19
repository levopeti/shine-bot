# environment/reward_functions.py

import numpy as np


class RewardFunctions:
    """
    Collection of different reward functions for trading agents
    """

    @staticmethod
    def simple_return(value_before, value_after):
        """
        Simple percentage return
        """
        return (value_after - value_before) / value_before

    @staticmethod
    def sharpe_based(returns_history, risk_free_rate=0.0):
        """
        Reward based on Sharpe ratio
        """
        if len(returns_history) < 2:
            return 0.0

        returns = np.array(returns_history)
        excess_return = np.mean(returns) - risk_free_rate
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        sharpe = excess_return / std_return
        return sharpe

    @staticmethod
    def risk_adjusted(value_before, value_after, volatility, risk_penalty=0.5):
        """
        Return adjusted for risk (volatility)
        """
        simple_return = (value_after - value_before) / value_before
        risk_adjusted_return = simple_return - risk_penalty * volatility
        return risk_adjusted_return

    @staticmethod
    def profit_with_drawdown_penalty(value_before, value_after, max_drawdown, penalty_factor=2.0):
        """
        Profit with penalty for drawdown
        """
        profit = (value_after - value_before) / value_before
        drawdown_penalty = penalty_factor * abs(max_drawdown)
        return profit - drawdown_penalty
