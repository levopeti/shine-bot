# strategies/rl_strategy.py

import numpy as np
import logging

logger = logging.getLogger(__name__)


class RLStrategy:
    """
    Wrapper for reinforcement learning trading strategies
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.position_history = []

    def execute(self, state: np.ndarray) -> np.ndarray:
        """
        Execute strategy to get trading actions
        """
        # Get raw action from agent
        action = self.agent.select_action(state)

        # Apply risk management filters
        action = self._apply_risk_management(action, state)

        return action

    def _apply_risk_management(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Apply risk management rules to actions
        """
        # Portfolio value and cash from state
        portfolio_value = state[0]
        cash = state[1]

        # Limit position sizes
        max_position_value = portfolio_value * self.config.MAX_POSITION_SIZE

        # Clip actions if needed
        action = np.clip(action, -1, 1)

        # Don't buy if low on cash
        if cash < portfolio_value * 0.05:  # Less than 5% cash
            action = np.where(action > 0, 0, action)  # Prevent buying

        return action
