# strategies/ensemble_strategy.py

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class EnsembleStrategy:
    """
    Combine predictions from multiple agents
    """

    def __init__(self, agents: List, weights: List[float] = None):
        self.agents = agents

        # Equal weights if not specified
        if weights is None:
            self.weights = [1.0 / len(agents)] * len(agents)
        else:
            assert len(weights) == len(agents), "Weights must match number of agents"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action by combining predictions from all agents
        """
        actions = []

        for agent in self.agents:
            action = agent.select_action(state)
            actions.append(action)

        # Weighted average of actions
        ensemble_action = np.zeros_like(actions[0])

        for action, weight in zip(actions, self.weights):
            ensemble_action += action * weight

        return ensemble_action

    def set_eval_mode(self):
        """
        Set all agents to evaluation mode
        """
        for agent in self.agents:
            agent.set_eval_mode()

    def set_train_mode(self):
        """
        Set all agents to training mode
        """
        for agent in self.agents:
            agent.set_train_mode()
