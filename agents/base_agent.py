# agents/base_agent.py

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    """

    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action based on current state
        """
        pass

    @abstractmethod
    def train_step(self) -> float:
        """
        Perform one training step
        Returns loss value
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """
        Save model to file
        """
        pass

    @abstractmethod
    def load_model(self, path: str):
        """
        Load model from file
        """
        pass

    def set_eval_mode(self):
        """
        Set agent to evaluation mode
        """
        pass

    def set_train_mode(self):
        """
        Set agent to training mode
        """
        pass
