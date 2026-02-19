# agents/llm_agent.py

import numpy as np
import logging

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM-based trading agent (placeholder for future implementation)
    Can integrate with OpenAI GPT, Claude, or other LLMs
    """

    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        logger.warning("LLM Agent is currently a placeholder implementation")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using LLM reasoning
        """
        # Placeholder: return random action
        # In full implementation, would:
        # 1. Convert state to natural language description
        # 2. Query LLM with market context
        # 3. Parse LLM response to action

        return np.random.uniform(-1, 1, self.action_dim)

    def _state_to_prompt(self, state: np.ndarray) -> str:
        """
        Convert numerical state to natural language prompt
        """
        prompt = f"""
        You are a professional trading agent. Based on the current market conditions:

        Portfolio Value: {state[0]:.2f}
        Cash Available: {state[1]:.2f}

        Analyze the market data and decide whether to buy, sell, or hold each asset.
        Provide your decision as a JSON with actions for each asset.
        """
        return prompt

    def save_model(self, path: str):
        """
        Save model configuration
        """
        logger.info("LLM agent doesn't require traditional model saving")

    def load_model(self, path: str):
        """
        Load model configuration
        """
        logger.info("LLM agent doesn't require traditional model loading")

    def set_eval_mode(self):
        pass

    def set_train_mode(self):
        pass
