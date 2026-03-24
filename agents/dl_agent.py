# agents/dl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for trading decisions
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepLearningAgent:
    """
    Deep Reinforcement Learning agent using DQN/DDQN
    """

    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Device configuration
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.RL_CONFIG['learning_rate']
        )

        # Experience replay memory
        self.memory = deque(maxlen=config.RL_CONFIG['memory_size'])

        # Training parameters
        self.batch_size = config.RL_CONFIG['batch_size']
        self.gamma = config.RL_CONFIG['gamma']
        self.epsilon = config.RL_CONFIG['epsilon_start']
        self.epsilon_min = config.RL_CONFIG['epsilon_min']
        self.epsilon_decay = config.RL_CONFIG['epsilon_decay']
        self.update_frequency = config.RL_CONFIG['update_frequency']

        self.steps = 0
        self.training_mode = True

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using epsilon-greedy policy
        """
        if self.training_mode and random.random() < self.epsilon:
            # Explore: random action
            return np.random.uniform(-1, 1, self.action_dim)
        else:
            # Exploit: use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)

                # Convert Q-values to actions in range [-1, 1]
                actions = torch.tanh(q_values).cpu().numpy()[0]
                return actions

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """
        Perform one training step
        Returns the loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        [print(len(x)) for x in states]
        breakpoint()
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states)

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss (MSE between current Q and target Q)
        loss = nn.MSELoss()(current_q.mean(dim=1), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, path: str):
        """
        Save model checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"Model loaded from {path}")

    def set_eval_mode(self):
        """
        Set agent to evaluation mode (no exploration)
        """
        self.training_mode = False
        self.policy_net.eval()

    def set_train_mode(self):
        """
        Set agent to training mode
        """
        self.training_mode = True
        self.policy_net.train()
