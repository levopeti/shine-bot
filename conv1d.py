import torch as t
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Conv1DFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, window_h=12):
        super().__init__(observation_space, features_dim)

        self.n_features = 6    # open, close, high, low, volume, rsi
        self.window = window_h * 12

        self.conv = nn.Sequential(
            nn.Conv1d(self.n_features, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )

        # egy dummy forward
        with t.no_grad():
            dummy = t.zeros(1, observation_space.shape[0])
            x = self._reshape_input(dummy)
            x = self.conv(x)
            n_flatten = x.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def _reshape_input(self, obs: t.Tensor) -> t.Tensor:
        # obs shape: (batch, 73) = 12*6 + 1
        batch_size = obs.shape[0]
        seq = obs[:, :self.window * self.n_features]     # az első 72 elem
        x = seq.view(batch_size, self.window, self.n_features)
        x = x.permute(0, 2, 1)
        # (batch, channels=6, length=12 * w)
        return x

    def forward(self, observations: t.Tensor) -> t.Tensor:
        x = self._reshape_input(observations)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

"""
from stable_baselines3 import DQN

policy_kwargs = dict(
    features_extractor_class=Conv1DFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[256, 128]   # ez már a CNN utáni MLP (Q-háló)
)

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
"""