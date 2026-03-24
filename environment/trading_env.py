# environment/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom trading environment compatible with OpenAI Gym
    Supports multiple assets across different markets

    MÓDOSÍTÁS: Rolling window support 30 éves adatokhoz
    """

    def __init__(self, data: pd.DataFrame, config, mode='train'):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.config = config
        self.mode = mode

        # Get list of tradeable assets from column names
        self.assets = self._extract_asset_names()
        self.n_assets = len(self.assets)

        # ====================================================================
        # MÓDOSÍTÁS: Rolling window beállítás
        # ====================================================================
        self.episode_length = getattr(config, 'EPISODE_LENGTH', len(data))

        if mode == 'train' and hasattr(config, 'EPISODE_LENGTH'):
            # Random kezdőpont számítása
            self.max_start_step = len(data) - self.episode_length - 1
            self.use_rolling_window = True
            print(f"\n{'=' * 70}")
            print(f"TradingEnvironment initialized with ROLLING WINDOW")
            print(f"{'=' * 70}")
            print(f"  Mode: {mode}")
            print(f"  Total data: {len(data)} days ({data.index[0].date()} - {data.index[-1].date()})")
            print(f"  Episode length: {self.episode_length} days (~{self.episode_length // 252} years)")
            print(f"  Possible start points: {self.max_start_step}")
        else:
            # Test módban vagy ha nincs EPISODE_LENGTH: teljes adat
            self.max_start_step = 0
            self.use_rolling_window = False
            print(f"\n{'=' * 70}")
            print(f"TradingEnvironment initialized (FULL DATA)")
            print(f"{'=' * 70}")
            print(f"  Mode: {mode}")
            print(f"  Total data: {len(data)} days")
        # ====================================================================

        # State: [portfolio_value, cash, positions..., market_features...]
        self.state_dim = 2 + self.n_assets + data.shape[1]

        # Action: [buy/sell/hold for each asset] - continuous values from -1 to 1
        # -1 = sell all, 0 = hold, 1 = buy max
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def _extract_asset_names(self) -> list:
        """
        Extract unique asset names from column headers
        """
        assets = set()
        for col in self.data.columns:
            # Extract asset name from column (e.g., 'stocks_AAPL_Close' -> 'stocks_AAPL')
            parts = col.split('_')
            if len(parts) >= 2:
                asset_name = f"{parts[0]}_{parts[1]}"
                assets.add(asset_name)
        return sorted(list(assets))

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state

        MÓDOSÍTÁS: Random kezdőpont ha rolling window van
        """
        # ====================================================================
        # MÓDOSÍTÁS: Rolling window - random kezdőpont
        # ====================================================================
        if self.use_rolling_window:
            # Random kezdőpont választása a 30 év között
            self.start_step = np.random.randint(0, self.max_start_step)
            self.end_step = self.start_step + self.episode_length
            self.current_step = self.start_step

            print(f"\n  Episode start: {self.data.index[self.start_step].date()} (step {self.start_step})")
            print(f"  Episode end:   {self.data.index[self.end_step - 1].date()} (step {self.end_step - 1})")
        else:
            # Teljes adat használata (test mód vagy nincs rolling window)
            self.start_step = 0
            self.end_step = len(self.data)
            self.current_step = 0
        # ====================================================================

        # Reset trading state
        self.cash = self.config.INITIAL_BALANCE
        self.positions = np.zeros(self.n_assets)  # Number of units held
        self.portfolio_value = self.cash
        self.total_trades = 0
        self.trade_history = []

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """ Get current state observation """
        # ====================================================================
        # MÓDOSÍTÁS: Boundary check az end_step-hez képest
        # ====================================================================
        if self.current_step >= self.end_step:
            self.current_step = self.end_step - 1
        # ====================================================================

        # Portfolio state
        portfolio_state = np.array([
            self.portfolio_value,
            self.cash
        ])

        # Current positions
        positions_state = self.positions.copy()

        # Market features (current row of data)
        market_state = self.data.iloc[self.current_step].values

        # Combine all states
        observation = np.concatenate([
            portfolio_state,
            positions_state,
            market_state
        ])
        return observation.astype(np.float32)

    def _get_asset_prices(self) -> np.ndarray:
        """ Get current prices for all assets """
        prices = []
        for asset in self.assets:
            # Look for Close price column for this asset
            close_col = [col for col in self.data.columns if asset in col and 'Close' in col]
            if close_col:
                price = self.data.iloc[self.current_step][close_col[0]]
                prices.append(price)
            else:
                prices.append(0.0)
        return np.array(prices)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: Array of actions for each asset (-1 to 1)

        Returns:
            observation, reward, done, info
        """
        # Get current prices
        current_prices = self._get_asset_prices()

        # Calculate portfolio value before action
        portfolio_value_before = self.cash + np.sum(self.positions * current_prices)

        # Execute trades based on actions
        for i, act in enumerate(action):
            if current_prices[i] <= 0:
                continue

            if act > 0.1:  # Buy signal
                # Calculate how much we can buy
                max_buy_value = self.cash * self.config.MAX_POSITION_SIZE
                buy_value = max_buy_value * act

                # Account for transaction costs
                buy_value_after_fee = buy_value * (1 - self.config.TRANSACTION_COST)

                # Calculate units to buy
                units_to_buy = buy_value_after_fee / current_prices[i]

                if buy_value <= self.cash:
                    self.positions[i] += units_to_buy
                    self.cash -= buy_value
                    self.total_trades += 1

                    self.trade_history.append({
                        'step': self.current_step,
                        'asset': self.assets[i],
                        'action': 'BUY',
                        'units': units_to_buy,
                        'price': current_prices[i],
                        'value': buy_value
                    })

            elif act < -0.1:  # Sell signal
                # Calculate how much to sell
                units_to_sell = self.positions[i] * abs(act)

                if units_to_sell > 0:
                    sell_value = units_to_sell * current_prices[i]

                    # Account for transaction costs
                    sell_value_after_fee = sell_value * (1 - self.config.TRANSACTION_COST)

                    self.positions[i] -= units_to_sell
                    self.cash += sell_value_after_fee
                    self.total_trades += 1

                    self.trade_history.append({
                        'step': self.current_step,
                        'asset': self.assets[i],
                        'action': 'SELL',
                        'units': units_to_sell,
                        'price': current_prices[i],
                        'value': sell_value
                    })

        # Move to next step
        self.current_step += 1

        # ====================================================================
        # MÓDOSÍTÁS: Done check az end_step alapján
        # ====================================================================
        # Check if episode is done (reached end of window or data)
        done = (self.current_step >= self.end_step - 1) or (self.current_step >= len(self.data) - 1)
        # ====================================================================

        # Calculate new portfolio value
        if self.current_step < len(self.data):
            new_prices = self._get_asset_prices()
            portfolio_value_after = self.cash + np.sum(self.positions * new_prices)
        else:
            portfolio_value_after = portfolio_value_before

        # Calculate reward
        reward = self._calculate_reward(portfolio_value_before, portfolio_value_after)

        # Update portfolio value
        self.portfolio_value = portfolio_value_after

        # ====================================================================
        # MÓDOSÍTÁS: Episode befejezési üzenet
        # ====================================================================
        if done:
            steps_taken = self.current_step - self.start_step
            return_pct = (self.portfolio_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE
            print(f"  Episode finished: Portfolio=${self.portfolio_value:,.0f} | "
                  f"Return={return_pct * 100:.2f}% | "
                  f"Steps={steps_taken} | "
                  f"Trades={self.total_trades}")
        # ====================================================================

        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_trades': self.total_trades,
            'return': (self.portfolio_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE
        }

        return self._get_observation(), reward, done, info

    def _calculate_reward(self, value_before: float, value_after: float) -> float:
        """
        Calculate reward for the agent
        Can be customized with different reward functions
        """
        # Simple return-based reward
        return_pct = (value_after - value_before) / value_before

        # Penalize excessive trading
        trade_penalty = -0.0001 * self.total_trades

        reward = return_pct + trade_penalty

        return reward

    def render(self, mode='human'):
        """
        Render the environment (optional)
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print("-" * 50)

