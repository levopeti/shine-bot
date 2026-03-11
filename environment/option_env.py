# environments/gold_options_env.py - ARANY + MAX 5 OPció

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Dict
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta


@dataclass
class GoldOptionConfig:
    INITIAL_BALANCE: float = 100000
    POSITION_SIZE: float = 0.1  # Fix 10% pot
    MAX_OPEN_OPTIONS: int = 5  # MAX 5 OPció!
    TRANSACTION_COST: float = 0.0005
    EPISODE_LENGTH: int = 78  # M5, 1 nap
    DEADZONE: float = 0.1  # -0.1 .. 0.1 = HOLD
    STRIKE_DISTANCE: float = 0.02  # 2% strike
    TP_DISTANCE: float = 0.015  # 1.5% TP
    SL_DISTANCE: float = 0.01  # 1% SL
    PREMIUM_PCT: float = 0.005  # 0.5% prémium


class GoldOptionsEnv(gym.Env):
    """
    ARANY trading MAX 5 OPcióval:
    - HÁLÓ: [-1,1] → CALL(-)/PUT(+)/HOLD
    - MAX 5 nyitott opció egyszerre
    - Automatikus SL/TP zárás
    """

    def __init__(self, gold_data: pd.DataFrame, config: GoldOptionConfig):
        super().__init__()
        self.data = gold_data
        self.config = config

        # Portfolio + opciók
        self.portfolio_value = config.INITIAL_BALANCE
        self.initial_balance = config.INITIAL_BALANCE
        self.open_options: List[Dict] = []
        self.current_step = 0

        # Action: [-1,1] egyetlen float!
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State: price + indicators + 5 option states (15 feature)
        self.state_dim = 20  # 5 price feat + 15 option feat
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.max_steps = config.EPISODE_LENGTH

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.open_options = []
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        action_value = action[0]  # [-1, 1]
        current_price = self.data['Close'].iloc[self.current_step]
        prev_portfolio = self.portfolio_value

        # DEAD ZONE: -0.1 .. 0.1 = HOLD
        if abs(action_value) < self.config.DEADZONE:
            reward = 0
        else:
            # ÚJ OPció nyitása HA < 5
            if len(self.open_options) < self.config.MAX_OPEN_OPTIONS:
                direction = -1 if action_value < 0 else 1  # -1=CALL, 1=PUT
                self._open_option(current_price, direction)

            reward = self._calculate_reward()

        # SL/TP ellenőrzés MINDEN opcióra
        self._check_options(current_price)

        # Portfolio frissítés
        total_premium = sum(opt['premium'] for opt in self.open_options)
        self.portfolio_value = self.initial_balance - total_premium

        # Reward finomhangolás
        portfolio_change = (self.portfolio_value - prev_portfolio) / prev_portfolio
        reward += portfolio_change * 0.1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            'portfolio_value': self.portfolio_value,
            'open_options': len(self.open_options),
            'current_price': current_price
        }

        return self._get_observation(), reward, done, False, info

    def _open_option(self, current_price: float, direction: int):
        """Új arany opció nyitása"""
        strike_price = current_price * (1 + self.config.STRIKE_DISTANCE * direction)
        tp_price = current_price * (1 + self.config.TP_DISTANCE * direction)
        sl_price = current_price * (1 - self.config.SL_DISTANCE * direction)

        premium = current_price * self.config.PREMIUM_PCT * self.config.POSITION_SIZE

        option = {
            'direction': direction,  # -1=CALL (fel), 1=PUT (le)
            'open_price': current_price,
            'strike_price': strike_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'premium': premium,
            'pnl': 0.0,
            'open_step': self.current_step
        }
        self.open_options.append(option)

    def _check_options(self, current_price: float):
        """SL/TP ellenőrzés és zárás"""
        i = 0
        while i < len(self.open_options):
            opt = self.open_options[i]

            # SL/TP HIT?
            hit_tp = (opt['direction'] == -1 and current_price >= opt['tp_price']) or \
                     (opt['direction'] == 1 and current_price <= opt['tp_price'])
            hit_sl = (opt['direction'] == -1 and current_price <= opt['sl_price']) or \
                     (opt['direction'] == 1 and current_price >= opt['sl_price'])

            if hit_tp or hit_sl:
                # Zárás
                profit_factor = 2.0 if hit_tp else -1.0
                opt['pnl'] = opt['premium'] * (profit_factor - 1)

                # Törlés
                self.open_options[i] = self.open_options[-1]
                self.open_options.pop()
            else:
                i += 1

    def _get_observation(self) -> np.ndarray:
        """State: price features + 5 option states"""
        if self.current_step >= len(self.data):
            price_data = self.data.iloc[-1]
        else:
            price_data = self.data.iloc[self.current_step]

        current_price = price_data['Close']

        # Price features (5)
        price_features = np.array([
            price_data['Close'] / 2000,  # Normalizált ár
            self.current_step / self.max_steps
        ])

        # Option states: 5 slot × 3 feature (direction, dist_to_strike, time_left)
        option_states = np.zeros(15)  # 5×3=15
        for i, opt in enumerate(self.open_options[:5]):
            dist_to_strike = (current_price - opt['strike_price']) / opt['strike_price']
            time_left = max(0, self.config.EPISODE_LENGTH - (
                        self.current_step - opt['open_step'])) / self.config.EPISODE_LENGTH
            option_states[i * 3:i * 3 + 3] = [opt['direction'], dist_to_strike, time_left]

        state = np.concatenate([price_features, option_states])
        return state.astype(np.float32)

    def _calculate_reward(self) -> float:
        """Reward = opciók P&L + portfolio"""
        if not self.open_options:
            return 0.0

        total_pnl = sum(opt['pnl'] for opt in self.open_options)
        return total_pnl * 0.1  # Skálázott


# 🚀 TESZT:
if __name__ == "__main__":
    config = GoldOptionConfig()

    # Arany M5
    gold_data = yf.download('GC=F',
                            start=datetime.now() - timedelta(days=45),
                            interval='5m',
                            prepost=True)

    env = GoldOptionsEnv(gold_data, config)

    obs, _ = env.reset()
    print(f"✅ State shape: {obs.shape} (20 feat)")
    print(f"MAX opciók: {config.MAX_OPEN_OPTIONS}")

    # Test lépések
    for i in range(10):
        action = np.array([0.3 * i / 10])  # Gradiens teszt
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Action={action[0]:.2f}, Opciók={info['open_options']}, "
              f"Portfólió=${info['portfolio_value']:,.0f}")
