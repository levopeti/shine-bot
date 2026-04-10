import json
import random
from time import time

import gymnasium as gym
import numpy as np
from pprint import pprint
from gymnasium import spaces

from config import Config


class ConfidentEnv(gym.Wrapper):
    def __init__(self, env, model, threshold=0.3):
        super().__init__(env)
        self.model = model
        self.threshold = threshold

    def step(self, action):
        obs = self.env._get_obs()

        _, q_values = self.model.predict(obs, deterministic=True)
        confidence = q_values.max() - q_values.mean()
        print(confidence)
        input()

        if confidence < self.threshold:
            action = 0  # Force HOLD

        return self.env.step(action)


class M5TradingEnv(gym.Env):
    """
    Observation: 12 M5 gyertya (open,close,high,low,volume,rsi) + aktuális open
    Action:      0=Hold, 1..9=Buy+TP/SL, 10..18=Sell+TP/SL
    Reward:      TP hit → +10, SL hit → -10, timeout → -1
    """
    metadata = {"render_modes": []}

    def __init__(self, df, features, mode="train"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.log_file = Config.EVAL_LOG_FILE
        self.window = Config.WINDOW_H * 12  # 12 × M5 = 1 óra lookback
        self.fwd_window = Config.FWD_WINDOW
        self.episode_steps = Config.TRAIN_EPISODE_STEPS
        self.episode_indices = list()
        self.current_step = 0
        self.global_step = 0
        self.episode_count = 0
        self.normalize = Config.NORMALIZE
        self.random_indices = Config.RANDOM_INDICES
        self.mode = mode
        self.features = features

        self.equity = 10_000.0
        self.max_equity = 10_000.0
        self.max_drawdown = 1

        self.TP_SL_RATIO = Config.TP_SL_RATIO
        self.SL_LEVELS = Config.SL_LEVELS
        self.max_tp = max(self.SL_LEVELS) * max(self.TP_SL_RATIO)

        n_combos = len(self.TP_SL_RATIO) * len(self.SL_LEVELS)
        self.action_space = spaces.Discrete(1 + 2 * n_combos)

        self.obs_dim = self.window * len(self.features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self._build_action_map()
        self.episode_stats = dict()

    def _build_action_map(self):
        """0=Hold; 1-12=Buy; 13-24=Sell"""
        self.action_map = {0: ("hold", 0, 0)}
        idx = 1
        for tp_sl in self.TP_SL_RATIO:
            for sl in self.SL_LEVELS:
                self.action_map[idx] = ("buy", tp_sl, sl)
                idx += 1
                self.action_map[idx] = ("sell", tp_sl, sl)
                idx += 1
        assert len(self.action_map) == 2 * len(self.TP_SL_RATIO) * len(self.SL_LEVELS) + 1

    def _get_obs(self):
        # TODO: current price
        idx = self.episode_indices[self.current_step]
        window_data = self.df.iloc[idx - self.window:idx]

        if self.normalize:
            window_data[["open", "close", "high", "low"]] -= window_data[["open", "close", "high", "low"]].iloc[0]

        # "open", "close", "high", "low", "volume", "rsi", "open", "close", "high", "low", "volume", "rsi", ...
        # obs = window_data[["open", "close", "high", "low", "volume", "rsi"]].values.flatten().astype(np.float32)
        obs = window_data[self.features].values.flatten().astype(np.float32)
        # obs = obs.reshape(self.window, len(self.features))
        return obs

    def reset(self, seed=None, options=None):
        self.episode_stats = {"tp": 0, "sl": 0, "timeout": 0, "holds": 0, "undefined": 0, "pl": 0, "wr": 0}
        super().reset(seed=seed)

        if len(self.episode_indices) > 0 and self.episode_indices[-1] == len(self.df) - 1:
            self.global_step = 0

        if self.mode == "train":
            if self.random_indices:
                self.episode_indices = random.sample(range(self.window, len(self.df)), self.episode_steps)
            else:
                start_idx = self.window + self.global_step
                end_idx = min(start_idx + self.episode_steps, len(self.df))
                self.episode_indices = range(start_idx, end_idx)
                print("{}, start idx: {}, end idx: {}".format(self.mode, min(self.episode_indices),
                                                              max(self.episode_indices)))
        else:
            self.episode_indices = range(self.window, len(self.df))
            print("{}, start idx: {}, end idx: {}".format(self.mode, min(self.episode_indices),
                                                          max(self.episode_indices)))

        self.current_step = 0
        self.episode_count += 1
        self.equity = 10_000.0
        self.max_equity = 10_000.0
        self.max_drawdown = 1
        return self._get_obs(), {"episode_stats": self.episode_stats}

    def step(self, action):
        direction, tp_sl, sl = self.action_map[action]
        idx = self.episode_indices[self.current_step]
        entry_price = self.df.loc[idx, "open"]

        reward = 0.0

        if direction != "hold":
            # tp_price = entry_price * (1 + tp_pct) if direction == "buy" else entry_price * (1 - tp_pct)
            # sl_price = entry_price * (1 - sl_pct) if direction == "buy" else entry_price * (1 + sl_pct)

            tp = sl * tp_sl
            tp_price = entry_price + tp if direction == "buy" else entry_price - tp
            sl_price = entry_price - sl if direction == "buy" else entry_price + sl

            # end_idx = min(self.current_step + self.fwd_window, len(self.df) - 1)
            fwd = self.df.iloc[idx + 1:idx + self.fwd_window]
            hit_tp = hit_sl = undefined = False
            for _, candle in fwd.iterrows():
                if direction == "buy":
                    if candle["high"] >= tp_price and candle["low"] <= sl_price:
                        undefined = True
                        break
                    if candle["low"] <= sl_price:
                        hit_sl = True
                        break
                    if candle["high"] >= tp_price:
                        hit_tp = True
                        break
                else:  # sell
                    if candle["high"] >= sl_price and candle["low"] <= tp_price:
                        undefined = True
                        break
                    if candle["high"] >= sl_price:
                        hit_sl = True
                        break
                    if candle["low"] <= tp_price:
                        hit_tp = True
                        break

            if hit_tp:
                pnl = tp
                self.equity += tp
                self.episode_stats["pl"] += tp
                self.episode_stats["tp"] += 1
            elif hit_sl:
                pnl = -sl
                self.equity -= sl
                self.episode_stats["pl"] -= sl
                self.episode_stats["sl"] += 1
            elif undefined:
                pnl = 0
                self.episode_stats["undefined"] += 1
            else:
                pnl = 0
                self.episode_stats["timeout"] += 1

            self.max_equity = max(self.max_equity, self.equity)
            drawdown = self.max_equity - self.equity
            self.max_drawdown = max(drawdown, self.max_drawdown)
            drawdown = (drawdown / self.max_drawdown) * self.max_tp * 0.2
            reward = pnl  # - drawdown
            # print(pnl, -drawdown, reward)
            # breakpoint()

            # print(fwd)
            # print("tp: {}, sl: {}".format(tp, sl))
            # print("{}, entry_price: {}, tp_price: {}, sl_price: {}".format(direction, entry_price, tp_price,
            #                                                                sl_price))
            # input()
        else:
            self.episode_stats["holds"] += 1

        self.current_step += 1
        self.global_step += 1
        truncated = self.current_step >= len(self.episode_indices)
        terminated = truncated

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        total_trades = self.episode_stats["tp"] + self.episode_stats["sl"] + self.episode_stats["timeout"] + \
                       self.episode_stats["undefined"]
        self.episode_stats["wr"] = self.episode_stats["tp"] / max(total_trades, 1)
        info = dict()
        info["episode_stats"] = self.episode_stats.copy()
        info["date"] = self.df.loc[self.current_step, "time"]
        info["episode_count"] = self.episode_count

        if terminated and self.mode == "val":
            print(self.mode, self.episode_stats)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(self.episode_stats) + "\n")

        return obs, reward, terminated, truncated, info
