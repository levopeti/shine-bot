import gymnasium as gym
import numpy as np
from gymnasium import spaces


class M5TradingEnv(gym.Env):
    """
    Observation: 12 M5 gyertya (open,close,high,low,volume,rsi) + aktuális open
    Action:      0=Hold, 1..9=Buy+TP/SL, 10..18=Sell+TP/SL
    Reward:      TP hit → +10, SL hit → -10, timeout → -1
    """
    metadata = {"render_modes": []}

    TP_LEVELS = [0.003, 0.006, 0.010]  # 0.3%, 0.6%, 1.0%
    SL_LEVELS = [0.001, 0.003, 0.006]  # 0.1%, 0.3%, 0.6%

    def __init__(self, df, window_h=12, fwd_window=1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_h * 12  # 12 × M5 = 1 óra lookback
        self.fwd_window = fwd_window * 12  # következő 1 óra = 12 × M5

        n_combos = len(self.TP_LEVELS) * len(self.SL_LEVELS)  # 9
        self.action_space = spaces.Discrete(1 + 2 * n_combos)  # 19

        self.obs_dim = self.window * 6 + 1  # 72 + 1 = 73
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self._build_action_map()

        self.episode_stats = {"tp": 0, "sl": 0, "timeout": 0, "holds": 0}

    def _build_action_map(self):
        """0=Hold; 1-9=Buy; 10-18=Sell"""
        self.action_map = {0: ("hold", 0, 0)}
        idx = 1
        for tp in self.TP_LEVELS:
            for sl in self.SL_LEVELS:
                self.action_map[idx] = ("buy", tp, sl)
                self.action_map[idx + 9] = ("sell", tp, sl)
                idx += 1

    def _get_obs(self):
        window_data = self.df.iloc[self.current_step - self.window: self.current_step]
        # "open", "close", "high", "low", "volume", "rsi", "open", "close", "high", "low", "volume", "rsi", ...
        features = window_data[["open", "close", "high", "low", "volume", "rsi"]].values.flatten()
        current_open = np.array([self.df.loc[self.current_step, "open"]])
        obs = np.concatenate([features, current_open]).astype(np.float32)
        # Egyszerű normalizálás: price feature-öket osszuk az első open-nel
        # ref = features[0] if features[0] != 0 else 1.0
        # obs[:self.window * 6] /= ref
        # obs[-1] /= ref
        return obs

    def reset(self, seed=None, options=None):
        self.episode_stats = {"tp": 0, "sl": 0, "timeout": 0, "holds": 0}
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_obs(), {"episode_stats": self.episode_stats}

    def step(self, action):
        direction, tp_pct, sl_pct = self.action_map[action]
        entry_price = self.df.loc[self.current_step, "open"]

        reward = 0.0

        if direction != "hold":
            tp_price = entry_price * (1 + tp_pct) if direction == "buy" else entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 - sl_pct) if direction == "buy" else entry_price * (1 + sl_pct)

            end_idx = min(self.current_step + self.fwd_window, len(self.df) - 1)
            fwd = self.df.iloc[self.current_step + 1: end_idx + 1]

            hit_tp = hit_sl = False
            for _, candle in fwd.iterrows():
                if direction == "buy":
                    if candle["low"] <= sl_price:
                        hit_sl = True
                        break
                    if candle["high"] >= tp_price:
                        hit_tp = True
                        break
                else:  # sell
                    if candle["high"] >= sl_price:
                        hit_sl = True
                        break
                    if candle["low"] <= tp_price:
                        hit_tp = True
                        break

            if hit_tp:
                reward = 10_000 * tp_pct
                self.episode_stats["tp"] += 1
            elif hit_sl:
                reward = -10_000 * sl_pct
                self.episode_stats["sl"] += 1
            else:
                reward = -3.0  # timeout
                self.episode_stats["timeout"] += 1
        else:
            self.episode_stats["holds"] += 1

        self.current_step += 1
        truncated = self.current_step >= len(self.df) - self.fwd_window - 1
        terminated = truncated

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = dict()
        info["episode_stats"] = self.episode_stats.copy()
        info["date"] = self.df.loc[self.current_step, "time"]
        return obs, reward, terminated, truncated, info
