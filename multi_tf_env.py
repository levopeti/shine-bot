import json
import random
import pandas as pd
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from config import Config
from pprint import pprint
from time import time


class MultyTFTradingEnv(gym.Env):
    """
    Observation: 12 M5 gyertya (open,close,high,low,volume,rsi) + aktuális open
    Action:      0=Hold, 1..9=Buy+TP/SL, 10..18=Sell+TP/SL
    Reward:      TP hit → +10, SL hit → -10, timeout → -1
    """
    metadata = {"render_modes": []}

    def __init__(self, df_dict, mode="train", save_log=True):
        super().__init__()
        n_macro = df_dict["macro"].shape[1] + 3  # +time (4) - "time" column (1)
        self.df_dict = df_dict
        self.log_file = Config.EVAL_LOG_FILE
        self.window_dict = Config.WINDOW_DICT
        self.fwd_window = Config.FWD_WINDOW
        self.episode_steps = Config.TRAIN_EPISODE_STEPS
        self.episode_indices = list()
        self.current_step = 0
        self.global_step = 0
        self.episode_count = 0
        # self.normalize = Config.NORMALIZE
        self.random_indices = Config.RANDOM_INDICES
        self.mode = mode
        self.features = Config.FEATURES
        self.save_log = save_log

        self.equity = 10_000.0
        self.max_equity = 10_000.0
        self.max_drawdown = 1

        self.TP_SL_RATIO = Config.TP_SL_RATIO
        self.SL_LEVELS = Config.SL_LEVELS
        self.max_tp = max(self.SL_LEVELS) * max(self.TP_SL_RATIO)

        if Config.CONT_ACTION:
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        else:
            n_combos = len(self.TP_SL_RATIO) * len(self.SL_LEVELS)
            self.action_space = spaces.Discrete(1 + 2 * n_combos)
            self._build_action_map()

        self.m5_obs_dim = self.window_dict["m5m"] * len(self.features)
        self.observation_space = spaces.Dict({
            timeframe: spaces.Box(low=-3, high=3,
                                  shape=(self.window_dict[timeframe] * len(self.features),), dtype=np.float32)
            for timeframe in self.window_dict.keys()
        })
        self.observation_space["macro"] = gym.spaces.Box(-np.inf, np.inf, shape=(n_macro,), dtype=np.float32)
        self.episode_stats = dict()
        self.episode_indices_list = list()
        if self.mode == "val":
            self.calculate_episode_indices()

    def calculate_episode_indices(self):
        episode_indices_full = range(self.window_dict["m5m"] + 1200, len(self.df_dict["m5m"]))
        for episode_count in range(Config.NUM_EVAL_EPISODES):
            episode_indices = episode_indices_full[self.episode_steps * episode_count:self.episode_steps * (episode_count + 1)]
            assert len(episode_indices) > 0, episode_count
            self.episode_indices_list.append(episode_indices)


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

    def _get_m_obs(self, timeframe, last_ts):
        df = self.df_dict[timeframe][self.df_dict[timeframe]["time"] <= last_ts]  # .reset_index(drop=True)
        window_data = df.iloc[-self.window_dict[timeframe]:]
        obs = window_data[self.features].values.flatten().astype(np.float32)
        return obs

    def _get_macro_obs(self, ts: pd.Timestamp) -> np.ndarray:
        # Macro feature-ök a DataFrame-ből
        idx = self.df_dict["macro"]["time"].searchsorted(ts, side="right") - 1
        idx = max(0, min(idx, len(self.df_dict["macro"]) - 1))
        macro_vals = self.df_dict["macro"].drop(columns="time").iloc[idx].values.astype(np.float32)

        # Sinusos időkódolás az aktuális timestamp-ből
        minutes = ts.hour * 60 + ts.minute
        day_of_week = ts.dayofweek  # 0=hétfő, 4=péntek

        time_sin = np.sin(2 * np.pi * minutes / 1440).astype(np.float32)
        time_cos = np.cos(2 * np.pi * minutes / 1440).astype(np.float32)
        day_sin = np.sin(2 * np.pi * day_of_week / 5).astype(np.float32)
        day_cos = np.cos(2 * np.pi * day_of_week / 5).astype(np.float32)

        time_enc = np.array([time_sin, time_cos, day_sin, day_cos], dtype=np.float32)
        return np.concatenate([macro_vals, time_enc])  # shape: (n_macro + 4,)

    def _get_obs(self):
        # TODO: current price
        try:
          idx = self.episode_indices[self.current_step]
        except IndexError:
            breakpoint()
        window_data = self.df_dict["m5m"].iloc[idx - self.window_dict["m5m"]:idx]

        # if self.normalize:
        #     window_data[["open", "close", "high", "low"]] -= window_data[["open", "close", "high", "low"]].iloc[0]

        # "open", "close", "high", "low", "volume", "rsi", "open", "close", "high", "low", "volume", "rsi", ...
        # obs = window_data[["open", "close", "high", "low", "volume", "rsi"]].values.flatten().astype(np.float32)
        obs = window_data[self.features].values.flatten().astype(np.float32)
        # obs = obs.reshape(self.window, len(self.features))
        last_ts = window_data["time"].iloc[-1]

        obs = {
            "m5m": obs,
            "m1h": self._get_m_obs("m1h", last_ts),
            "m4h": self._get_m_obs("m4h", last_ts),
            "macro": self._get_macro_obs(last_ts),
        }
        return obs

    def reset(self, seed=None, options=None):
        if self.mode == "train" or (self.mode == "val" and  self.episode_count == 0):
            self.episode_stats = {"buy": 0, "sell": 0, "holds": 0, "tp": 0, "sl": 0, "timeout": 0, "undefined": 0,
                                  "pl": 0, "wr": 0}
        super().reset(seed=seed)

        if len(self.episode_indices) > 0 and self.episode_indices[-1] == len(self.df_dict["m5m"]) - 1:
            self.global_step = 0

        if self.mode == "train":
            if self.random_indices:
                self.episode_indices = random.sample(range(self.window_dict["m5m"], len(self.df_dict["m5m"])),
                                                     self.episode_steps)
            else:
                # start_idx = self.window + self.global_step
                start_idx = random.randint(self.window_dict["m5m"] + 1200,  # 1000 for sure
                                           len(self.df_dict["m5m"]) - self.episode_steps - self.window_dict["m5m"])
                end_idx = min(start_idx + self.episode_steps, len(self.df_dict["m5m"]))
                self.episode_indices = range(start_idx, end_idx)
                # print("{}, start idx: {}, end idx: {}".format(self.mode, min(self.episode_indices),
                #                                               max(self.episode_indices)))
            self.episode_count += 1
        else:
            if self.episode_count >= len(self.episode_indices_list):
                self.episode_count = 0
            else:
                self.episode_indices = self.episode_indices_list[self.episode_count]
                self.episode_count += 1
            # print("{}, {}, start idx: {}, end idx: {}".format(self.mode, self.episode_count, min(self.episode_indices),
            #                                               max(self.episode_indices)))

        self.current_step = 0

        self.equity = 10_000.0
        self.max_equity = 10_000.0
        self.max_drawdown = 1
        return self._get_obs(), {"episode_stats": self.episode_stats}

    def step(self, action):
        if Config.CONT_ACTION:
            action = action[0]
            if action <= -0.1:
                direction = "sell"
            elif action >= 0.1:
                direction = "buy"
            else:
                direction = "hold"
            tp_sl = abs(action) * 10
        else:
            direction, tp_sl, sl = self.action_map[action]

        idx = self.episode_indices[self.current_step]
        entry_price = self.df_dict["m5m"].loc[idx, "open"]
        atr = self.df_dict["m5m"].loc[idx, "atr"]

        if np.isnan(atr) or atr < 1e-8:
            direction = "hold"

        reward = Config.HOLD_MLT * atr

        if direction != "hold":
            # tp_price = entry_price * (1 + tp_pct) if direction == "buy" else entry_price * (1 - tp_pct)
            # sl_price = entry_price * (1 - sl_pct) if direction == "buy" else entry_price * (1 + sl_pct)

            # tp = sl * tp_sl
            tp = atr * tp_sl
            sl = atr * 1.0
            tp_price = entry_price + tp if direction == "buy" else entry_price - tp
            sl_price = entry_price - sl if direction == "buy" else entry_price + sl

            # end_idx = min(self.current_step + self.fwd_window, len(self.df) - 1)
            fwd = self.df_dict["m5m"].iloc[idx + 1:idx + self.fwd_window]
            hit_tp = hit_sl = undefined = False

            if direction == "buy":
                for _, candle in fwd.iterrows():
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
                for _, candle in fwd.iterrows():
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
                pnl = tp_sl  # pl. +1.5 (TP/SL arány)
                self.equity += tp
                self.episode_stats["pl"] += float(tp)
                self.episode_stats["tp"] += 1
                if direction == "buy":
                    self.episode_stats["buy"] += 1
                elif direction == "sell":
                    self.episode_stats["sell"] += 1
            elif hit_sl:
                pnl = -1
                self.equity -= sl
                self.episode_stats["pl"] -= float(sl)
                self.episode_stats["sl"] += 1
                if direction == "buy":
                    self.episode_stats["buy"] += 1
                elif direction == "sell":
                    self.episode_stats["sell"] += 1
            elif undefined:
                pnl = 0
                self.episode_stats["undefined"] += 1
            else:
                pnl = -0.1
                self.episode_stats["timeout"] += 1

            self.max_equity = max(self.max_equity, self.equity)
            drawdown = (self.max_equity - self.equity) / max(self.max_equity, 1e-8)
            self.max_drawdown = max(drawdown, self.max_drawdown)
            # drawdown = (drawdown / self.max_drawdown) * self.max_tp * 0.2
            reward = pnl - drawdown * 0.1
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

        obs = self._get_obs() if not terminated else np.zeros(self.m5_obs_dim, dtype=np.float32)
        total_trades = self.episode_stats["tp"] + self.episode_stats["sl"] + self.episode_stats["timeout"] + \
                       self.episode_stats["undefined"]
        self.episode_stats["wr"] = self.episode_stats["tp"] / max(total_trades, 1)
        info = dict()
        info["episode_stats"] = self.episode_stats.copy()
        info["date"] = self.df_dict["m5m"].loc[self.current_step, "time"]
        info["episode_count"] = self.episode_count

        if terminated and self.mode == "val" and self.episode_count >= len(self.episode_indices_list):
            self.global_step = 0
            if self.save_log:
                print(self.mode, self.episode_stats)
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(self.episode_stats) + "\n")

        return obs, reward, terminated, truncated, info
