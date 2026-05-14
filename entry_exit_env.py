import json
import random
import pandas as pd
import gymnasium as gym
import numpy as np

from gymnasium import spaces
from config import Config
from pprint import pprint
from time import time

from multi_tf_env import MultyTFTradingEnv


class EntryExitEnv(MultyTFTradingEnv):
    def __init__(self, df_dict, mode="train"):
        super().__init__(df_dict, mode)
        self.position = None
        self.action_space = spaces.Discrete(4)  # 0=buy, 1=sell, 2=hold, 3=close
        self.action_map = {0: "buy", 1: "sell", 2: "hold", 3: "close"}

        n_macro = df_dict["macro"].shape[1] + 4 - 1 + 4  # +time (4) - "time" column (1) + position features (4)
        self.observation_space["macro"] = gym.spaces.Box(-np.inf, np.inf, shape=(n_macro,), dtype=np.float32)

    def _get_position_features(self) -> np.ndarray:
        if self.position is None:
            return np.array([
                0.0,  # is_in_trade: 0 = flat
                0.0,  # direction: 0 = nincs
                0.0,  # unrealized_pnl (ATR-ban normálva)
                0.0,  # steps_held (normálva)
            ], dtype=np.float32)
        else:
            pos = self.position
            idx = self.episode_indices[self.current_step]
            current_price = self.df_dict["m5m"].loc[idx, "close"]
            atr = self.df_dict["m5m"].loc[idx, "atr"]

            if pos["direction"] == "buy":
                unrealized = current_price - pos["entry"]
            else:
                unrealized = pos["entry"] - current_price

            unrealized_norm = unrealized / max(atr, 1e-8)  # ATR-egységben
            steps_held = self.current_step - pos["entry_step"]
            steps_norm = steps_held / self.fwd_window  # 0..1 között

            return np.array([
                1.0,  # is_in_trade: 1 = nyitott pozíció
                1.0 if pos["direction"] == "buy" else -1.0,  # irány
                np.clip(unrealized_norm, -5.0, 5.0),  # PnL ATR-ban
                steps_norm,  # meddig tart már
            ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.position = None
        # position = {
        #   'direction': 'buy'/'sell',
        #   'entry': float,
        #   'sl_price': float,
        #   'sl_size': float,       # abs(entry - sl_price)
        #   'entry_step': int,
        #   'prev_unrealized': 0.0  # előző lépés unrealized PnL-je
        # }
        return super().reset(seed, options)

    def action_masks(self) -> np.ndarray:
        if self.position is None:
            return np.array([True, True, True, False])
        else:
            return np.array([False, False, True, True])

    def _get_obs(self):
        obs = super()._get_obs()
        obs["macro"] = np.concatenate([obs["macro"], self._get_position_features()])
        return obs

    def step(self, action):
        action = self.action_map[action]
        idx = self.episode_indices[self.current_step]
        row = self.df_dict["m5m"].loc[idx]
        entry_price = row["open"]
        atr = row["atr"]

        if np.isnan(atr) or atr < 1e-8:
            action = "hold"

        # ── FLAT állapot ──────────────────────────────────────────
        if self.position is None:
            if action in ("buy", "sell"):
                direction = "buy" if action == "buy" else "sell"
                sl_size = atr * Config.SL_MLT
                sl_price = (entry_price - sl_size if direction == "buy"
                            else entry_price + sl_size)
                self.position = {
                    "direction": direction,
                    "entry": entry_price,
                    "sl_price": sl_price,
                    "sl_size": sl_size,
                    "entry_step": self.current_step,
                    "prev_unrealized": 0.0,
                }
                reward = 0.0  # nyitáskor semleges
                if direction == "buy":
                    self.episode_stats["buy"] += 1
                elif direction == "sell":
                    self.episode_stats["sell"] += 1
            else:  # hold flat
                reward = Config.HOLD_MLT * atr  # kis büntetés tétlenségért
                self.episode_stats["holds"] += 1

        # ── IN_TRADE állapot ──────────────────────────────────────
        else:
            pos = self.position
            current_price = row["close"]

            # SL automatikus ellenőrzés (elsőbbsége van az action előtt)
            sl_hit = (pos["direction"] == "buy" and row["low"] <= pos["sl_price"]) or \
                     (pos["direction"] == "sell" and row["high"] >= pos["sl_price"])

            if sl_hit:
                steps_held = self.current_step - pos["entry_step"]
                time_penalty = steps_held / self.fwd_window * Config.TIME_PENALTY_MLT  # kis idő-diszkont
                reward = -1.0 - time_penalty  # SL is büntetve van az eltöltött időért
                self.equity -= pos["sl_size"]
                self.episode_stats["pl"] -= pos["sl_size"]
                # self._update_stats("sl", pos["direction"])
                self.position = None
                self.episode_stats["sl"] += 1

            elif action == "close":
                # Ágens döntött a zárásról
                if pos["direction"] == "buy":
                    pnl = current_price - pos["entry"]
                else:
                    pnl = pos["entry"] - current_price

                pnl_norm = pnl / pos["sl_size"]  # pl. +2.5 vagy -0.3
                steps_held = self.current_step - pos["entry_step"]
                time_penalty = steps_held / self.fwd_window * Config.TIME_PENALTY_MLT  # kis idő-diszkont

                reward = pnl_norm - time_penalty
                self.equity += pnl
                self.episode_stats["pl"] += pnl
                # self._update_stats("close", pos["direction"])
                self.position = None
                if pnl >= 0:
                    self.episode_stats["tp"] += 1
                else:
                    self.episode_stats["sl"] += 1

            else:  # ACTION_HOLD – pozíció él tovább
                # Step-wise unrealized PnL delta (potenciál-alapú shaping)
                if pos["direction"] == "buy":
                    unrealized = current_price - pos["entry"]
                else:
                    unrealized = pos["entry"] - current_price

                delta = (unrealized - pos["prev_unrealized"]) / pos["sl_size"]
                pos["prev_unrealized"] = unrealized

                # Drawdown büntetés: ha unrealized negatív
                unrealized_norm = unrealized / pos["sl_size"]  # -1.0 = SL-en
                if unrealized_norm < 0:
                    drawdown_penalty = (unrealized_norm ** 2) * Config.DD_PENALTY_MLT * -1
                else:
                    drawdown_penalty = 0.0

                reward = delta + drawdown_penalty

        # ── Drawdown követés az equity-n ──────────────────────────
        self.max_equity = max(self.max_equity, self.equity)
        if self.position is None:  # csak lezárt trade után
            dd = (self.max_equity - self.equity) / max(self.max_equity, 1e-8)
            self.max_drawdown = max(dd, self.max_drawdown)
            # Plusz globális DD büntetés, ha nagy a drawdown
            reward -= dd * Config.DD_MLT

        self.current_step += 1
        self.global_step += 1
        truncated = self.current_step >= len(self.episode_indices)
        terminated = truncated

        obs = self._get_obs() if not terminated else None
        total_trades = self.episode_stats["tp"] + self.episode_stats["sl"] + self.episode_stats["timeout"] + \
                       self.episode_stats["undefined"]
        self.episode_stats["wr"] = self.episode_stats["tp"] / max(total_trades, 1)
        info = dict()
        info["episode_stats"] = self.episode_stats.copy()
        info["date"] = self.df_dict["m5m"].loc[self.current_step, "time"]
        info["episode_count"] = self.episode_count

        if terminated and self.mode == "val" and self.episode_count >= len(self.episode_indices_list):
            self.global_step = 0
            print(self.mode, self.episode_stats)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(self.episode_stats) + "\n")

        return obs, reward, terminated, truncated, info
