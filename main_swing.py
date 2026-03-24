import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

import warnings
warnings.filterwarnings("ignore")


# ── Environment ──────────────────────────────────────────────────────────────
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
        features = window_data[["open", "close", "high", "low", "volume", "rsi"]].values.flatten()
        current_open = np.array([self.df.loc[self.current_step, "open"]])
        obs = np.concatenate([features, current_open]).astype(np.float32)
        # Egyszerű normalizálás: price feature-öket osszuk az első open-nel
        ref = features[0] if features[0] != 0 else 1.0
        obs[:self.window * 6] /= ref
        obs[-1] /= ref
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_obs(), {}

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
                reward = +10.0
            elif hit_sl:
                reward = -10.0
            else:
                reward = -1.0  # timeout büntetés

        self.current_step += 1
        truncated = self.current_step >= len(self.df) - self.fwd_window - 1
        terminated = truncated

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, {}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_gold_m5(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=";",  # tab elválasztó
        parse_dates=["Date"],
        dayfirst=False
    )

    # Oszlopnevek kisbetűsítése
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "time"}, inplace=True)

    # Csak a szükséges oszlopok
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # RSI hozzáadása
    df["rsi"] = compute_rsi(df["close"])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    df = load_gold_m5("/content/drive/MyDrive/XAU_5m_data.csv")

    # Leválasztás év alapján
    test_start_year = 2022

    train_df = df[df["time"].dt.year < test_start_year].reset_index(drop=True)
    test_df = df[df["time"].dt.year >= test_start_year].reset_index(drop=True)

    print(f"Train: {train_df['time'].iloc[0].year} – {train_df['time'].iloc[-1].year} | {len(train_df):,} gyertya")
    print(f"Test:  {test_df['time'].iloc[0].year} – {test_df['time'].iloc[-1].year} | {len(test_df):,} gyertya")

    # Env létrehozás
    from stable_baselines3.common.monitor import Monitor

    # Monitor wrap-be kell tenni az env-et, hogy loggoljon
    train_env = Monitor(M5TradingEnv(train_df))
    test_env = Monitor(M5TradingEnv(test_df))
    # train_env = M5TradingEnv(train_df)
    # test_env = M5TradingEnv(test_df)

    eval_callback = EvalCallback(
        test_env,
        eval_freq=500_000,  # minden 200k lépésnél kiértékel
        n_eval_episodes=1,
        verbose=1,
        render=True
    )

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # MLP rétegek
        tensorboard_log="./tensorboard_logs/"
    )
    print("learning")
    model.learn(total_timesteps=1_000_000,
                # callback=eval_callback,
                progress_bar=True)
    model.save("/content/drive/MyDrive/m5_dqn_trader_1m")
