from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from typing import Callable
from config import Config
from inception_time import InceptionTime
from inception_time2d import InceptionTime2D
from only_gold_env import M5TradingEnv
from synthetic_data import generate_synthetic_candles
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total, desc="Train")

    def _on_step(self) -> bool:
        postfix = dict()
        # if hasattr(self.logger, "name") and self.logger.name:
        # postfix.update({
        #     # "eps": f"{self.training_env.get_attr('epsilon')[0]:.3f}" if hasattr(self.training_env, 'get_attr') else "?",
        #     "loss": f"{self.locals.get('loss', '?')}",
        #     "mean_r": f"{self.locals.get('episode_reward', '?')}"
        # })

        # if self.locals.get("infos", [{}])[0].get("is_success") or self.locals.get("dones", [False])[0]:
        # self.episode_count += 1
        stats = self.locals["infos"][0]["episode_stats"]
        episode_count = self.locals["infos"][0]["episode_count"]

        postfix.update({
            "ep": episode_count,
            "BUY": f"{stats['buy']}",
            "SELL": f"{stats['sell']}",
            "H": f"{stats['holds']}",
            "TP": f"{stats['tp']}",
            "SL": f"{stats['sl']}",
            "TO": f"{stats['timeout']}",
            "UD": f"{stats['undefined']}",
            "PL": f"{stats['pl']:.0f}",
            "WR": f"{stats['wr'] * 100:.0f}%"
        })

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class SyncNormalizeEvalCallback(EvalCallback):
    """
    EvalCallback, ami minden eval előtt szinkronizálja a VecNormalize
    statisztikákat a train env-ből az eval env-be.
    """

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Train → Eval statisztika szinkron
            sync_envs_normalization(self.training_env, self.eval_env)
            # Eval módba kapcsolás
            self.eval_env.training = False
            self.eval_env.norm_reward = False
        return super()._on_step()


def cosine_schedule(initial_value: float, final_value: float = 1e-5) -> Callable:
    def func(progress_remaining: float) -> float:
        # Koszinusz görbe: lassan indul, gyorsan csökken középen, lassan végez
        cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
        return final_value + cosine_factor * (initial_value - final_value)

    return func


def add_normalized_features(
        df: pd.DataFrame,
        window: int = 20,
        clip: float = 3.0
) -> pd.DataFrame:
    """
    Előre kiszámolja az összes normalizált OHLCV feature-t és új oszlopokban tárolja.

    Paraméterek
    -----------
    df     : DataFrame 'open', 'high', 'low', 'close', 'volume' oszlopokkal
    window : Rolling z-score ablak mérete gyertyában (alapértelmezett: 20)
    clip   : Kiugró értékek levágása ± ennyi szórásra (alapértelmezett: 3.0)

    Visszatérési érték
    ------------------
    DataFrame az eredeti oszlopokkal + az alábbi új feature oszlopokkal:
        feat_close_ret    – záróár változás z-score (momentum)
        feat_body         – gyertya test iránya és mérete z-score
        feat_upper_wick   – felső kanóc relatív mérete z-score
        feat_lower_wick   – alsó kanóc relatív mérete z-score
        feat_volume       – volumen momentum z-score
        feat_hl_range     – gyertya méret (high-low) z-score
        feat_gap          – nyitógap az előző záráshoz képest z-score
    """
    out = df.copy()

    # --- Nyers relatív értékek ---
    close_ret = out['close'].pct_change()
    body = out['close'] / out['open'] - 1
    upper_wick = out['high'] / out[['open', 'close']].max(axis=1) - 1
    lower_wick = out[['open', 'close']].min(axis=1) / out['low'] - 1
    vol_log_ret = np.log(out['volume'] / out['volume'].shift(1))
    hl_range = (out['high'] - out['low']) / out['close']
    gap = (out['open'] / out['close'].shift(1) - 1)
    atr = (out['high'] - out['low']).rolling(14).mean()

    # --- Rolling z-score segédfüggvény ---
    def rolling_zscore(series: pd.Series, w: int) -> pd.Series:
        mean = series.rolling(w, min_periods=w).mean()
        std = series.rolling(w, min_periods=w).std()
        return (series - mean) / (std + 1e-8)

    # --- Feature oszlopok számítása és cliplelése ---
    out['feat_close_ret'] = rolling_zscore(close_ret, window).clip(-clip, clip)
    out['feat_body'] = rolling_zscore(body, window).clip(-clip, clip)
    out['feat_upper_wick'] = rolling_zscore(upper_wick, window).clip(-clip, clip)
    out['feat_lower_wick'] = rolling_zscore(lower_wick, window).clip(-clip, clip)
    out['feat_volume'] = rolling_zscore(vol_log_ret, window).clip(-clip, clip)
    out['feat_hl_range'] = rolling_zscore(hl_range, window).clip(-clip, clip)
    out['feat_gap'] = rolling_zscore(gap, window).clip(-clip, clip)
    out['feat_atr_ratio'] = atr / out['close']  # skálafüggetlen

    # Az első 'window' sor NaN lesz (warmup periódus) — droppolható
    out.dropna(subset=[c for c in out.columns if c.startswith('feat_')], inplace=True)
    out.reset_index(drop=True, inplace=True)

    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,  # gyertya terjedelme
        (high - prev_close).abs(),  # gap felfelé
        (low - prev_close).abs(),  # gap lefelé
    ], axis=1).max(axis=1)

    # Wilder-féle simított mozgóátlag (RMA)
    df['atr'] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


if __name__ == "__main__":
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    features = [
        'feat_close_ret',
        'feat_body',
        'feat_upper_wick',
        'feat_lower_wick',
        'feat_volume',
        'feat_hl_range',
        'feat_gap',
        'feat_atr_ratio',
    ]

    Config.FEATURES = features
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.save_json(Config.MODEL_DIR / "config.json")
    pprint(Config.to_dict())

    train_df = generate_synthetic_candles(n_candles=500000, amplitude_range=(5.0, 50.0), random_seed=6)
    train_df = add_normalized_features(train_df, window=20, clip=3.0)
    print(train_df[features].describe().round(3))

    test_df = generate_synthetic_candles(n_candles=10000, amplitude_range=(5.0, 50.0), random_seed=8)
    test_df = add_normalized_features(test_df, window=20, clip=3.0)

    # train_env = M5TradingEnv(train_df, features)
    # obs = train_env.reset()
    # total_reward = 0
    # for _ in range(1000):
    #     action = train_env.action_space.sample()  # teljesen véletlenszerű
    #     obs, reward, terminated, truncated, info = train_env.step(action)
    #     total_reward += reward
    #     if terminated:
    #         break
    # print(f"Random policy total reward: {total_reward:.4f}")
    # exit()

    vec_env = make_vec_env(lambda: M5TradingEnv(train_df, features), n_envs=Config.N_ENVS)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,  # observation normalizálás
        norm_reward=True,  # reward normalizálás — EZ A KULCS
        clip_reward=10.0,  # ne legyenek extrém reward értékek
        gamma=0.99
    )

    test_env = make_vec_env(
        lambda: M5TradingEnv(test_df, features, mode="val"), n_envs=1
    )
    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,  # ← eval-nál MINDIG False — igazi reward-ot akarunk
        clip_reward=10.0,
        gamma=0.99,
        training=False,  # ← statisztika NE frissüljön az eval env-ben
    )

    eval_callback = SyncNormalizeEvalCallback(
        test_env,
        eval_freq=Config.EVAL_FREQ,
        n_eval_episodes=1,
        deterministic=True,
        log_path=Config.MODEL_DIR / "eval_log",
        best_model_save_path=Config.MODEL_DIR / "best_model",
        verbose=1,
    )

    inception_policy_kwargs = dict(
        features_extractor_class=InceptionTime,
        features_extractor_kwargs=dict(features_dim=Config.FEATURE_DIM, in_channels=len(features),
                                       window_h=Config.WINDOW_H, kernel_sizes=Config.KERNEL_SIZES,
                                       n_filters=Config.N_FILTERS,
                                       bottleneck_channels=Config.BOTTLENECK_CHANNELS),
        net_arch=Config.NET_ARCH  # MLP
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # --- Tanulás ---
        learning_rate=cosine_schedule(Config.START_LR, Config.END_LR),  # kezdeti LR, lineárisan csökkenjen
        n_steps=Config.N_STEPS,  # rollout buffer mérete (env-enként)
        batch_size=Config.BATCH_SIZE,  # mini-batch a frissítéshez
        n_epochs=Config.N_EPOCHS,  # hányszor iterál a bufferen, 10
        gamma=Config.GAMMA,  # diszkont faktor
        gae_lambda=Config.GEA_LAMBDA,  # GAE lambda
        # --- Stabilitás ---
        clip_range=Config.CLIP_RANGE,  # PPO clip 0.02
        ent_coef=Config.PPO_ENT_COEF,  # entrópia bónusz — NE állítsd 0-ra!
        vf_coef=Config.VF_COEF,
        max_grad_norm=Config.MAX_GRAD_NORM,
        normalize_advantage=True,
        # --- Hálózat ---
        policy_kwargs=inception_policy_kwargs,
        verbose=1,
        target_kl=Config.TARGET_KL,  # ha kl > 0.02, leállítja a frissítést az epochon belül
    )
    model.learn(total_timesteps=Config.TOTAL_STEPS * Config.N_ENVS,
                callback=CallbackList([ProgressCallback(Config.TOTAL_STEPS), eval_callback]))
