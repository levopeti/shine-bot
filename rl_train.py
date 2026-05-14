from pprint import pprint
import warnings

from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize

from conv1d import Conv1DFeaturesExtractor
from data_utils import create_features, load_gold
from inception_time import InceptionTime
from inception_time2d import InceptionTime2D
from only_gold_env import M5TradingEnv, ConfidentEnv
from config import Config
from synthetic_data import generate_synthetic_candles
from rl_debug import SyncNormalizeEvalCallback, cosine_schedule, add_normalized_features, ProgressCallback, add_atr

warnings.filterwarnings("ignore")



if __name__ == "__main__":
    # train_df = generate_synthetic_candles(n_candles=500000, amplitude_range=(5.0, 50.0), random_seed=6)
    # test_df = generate_synthetic_candles(n_candles=10000, amplitude_range=(5.0, 50.0), random_seed=8)

    # train_df.plot(x='time')
    # plt.show()
    # exit()


    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.save_json(Config.MODEL_DIR / "config.json")
    pprint(Config.to_dict())



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
        features_extractor_kwargs=dict(features_dim=Config.FEATURE_DIM, in_channels=len(features), window=Config.WINDOW, kernel_sizes=Config.KERNEL_SIZES, n_filters=Config.N_FILTERS,
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
                tb_log_name="inception",
                callback=CallbackList([ProgressCallback(Config.TOTAL_STEPS), eval_callback]),
                log_interval=1,
                progress_bar=False)
    model.save(Config.MODEL_DIR / "m5_ppo_trader_5M_inception_final")
