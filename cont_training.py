import warnings
import gym
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from pprint import pprint

from data_utils import get_df_dict
from inception_time_ori import InceptionTime
from multi_tf_env import MultyTFTradingEnv
from config import Config
from rl_debug import SyncNormalizeEvalCallback, cosine_schedule, ProgressCallback

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    train_df_dict, test_df_dict = get_df_dict()

    conf_path = "./models/2026-05-06-16-57/config.json"
    Config.load_json(conf_path)
    pprint(Config.to_dict())

    vec_env = make_vec_env(lambda: MultyTFTradingEnv(train_df_dict), n_envs=Config.N_ENVS)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,  # observation normalizálás
        norm_reward=True,  # reward normalizálás — EZ A KULCS
        clip_reward=10.0,  # ne legyenek extrém reward értékek
        gamma=0.99
    )

    test_env = make_vec_env(
        lambda: MultyTFTradingEnv(test_df_dict, mode="val"), n_envs=1
    )
    test_env = VecNormalize(
        test_env,
        norm_obs=False,
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

    model = PPO.load(
        Config.MODEL_DIR / "mtf_ppo_trader_5M_inception_final",
        env=vec_env,
        tensorboard_log=Config.MODEL_DIR / "tensorboard_log",
    )

    model.learning_rate = 2e-05

    model.learn(
        total_timesteps=Config.TOTAL_STEPS * Config.N_ENVS,  # 8M további lépés
        reset_num_timesteps=False,  # ← folytatja 16M-től
        tb_log_name="run_continued",
        callback=CallbackList([ProgressCallback(Config.TOTAL_STEPS), eval_callback]),
    )
    model.save(Config.MODEL_DIR / "mtf_ppo_trader_5M_inception_final")
