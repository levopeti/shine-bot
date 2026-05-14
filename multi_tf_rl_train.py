import warnings
import gym
import torch

from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from pprint import pprint

from data_utils import get_df_dict
from entry_exit_env import EntryExitEnv
from inception_time_ori import InceptionTime
from multi_tf_env import MultyTFTradingEnv
from config import Config
from rl_debug import SyncNormalizeEvalCallback, cosine_schedule, ProgressCallback

warnings.filterwarnings("ignore")


class MultiTFExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        self.extractors = torch.nn.ModuleDict({
            timeframe: InceptionTime(features_dim=Config.FEATURE_DIM[timeframe],
                                     in_channels=len(Config.FEATURES),
                                     window=Config.WINDOW_DICT[timeframe],
                                     kernel_sizes=Config.KERNEL_SIZES[timeframe],
                                     n_filters=Config.N_FILTERS,
                                     bottleneck_channels=Config.BOTTLENECK_CHANNELS) for timeframe in Config.WINDOW_DICT
        })

        self.extractors["macro"] = torch.nn.Sequential(torch.nn.Flatten(),
                                                       torch.nn.Linear(observation_space["macro"].shape[0],
                                                                       Config.FEATURE_DIM["macro"]),
                                                       torch.nn.ReLU(),
                                                       )

        self._features_dim = sum(Config.FEATURE_DIM.values())  # concatenált kimenet mérete

    def forward(self, observations):
        outputs = []
        for key, extractor in self.extractors.items():
            output = extractor(observations[key])

            if output.dim() == 1:
                output = output.unsqueeze(0)
            outputs.append(output)
        return torch.cat(outputs, dim=1)  # shape: (batch, 192)


if __name__ == "__main__":
    train_df_dict, test_df_dict = get_df_dict()

    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.save_json(Config.MODEL_DIR / "config.json")
    pprint(Config.to_dict())

    env = EntryExitEnv if Config.ENTRY_EXIT else MultyTFTradingEnv

    vec_env = make_vec_env(lambda: env(train_df_dict), n_envs=Config.N_ENVS)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,  # observation normalizálás
        norm_reward=True,  # reward normalizálás — EZ A KULCS
        clip_reward=10.0,  # ne legyenek extrém reward értékek
        gamma=0.99
    )

    test_env = make_vec_env(lambda: env(test_df_dict, mode="val"), n_envs=1)
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
        n_eval_episodes=Config.NUM_EVAL_EPISODES,
        deterministic=True,
        log_path=Config.MODEL_DIR / "eval_log",
        best_model_save_path=Config.MODEL_DIR / "best_model",
        verbose=1,
    )

    multy_inception_policy_kwargs = dict(
        features_extractor_class=MultiTFExtractor,
        net_arch=Config.NET_ARCH
    )

    ppo = MaskablePPO if Config.ENTRY_EXIT else PPO
    model = ppo(
        policy="MultiInputPolicy",
        env=vec_env,
        tensorboard_log=Config.MODEL_DIR / "tensorboard_log",
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
        policy_kwargs=multy_inception_policy_kwargs,
        verbose=1,
        target_kl=Config.TARGET_KL,  # ha kl > 0.02, leállítja a frissítést az epochon belül
    )
    model.learn(total_timesteps=Config.TOTAL_STEPS * Config.N_ENVS,
                tb_log_name="inception",
                callback=CallbackList([ProgressCallback(Config.TOTAL_STEPS), eval_callback]),
                log_interval=1,
                progress_bar=False)
    model.save(Config.MODEL_DIR / "mtf_ppo_trader_5M_inception_final")
