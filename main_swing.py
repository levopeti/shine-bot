from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from conv1d import Conv1DFeaturesExtractor
from data_utils import load_gold_m5
from inception_time import InceptionTime
from only_gold_env import M5TradingEnv

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.episode_count = 0
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
        total_trades = stats["tp"] + stats["sl"] + stats["timeout"]

        win_rate = stats["tp"] / max(total_trades, 1) * 100

        postfix.update({
            # "ep": self.episode_count,
            "TP": f"{stats['tp']}",
            "SL": f"{stats['sl']}",
            "TO": f"{stats['timeout']}",
            "WR": f"{win_rate:.0f}%"
        })

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos", [{}])[0].get("is_success") or self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            stats = self.locals["infos"][0]["episode_stats"]
            total_trades = stats["tp"] + stats["sl"] + stats["timeout"]

            win_rate = stats["tp"] / max(total_trades, 1) * 100

            postfix = {
                "ep": self.episode_count,
                "TP": f"{stats['tp']}",
                "SL": f"{stats['sl']}",
                "TO": f"{stats['timeout']}",
                "WR": f"{win_rate:.0f}%"
            }
            self.pbar.set_postfix(postfix)

        return True


if __name__ == "__main__":
    df = load_gold_m5("./XAU_5m_data.csv")

    test_start_year = 2005
    train_df = df[df["time"].dt.year < test_start_year].reset_index(drop=True)
    test_df = df[df["time"].dt.year >= test_start_year].reset_index(drop=True)

    print(f"Train: {train_df['time'].iloc[0].year} – {train_df['time'].iloc[-1].year} | {len(train_df):,} gyertya")
    print(f"Test:  {test_df['time'].iloc[0].year} – {test_df['time'].iloc[-1].year} | {len(test_df):,} gyertya")

    train_env = Monitor(M5TradingEnv(train_df, fwd_window=2))
    # test_env = Monitor(M5TradingEnv(test_df))
    # train_env = M5TradingEnv(train_df)
    # test_env = M5TradingEnv(test_df)

    mlp_policy_kwargs = dict(net_arch=[1024, 512, 128])
    cnn_policy_kwargs = dict(
        features_extractor_class=Conv1DFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=[512, 256]  # MLP
    )
    inception_policy_kwargs = dict(
        features_extractor_class=InceptionTime,
        features_extractor_kwargs=dict(features_dim=256, in_channels=6),
        net_arch=[128]  # MLP
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
        policy_kwargs=cnn_policy_kwargs,
        tensorboard_log="./tensorboard_logs/"
    )
    policy = model.policy
    print(policy.features_extractor)
    print(policy.q_net)

    print("learning")
    # eval_callback = EvalCallback(
    #     test_env,
    #     eval_freq=500_000,
    #     n_eval_episodes=1,
    #     verbose=1,
    #     render=True
    # )
    TOTAL = 5_000_000
    model.learn(total_timesteps=TOTAL,
                tb_log_name="inception",
                callback=ProgressCallback(TOTAL),
                log_interval=1000,
                progress_bar=False)
    model.save("./m5_dqn_trader_5M_inception")
