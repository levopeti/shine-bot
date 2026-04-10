from pprint import pprint
import warnings
from tqdm import tqdm
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList

from conv1d import Conv1DFeaturesExtractor
from data_utils import load_gold_m5, create_features
from inception_time import InceptionTime
from only_gold_env import M5TradingEnv, ConfidentEnv
from config import Config

warnings.filterwarnings("ignore")


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
            "TP": f"{stats['tp']}",
            "SL": f"{stats['sl']}",
            "H": f"{stats['holds']}",
            "TO": f"{stats['timeout']}",
            "UD": f"{stats['undefined']}",
            "PL": f"{stats['pl']}",
            "WR": f"{stats['wr'] * 100:.0f}%"
        })

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


if __name__ == "__main__":
    # 2025-09-12 23:45:00 2025-10-15 07:55:00, 49424
    df = load_gold_m5(Config.DATA_CSV_PATH)
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.save_json(Config.MODEL_DIR / "config.json")
    pprint(Config.to_dict())

    from_train = datetime(2008, 9, 12)
    from_test = datetime(2024, 9, 12)
    from_drop = datetime(2025, 9, 13)

    df = df.loc[df["time"] < from_drop]

    train_df = df[df["time"] < from_test].reset_index(drop=True)
    train_df = train_df[train_df["time"] > from_train].reset_index(drop=True)

    test_df = df[df["time"] >= from_test].reset_index(drop=True)

    train_df = create_features(train_df)
    train_df.dropna(inplace=True)

    test_df = create_features(test_df)
    test_df.dropna(inplace=True)

    print(f"Train: {train_df['time'].iloc[0]} – {train_df['time'].iloc[-1]} | {len(train_df):,}")
    print(f"Test:  {test_df['time'].iloc[0]} – {test_df['time'].iloc[-1]} | {len(test_df):,}")

    features = [
        'open', 'high', 'low', 'close', 'volume',
        'log_ret', 'vol_20',
        'close_over_ma', 'body_ratio',
        'rsi_14', 'rsi_7',
        'macd', 'macd_sig',
        'atr_14', 'bb_z',
        'donch_high_ratio', 'donch_low_ratio',
        # 'ma_15m_20'  # multi‑tf feature
    ]

    train_env = M5TradingEnv(train_df, features)
    test_env = M5TradingEnv(test_df, features, mode="val")


    mlp_policy_kwargs = dict(net_arch=[1024, 512, 128])
    cnn_policy_kwargs = dict(
        features_extractor_class=Conv1DFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=[512, 256]  # MLP
    )
    inception_policy_kwargs = dict(
        features_extractor_class=InceptionTime,
        features_extractor_kwargs=dict(features_dim=256, in_channels=len(features), window_h=Config.WINDOW_H),
        net_arch=Config.NET_ARCH  # MLP
    )

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=Config.LR,
        # buffer_size=Config.BUFFER_SIZE,
        batch_size=Config.BATCH_SIZE,
        gamma=Config.GAMMA,
        # exploration_fraction=Config.EXPLR_FRACTION,
        # exploration_final_eps=Config.EXPLR_FINAL_EPS,
        # train_freq=Config.TRAIN_FREQ,
        # target_update_interval=Config.T_U_I,
        verbose=0,
        policy_kwargs=inception_policy_kwargs,
        tensorboard_log=Config.MODEL_DIR / "tensorboard_logs/",
        device="cuda"
    )
    policy = model.policy
    # print(policy.features_extractor)
    # print(policy.q_net)
    # exit()

    # confident_env = ConfidentEnv(train_env, model)
    # model.set_env(confident_env)

    print("learning")
    eval_callback = EvalCallback(
        test_env,
        eval_freq=Config.EVAL_FREQ,
        n_eval_episodes=1,
        deterministic=True,
        log_path=Config.MODEL_DIR / "eval_log",
        best_model_save_path=Config.MODEL_DIR / "best_model",
        verbose=1,
        render=True
    )
    datetime.now().strftime('%Y-%m-%d-%H-%M')

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=Config.MODEL_DIR / 'saved_models',
        name_prefix='dqn_swing',
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    model.learn(total_timesteps=Config.TOTAL_STEPS,
                tb_log_name="inception",
                callback=CallbackList([ProgressCallback(Config.TOTAL_STEPS), eval_callback, checkpoint_callback]),
                log_interval=1,
                progress_bar=False)
    model.save(Config.MODEL_DIR / "m5_dqn_trader_5M_inception_final")
