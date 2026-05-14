import json
from datetime import datetime
from pathlib import Path


class Config:
    DATA_CSV_FOLDER = Path("./archive")
    MODEL_DIR = Path("./models") / datetime.now().strftime('%Y-%m-%d-%H-%M')
    EVAL_LOG_FILE = MODEL_DIR / "eval.log"

    # ENV TODO: lot multiplier
    # TODO: adaptive episode length
    """
    initial_ep_length = 100
    max_ep_length = 1000

    # Ha az entropy csökken (policy konvergál), növeld az epizódot
    if policy_entropy < entropy_threshold:
        episode_length = min(episode_length * 1.5, max_ep_length)
    """
    ENTRY_EXIT = True
    TRAIN_EPISODE_STEPS = 1000
    WINDOW_DICT = {
        "m5m": 8 * 12,  # 8 hours
        "m1h": 48,  # 2 days
        "m4h": 4 * 6,  # 4 days
    }
    TOTAL_STEPS = 4_000_000
    EVAL_FREQ = 10_000
    NUM_EVAL_EPISODES = 60
    N_ENVS = 8

    FROM_TRAIN = datetime(2008, 9, 12)  # datetime(2008, 9, 12)
    FROM_TEST = datetime(2023, 12, 24)  # datetime(2024, 9, 12)
    FROM_DROP = datetime(2024, 12, 24)  # datetime(2025, 9, 13)

    FWD_WINDOW = 5001
    TP_SL_RATIO = [1.5]  # [1.5, 2, 3], [2], [1, 1.5, 2, 3]
    SL_LEVELS = [10]  # [3, 4, 5, 8], [10]
    CONT_ACTION = False
    RANDOM_INDICES = False
    HOLD_MLT = 0  # -0.001
    SL_MLT = 1.5
    TIME_PENALTY_MLT = 0.002
    DD_PENALTY_MLT = 0.05
    DD_MLT = 0.05

    # MODEL
    FEATURE_DIM = {
        "m5m": 128,
        "m1h": 64,
        "m4h": 32,
        "macro": 32
    }
    KERNEL_SIZES = {
        "m5m": (9, 19, 39),
        "m1h": (5, 11, 19),
        "m4h": (3, 5, 9),
    }
    BOTTLENECK_CHANNELS = 32
    N_FILTERS = 64

    NET_ARCH = [128]
    START_LR = 5e-5
    END_LR = 1e-5
    BATCH_SIZE = 512  # cnn1d 256
    GAMMA = 0.99
    N_EPOCHS = 4

    # DQN
    BUFFER_SIZE = 100_000
    EXPLR_FRACTION = 0.25
    EXPLR_FINAL_EPS = 0.01
    TRAIN_FREQ = 4
    T_U_I = 1000

    # PPO
    N_STEPS = 2048
    GEA_LAMBDA = 0.92
    CLIP_RANGE = 0.2
    PPO_ENT_COEF = 0.02
    VF_COEF = 0.7
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.02

    FEATURES = [
        # 'open', 'high', 'low', 'close',  # 'volume',
        # 'log_ret',
        'vol_20',
        'close_over_ma',
        # 'body_ratio',
        'rsi_14',
        'rsi_7',
        'macd',
        'macd_sig',
        # 'atr_14',
        'bb_z',
        'donch_high_ratio',
        'donch_low_ratio',
        'feat_close_ret',
        'feat_body',
        'feat_upper_wick',
        'feat_lower_wick',
        'feat_volume',
        'feat_hl_range',
        'feat_gap',
        'feat_atr_ratio',
        'adx_14',
        'dmp_14',
        'dmn_14',
    ]

    @classmethod
    def to_dict(cls) -> dict:
        d = {}
        for k, v in cls.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            if isinstance(v, classmethod):
                continue
            # Path -> str (JSON‑barát)
            if isinstance(v, (Path, datetime)):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict) -> None:
        for k, v in data.items():
            if not hasattr(cls, k):
                continue
            if k == "MODEL_DIR":
                setattr(cls, k, Path(v))
            else:
                setattr(cls, k, v)

    @classmethod
    def save_json(cls, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(cls.to_dict())
        with path.open("w", encoding="utf-8") as f:
            json.dump(cls.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path: str | Path) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cls.from_dict(data)
