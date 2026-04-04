import json
from datetime import datetime
from pathlib import Path


class Config:
    DATA_CSV_PATH = "./XAU_5m_data.csv"
    MODEL_DIR = Path("./models") / datetime.now().strftime('%Y-%m-%d-%H-%M')

    # ENV TODO: lot multiplier
    TRAIN_EPISODE_STEPS = 10_000
    WINDOW_H = 48
    FWD_WINDOW = 5001
    TP_SL_RATIO = [1.5, 2, 3] #  [1.5, 2, 3], [2]
    SL_LEVELS = [3, 4, 5, 8] #  [3, 4, 5, 8], [10]
    NORMALIZE = False
    RANDOM_INDICES = False

    # MODEL
    NET_ARCH = [128]
    LR = 1e-4
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 64
    GAMMA = 0.99
    EXPLR_FRACTION = 0.25
    EXPLR_FINAL_EPS = 0.01
    TRAIN_FREQ = 4
    T_U_I = 1000
    TOTAL_STEPS = 5_000_000

    EVAL_FREQ = 200_000

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
            if isinstance(v, Path):
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

