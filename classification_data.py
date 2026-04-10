import numpy as np
import pandas as pd
from tqdm import tqdm

from rl_train import load_gold_m5


def make_labels(df: pd.DataFrame, frw_window: int, tp_limit: int, sl_limit: int) -> pd.DataFrame:
    # current_idx = 0
    # labels = list()
    df["label"] = None
    for current_idx in tqdm(range(0, len(df))):
        # current_high = df.loc[current_idx, "high"]
        # current_low = df.loc[current_idx, "low"]
        current_price = df.loc[current_idx, "close"]
        start_idx = current_idx + 1
        end_idx = start_idx + frw_window

        if end_idx >= len(df):
            break
        fwd = df.iloc[start_idx:end_idx]

        if fwd["high"].max() < current_price + sl_limit and fwd["low"].min() <= current_price - tp_limit:
            # labels.append((current_idx, "sell"))
            df.loc[current_idx, "label"] = "sell"
        elif fwd["low"].min() > current_price - sl_limit and fwd["high"].max() >= current_price + tp_limit:
            # labels.append((current_idx, "buy"))
            df.loc[current_idx, "label"] = "buy"
        else:
            # labels.append((current_idx, "hold"))
            df.loc[current_idx, "label"] = "hold"

    sell = (df["label"] == "sell").sum()
    buy = (df["label"] == "buy").sum()
    hold = (df["label"] == "hold").sum()
    print("sell: {}, {:.2}%".format(sell, sell / len(df)))
    print("buy: {}, {:.2}%".format(buy, buy / len(df)))
    print("hold: {}, {:.2}%".format(hold, hold / len(df)))
    return df

def make_labels_2(df: pd.DataFrame, frw_window: int, tp_thresh: float, sl_thresh: float) -> pd.DataFrame:
    df['future_max'] = df['close'].rolling(frw_window).max() / df['close'] - 1
    df['future_min'] = df['close'].rolling(frw_window).min() / df['close'] - 1
    df['future_return'] = df['close'].shift(-frw_window) / df['close'] - 1

    df['label'] = "hold"  # hold
    df.loc[df['future_max'] >= tp_thresh, 'label'] = "buy"  # buy
    df.loc[df['future_min'] <= -sl_thresh, 'label'] = "sell"  # sell

    df['y_tp'] = np.maximum(df['future_max'], 0.01)  # min 1%
    df['y_sl'] = np.abs(np.minimum(df['future_min'], -0.01))  # abs min 1%

    df.dropna(inplace=True)  # Train data kész
    return df


if __name__ == "__main__":
    _frw_window = 12 * 12
    _tp_limit = 6
    _sl_limit = 3
    _df = load_gold_m5("./XAU_5m_data.csv")  # [:200000]
    _df = make_labels(_df, frw_window=_frw_window, tp_limit=_tp_limit, sl_limit=_sl_limit)
    _df.to_csv("./XAU_5m_data_labels_{}_{}_{}.csv".format(_frw_window, _tp_limit, _sl_limit), index=False)
    # print(_df)