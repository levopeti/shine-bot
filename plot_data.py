from datetime import datetime

import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from classification_data import make_labels_2
from data_utils import rsi, create_features
import talib as ta


df = pd.read_csv("./archive/XAU_1m_data.csv", sep=";")#[:100000]
print(df)
df["time"] = pd.to_datetime(df["Date"])
df_1 = df.set_index('time')

df = pd.read_csv("./archive/XAU_5m_data.csv", sep=";")#[:100000]
print(df)
df["time"] = pd.to_datetime(df["Date"])
df_5 = df.set_index('time')

df = pd.read_csv("./archive/XAU_15m_data.csv", sep=";")#[:100000]
print(df)
df["time"] = pd.to_datetime(df["Date"])
df_15 = df.set_index('time')
# df.index = pd.DatetimeIndex(df.time)
# df = make_labels_2(df, frw_window=12 * 48, tp_thresh=0.03, sl_thresh=0.02)
# df = create_features(df)
# # print(len(df), df.isna().sum())
# print(df.index.diff().value_counts().sort_values())
# df.dropna(inplace=True)
# print(df.index.diff().value_counts().sort_values())
# print(df)
# exit()

from_drop = datetime(2025, 9, 12)
# df = df.loc[df["time"] > from_drop]

from_test = datetime(2024, 9, 11)
# df = df[df["time"] < from_test].reset_index(drop=True)
# df = df[df["time"] >= from_test].reset_index(drop=True)

# sell = (df["label"] == "sell").sum()
# buy = (df["label"] == "buy").sum()
# hold = (df["label"] == "hold").sum()
#
# print("sell: {}, {:.2f}%".format(sell, sell / len(df) * 100))
# print("buy: {}, {:.2f}%".format(buy, buy / len(df) * 100))
# print("hold: {}, {:.2f}%".format(hold, hold / len(df) * 100))

color_map = {
    "sell": "r",
    "buy": "g",
    "hold": "b",
}

# c = [color_map.get(i, "k") for i in df["label"].values]

# df.plot(x="time", y="close", color=c)
plt.scatter(df_1.index,
            df_1["Close"],
            color="b",
            s=1)
plt.scatter(df_5.index,
            df_5["Close"],
            color="r",
            s=1)
plt.scatter(df_15.index,
            df_15["Close"],
            color="g",
            s=1)
plt.grid()
plt.show()
