import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from data_utils import rsi, create_features
import talib as ta


df = pd.read_csv("./XAU_5m_data_labels_24_6_3.csv")#[:100000]
df["time"] = pd.to_datetime(df["time"])
df = df.set_index('time')
df.index = pd.DatetimeIndex(df.index)
df = create_features(df)
# print(len(df), df.isna().sum())
print(df.index.diff().value_counts().sort_values())
df.dropna(inplace=True)
print(df.index.diff().value_counts().sort_values())
exit()

df["time"] = pd.to_datetime(df["time"])

color_map = {
    "sell": "r",
    "buy": "g",
    "hold": "b",
}

c = [color_map.get(i, "k") for i in df["label"].values]

# df.plot(x="time", y="close", color=c)
plt.scatter(df["time"], df["close"], color=c, s=1)
plt.grid()
plt.show()
