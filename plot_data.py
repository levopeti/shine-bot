import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv("./XAU_5m_data_labels_24_6_3.csv")[:100000]
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
