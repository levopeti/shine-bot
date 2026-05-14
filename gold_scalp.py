import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from backtesting import Backtest, Strategy

from config import Config

# Elvárt oszlopok: Open, High, Low, Close, Volume
df = pd.read_csv("./archive/XAU_1m_data.csv", sep=";", parse_dates=["Date"], dayfirst=False)
df = df[df["Date"] > Config.FROM_DROP]
df = df.rename(columns={
    "Date": "datetime",
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume"
}).set_index("datetime")

class GoldScalpStrategy(Strategy):
    ema_fast = 9
    ema_slow = 21
    rsi_period = 7
    rsi_long_level = 55
    rsi_short_level = 45
    atr_period = 14
    sl_atr_mult = 1.2
    tp_atr_mult = 1.8

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.ema_fast_series = self.I(
            lambda x: EMAIndicator(pd.Series(x), window=self.ema_fast).ema_indicator().values,
            self.data.Close
        )
        self.ema_slow_series = self.I(
            lambda x: EMAIndicator(pd.Series(x), window=self.ema_slow).ema_indicator().values,
            self.data.Close
        )
        self.rsi_series = self.I(
            lambda x: RSIIndicator(pd.Series(x), window=self.rsi_period).rsi().values,
            self.data.Close
        )
        self.atr_series = self.I(
            lambda h, l, c: AverageTrueRange(
                pd.Series(h), pd.Series(l), pd.Series(c), window=self.atr_period
            ).average_true_range().values,
            self.data.High, self.data.Low, self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]
        ema_fast = self.ema_fast_series[-1]
        ema_slow = self.ema_slow_series[-1]
        rsi = self.rsi_series[-1]
        atr = self.atr_series[-1]

        if self.position:
            return

        # Long setup
        if ema_fast > ema_slow and price > ema_slow and rsi > self.rsi_long_level:
            sl = price - atr * self.sl_atr_mult
            tp = price + atr * self.tp_atr_mult
            self.buy(sl=sl, tp=tp)

        # Short setup
        elif ema_fast < ema_slow and price < ema_slow and rsi < self.rsi_short_level:
            sl = price + atr * self.sl_atr_mult
            tp = price - atr * self.tp_atr_mult
            self.sell(sl=sl, tp=tp)

bt = Backtest(df, GoldScalpStrategy, cash=10000, commission=0.0005, trade_on_close=False)

stats = bt.run()
print(stats)

best = bt.optimize(
    ema_fast=range(5, 13, 2),
    ema_slow=range(18, 41, 4),
    rsi_period=range(5, 11),
    rsi_long_level=range(52, 61, 2),
    rsi_short_level=range(39, 48, 2),
    sl_atr_mult=[0.8, 1.0, 1.2, 1.5],
    tp_atr_mult=[1.0, 1.5, 1.8, 2.0, 2.5],
    maximize="Equity Final [$]"
)

print(best)