import numpy as np
import pandas as pd
import talib as ta


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def macd(close, fast=12, slow=16, sig=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal


def atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def bollinger_bands(close, window=20, k=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + k * std
    lower = ma - k * std
    bands = (close - ma) / std  # z‑score a ből
    return upper, lower, bands


def resample_ohlc(df, freq="5T"):
    return df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def load_gold_m5(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=";",  # tab elválasztó
        parse_dates=["Date"],
        dayfirst=False
    )

    # Oszlopnevek kisbetűsítése
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "time"}, inplace=True)

    # Csak a szükséges oszlopok
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # RSI hozzáadása
    df["rsi"] = compute_rsi(df["close"])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Log return
    df['log_ret'] = np.log(df['close']).diff()

    # 2. Retrurn std (pl. 20 lépéses vola)
    df['vol_20'] = df['log_ret'].rolling(20).std()

    # 3. Ár / 20‑napi MA (normális trend + erő)
    df['ma_20'] = df['close'].rolling(20).mean()
    df['close_over_ma'] = df['close'] / df['ma_20'] - 1

    # 4. Candle‑forma
    df['body'] = df['close'] - df['open']  # signed
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, 0.00001)

    # 5. Volume
    df['vol_change'] = df['volume'].pct_change()

    # 6. rsi
    df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
    df['rsi_7'] = ta.RSI(df['close'], timeperiod=7)

    # 7. MACD (12‑26‑9)
    df['macd'], df['macd_sig'] = macd(df['close'])

    df['atr_14'] = atr(df, 14)
    df['bb_upper'], df['bb_lower'], df['bb_z'] = bollinger_bands(df['close'])
    df['donchian_high'] = df['high'].rolling(20).max()
    df['donchian_low'] = df['low'].rolling(20).min()
    df['donch_high_ratio'] = df['close'] / df['donchian_high'] - 1
    df['donch_low_ratio'] = df['close'] / df['donchian_low'] - 1

    # 5‑perces MA 1‑perces adatból
    # df_5m = resample_ohlc(df, "5T").dropna()
    # df_5m['ma_5m_20'] = df_5m['close'].rolling(20).mean()

    # 15‑perces MA
    # df_15m = resample_ohlc(df, "15min").dropna()
    # df_15m['ma_15m_20'] = df_15m['close'].rolling(20).mean()

    # 1‑perces df‑ben újraindex, és merge a 15‑perces MA‑val
    # df['ma_15m_20'] = df.resample("5min").ffill().merge(df_15m[['ma_15m_20']], left_index=True, right_index=True, how="left")
    return df
