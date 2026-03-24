import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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
