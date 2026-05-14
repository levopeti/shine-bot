from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta as pta

from config import Config
from download_macro import download_macro


# ---------------------------------------------------------------------------
# Rolling z-score segéd
# ---------------------------------------------------------------------------
def _rolling_zscore(series: pd.Series, w: int, clip: float = 3.0) -> pd.Series:
    mean = series.rolling(w, min_periods=w).mean()
    std = series.rolling(w, min_periods=w).std()
    return ((series - mean) / (std + 1e-8)).clip(-clip, clip)


# ---------------------------------------------------------------------------
# Fő feature pipeline (pandas-ta alapú)
# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame, zscore_window: int = 20, clip: float = 3.0) -> pd.DataFrame:
    """
    Kiszámítja az összes indikátort és normalizált feature-t.
    Bemeneti oszlopok: open, high, low, close, volume
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # ------------------------------------------------------------------ #
    # 1. Volatilitás és trend                                             #
    # ------------------------------------------------------------------ #
    log_ret = np.log(close).diff()
    df["vol_20"] = log_ret.rolling(20).std()

    ma_20 = pta.sma(close, length=20)
    df["close_over_ma"] = close / ma_20 - 1

    # ------------------------------------------------------------------ #
    # 2. RSI                                                              #
    # ------------------------------------------------------------------ #
    df["rsi_14"] = pta.rsi(close, length=14)
    df["rsi_7"] = pta.rsi(close, length=7)

    # ------------------------------------------------------------------ #
    # 3. MACD                                                             #
    # ------------------------------------------------------------------ #
    macd_df = pta.macd(close, fast=12, slow=26, signal=9)
    df["macd"] = macd_df["MACD_12_26_9"]
    df["macd_sig"] = macd_df["MACDs_12_26_9"]

    # ------------------------------------------------------------------ #
    # 4. Bollinger Bands (z-score)                                        #
    # ------------------------------------------------------------------ #
    bb_df = pta.bbands(close, length=20, std=2)
    df["bb_z"] = (close - bb_df["BBM_20_2.0_2.0"]) / (bb_df["BBU_20_2.0_2.0"] - bb_df["BBL_20_2.0_2.0"] + 1e-8)

    # ------------------------------------------------------------------ #
    # 5. Donchian Channel                                                 #
    # ------------------------------------------------------------------ #
    dc_df = pta.donchian(high, low, upper_length=20, lower_length=20)
    df["donch_high_ratio"] = close / dc_df["DCU_20_20"] - 1
    df["donch_low_ratio"] = close / dc_df["DCL_20_20"] - 1

    # ------------------------------------------------------------------ #
    # 6. ATR                                                              #
    # ------------------------------------------------------------------ #
    df["atr"] = pta.atr(high, low, close, length=14)
    df["feat_atr_ratio"] = df["atr"] / (close + 1e-8)  # skálafüggetlen

    # ------------------------------------------------------------------ #
    # 7. ADX (trend erősség) — új                                        #
    # ------------------------------------------------------------------ #
    adx_df = pta.adx(high, low, close, length=14)
    df["adx_14"] = adx_df["ADX_14"]
    df["dmp_14"] = adx_df["DMP_14"]  # +DI (bullish nyomás)
    df["dmn_14"] = adx_df["DMN_14"]  # -DI (bearish nyomás)

    # ------------------------------------------------------------------ #
    # 8. Normalizált gyertya feature-ök (z-score)                        #
    # ------------------------------------------------------------------ #
    close_ret = close.pct_change()
    body = close / open_ - 1
    upper_wick = high / pd.concat([open_, close], axis=1).max(axis=1) - 1
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) / (low + 1e-8) - 1
    vol_log_ret = np.log(volume / volume.shift(1))
    hl_range = (high - low) / (close + 1e-8)
    gap = open_ / close.shift(1) - 1

    df["feat_close_ret"] = _rolling_zscore(close_ret, zscore_window, clip)
    df["feat_body"] = _rolling_zscore(body, zscore_window, clip)
    df["feat_upper_wick"] = _rolling_zscore(upper_wick, zscore_window, clip)
    df["feat_lower_wick"] = _rolling_zscore(lower_wick, zscore_window, clip)
    df["feat_volume"] = _rolling_zscore(vol_log_ret, zscore_window, clip)
    df["feat_hl_range"] = _rolling_zscore(hl_range, zscore_window, clip)
    df["feat_gap"] = _rolling_zscore(gap, zscore_window, clip)
    return df


# ---------------------------------------------------------------------------
# ATR a stop loss számításhoz (add_atr megtartva backward compat miatt)
# ---------------------------------------------------------------------------
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR hozzáadása Wilder-féle EWM-mel (create_features-ben már benne van)."""
    if "atr" not in df.columns:
        df["atr"] = pta.atr(df["high"], df["low"], df["close"], length=period)
    return df


# ---------------------------------------------------------------------------
# Betöltés
# ---------------------------------------------------------------------------
def load_gold(folder, timeframe: str) -> pd.DataFrame:
    filename_dic = {
        "m5m": "XAU_5m_data.csv",
        "m1h": "XAU_1h_data.csv",
        "m4h": "XAU_4h_data.csv",
    }
    df = pd.read_csv(
        folder / filename_dic[timeframe],
        sep=";",
        parse_dates=["Date"],
        dayfirst=False,
    )
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "time"}, inplace=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Szinkronizálás
# ---------------------------------------------------------------------------
def synchronize(df_dict: dict) -> dict:
    # macro-t kihagyjuk az időszinkronból (külön indexe van)
    price_keys = [k for k in df_dict if k != "macro"]
    min_time = max(df_dict[k]["time"].min() for k in price_keys)
    max_time = min(df_dict[k]["time"].max() for k in price_keys)

    for k in price_keys:
        df = df_dict[k]
        df_dict[k] = df[(df["time"] >= min_time) & (df["time"] <= max_time)].reset_index(drop=True)
    return df_dict


# ---------------------------------------------------------------------------
# Fő belépési pont
# ---------------------------------------------------------------------------
def get_df_dict() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train_df_dict: Dict[str, pd.DataFrame] = {}
    test_df_dict: Dict[str, pd.DataFrame] = {}

    for timeframe in Config.WINDOW_DICT:
        print(f"── {timeframe} ──")
        df = load_gold(Path(Config.DATA_CSV_FOLDER), timeframe)
        df = df[df["time"] < Config.FROM_DROP]

        train_df = df[(df["time"] >= Config.FROM_TRAIN) & (df["time"] < Config.FROM_TEST)].reset_index(drop=True)
        test_df = df[df["time"] >= Config.FROM_TEST].reset_index(drop=True)

        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            split_df = create_features(split_df)
            split_df = add_atr(split_df)  # backward compat, no-op ha már van atr
            split_df.dropna(inplace=True)
            split_df.reset_index(drop=True, inplace=True)

            if split_name == "train":
                print(split_df[Config.FEATURES].describe().round(3))
                train_df_dict[timeframe] = split_df
            else:
                print(split_df[Config.FEATURES].describe().round(3))
                test_df_dict[timeframe] = split_df

    # ------------------------------------------------------------------ #
    # Macro                                                               #
    # ------------------------------------------------------------------ #
    macro_path = Path("./archive/macro_m1h.csv")
    if macro_path.is_file():
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro = macro.reset_index().rename(columns={"index": "time"})
    else:
        m1h_df = load_gold(Config.DATA_CSV_FOLDER, "m1h")
        macro = download_macro(
            start="2004-01-01",
            end="2026-01-01",
            m1h_df=m1h_df,
            save_path=str(macro_path),
        )
        macro = macro.reset_index().rename(columns={"index": "time"})

    macro["time"] = pd.to_datetime(macro["time"])

    train_macro = macro[(macro["time"] >= Config.FROM_TRAIN) & (macro["time"] < Config.FROM_TEST)].reset_index(
        drop=True)
    test_macro = macro[macro["time"] >= Config.FROM_TEST].reset_index(drop=True)

    train_df_dict["macro"] = train_macro
    test_df_dict["macro"] = test_macro

    # ------------------------------------------------------------------ #
    # Szinkronizálás és összefoglaló                                      #
    # ------------------------------------------------------------------ #
    train_df_dict = synchronize(train_df_dict)
    test_df_dict = synchronize(test_df_dict)

    for tf in train_df_dict:
        tr = train_df_dict[tf]
        te = test_df_dict[tf]
        print(f"{tf}")
        if "time" in tr.columns:
            print(f"  Train: {tr['time'].iloc[0]} – {tr['time'].iloc[-1]} | {len(tr):,}")
            print(f"  Test:  {te['time'].iloc[0]} – {te['time'].iloc[-1]} | {len(te):,}")
        else:
            print(f"  Train: {tr.index[0]} – {tr.index[-1]} | {len(tr):,}")
            print(f"  Test:  {te.index[0]} – {te.index[-1]} | {len(te):,}")

    return train_df_dict, test_df_dict
