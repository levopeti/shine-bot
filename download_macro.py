import yfinance as yf
import pandas as pd
import numpy as np


TICKERS = {
    "dxy":    "DX-Y.NYB",
    "xag":    "SI=F",
    "us10y":  "^TNX",
    "wti":    "CL=F",
    "usdjpy": "JPY=X",
}


def _fix_yfinance_df(df: pd.DataFrame, name: str) -> pd.Series:
    """yfinance MultiIndex oszlop és index tisztítása → egyszerű Series."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].copy()

    if isinstance(close.index, pd.MultiIndex):
        close.index = close.index.get_level_values(0)

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close.name = name
    close = close[~close.index.duplicated(keep="last")]
    return close


def download_macro(
    start: str = "2004-01-01",
    end: str = "2026-01-01",
    m1h_df: pd.DataFrame | None = None,
    gold_close_col: str = "close",
    shift_days: int = 1,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Letölti a macro adatokat (napi), interpolálja és kiszámolja az indikátorokat.

    Args:
        start: Letöltés kezdő dátuma.
        end: Letöltés végső dátuma.
        m1h_df: Az arany M1H DataFrame. Ha megadod:
            - target index = m1h_df.index
            - gold/silver ratio is kiszámolódik
        gold_close_col: Az arany záróár oszlop neve m1h_df-ben.
        shift_days: Look-ahead bias elkerülésére hány napot shifteljen.
        save_path: Opcionális CSV mentés.

    Returns:
        pd.DataFrame: Macro feature-ök, M1H indexre igazítva ha m1h_df adott.
    """

    series_list = []
    raw_xag = None

    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print(f"WARNING: {ticker} üres, kihagyva.")
            continue

        s = _fix_yfinance_df(df, name)
        series_list.append(s)

        if name == "xag":
            raw_xag = s.copy()

    macro = pd.concat(series_list, axis=1)
    macro = macro.sort_index().ffill().bfill()

    # Look-ahead bias ellen
    macro = macro.shift(shift_days)

    # Feature-ök
    macro["dxy_log_ret"] = np.log(macro["dxy"] / macro["dxy"].shift(1))
    macro["dxy_over_ma20"] = macro["dxy"] / macro["dxy"].rolling(20).mean() - 1
    macro["dxy_over_ma50"] = macro["dxy"] / macro["dxy"].rolling(50).mean() - 1

    macro["xag_log_ret"] = np.log(macro["xag"] / macro["xag"].shift(1))

    macro["us10y_delta_1d"] = macro["us10y"].diff(1)
    macro["us10y_delta_5d"] = macro["us10y"].diff(5)
    macro["us10y_over_ma20"] = macro["us10y"] / macro["us10y"].rolling(20).mean() - 1

    macro["wti_log_ret"] = np.log(macro["wti"] / macro["wti"].shift(1))
    macro["wti_over_ma20"] = macro["wti"] / macro["wti"].rolling(20).mean() - 1

    macro["usdjpy_log_ret"] = np.log(macro["usdjpy"] / macro["usdjpy"].shift(1))
    macro["usdjpy_over_ma20"] = macro["usdjpy"] / macro["usdjpy"].rolling(20).mean() - 1

    if m1h_df is not None:
        target_index = pd.DatetimeIndex(m1h_df["time"])
        if target_index.tz is not None:
            target_index = target_index.tz_localize(None)
        target_index = target_index[~target_index.duplicated()]

        combined_idx = macro.index.union(target_index).sort_values()
        macro = macro.reindex(combined_idx).ffill().reindex(target_index).ffill().bfill()

        if raw_xag is not None:
            gold = m1h_df[gold_close_col].copy()
            gold.index = pd.to_datetime(gold.index)
            if gold.index.tz is not None:
                gold.index = gold.index.tz_localize(None)
            gold = gold[~gold.index.duplicated(keep="last")]

            xag_m1h = (
                raw_xag.shift(shift_days)
                .reindex(raw_xag.index.union(macro.index))
                .ffill()
                .reindex(macro.index)
            )

            gold_arr = gold.reindex(macro.index, method="ffill").to_numpy(dtype=float).ravel()
            xag_arr = xag_m1h.to_numpy(dtype=float).ravel()

            assert gold_arr.shape == xag_arr.shape, (
                f"Shape mismatch: gold={gold_arr.shape}, xag={xag_arr.shape}"
            )

            gs_ratio = gold_arr / np.where(xag_arr > 0, xag_arr, np.nan)
            gs_mean = np.nanmean(gs_ratio)
            gs_std = np.nanstd(gs_ratio)

            macro["gold_silver_ratio_z"] = (gs_ratio - gs_mean) / (gs_std + 1e-8)

    drop_raw = [c for c in TICKERS.keys() if c in macro.columns]
    macro_features = macro.drop(columns=drop_raw, errors="ignore")

    macro_features = macro_features.ffill().bfill()

    if save_path:
        macro_features.to_csv(save_path)
        print(f"Mentve: {save_path}")

    print(f"Kész. Shape: {macro_features.shape}")
    print(f"Columns: {list(macro_features.columns)}")

    return macro_features