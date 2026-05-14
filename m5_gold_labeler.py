#!/usr/bin/env python3
"""
M5 XAUUSD gyertya labelező
Labels: hold | buy1 | buy2 | sell1 | sell2
"""

import pandas as pd
import numpy as np
import argparse
import sys

ATR_PERIOD = 14
ATR1_MULT = 1.5
ATR2_MULT = 3.0
WINDOW = 12
MAX_GAP_BARS = 6
M5_MINUTES = 5


def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def has_gap(times):
    diffs = times.diff().dropna()
    return (diffs > pd.Timedelta(minutes=MAX_GAP_BARS * M5_MINUTES)).any()


def assign_label(i, df, atr1, atr2):
    window = df.iloc[i + 1: i + 1 + WINDOW]
    if len(window) < WINDOW:
        return "hold"
    if has_gap(window["time"]):
        return "hold"

    ref_price = df.iloc[i]["close"]

    up_direction = (
            (window["open"] >= ref_price).all() and
            (window["close"] >= ref_price).all() and
            (window["low"] >= ref_price - atr1).all()
    )
    if up_direction:
        max_up = window["high"].max() - ref_price
        if max_up > atr2:
            return "buy2"
        elif max_up > atr1:
            return "buy1"

    down_direction = (
            (window["open"] <= ref_price).all() and
            (window["close"] <= ref_price).all() and
            (window["high"] <= ref_price + atr1).all()
    )
    if down_direction:
        max_down = ref_price - window["low"].min()
        if max_down > atr2:
            return "sell2"
        elif max_down > atr1:
            return "sell1"

    return "hold"


def print_label_stats(labels: pd.Series, total: int):
    LABEL_ORDER = ["buy2", "buy1", "hold", "sell1", "sell2"]
    counts = labels.value_counts()

    print()
    print("=" * 48)
    print(f"  LABEL STATISZTIKA  (összesen: {total} gyertya)")
    print("=" * 48)
    print(f"  {'Label':<10} {'Darab':>8}  {'Arány':>8}")
    print("-" * 48)
    for lbl in LABEL_ORDER:
        cnt = counts.get(lbl, 0)
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl:<10} {cnt:>8}  {pct:>7.2f}%  {bar}")
    print("-" * 48)
    labeled = total - counts.get("hold", 0)
    print(f"  {'Nem-hold':<10} {labeled:>8}  {labeled / total * 100:>7.2f}%")
    print(f"  {'hold':<10} {counts.get('hold', 0):>8}  {counts.get('hold', 0) / total * 100:>7.2f}%")
    print("=" * 48)
    print()


def label_candles(input_csv, output_csv):
    df = pd.read_csv(input_csv, sep=';')
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Hiányzó oszlopok: {missing}")

    df["time"] = pd.to_datetime(df["date"])
    df = df.sort_values("time").reset_index(drop=True)

    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["atr1"] = df["atr"] * ATR1_MULT
    df["atr2"] = df["atr"] * ATR2_MULT

    labels = []
    n = len(df)
    for i in range(n):
        if pd.isna(df.at[i, "atr"]):
            labels.append("hold")
            continue
        lbl = assign_label(i, df, df.at[i, "atr1"], df.at[i, "atr2"])
        labels.append(lbl)

    df["label"] = labels
    df.to_csv(output_csv, index=False)

    print(f"Kész → {output_csv}")
    print_label_stats(df["label"], n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 arany gyertya labelező")
    parser.add_argument("input", help="Bemeneti CSV fájl")
    parser.add_argument("output", help="Kimeneti CSV fájl")
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    parser.add_argument("--atr1-mult", type=float, default=ATR1_MULT)
    parser.add_argument("--atr2-mult", type=float, default=ATR2_MULT)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--max-gap", type=int, default=MAX_GAP_BARS)
    args = parser.parse_args()

    ATR_PERIOD = args.atr_period
    ATR1_MULT = args.atr1_mult
    ATR2_MULT = args.atr2_mult
    WINDOW = args.window
    MAX_GAP_BARS = args.max_gap

    label_candles(args.input, args.output)
