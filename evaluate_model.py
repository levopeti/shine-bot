#!/usr/bin/env python3
"""
Model Evaluator Script
======================
Kiértékeli a PPO modellt val/test idősoron.
- Konfiguráló: MAX_POSITIONS (1 = csak 1 nyitott pozíció egyszerre)
- CONFIDENCE_THRESHOLD: csak ha a háló softmax max prob >= threshold
- Vizualizáció: plotly interaktív chart

Használat:
    python evaluate_model.py --mode val   # validációs idősorra
    python evaluate_model.py --mode test  # test idősorra
    python evaluate_model.py --mode both  # mindkettő
"""

import argparse
import json
import os
import torch
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Konfiguráció — módosítsd igény szerint
# ---------------------------------------------------------------------------
MODEL_PATH = Path("./models/2026-05-06-16-57/best_model/best_model")   # .zip nélkül
MAX_POSITIONS = 1          # 1 = egyszerre max 1 nyitott pozíció; None = korlátlan
CONFIDENCE_THR = 0.4  # 0.0–1.0; csak ha softmax max >= ez az érték
OUTPUT_DIR = MODEL_PATH.parent / Path("eval_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Confidence kinyerése a policy-ból
# ---------------------------------------------------------------------------
def get_action_and_probs(model: PPO, obs: dict) -> tuple[int, float, np.ndarray]:
    """
    Visszaadja: (action, confidence, probs)
    confidence = softmax max valószínűsége
    """
    obs_th = {
        k: torch.as_tensor(np.array(v)[np.newaxis], dtype=torch.float32).to(model.device)
        for k, v in obs.items()
    }
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_th)
        probs = dist.distribution.probs.cpu().numpy()[0]  # [n_actions]

    action = int(np.argmax(probs))
    confidence = float(probs[action])
    return action, confidence, probs


# ---------------------------------------------------------------------------
# Akció értelmezése az action_map alapján
# ---------------------------------------------------------------------------
def decode_action(action_map: dict, action: int) -> tuple[str, float, float]:
    """Visszaadja: (direction, tp_sl_ratio, sl_level)"""
    return action_map.get(action, ("hold", 0, 0))


# ---------------------------------------------------------------------------
# Trade tracker
# ---------------------------------------------------------------------------
class TradeTracker:
    def __init__(self, max_positions):
        self.max_positions = max_positions
        self.open_trades = []
        self.closed_trades = []

    @property
    def n_open(self):
        return len(self.open_trades)

    def can_open(self):
        if self.max_positions is None:
            return True
        return self.n_open < self.max_positions

    def open_trade(self, step, time, price, direction, confidence, action, tp_price, sl_price):
        self.open_trades.append({
            "open_step": step,
            "open_time": time,
            "open_price": price,
            "direction": direction,
            "confidence": confidence,
            "action": action,
            "tp_price": tp_price,
            "sl_price": sl_price,
        })

    def check_and_close(self, step, time, candle_high, candle_low, candle_close):
        """TP/SL ellenőrzés minden nyitott pozícióra."""
        remaining = []
        for t in self.open_trades:
            direction = t["direction"]
            tp = t["tp_price"]
            sl = t["sl_price"]
            closed = False

            if direction == "buy":
                if candle_low <= sl and candle_high >= tp:
                    reason, pnl = "undefined", 0.0
                    closed = True
                elif candle_low <= sl:
                    reason, pnl = "sl", sl - t["open_price"]
                    closed = True
                elif candle_high >= tp:
                    reason, pnl = "tp", tp - t["open_price"]
                    closed = True
            else:  # sell
                if candle_high >= sl and candle_low <= tp:
                    reason, pnl = "undefined", 0.0
                    closed = True
                elif candle_high >= sl:
                    reason, pnl = "sl", t["open_price"] - sl
                    closed = True
                elif candle_low <= tp:
                    reason, pnl = "tp", t["open_price"] - tp
                    closed = True

            if closed:
                close_price = tp if reason == "tp" else (sl if reason == "sl" else candle_close)
                t.update({
                    "close_step": step,
                    "close_time": time,
                    "close_price": close_price,
                    "pnl": pnl,
                    "reason": reason,
                })
                self.closed_trades.append(t)
            else:
                remaining.append(t)
        self.open_trades = remaining

    def close_conflicting(self, step, time, price, new_direction):
        remaining = []
        for t in self.open_trades:
            if t["direction"] != new_direction:
                pnl = (price - t["open_price"]) if t["direction"] == "buy" else (t["open_price"] - price)
                t.update({
                    "close_step": step, "close_time": time,
                    "close_price": price, "pnl": pnl, "reason": "reversed"
                })
                self.closed_trades.append(t)
            else:
                remaining.append(t)
        self.open_trades = remaining

    def close_all(self, step, time, price, reason="end_of_data"):
        for t in self.open_trades:
            pnl = (price - t["open_price"]) if t["direction"] == "buy" else (t["open_price"] - price)
            t.update({
                "close_step": step, "close_time": time,
                "close_price": price, "pnl": pnl, "reason": reason
            })
            self.closed_trades.append(t)
        self.open_trades = []


# ---------------------------------------------------------------------------
# Fő kiértékelő — env-alapú
# ---------------------------------------------------------------------------
def evaluate_with_env(
    model: PPO,
    env,                        # MultyTFTradingEnv instance, mode="val" vagy "test"
    m5m_df: pd.DataFrame,       # az m5m dataframe az ár és időadatokhoz
    label: str = "eval",
    max_positions: int = 1,
    confidence_threshold: float = 0.40,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    tracker = TradeTracker(max_positions=max_positions)
    action_map = env.action_map

    records = []
    episode = 0
    total_episodes = len(env.episode_indices_list)

    print(f"  {total_episodes} epizód kiértékelése...")

    obs, info = env.reset()

    while True:
        # --- akció és confidence ---
        action, confidence, probs = get_action_and_probs(model, obs)

        direction, tp_sl_ratio, sl_level = decode_action(action_map, action)
        effective_direction = direction

        # confidence threshold — ha nem elég biztos, hold
        if confidence < confidence_threshold and direction != "hold":
            effective_action = 0  # hold
            effective_direction = "hold"
        else:
            effective_action = action

        # --- aktuális ár és idő ---
        step_idx = env.episode_indices[env.current_step] if env.current_step < len(env.episode_indices) else env.episode_indices[-1]
        row = m5m_df.iloc[step_idx]
        price = float(row["close"])
        time  = row["time"]
        atr   = float(row.get("atr", 0))

        # --- TP/SL árak kiszámítása ---
        tp_price = sl_price = None
        if effective_direction in ("buy", "sell") and atr > 1e-8:
            tp = atr * tp_sl_ratio * (1 + confidence)
            sl = atr * (1 + confidence)
            if effective_direction == "buy":
                tp_price = price + tp
                sl_price = price - sl
            else:
                tp_price = price - tp
                sl_price = price + sl

        # --- Nyitott pozíciók TP/SL ellenőrzése ---
        tracker.check_and_close(
            step_idx, time,
            float(row["high"]), float(row["low"]), price
        )

        # --- Pozíció kezelés ---
        if effective_direction in ("buy", "sell"):
            tracker.close_conflicting(step_idx, time, price, effective_direction)
            if tracker.can_open() and tp_price is not None:
                tracker.open_trade(
                    step_idx, time, price, effective_direction,
                    confidence, effective_action, tp_price, sl_price
                )

        records.append({
            "step_idx": step_idx,
            "time": time,
            "price": price,
            "high": float(row["high"]),
            "low": float(row["low"]),
            "raw_action": action,
            "effective_action": effective_action,
            "direction": effective_direction,
            "confidence": confidence,
            "prob_hold": float(probs[0]),
            "prob_best": float(np.max(probs[1:])) if len(probs) > 1 else 0.0,
            "n_open": tracker.n_open,
        })

        # --- step ---
        obs, reward, terminated, truncated, info = env.step(effective_action)
        done = terminated or truncated

        if done:
            episode += 1
            print(f"    Epizód {episode}/{total_episodes} kész | "
                  f"stats: {info['episode_stats']}")
            if episode >= total_episodes:
                break
            obs, info = env.reset()

    # Maradék nyitott pozíciók zárása
    if records:
        last = records[-1]
        tracker.close_all(last["step_idx"], last["time"], last["price"])

    result_df = pd.DataFrame(records)
    trades_df  = pd.DataFrame(tracker.closed_trades)

    result_df.to_csv(OUTPUT_DIR / f"{label}_signals.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_DIR / f"{label}_trades.csv", index=False)

    # --- Összefoglaló ---
    print(f"\n{'='*52}")
    print(f"  {label.upper()} összefoglaló")
    print(f"{'='*52}")
    print(f"  Lépések:         {len(result_df):,}")
    if not trades_df.empty:
        n  = len(trades_df)
        tp = (trades_df["reason"] == "tp").sum()
        sl = (trades_df["reason"] == "sl").sum()
        wr = tp / n if n else 0
        pnl = trades_df["pnl"].sum()
        print(f"  Zárt tradek:     {n}")
        print(f"  TP / SL / egyéb: {tp} / {sl} / {n - tp - sl}")
        print(f"  Win Rate:        {wr:.1%}")
        print(f"  Total PnL (pip): {pnl:.2f}")
        print(f"  Avg PnL/trade:   {pnl/n:.2f}")
    print(f"{'='*52}\n")

    return result_df, trades_df


# ---------------------------------------------------------------------------
# Vizualizáció
# ---------------------------------------------------------------------------
def plot_evaluation(
    result_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    label: str,
    confidence_threshold: float,
    max_points: int = 5000,
):
    df = result_df.copy()
    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        df = df.iloc[::step].reset_index(drop=True)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[
            f"Ár + jelzések ({label})",
            "Confidence",
            "Kumulatív PnL (pip)",
        ],
        vertical_spacing=0.06,
    )

    # Ár
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["price"],
        mode="lines", name="Ár",
        line=dict(width=1.2, color="#94a3b8"),
    ), row=1, col=1)

    # BUY jelzők
    buy_df = df[df["direction"] == "buy"]
    fig.add_trace(go.Scatter(
        x=buy_df["time"], y=buy_df["price"],
        mode="markers", name="BUY",
        marker=dict(symbol="triangle-up", size=9, color="#22c55e"),
        hovertemplate="BUY<br>%{x}<br>Ár: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # SELL jelzők
    sell_df = df[df["direction"] == "sell"]
    fig.add_trace(go.Scatter(
        x=sell_df["time"], y=sell_df["price"],
        mode="markers", name="SELL",
        marker=dict(symbol="triangle-down", size=9, color="#ef4444"),
        hovertemplate="SELL<br>%{x}<br>Ár: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Trade záró vonalak
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            color = "#22c55e" if t.get("pnl", 0) > 0 else "#ef4444"
            try:
                fig.add_shape(
                    type="line",
                    x0=t["open_time"], x1=t["close_time"],
                    y0=t["open_price"], y1=t["close_price"],
                    line=dict(color=color, width=1, dash="dot"),
                    row=1, col=1,
                )
            except Exception:
                pass

    # Confidence
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["confidence"],
        mode="lines", name="Confidence",
        line=dict(width=1, color="#818cf8"),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.12)",
    ), row=2, col=1)

    fig.add_hline(
        y=confidence_threshold,
        line_dash="dash", line_color="#f59e0b", line_width=1.5,
        annotation_text=f"thr={confidence_threshold}",
        row=2, col=1,
    )

    # Kumulatív PnL
    if not trades_df.empty:
        cum = trades_df.sort_values("close_time").copy()
        cum["cum_pnl"] = cum["pnl"].cumsum()
        final_pnl = cum["cum_pnl"].iloc[-1]
        line_color = "#22c55e" if final_pnl > 0 else "#ef4444"
        fig.add_trace(go.Scatter(
            x=cum["close_time"], y=cum["cum_pnl"],
            mode="lines", name="Kum. PnL",
            line=dict(width=1.8, color=line_color),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(line_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
        ), row=3, col=1)

    fig.update_layout(
        title=f"Model kiértékelés — {label} | max_pos={MAX_POSITIONS} | conf_thr={confidence_threshold}",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=950,
    )
    fig.update_xaxes(title_text="Idő", row=3, col=1)
    fig.update_yaxes(title_text="Ár", row=1, col=1)
    fig.update_yaxes(title_text="Conf.", row=2, col=1)
    fig.update_yaxes(title_text="PnL (pip)", row=3, col=1)

    html_path = OUTPUT_DIR / f"{label}_chart.html"
    fig.write_html(str(html_path))
    print(f"  Interaktív chart: {html_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global MAX_POSITIONS, CONFIDENCE_THR
    parser = argparse.ArgumentParser(description="PPO Model Evaluator (env-alapú)")
    parser.add_argument("--mode",             choices=["val", "test", "both"], default="both")
    parser.add_argument("--model_path",       default=MODEL_PATH)
    parser.add_argument("--max_positions",    type=int,   default=MAX_POSITIONS)
    parser.add_argument("--confidence",       type=float, default=CONFIDENCE_THR)
    parser.add_argument("--max_chart_points", type=int,   default=5000)
    args = parser.parse_args()

    print(f"Modell betöltése: {args.model_path}")
    model = PPO.load(args.model_path)
    print("  OK")

    from data_utils import get_df_dict
    from config import Config
    from multi_tf_env import MultyTFTradingEnv  # igazítsd az import path-hoz

    print("Adatok betöltése...")
    train_df_dict, test_df_dict = get_df_dict()

    Config.load_json(MODEL_PATH.parent.parent / "config.json")

    modes = []
    if args.mode in ("val", "both"):
        modes.append(("val", train_df_dict))
    if args.mode in ("test", "both"):
        modes.append(("test", test_df_dict))

    # MAX_POSITIONS globál frissítése args-ból

    MAX_POSITIONS  = args.max_positions
    CONFIDENCE_THR = args.confidence

    for label, df_dict in modes:
        print(f"\n--- {label.upper()} kiértékelés ---")
        env = MultyTFTradingEnv(df_dict, mode="val", save_log=False)  # "val" mód → episode_indices_list

        result_df, trades_df = evaluate_with_env(
            model=model,
            env=env,
            m5m_df=df_dict["m5m"],
            label=label,
            max_positions=args.max_positions,
            confidence_threshold=args.confidence,
        )

        plot_evaluation(
            result_df, trades_df,
            label=label,
            confidence_threshold=args.confidence,
            max_points=args.max_chart_points,
        )

    print("\nKész. Outputok:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()