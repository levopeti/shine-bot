#!/usr/bin/env python3
"""
Confidence vs Win Rate elemzo
Hasznalat:
    python confidence_analysis.py --trades eval_output/test_trades.csv
    python confidence_analysis.py --trades eval_output/val_trades.csv --bins 0.05
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze(trades_path, bin_size=0.10, min_trades=5, out_dir="eval_output"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df = pd.read_csv(trades_path)

    required = {"confidence", "reason", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hianyzo oszlopok: {missing}")

    bins = np.arange(0.0, 1.0 + bin_size, bin_size)
    bin_labels = [f"{b:.0%}-{b+bin_size:.0%}" for b in bins[:-1]]
    df["conf_bin"] = pd.cut(df["confidence"], bins=bins, labels=bin_labels, include_lowest=True)

    grouped = df.groupby("conf_bin", observed=True).agg(
        n_trades=("pnl", "count"),
        n_win=("reason", lambda x: (x == "tp").sum()),
        n_sl=("reason", lambda x: (x == "sl").sum()),
        n_undef=("reason", lambda x: (~x.isin(["tp","sl"])).sum()),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    grouped["wr"] = grouped["n_win"] / grouped["n_trades"].clip(lower=1)
    grouped = grouped[grouped["n_trades"] >= min_trades].copy()

    if grouped.empty:
        print("Nincs elegendo adat.")
        return

    print(grouped[["conf_bin","n_trades","wr","avg_pnl","total_pnl"]].to_string(index=False))
    grouped.to_csv(out / "confidence_analysis.csv", index=False)
    label = Path(trades_path).stem

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.42, 0.32, 0.26], vertical_spacing=0.12,
                        subplot_titles=[
                            "Win Rate / confidence bin",
                            "Trade-ek szama (TP / SL / egyeb)",
                            "Atlag PnL / trade",
                        ])

    wr_colors = ["#22c55e" if w >= 0.5 else "#ef4444" for w in grouped["wr"]]
    fig.add_trace(go.Bar(x=grouped["conf_bin"], y=grouped["wr"], name="Win Rate",
                         marker_color=wr_colors,
                         text=[f"{w:.1%}" for w in grouped["wr"]],
                         textposition="outside", textfont=dict(size=13)), row=1, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(grouped)-0.5, y0=0.5, y1=0.5,
                  line=dict(color="#f59e0b", width=2, dash="dash"), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1], row=1, col=1)

    fig.add_trace(go.Bar(x=grouped["conf_bin"], y=grouped["n_win"], name="TP",
                         marker_color="#22c55e", text=grouped["n_win"],
                         textposition="inside", textfont=dict(size=12)), row=2, col=1)
    fig.add_trace(go.Bar(x=grouped["conf_bin"], y=grouped["n_sl"], name="SL",
                         marker_color="#ef4444", text=grouped["n_sl"],
                         textposition="inside", textfont=dict(size=12)), row=2, col=1)
    fig.add_trace(go.Bar(x=grouped["conf_bin"], y=grouped["n_undef"], name="Egyeb",
                         marker_color="#94a3b8", text=grouped["n_undef"],
                         textposition="inside", textfont=dict(size=12)), row=2, col=1)

    pnl_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in grouped["avg_pnl"]]
    fig.add_trace(go.Bar(x=grouped["conf_bin"], y=grouped["avg_pnl"], name="Atlag PnL",
                         marker_color=pnl_colors,
                         text=[f"{p:.2f}" for p in grouped["avg_pnl"]],
                         textposition="outside", textfont=dict(size=13)), row=3, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(grouped)-0.5, y0=0, y1=0,
                  line=dict(color="#94a3b8", width=1), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"Confidence vs Win Rate  |  {label}", font=dict(size=20)),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
        font=dict(size=13),
    )
    fig.update_xaxes(title_text="Confidence tartomany", tickfont=dict(size=12), row=3, col=1)
    fig.update_xaxes(tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=12), row=2, col=1)
    fig.update_yaxes(title_text="Win Rate", title_font=dict(size=13), row=1, col=1)
    fig.update_yaxes(title_text="Trade db", title_font=dict(size=13), row=2, col=1)
    fig.update_yaxes(title_text="Atlag PnL", title_font=dict(size=13), row=3, col=1)
    fig.update_traces(cliponaxis=False)

    html_path = out / f"{label}_confidence_analysis.html"
    fig.write_html(str(html_path))
    print(f"Chart: {html_path}")
    print(f"CSV:   {out / 'confidence_analysis.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades",     required=True)
    parser.add_argument("--bins",       type=float, default=0.10)
    parser.add_argument("--min_trades", type=int,   default=5)
    parser.add_argument("--out",        default="eval_output")
    args = parser.parse_args()
    analyze(args.trades, bin_size=args.bins, min_trades=args.min_trades, out_dir=args.out)


if __name__ == "__main__":
    main()
