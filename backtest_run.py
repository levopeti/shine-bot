import numpy as np
from stable_baselines3 import DQN
from backtesting import Backtest, Strategy
from torch.backends.mkl import verbose

from main_swing import load_gold_m5, M5TradingEnv


def collect_trades(model, env):
    obs, _ = env.reset()
    done = False
    step = env.window  # current_step kezdeti értéke

    trades = []
    rewards = []
    tp_hit = sl_hit = timeout = holds = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        direction, tp_pct, sl_pct = env.action_map[action.tolist()]
        obs, reward, terminated, truncated, date_dict = env.step(action.tolist())
        done = terminated or truncated

        rewards.append(reward)

        if direction == "hold":
            holds += 1
        else:
            entry_price = env.df.loc[step, "open"]
            trades.append({
                "date": date_dict["date"],
                "step": step,
                "direction": direction,
                "entry": entry_price,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "reward": reward
            })

            if reward == 10_000 * tp_pct:
                tp_hit += 1
            elif reward == -10_000 * sl_pct:
                sl_hit += 1
            else:
                timeout += 1

        step += 1

    return trades, rewards, {"tp_hit": tp_hit, "sl_hit": sl_hit,
                             "timeout": timeout, "holds": holds}


if "__main__" == __name__:
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    df = load_gold_m5("./XAU_5m_data.csv")

    test_start_year = 2025
    train_df = df[df["time"].dt.year < test_start_year].reset_index(drop=True)
    test_df = df[df["time"].dt.year >= test_start_year].reset_index(drop=True)
    # test_df = df[df["time"].dt.year < test_start_year].reset_index(drop=True)

    print(f"Train: {train_df['time'].iloc[0].year} – {train_df['time'].iloc[-1].year} | {len(train_df):,} gyertya")
    print(f"Test:  {test_df['time'].iloc[0].year} – {test_df['time'].iloc[-1].year} | {len(test_df):,} gyertya")

    test_env = M5TradingEnv(test_df)
    # model = DQN.load("/home/salusmo/Downloads/m5_dqn_trader.zip")
    model = DQN.load("./m5_dqn_trader_5M_cnn.zip")

    trades, rewards, stats = collect_trades(model, test_env)

    # ── 2. Saját metrikák kiírása ─────────────────────────────────────────────────

    total_trades = stats["tp_hit"] + stats["sl_hit"] + stats["timeout"]
    win_rate = stats["tp_hit"] / max(total_trades, 1) * 100

    print("=" * 45)
    print("         MODELL KIÉRTÉKELÉS")
    print("=" * 45)
    print(f"Összes lépés:      {len(rewards):,}")
    print(f"Összes trade:      {total_trades:,}")
    print(f"Hold:              {stats['holds']:,}")
    print(f"─────────────────────────────────────────")
    print(f"TP hit:            {stats['tp_hit']:,}  ({stats['tp_hit'] / max(total_trades, 1) * 100:.1f}%)")
    print(f"SL hit:            {stats['sl_hit']:,}  ({stats['sl_hit'] / max(total_trades, 1) * 100:.1f}%)")
    print(f"Timeout:           {stats['timeout']:,}  ({stats['timeout'] / max(total_trades, 1) * 100:.1f}%)")
    print(f"─────────────────────────────────────────")
    print(f"Win rate:          {win_rate:.1f}%")
    print(f"Összes reward:     {sum(rewards):.1f}")
    print(f"Átlag reward:      {np.mean(rewards):.4f}")
    print(
        f"Legjobb streak:    {max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(rewards, lambda x: x > 0) if k), default=0)}")
    print("=" * 45)

    # ── 3. Backtesting.py stratégia ───────────────────────────────────────────────

    # trade_lookup: step → trade info
    trade_lookup = {t["step"]: t for t in trades}


    # trade_lookup = {t["date"]: t for t in trades}

    class DQNStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            current_step = len(self.data) - 1  # + test_env.window

            if current_step not in trade_lookup:
                return

            trade = trade_lookup[current_step]
            entry = trade["entry"]

            try:
                if trade["direction"] == "buy":
                    tp = entry * (1 + trade["tp_pct"])
                    sl = entry * (1 - trade["sl_pct"])
                    self.buy(tp=tp, sl=sl, size=0.01)
                else:
                    tp = entry * (1 - trade["tp_pct"])
                    sl = entry * (1 + trade["sl_pct"])
                    self.sell(tp=tp, sl=sl, size=0.01)
            except ValueError as e:
                print(e)
                print(self.data)
                print(trade)
                print()


    # Futtatás
    bt_df = test_df[["time", "open", "high", "low", "close", "volume"]].copy()
    bt_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    bt_df.set_index("Date", inplace=True)
    bt = Backtest(bt_df, DQNStrategy, cash=1_000, commission=0.0002, margin=0.002, finalize_trades=True)
    result = bt.run()

    # Releváns metrikák a result-ból
    print("=" * 45)
    print("         BACKTESTING EREDMÉNYEK")
    print("=" * 45)
    print(f"Végső tőke:        ${result['Equity Final [$]']:,.2f}")
    print(f"Hozam:             {result['Return [%]']:.2f}%")
    print(f"Max Drawdown:      {result['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate:          {result['Win Rate [%]']:.2f}%")
    print(f"Összes trade:      {result['# Trades']}")
    print(f"Sharpe Ratio:      {result['Sharpe Ratio']:.3f}")
    print(f"Profit Factor:     {result['Profit Factor']:.3f}")
    print("=" * 45)

    # print("Nyereséges trade-ek:")
    # print(result[result['PnL'] > 0][['EntryTime', 'PnL', 'PnL%']].round(4))
    #
    # # Csak veszteséges
    # print("Veszteséges trade-ek:")
    # print(result[result['PnL'] < 0][['EntryTime', 'PnL', 'PnL%']].round(4))

    breakpoint()

    # bt.plot()
