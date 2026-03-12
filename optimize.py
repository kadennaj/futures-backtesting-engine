import pandas as pd
import numpy as np
from itertools import product

# =========================
# LOAD DATA ONCE
# =========================
df_raw = pd.read_csv("spy_data.csv")
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
df_raw["timestamp_ny"] = df_raw["timestamp"].dt.tz_convert("America/New_York")
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
df_raw["date"] = df_raw["timestamp_ny"].dt.date
df_raw["time"] = df_raw["timestamp_ny"].dt.time

SESSION_START = pd.to_datetime("09:40").time()
SESSION_END = pd.to_datetime("15:50").time()
ATR_PERIOD = 14

def run_backtest(fast_ema, slow_ema, trend_ema, stop_loss_pct, take_profit_pct):
    df = df_raw.copy()

    # Indicators
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=trend_ema, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    trend_filter = df["close"] > df["ema_trend"]
    momentum_filter = df["ema_fast"] > df["ema_slow"]
    vol_filter = df["atr_pct"] > 0.0008
    time_filter = (df["time"] >= SESSION_START) & (df["time"] < SESSION_END)

    df["entry_signal"] = (trend_filter & momentum_filter & vol_filter & time_filter).astype(int)

    position = 0
    entry_price = 0.0
    signals = []
    trade_returns = []

    for i in range(len(df)):
        signal = 0
        ret = 0.0

        current_close = df.at[i, "close"]
        current_high = df.at[i, "high"]
        current_low = df.at[i, "low"]
        current_time = df.at[i, "time"]

        if i == 0:
            signals.append(signal)
            trade_returns.append(ret)
            continue

        prev_close_bar = df.at[i - 1, "close"]

        if position == 1:
            ret = (current_close / prev_close_bar) - 1

        if position == 1:
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)

            hit_stop = current_low <= stop_price
            hit_target = current_high >= target_price
            end_of_day_exit = current_time >= SESSION_END
            trend_exit = df.at[i, "ema_fast"] < df.at[i, "ema_slow"]

            if hit_stop:
                ret = (stop_price / prev_close_bar) - 1
                position = 0
            elif hit_target:
                ret = (target_price / prev_close_bar) - 1
                position = 0
            elif end_of_day_exit:
                ret = (current_close / prev_close_bar) - 1
                position = 0
            elif trend_exit:
                ret = (current_close / prev_close_bar) - 1
                position = 0

        if position == 0 and df.at[i, "entry_signal"] == 1:
            position = 1
            entry_price = current_close
            signal = 1
        elif position == 1:
            signal = 1

        signals.append(signal)
        trade_returns.append(ret)

    df["signal"] = signals
    df["strategy_return"] = trade_returns
    df["market_return"] = df["close"].pct_change().fillna(0)

    df["strategy_equity"] = (1 + pd.Series(df["strategy_return"]).fillna(0)).cumprod()
    df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()

    entries = ((pd.Series(df["signal"]) == 1) & (pd.Series(df["signal"]).shift(1).fillna(0) == 0)).sum()
    exits = ((pd.Series(df["signal"]) == 0) & (pd.Series(df["signal"]).shift(1).fillna(0) == 1)).sum()

    running_max = df["strategy_equity"].cummax()
    drawdown = df["strategy_equity"] / running_max - 1
    max_drawdown = drawdown.min()

    trade_list = []
    in_trade = False
    entry_eq = None

    for i in range(len(df)):
        curr_sig = df.at[i, "signal"]
        prev_sig = df.at[i - 1, "signal"] if i > 0 else 0

        if curr_sig == 1 and prev_sig == 0:
            in_trade = True
            entry_eq = df.at[i, "strategy_equity"]

        if curr_sig == 0 and prev_sig == 1 and in_trade:
            exit_eq = df.at[i, "strategy_equity"]
            trade_list.append((exit_eq / entry_eq) - 1)
            in_trade = False

    wins = sum(1 for t in trade_list if t > 0)
    win_rate = wins / len(trade_list) if trade_list else 0

    return {
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
        "trend_ema": trend_ema,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "final_equity": df["strategy_equity"].iloc[-1],
        "buy_hold_equity": df["buy_hold_equity"].iloc[-1],
        "entries": int(entries),
        "exits": int(exits),
        "closed_trades": len(trade_list),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown
    }

# =========================
# PARAMETER GRID
# =========================
fast_emas = [8, 12, 16]
slow_emas = [21, 36, 50]
trend_emas = [100, 150]
stop_losses = [0.003, 0.004, 0.005]
take_profits = [0.006, 0.008, 0.010]

results = []

for fast, slow, trend, sl, tp in product(fast_emas, slow_emas, trend_emas, stop_losses, take_profits):
    if fast >= slow:
        continue

    result = run_backtest(fast, slow, trend, sl, tp)
    results.append(result)
    print(
        f"Tested fast={fast}, slow={slow}, trend={trend}, "
        f"SL={sl}, TP={tp} -> equity={result['final_equity']:.4f}, "
        f"dd={result['max_drawdown']:.4f}, trades={result['closed_trades']}"
    )

results_df = pd.DataFrame(results)

# Rank: good equity, controlled drawdown, enough trades
results_df = results_df.sort_values(
    by=["final_equity", "max_drawdown", "closed_trades"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("\nTop 10 Results:")
print(results_df.head(10))

results_df.to_csv("optimization_results.csv", index=False)
print("\nSaved optimization_results.csv")
