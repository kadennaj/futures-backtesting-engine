import pandas as pd
import numpy as np
from itertools import product

# =========================
# LOAD NQ DATA
# =========================
df = pd.read_csv("nq_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")

df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp_ny"].dt.date
df["time"] = df["timestamp_ny"].dt.time

# =========================
# COMMON INDICATORS
# =========================
prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()

df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()
df["atr_pct"] = df["atr"] / df["close"]

# =========================
# COSTS
# keep same quick comparison model
# =========================
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# BACKTEST FUNCTION
# =========================
def run_backtest(
    fast,
    slow,
    trend,
    session_end_str,
    atr_filter,
    ema_sep,
    stop_points,
    target_points
):
    df_bt = df.copy()

    df_bt["ema_fast"] = df_bt["close"].ewm(span=fast, adjust=False).mean()
    df_bt["ema_slow"] = df_bt["close"].ewm(span=slow, adjust=False).mean()
    df_bt["ema_trend"] = df_bt["close"].ewm(span=trend, adjust=False).mean()

    session_start = pd.to_datetime("09:35").time()
    session_end = pd.to_datetime(session_end_str).time()

    trend_filter = df_bt["close"] > df_bt["ema_trend"]
    momentum_filter = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) > ema_sep
    vol_filter = df_bt["atr_pct"] > atr_filter
    time_filter = (df_bt["time"] >= session_start) & (df_bt["time"] < session_end)

    entry_signal = trend_filter & momentum_filter & vol_filter & time_filter
    exit_signal = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) < 0

    position = 0
    entry_price = 0.0
    signals = []
    trade_returns = []

    for i in range(len(df_bt)):
        signal = 0
        ret = 0.0

        current_close = df_bt.iloc[i]["close"]
        current_high = df_bt.iloc[i]["high"]
        current_low = df_bt.iloc[i]["low"]
        current_time = df_bt.iloc[i]["time"]

        if i == 0:
            signals.append(signal)
            trade_returns.append(ret)
            continue

        prev_close_bar = df_bt.iloc[i - 1]["close"]

        if position == 1:
            ret = (current_close / prev_close_bar) - 1

        if position == 1:
            stop_price = entry_price - stop_points
            target_price = entry_price + target_points

            hit_stop = current_low <= stop_price
            hit_target = current_high >= target_price
            session_exit = current_time >= session_end
            custom_exit = bool(exit_signal.iloc[i])

            if hit_stop:
                ret = (stop_price / prev_close_bar) - 1 - EXIT_COST_PCT
                position = 0
            elif hit_target:
                ret = (target_price / prev_close_bar) - 1 - EXIT_COST_PCT
                position = 0
            elif session_exit or custom_exit:
                ret = (current_close / prev_close_bar) - 1 - EXIT_COST_PCT
                position = 0

        if position == 0 and bool(entry_signal.iloc[i]):
            position = 1
            entry_price = current_close
            signal = 1
            ret -= ENTRY_COST_PCT
        elif position == 1:
            signal = 1

        signals.append(signal)
        trade_returns.append(ret)

    df_bt["signal"] = signals
    df_bt["strategy_return"] = trade_returns
    df_bt["market_return"] = df_bt["close"].pct_change().fillna(0)

    df_bt["strategy_equity"] = (1 + pd.Series(df_bt["strategy_return"]).fillna(0)).cumprod()
    df_bt["buy_hold_equity"] = (1 + df_bt["market_return"]).cumprod()

    entries = ((pd.Series(df_bt["signal"]) == 1) & (pd.Series(df_bt["signal"]).shift(1).fillna(0) == 0)).sum()

    running_max = df_bt["strategy_equity"].cummax()
    drawdown = df_bt["strategy_equity"] / running_max - 1
    max_drawdown = drawdown.min()

    trade_list = []
    in_trade = False
    entry_eq = None

    for i in range(len(df_bt)):
        curr_sig = df_bt.iloc[i]["signal"]
        prev_sig = df_bt.iloc[i - 1]["signal"] if i > 0 else 0

        if curr_sig == 1 and prev_sig == 0:
            in_trade = True
            entry_eq = df_bt.iloc[i]["strategy_equity"]

        if curr_sig == 0 and prev_sig == 1 and in_trade:
            exit_eq = df_bt.iloc[i]["strategy_equity"]
            trade_list.append((exit_eq / entry_eq) - 1)
            in_trade = False

    wins = sum(1 for t in trade_list if t > 0)
    win_rate = wins / len(trade_list) if trade_list else 0

    avg_win = sum(t for t in trade_list if t > 0) / max(1, sum(1 for t in trade_list if t > 0))
    avg_loss = sum(t for t in trade_list if t <= 0) / max(1, sum(1 for t in trade_list if t <= 0))
    profit_factor = (
        sum(t for t in trade_list if t > 0) / abs(sum(t for t in trade_list if t < 0))
        if any(t < 0 for t in trade_list) else float("inf")
    )

    return {
        "equity": float(df_bt["strategy_equity"].iloc[-1]),
        "buy_hold_equity": float(df_bt["buy_hold_equity"].iloc[-1]),
        "entries": int(entries),
        "closed_trades": len(trade_list),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
    }

# =========================
# NQ PARAM GRID
# =========================
FAST_VALS = [12, 16, 20]
SLOW_VALS = [21, 25, 30]
TREND_VALS = [150, 200, 250]
SESSION_END_VALS = ["11:30", "12:00", "12:30"]
ATR_FILTER_VALS = [0.0007, 0.0009, 0.0011]
EMA_SEP_VALS = [0.00015, 0.0002, 0.00025]

# NQ point-based exits
STOP_POINTS_VALS = [20, 30, 40, 50]
TARGET_POINTS_VALS = [40, 60, 80, 100]

# =========================
# OPTIMIZATION
# =========================
results = []

combos = list(product(
    FAST_VALS,
    SLOW_VALS,
    TREND_VALS,
    SESSION_END_VALS,
    ATR_FILTER_VALS,
    EMA_SEP_VALS,
    STOP_POINTS_VALS,
    TARGET_POINTS_VALS
))

print(f"Total raw combinations: {len(combos)}")

tested = 0

for fast, slow, trend, session_end, atr_filter, ema_sep, stop_points, target_points in combos:
    if fast >= slow:
        continue
    if target_points <= stop_points:
        continue

    stats = run_backtest(
        fast,
        slow,
        trend,
        session_end,
        atr_filter,
        ema_sep,
        stop_points,
        target_points
    )

    row = {
        "strategy": "NQ_EMA_MORNING",
        "fast": fast,
        "slow": slow,
        "trend": trend,
        "session_end": session_end,
        "atr_filter": atr_filter,
        "ema_sep": ema_sep,
        "stop_points": stop_points,
        "target_points": target_points,
        "equity": stats["equity"],
        "buy_hold_equity": stats["buy_hold_equity"],
        "entries": stats["entries"],
        "closed_trades": stats["closed_trades"],
        "win_rate": stats["win_rate"],
        "avg_win": stats["avg_win"],
        "avg_loss": stats["avg_loss"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown": stats["max_drawdown"],
    }
    results.append(row)

    tested += 1
    if tested % 100 == 0:
        best_equity = max(r["equity"] for r in results)
        print(f"Tested {tested} valid strategies... best equity so far: {best_equity:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["equity", "profit_factor", "max_drawdown"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("\nTOP 15 NQ RESULTS\n")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("nq_optimization_results.csv", index=False)
print("\nSaved nq_optimization_results.csv")
