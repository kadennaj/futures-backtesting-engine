import pandas as pd
import numpy as np
from itertools import product

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("spy_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

typical_price = (df["high"] + df["low"] + df["close"]) / 3
df["tpv"] = typical_price * df["volume"]
df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()
df["cum_vol"] = df.groupby("date")["volume"].cumsum()
df["vwap"] = df["cum_tpv"] / df["cum_vol"]

# Opening range 9:30-9:45
opening_mask = (
    (df["time"] >= pd.to_datetime("09:30").time()) &
    (df["time"] < pd.to_datetime("09:45").time())
)

opening_range = (
    df.loc[opening_mask]
    .groupby("date")
    .agg(opening_high=("high", "max"), opening_low=("low", "min"))
    .reset_index()
)

df = df.merge(opening_range, on="date", how="left")

# =========================
# COSTS
# =========================
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# BACKTEST ENGINE
# =========================
def run_backtest(df_in, entry_signal, session_end_str, stop_loss_pct, take_profit_pct, exit_signal=None):
    df_bt = df_in.copy()

    session_end = pd.to_datetime(session_end_str).time()

    position = 0
    entry_price = 0.0
    signals = []
    trade_returns = []

    for i in range(len(df_bt)):
        signal = 0
        ret = 0.0

        current_close = df_bt.at[i, "close"]
        current_high = df_bt.at[i, "high"]
        current_low = df_bt.at[i, "low"]
        current_time = df_bt.at[i, "time"]

        if i == 0:
            signals.append(signal)
            trade_returns.append(ret)
            continue

        prev_close_bar = df_bt.at[i - 1, "close"]

        if position == 1:
            ret = (current_close / prev_close_bar) - 1

        if position == 1:
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)

            hit_stop = current_low <= stop_price
            hit_target = current_high >= target_price
            session_exit = current_time >= session_end
            custom_exit = False if exit_signal is None else bool(exit_signal.iloc[i])

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
    exits = ((pd.Series(df_bt["signal"]) == 0) & (pd.Series(df_bt["signal"]).shift(1).fillna(0) == 1)).sum()

    running_max = df_bt["strategy_equity"].cummax()
    drawdown = df_bt["strategy_equity"] / running_max - 1
    max_drawdown = drawdown.min()

    trade_list = []
    in_trade = False
    entry_eq = None

    for i in range(len(df_bt)):
        curr_sig = df_bt.at[i, "signal"]
        prev_sig = df_bt.at[i - 1, "signal"] if i > 0 else 0

        if curr_sig == 1 and prev_sig == 0:
            in_trade = True
            entry_eq = df_bt.at[i, "strategy_equity"]

        if curr_sig == 0 and prev_sig == 1 and in_trade:
            exit_eq = df_bt.at[i, "strategy_equity"]
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
        "exits": int(exits),
        "closed_trades": len(trade_list),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
    }

# =========================
# STRATEGY 1: EMA MORNING
# =========================
def test_ema_strategy():
    results = []

    fast_vals = [12, 16, 20]
    slow_vals = [21, 25, 30]
    trend_vals = [150, 200, 250]
    session_end_vals = ["11:30", "12:00", "12:30"]
    atr_vals = [0.0007, 0.0009, 0.0011]
    sep_vals = [0.00015, 0.0002, 0.00025]
    stop_vals = [0.004, 0.005]
    tp_vals = [0.008, 0.010]

    total = len(list(product(fast_vals, slow_vals, trend_vals, session_end_vals, atr_vals, sep_vals, stop_vals, tp_vals)))
    tested = 0

    for fast, slow, trend, session_end, atr_filter, ema_sep, stop, tp in product(
        fast_vals, slow_vals, trend_vals, session_end_vals, atr_vals, sep_vals, stop_vals, tp_vals
    ):
        if fast >= slow:
            continue

        df_bt = df.copy()
        df_bt["ema_fast"] = df_bt["close"].ewm(span=fast, adjust=False).mean()
        df_bt["ema_slow"] = df_bt["close"].ewm(span=slow, adjust=False).mean()
        df_bt["ema_trend"] = df_bt["close"].ewm(span=trend, adjust=False).mean()

        session_start = pd.to_datetime("09:35").time()
        session_end_time = pd.to_datetime(session_end).time()

        trend_filter = df_bt["close"] > df_bt["ema_trend"]
        momentum_filter = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) > ema_sep
        vol_filter = df_bt["atr_pct"] > atr_filter
        time_filter = (df_bt["time"] >= session_start) & (df_bt["time"] < session_end_time)

        entry_signal = trend_filter & momentum_filter & vol_filter & time_filter
        exit_signal = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) < 0

        stats = run_backtest(df_bt, entry_signal, session_end, stop, tp, exit_signal)
        stats.update({
            "strategy": "EMA_MORNING",
            "fast": fast,
            "slow": slow,
            "trend": trend,
            "session_end": session_end,
            "atr_filter": atr_filter,
            "ema_sep": ema_sep,
            "stop": stop,
            "tp": tp
        })
        results.append(stats)

        tested += 1
        if tested % 100 == 0:
            print(f"EMA tested {tested}... best equity so far: {max(r['equity'] for r in results):.4f}")

    return results

# =========================
# STRATEGY 2: ORB
# =========================
def test_orb_strategy():
    results = []

    breakout_buffer_vals = [0.0, 0.0002, 0.0005]
    session_end_vals = ["11:30", "12:00", "12:30"]
    atr_vals = [0.0007, 0.0009, 0.0011]
    stop_vals = [0.004, 0.005]
    tp_vals = [0.008, 0.010]

    tested = 0

    for breakout_buffer, session_end, atr_filter, stop, tp in product(
        breakout_buffer_vals, session_end_vals, atr_vals, stop_vals, tp_vals
    ):
        df_bt = df.copy()

        session_start = pd.to_datetime("09:45").time()
        session_end_time = pd.to_datetime(session_end).time()

        breakout_level = df_bt["opening_high"] * (1 + breakout_buffer)

        time_filter = (df_bt["time"] >= session_start) & (df_bt["time"] < session_end_time)
        vol_filter = df_bt["atr_pct"] > atr_filter
        breakout_filter = df_bt["close"] > breakout_level

        entry_signal = time_filter & vol_filter & breakout_filter
        exit_signal = pd.Series(False, index=df_bt.index)

        stats = run_backtest(df_bt, entry_signal, session_end, stop, tp, exit_signal)
        stats.update({
            "strategy": "ORB",
            "breakout_buffer": breakout_buffer,
            "session_end": session_end,
            "atr_filter": atr_filter,
            "stop": stop,
            "tp": tp
        })
        results.append(stats)

        tested += 1
        if tested % 50 == 0:
            print(f"ORB tested {tested}... best equity so far: {max(r['equity'] for r in results):.4f}")

    return results

# =========================
# STRATEGY 3: VWAP RECLAIM
# =========================
def test_vwap_reclaim_strategy():
    results = []

    reclaim_margin_vals = [0.0, 0.0001, 0.0002]
    session_end_vals = ["11:30", "12:00", "12:30"]
    atr_vals = [0.0007, 0.0009, 0.0011]
    stop_vals = [0.004, 0.005]
    tp_vals = [0.008, 0.010]

    tested = 0

    for reclaim_margin, session_end, atr_filter, stop, tp in product(
        reclaim_margin_vals, session_end_vals, atr_vals, stop_vals, tp_vals
    ):
        df_bt = df.copy()

        session_start = pd.to_datetime("09:35").time()
        session_end_time = pd.to_datetime(session_end).time()

        time_filter = (df_bt["time"] >= session_start) & (df_bt["time"] < session_end_time)
        vol_filter = df_bt["atr_pct"] > atr_filter
        above_vwap = df_bt["close"] > (df_bt["vwap"] * (1 + reclaim_margin))
        reclaim = (df_bt["close"].shift(1) <= df_bt["vwap"].shift(1)) & above_vwap

        entry_signal = time_filter & vol_filter & reclaim
        exit_signal = df_bt["close"] < df_bt["vwap"]

        stats = run_backtest(df_bt, entry_signal, session_end, stop, tp, exit_signal)
        stats.update({
            "strategy": "VWAP_RECLAIM",
            "reclaim_margin": reclaim_margin,
            "session_end": session_end,
            "atr_filter": atr_filter,
            "stop": stop,
            "tp": tp
        })
        results.append(stats)

        tested += 1
        if tested % 50 == 0:
            print(f"VWAP tested {tested}... best equity so far: {max(r['equity'] for r in results):.4f}")

    return results

# =========================
# RUN ALL
# =========================
all_results = []
print("Testing EMA Morning...")
all_results.extend(test_ema_strategy())

print("Testing ORB...")
all_results.extend(test_orb_strategy())

print("Testing VWAP Reclaim...")
all_results.extend(test_vwap_reclaim_strategy())

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["equity", "profit_factor", "max_drawdown"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("\nTOP 15 OVERALL STRATEGY RESULTS\n")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("strategy_comparison_results.csv", index=False)
print("\nSaved strategy_comparison_results.csv")
