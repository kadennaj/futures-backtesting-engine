import pandas as pd
import numpy as np
from itertools import product

# ========================
# LOAD DATA
# ========================
df = pd.read_csv("spy_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
df = df.sort_values("timestamp").reset_index(drop=True)
df["time"] = df["timestamp_ny"].dt.time

# ========================
# COSTS
# ========================
COMMISSION = 0.00025
SLIPPAGE = 0.00025
ENTRY_COST = COMMISSION + SLIPPAGE
EXIT_COST = COMMISSION + SLIPPAGE

# ========================
# ATR
# ========================
prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()

df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()
df["atr_pct"] = df["atr"] / df["close"]

# ========================
# ORIGINAL PARAMETER GRID
# ========================
fast_ema_vals = [8, 12, 16, 20]
slow_ema_vals = [21, 30, 40, 50]
trend_ema_vals = [100, 150, 200]

session_end_vals = ["11:30", "12:00", "12:30"]

atr_filter_vals = [0.0007, 0.0009, 0.0011]

ema_sep_vals = [0.0001, 0.00015, 0.0002]

stop_vals = [0.003, 0.004, 0.005]

tp_vals = [0.006, 0.008, 0.010]

# ========================
# BACKTEST FUNCTION
# ========================
def run_backtest(fast, slow, trend, session_end_str, atr_filter, ema_sep, stop, tp):
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

    entry = trend_filter & momentum_filter & vol_filter & time_filter

    position = 0
    entry_price = 0.0
    returns = []

    for i in range(len(df_bt)):
        r = 0.0
        price = df_bt["close"].iloc[i]

        if i == 0:
            returns.append(r)
            continue

        prev_price = df_bt["close"].iloc[i - 1]

        if position == 1:
            r = price / prev_price - 1

        if position == 1:
            stop_price = entry_price * (1 - stop)
            tp_price = entry_price * (1 + tp)

            low = df_bt["low"].iloc[i]
            high = df_bt["high"].iloc[i]
            now_time = df_bt["time"].iloc[i]

            if low <= stop_price:
                r = stop_price / prev_price - 1 - EXIT_COST
                position = 0
            elif high >= tp_price:
                r = tp_price / prev_price - 1 - EXIT_COST
                position = 0
            elif now_time >= session_end:
                r = price / prev_price - 1 - EXIT_COST
                position = 0

        if position == 0 and entry.iloc[i]:
            position = 1
            entry_price = price
            r -= ENTRY_COST

        returns.append(r)

    equity = (1 + pd.Series(returns)).cumprod()
    trades = int(((entry.astype(int).diff().fillna(0) == 1)).sum())
    max_dd = (equity / equity.cummax() - 1).min()

    return float(equity.iloc[-1]), trades, float(max_dd)

# ========================
# OPTIMIZATION LOOP
# ========================
results = []

combos = list(product(
    fast_ema_vals,
    slow_ema_vals,
    trend_ema_vals,
    session_end_vals,
    atr_filter_vals,
    ema_sep_vals,
    stop_vals,
    tp_vals
))

total = len(combos)
print(f"Total raw combinations: {total}")

tested = 0

for idx, (fast, slow, trend, sess, atrf, emas, stop, tp) in enumerate(combos, start=1):
    if fast >= slow:
        continue

    equity, trades, dd = run_backtest(fast, slow, trend, sess, atrf, emas, stop, tp)

    results.append([
        fast, slow, trend, sess, atrf, emas, stop, tp,
        equity, trades, dd
    ])

    tested += 1

    if tested % 100 == 0:
        print(f"Tested {tested} valid strategies... latest equity={equity:.4f}")

results_df = pd.DataFrame(results, columns=[
    "fast", "slow", "trend", "session_end", "atr_filter",
    "ema_sep", "stop", "tp",
    "equity", "trades", "drawdown"
])

results_df = results_df.sort_values("equity", ascending=False)

print("\nTop 15 Results\n")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("optimization_results.csv", index=False)
print("\nSaved optimization_results.csv")
