import pandas as pd
import numpy as np

# =========================
# SETTINGS
# =========================
OPEN_RANGE_START = "09:30"
OPEN_RANGE_END = "09:45"

ENTRY_START = "09:45"
SESSION_END = "12:00"

ATR_PERIOD = 14
ATR_FILTER_PCT = 0.0007

# NQ point-based exits
STOP_POINTS = 40
TARGET_POINTS = 100

# confirmation settings
BREAKOUT_BUFFER_POINTS = 10
RETEST_BUFFER_POINTS = 5

# friction model
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("nq_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")

df = df.sort_values("timestamp").reset_index(drop=True)

df["date"] = df["timestamp_ny"].dt.date
df["time"] = df["timestamp_ny"].dt.time

# =========================
# ATR
# =========================
prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()

df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
df["atr_pct"] = df["atr"] / df["close"]

# =========================
# OPENING RANGE
# =========================
or_start = pd.to_datetime(OPEN_RANGE_START).time()
or_end = pd.to_datetime(OPEN_RANGE_END).time()

opening_mask = (df["time"] >= or_start) & (df["time"] < or_end)

opening_range = (
    df.loc[opening_mask]
    .groupby("date")
    .agg(
        or_high=("high", "max"),
        or_low=("low", "min")
    )
    .reset_index()
)

df = df.merge(opening_range, on="date", how="left")

# =========================
# ENTRY CONDITIONS
# =========================
entry_start = pd.to_datetime(ENTRY_START).time()
session_end = pd.to_datetime(SESSION_END).time()

time_filter = (df["time"] >= entry_start) & (df["time"] < session_end)
vol_filter = df["atr_pct"] > ATR_FILTER_PCT

# Breakout and confirmation logic
df["breakout_level"] = df["or_high"] + BREAKOUT_BUFFER_POINTS
df["retest_level"] = df["or_high"] + RETEST_BUFFER_POINTS

# first, we want market above breakout level
above_breakout_now = df["close"] > df["breakout_level"]
above_breakout_prev = df["close"].shift(1) > df["breakout_level"].shift(1)

# current bar should also hold above the retest level
holds_retest = df["low"] > df["retest_level"]

# confirmed continuation: previous bar already closed above breakout,
# and current bar is still holding above OR high/retest zone
confirmed_breakout = above_breakout_prev & holds_retest & (df["close"] > df["retest_level"])

df["entry_signal"] = (time_filter & vol_filter & confirmed_breakout).astype(int)

# =========================
# TRADE ENGINE
# =========================
position = 0
entry_price = 0.0
entry_day = None

signals = []
trade_returns = []
exit_reasons = []

for i in range(len(df)):
    signal = 0
    ret = 0.0
    exit_reason = ""

    current_close = df.at[i, "close"]
    current_high = df.at[i, "high"]
    current_low = df.at[i, "low"]
    current_time = df.at[i, "time"]
    current_date = df.at[i, "date"]
    current_or_high = df.at[i, "or_high"]

    if i == 0:
        signals.append(signal)
        trade_returns.append(ret)
        exit_reasons.append(exit_reason)
        continue

    prev_close_bar = df.at[i - 1, "close"]

    if position == 1:
        ret = (current_close / prev_close_bar) - 1

    if position == 1:
        stop_price = entry_price - STOP_POINTS
        target_price = entry_price + TARGET_POINTS

        hit_stop = current_low <= stop_price
        hit_target = current_high >= target_price
        session_exit = current_time >= session_end
        new_day_exit = current_date != entry_day

        # if it loses OR high decisively, get out
        failed_retest_exit = current_close < current_or_high

        if hit_stop:
            ret = (stop_price / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
            exit_reason = "STOP"
        elif hit_target:
            ret = (target_price / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
            exit_reason = "TARGET"
        elif session_exit or new_day_exit:
            ret = (current_close / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
            exit_reason = "TIME"
        elif failed_retest_exit:
            ret = (current_close / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
            exit_reason = "FAILED_RETEST"

    # only one entry per day
    if position == 0 and df.at[i, "entry_signal"] == 1 and entry_day != current_date:
        position = 1
        entry_price = current_close
        entry_day = current_date
        signal = 1
        ret -= ENTRY_COST_PCT
    elif position == 1:
        signal = 1

    signals.append(signal)
    trade_returns.append(ret)
    exit_reasons.append(exit_reason)

    if position == 0 and current_time >= session_end:
        entry_day = None

# =========================
# RESULTS
# =========================
df["signal"] = signals
df["strategy_return"] = trade_returns
df["exit_reason"] = exit_reasons
df["market_return"] = df["close"].pct_change().fillna(0)

df["strategy_equity"] = (1 + pd.Series(df["strategy_return"]).fillna(0)).cumprod()
df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()

entries = ((df["signal"] == 1) & (df["signal"].shift(1).fillna(0) == 0)).sum()
exits = ((df["signal"] == 0) & (df["signal"].shift(1).fillna(0) == 1)).sum()

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

avg_win = sum(t for t in trade_list if t > 0) / max(1, sum(1 for t in trade_list if t > 0))
avg_loss = sum(t for t in trade_list if t <= 0) / max(1, sum(1 for t in trade_list if t <= 0))
profit_factor = (
    sum(t for t in trade_list if t > 0) / abs(sum(t for t in trade_list if t < 0))
    if any(t < 0 for t in trade_list) else float("inf")
)

print("NQ ORB Confirmed Test")
print("Final Buy & Hold Equity:", round(df["buy_hold_equity"].iloc[-1], 4))
print("Final Strategy Equity:", round(df["strategy_equity"].iloc[-1], 4))
print("Entries:", int(entries))
print("Exits:", int(exits))
print("Closed Trades:", len(trade_list))
print("Win Rate:", round(win_rate * 100, 2), "%")
print("Avg Win:", round(avg_win, 5))
print("Avg Loss:", round(avg_loss, 5))
print("Profit Factor:", round(profit_factor, 3))
print("Max Drawdown:", round(max_drawdown, 4))

print("\nExit reason counts:")
print(df["exit_reason"].value_counts())

df.to_csv("nq_orb_confirmed_results.csv", index=False)
print("\nSaved nq_orb_confirmed_results.csv")
