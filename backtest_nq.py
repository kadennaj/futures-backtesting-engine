import pandas as pd
import numpy as np

# =========================
# SETTINGS
# =========================
FAST_EMA = 16
SLOW_EMA = 21
TREND_EMA = 250
ATR_PERIOD = 14

STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.008

SESSION_START = "09:35"
SESSION_END = "12:00"

# Keep same friction model for quick comparison
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

ATR_FILTER_PCT = 0.0009
EMA_SEPARATION_PCT = 0.0002

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
# INDICATORS
# =========================
df["ema_fast"] = df["close"].ewm(span=FAST_EMA, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=SLOW_EMA, adjust=False).mean()
df["ema_trend"] = df["close"].ewm(span=TREND_EMA, adjust=False).mean()

prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()
df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
df["atr_pct"] = df["atr"] / df["close"]

# =========================
# ENTRY CONDITIONS
# =========================
session_start = pd.to_datetime(SESSION_START).time()
session_end = pd.to_datetime(SESSION_END).time()

trend_filter = df["close"] > df["ema_trend"]
momentum_filter = ((df["ema_fast"] - df["ema_slow"]) / df["close"]) > EMA_SEPARATION_PCT
vol_filter = df["atr_pct"] > ATR_FILTER_PCT
time_filter = (df["time"] >= session_start) & (df["time"] < session_end)

df["entry_signal"] = (trend_filter & momentum_filter & vol_filter & time_filter).astype(int)

# =========================
# TRADE ENGINE
# =========================
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
        stop_price = entry_price * (1 - STOP_LOSS_PCT)
        target_price = entry_price * (1 + TAKE_PROFIT_PCT)

        hit_stop = current_low <= stop_price
        hit_target = current_high >= target_price
        session_exit = current_time >= session_end

        ema_sep_now = (df.at[i, "ema_fast"] - df.at[i, "ema_slow"]) / current_close
        momentum_exit = ema_sep_now < 0

        if hit_stop:
            ret = (stop_price / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
        elif hit_target:
            ret = (target_price / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
        elif session_exit:
            ret = (current_close / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0
        elif momentum_exit:
            ret = (current_close / prev_close_bar) - 1 - EXIT_COST_PCT
            position = 0

    if position == 0 and df.at[i, "entry_signal"] == 1:
        position = 1
        entry_price = current_close
        signal = 1
        ret -= ENTRY_COST_PCT
    elif position == 1:
        signal = 1

    signals.append(signal)
    trade_returns.append(ret)

# =========================
# RESULTS
# =========================
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

avg_win = sum(t for t in trade_list if t > 0) / max(1, sum(1 for t in trade_list if t > 0))
avg_loss = sum(t for t in trade_list if t <= 0) / max(1, sum(1 for t in trade_list if t <= 0))
profit_factor = (
    sum(t for t in trade_list if t > 0) / abs(sum(t for t in trade_list if t < 0))
    if any(t < 0 for t in trade_list) else float("inf")
)

print("NQ Quick Test")
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

df.to_csv("nq_backtest_results.csv", index=False)
print("\nSaved nq_backtest_results.csv")
