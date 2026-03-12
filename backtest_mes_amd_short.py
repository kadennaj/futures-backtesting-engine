import pandas as pd
import numpy as np

print("MES AMD Short Strategy")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("mes_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")

df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp_ny"].dt.date
df["time"] = df["timestamp_ny"].dt.time

# =========================
# SETTINGS
# =========================
SESSION_START = "09:40"
SESSION_END = "11:30"

ACCUM_BARS = 10                  # lookback bars for accumulation box
MAX_RANGE_PCT = 0.003          # accumulation must be tighter than 0.25%
SWEEP_BUFFER_PCT = 0.0002       # price must sweep above box by at least 0.04%
STOP_BUFFER_PCT = 0.0004        # stop slightly above sweep high
R_MULTIPLE = 2.0                # target = 2R

COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# SESSION FILTER
# =========================
session_start = pd.to_datetime(SESSION_START).time()
session_end = pd.to_datetime(SESSION_END).time()

df["in_session"] = (df["time"] >= session_start) & (df["time"] < session_end)

# =========================
# BUILD ACCUMULATION BOX
# =========================
rolling_high = df["high"].rolling(ACCUM_BARS).max().shift(1)
rolling_low = df["low"].rolling(ACCUM_BARS).min().shift(1)

df["accum_high"] = rolling_high
df["accum_low"] = rolling_low
df["accum_mid"] = (df["accum_high"] + df["accum_low"]) / 2.0
df["accum_range"] = df["accum_high"] - df["accum_low"]
df["accum_range_pct"] = df["accum_range"] / df["close"]

# tight box = accumulation
df["is_accumulation"] = df["accum_range_pct"] < MAX_RANGE_PCT

# =========================
# AMD SHORT ENTRY LOGIC
# =========================
# Manipulation:
# price sweeps above accumulation high
df["sweep_above"] = df["high"] > (df["accum_high"] * (1.0 + SWEEP_BUFFER_PCT))

# Distribution confirmation:
# close comes back below accumulation high
df["rejection_bar"] = (
    (df["high"] > (df["accum_high"] * (1.0 + SWEEP_BUFFER_PCT))) &
    (df["close"] < df["accum_high"])
)

df["short_entry_signal"] = (
    df["in_session"] &
    df["is_accumulation"] &
    df["rejection_bar"]
).astype(int)
# =========================
# TRADE ENGINE
# =========================
position = 0
entry_price = 0.0
entry_time = None
entry_date = None
stop_price = 0.0
target_price = 0.0

signals = []
strategy_returns = []
exit_reasons = []
trade_log = []

for i in range(len(df)):
    signal = 0
    ret = 0.0
    exit_reason = ""

    current_close = df.at[i, "close"]
    current_high = df.at[i, "high"]
    current_low = df.at[i, "low"]
    current_time = df.at[i, "time"]
    current_ts = df.at[i, "timestamp_ny"]
    current_date = df.at[i, "date"]

    if i == 0:
        signals.append(signal)
        strategy_returns.append(ret)
        exit_reasons.append(exit_reason)
        continue

    prev_close_bar = df.at[i - 1, "close"]

    # Mark to market for short
    if position == -1:
        ret = (prev_close_bar / current_close) - 1.0

    # Force flat on date change
    if position == -1 and current_date != entry_date:
        ret = -EXIT_COST_PCT
        trade_log.append({
            "side": "SHORT",
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": df.at[i - 1, "timestamp_ny"],
            "exit_price": prev_close_bar,
            "exit_reason": "DAY_CHANGE_FORCE_EXIT",
        })
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_date = None
        stop_price = 0.0
        target_price = 0.0

    # Exit logic
    if position == -1:
        session_exit = current_time >= session_end

        if current_high >= stop_price:
            ret = (prev_close_bar / stop_price) - 1.0 - EXIT_COST_PCT
            exit_reason = "STOP"
            trade_log.append({
                "side": "SHORT",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": stop_price,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_date = None
            stop_price = 0.0
            target_price = 0.0

        elif current_low <= target_price:
            ret = (prev_close_bar / target_price) - 1.0 - EXIT_COST_PCT
            exit_reason = "TARGET"
            trade_log.append({
                "side": "SHORT",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": target_price,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_date = None
            stop_price = 0.0
            target_price = 0.0

        elif session_exit:
            ret = (prev_close_bar / current_close) - 1.0 - EXIT_COST_PCT
            exit_reason = "TIME"
            trade_log.append({
                "side": "SHORT",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": current_close,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_date = None
            stop_price = 0.0
            target_price = 0.0

    # Entry logic
    if position == 0 and df.at[i, "short_entry_signal"] == 1:
        entry_price = current_close
        entry_time = current_ts
        entry_date = current_date

        # stop above manipulation high
        raw_stop = current_high * (1.0 + STOP_BUFFER_PCT)
        risk = raw_stop - entry_price

        # skip broken setups with no valid risk
        if risk > 0:
            stop_price = raw_stop
            target_price = entry_price - (risk * R_MULTIPLE)
            position = -1
            signal = -1
            ret -= ENTRY_COST_PCT

    elif position == -1:
        signal = -1

    signals.append(signal)
    strategy_returns.append(ret)
    exit_reasons.append(exit_reason)

# =========================
# RESULTS
# =========================
df["signal"] = signals
df["strategy_return"] = strategy_returns
df["exit_reason"] = exit_reasons
df["market_return"] = df["close"].pct_change().fillna(0)

df["strategy_equity"] = (1 + pd.Series(df["strategy_return"]).fillna(0)).cumprod()
df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()

trade_df = pd.DataFrame(trade_log)

if not trade_df.empty:
    trade_df["pnl_pct"] = (trade_df["entry_price"] / trade_df["exit_price"]) - 1.0
else:
    trade_df = pd.DataFrame(columns=["side", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "pnl_pct"])

closed_trades = len(trade_df)
wins = (trade_df["pnl_pct"] > 0).sum() if closed_trades > 0 else 0
losses = (trade_df["pnl_pct"] <= 0).sum() if closed_trades > 0 else 0
win_rate = wins / closed_trades if closed_trades > 0 else 0.0

avg_win = trade_df.loc[trade_df["pnl_pct"] > 0, "pnl_pct"].mean() if wins > 0 else 0.0
avg_loss = trade_df.loc[trade_df["pnl_pct"] <= 0, "pnl_pct"].mean() if losses > 0 else 0.0

gross_win = trade_df.loc[trade_df["pnl_pct"] > 0, "pnl_pct"].sum() if wins > 0 else 0.0
gross_loss = abs(trade_df.loc[trade_df["pnl_pct"] <= 0, "pnl_pct"].sum()) if losses > 0 else 0.0
profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

running_max = df["strategy_equity"].cummax()
drawdown = df["strategy_equity"] / running_max - 1.0
max_drawdown = drawdown.min()

entries = ((df["signal"] == -1) & (df["signal"].shift(1).fillna(0) == 0)).sum()
exits = closed_trades

print("Final Buy & Hold Equity:", round(df["buy_hold_equity"].iloc[-1], 4))
print("Final Strategy Equity:", round(df["strategy_equity"].iloc[-1], 4))
print("Entries:", int(entries))
print("Exits:", int(exits))
print("Closed Trades:", int(closed_trades))
print("Win Rate:", round(win_rate * 100, 2), "%")
print("Avg Win:", round(float(avg_win), 5))
print("Avg Loss:", round(float(avg_loss), 5))
print("Profit Factor:", round(float(profit_factor), 3))
print("Max Drawdown:", round(float(max_drawdown), 4))

print("\nExit reason counts:")
print(trade_df["exit_reason"].value_counts() if not trade_df.empty else "No trades")

print("\nFirst 10 trades:")
if not trade_df.empty:
    print(trade_df.head(10).to_string(index=False))
else:
    print("No trades")

df.to_csv("mes_amd_short_results.csv", index=False)
trade_df.to_csv("mes_amd_short_trade_log.csv", index=False)

print("\nSaved mes_amd_short_results.csv")
print("Saved mes_amd_short_trade_log.csv")
