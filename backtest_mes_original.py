import pandas as pd
import numpy as np

print("MES Strategy - Long + Short (Strict Intraday)")

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
FAST_EMA = 20
SLOW_EMA = 27
TREND_EMA = 250
ATR_PERIOD = 14

SESSION_START = "09:40"
SESSION_END = "12:00"

ATR_FILTER_PCT = 0.0012
EMA_SEPARATION_PCT = 0.00015

STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.008

COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

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

sep_series = (df["ema_fast"] - df["ema_slow"]) / df["close"]

# =========================
# ENTRY CONDITIONS
# =========================
session_start = pd.to_datetime(SESSION_START).time()
session_end = pd.to_datetime(SESSION_END).time()

time_filter = (df["time"] >= session_start) & (df["time"] < session_end)

long_entry_signal = (
    (df["close"] > df["ema_trend"]) &
    (sep_series > EMA_SEPARATION_PCT) &
    (df["atr_pct"] > ATR_FILTER_PCT) &
    time_filter
)

short_entry_signal = (
    (df["close"] < df["ema_trend"]) &
    ((-sep_series) > EMA_SEPARATION_PCT) &
    (df["atr_pct"] > ATR_FILTER_PCT) &
    time_filter
)

# =========================
# TRADE ENGINE
# =========================
position = 0   # 1 = long, -1 = short, 0 = flat
entry_price = 0.0
entry_time = None
entry_side = ""
entry_date = None

signals = []
strategy_returns = []
exit_reasons_series = []
position_side_series = []

trade_log = []

long_entries = 0
short_entries = 0

for i in range(len(df)):
    signal = 0
    ret = 0.0
    exit_reason = ""
    side_label = ""

    current_close = df.at[i, "close"]
    current_high = df.at[i, "high"]
    current_low = df.at[i, "low"]
    current_time = df.at[i, "time"]
    current_ts = df.at[i, "timestamp_ny"]
    current_date = df.at[i, "date"]

    if i == 0:
        signals.append(signal)
        strategy_returns.append(ret)
        exit_reasons_series.append(exit_reason)
        position_side_series.append(side_label)
        continue

    prev_close_bar = df.at[i - 1, "close"]
    prev_date = df.at[i - 1, "date"]

    # Mark to market
    if position == 1:
        ret = (current_close / prev_close_bar) - 1.0
    elif position == -1:
        ret = (prev_close_bar / current_close) - 1.0

    # Force flat on day change first
    if position != 0 and current_date != entry_date:
        if position == 1:
            ret = (prev_close_bar / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_price = prev_close_bar
        else:
            ret = (prev_close_bar / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_price = prev_close_bar

        exit_reason = "DAY_CHANGE_FORCE_EXIT"
        trade_log.append({
            "side": entry_side,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": df.at[i - 1, "timestamp_ny"],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
        })

        position = 0
        entry_price = 0.0
        entry_time = None
        entry_side = ""
        entry_date = None

    # =====================
    # EXIT LOGIC - LONG
    # =====================
    if position == 1:
        stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
        target_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
        momentum_exit = sep_series.iloc[i] < 0
        session_exit = current_time >= session_end

        if current_low <= stop_price:
            ret = (stop_price / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_reason = "STOP"
            trade_log.append({
                "side": "LONG",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": stop_price,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_side = ""
            entry_date = None

        elif current_high >= target_price:
            ret = (target_price / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_reason = "TARGET"
            trade_log.append({
                "side": "LONG",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": target_price,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_side = ""
            entry_date = None

        elif session_exit:
            ret = (current_close / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_reason = "TIME"
            trade_log.append({
                "side": "LONG",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": current_close,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_side = ""
            entry_date = None

        elif momentum_exit:
            ret = (current_close / prev_close_bar) - 1.0 - EXIT_COST_PCT
            exit_reason = "MOMENTUM_EXIT"
            trade_log.append({
                "side": "LONG",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": current_ts,
                "exit_price": current_close,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
            entry_side = ""
            entry_date = None

    # =====================
    # EXIT LOGIC - SHORT
    # =====================
    elif position == -1:
        stop_price = entry_price * (1.0 + STOP_LOSS_PCT)
        target_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
        momentum_exit = sep_series.iloc[i] > 0
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
            entry_side = ""
            entry_date = None

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
            entry_side = ""
            entry_date = None

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
            entry_side = ""
            entry_date = None

        elif momentum_exit:
            ret = (prev_close_bar / current_close) - 1.0 - EXIT_COST_PCT
            exit_reason = "MOMENTUM_EXIT"
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
            entry_side = ""
            entry_date = None

    # =====================
    # ENTRY LOGIC
    # =====================
    if position == 0:
        if long_entry_signal.iloc[i]:
            position = 1
            entry_price = current_close
            entry_time = current_ts
            entry_side = "LONG"
            entry_date = current_date
            signal = 1
            ret -= ENTRY_COST_PCT
            long_entries += 1

        elif short_entry_signal.iloc[i]:
            position = -1
            entry_price = current_close
            entry_time = current_ts
            entry_side = "SHORT"
            entry_date = current_date
            signal = -1
            ret -= ENTRY_COST_PCT
            short_entries += 1

    elif position == 1:
        signal = 1
    elif position == -1:
        signal = -1

    if position == 1:
        side_label = "LONG"
    elif position == -1:
        side_label = "SHORT"
    else:
        side_label = ""

    signals.append(signal)
    strategy_returns.append(ret)
    exit_reasons_series.append(exit_reason)
    position_side_series.append(side_label)

# =========================
# RESULTS
# =========================
df["signal"] = signals
df["strategy_return"] = strategy_returns
df["exit_reason"] = exit_reasons_series
df["position_side"] = position_side_series
df["market_return"] = df["close"].pct_change().fillna(0)

df["strategy_equity"] = (1 + pd.Series(df["strategy_return"]).fillna(0)).cumprod()
df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()

trade_df = pd.DataFrame(trade_log)

if not trade_df.empty:
    trade_df["pnl_pct"] = np.where(
        trade_df["side"] == "LONG",
        (trade_df["exit_price"] / trade_df["entry_price"]) - 1.0,
        (trade_df["entry_price"] / trade_df["exit_price"]) - 1.0,
    )
else:
    trade_df["pnl_pct"] = []

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
drawdown = df["strategy_equity"] / running_max - 1
max_drawdown = drawdown.min()

entries = long_entries + short_entries
exits = closed_trades

long_trade_df = trade_df[trade_df["side"] == "LONG"].copy() if not trade_df.empty else pd.DataFrame()
short_trade_df = trade_df[trade_df["side"] == "SHORT"].copy() if not trade_df.empty else pd.DataFrame()

long_win_rate = (long_trade_df["pnl_pct"] > 0).mean() if len(long_trade_df) > 0 else 0.0
short_win_rate = (short_trade_df["pnl_pct"] > 0).mean() if len(short_trade_df) > 0 else 0.0

print("Final Buy & Hold Equity:", round(df["buy_hold_equity"].iloc[-1], 4))
print("Final Strategy Equity:", round(df["strategy_equity"].iloc[-1], 4))
print("Entries:", int(entries))
print("Exits:", int(exits))
print("Closed Trades:", int(closed_trades))
print("Long Entries:", int(long_entries))
print("Short Entries:", int(short_entries))
print("Win Rate:", round(win_rate * 100, 2), "%")
print("Long Win Rate:", round(long_win_rate * 100, 2), "%")
print("Short Win Rate:", round(short_win_rate * 100, 2), "%")
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

df.to_csv("mes_backtest_results_long_short_fixed2.csv", index=False)
trade_df.to_csv("mes_trade_log_long_short_fixed2.csv", index=False)

print("\nSaved mes_backtest_results_long_short_fixed2.csv")
print("Saved mes_trade_log_long_short_fixed2.csv")
