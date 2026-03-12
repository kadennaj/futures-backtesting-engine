import pandas as pd
import numpy as np
from itertools import product

print("FAST OPTIMIZER - Combined Long Trend + AMD Short")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("spy_data_massive_full.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
df = df.sort_values("timestamp").reset_index(drop=True)

df["date"] = df["timestamp_ny"].dt.date
df["time"] = df["timestamp_ny"].dt.time

# =========================
# FIXED BASE SETTINGS
# =========================
FAST_EMA = 20
SLOW_EMA = 27
TREND_EMA = 250
ATR_PERIOD = 14

LONG_SESSION_START = "09:35"
SHORT_SESSION_START = "09:40"

MAX_LONG_TRADES_PER_DAY = 3
MAX_SHORT_TRADES_PER_DAY = 1

COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST = COMMISSION_PCT + SLIPPAGE_PCT

SWEEP_BUFFER_PCT = 0.0002

# =========================
# PRECOMPUTE INDICATORS ONCE
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
df["sep"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

# numpy arrays for speed
close_arr = df["close"].to_numpy()
high_arr = df["high"].to_numpy()
low_arr = df["low"].to_numpy()
atr_pct_arr = df["atr_pct"].to_numpy()
sep_arr = df["sep"].to_numpy()
ema_trend_arr = df["ema_trend"].to_numpy()

time_arr = df["time"].to_numpy()
date_arr = df["date"].to_numpy()
ts_arr = df["timestamp_ny"].to_numpy()

market_return = df["close"].pct_change().fillna(0).to_numpy()
buy_hold_equity = np.cumprod(1 + market_return)[-1]

# =========================
# PARAM GRID
# =========================
LONG_SESSION_END_VALS = ["12:00", "13:00", "15:30"]
LONG_ATR_FILTER_VALS = [0.0008, 0.0010, 0.0012]
LONG_EMA_SEP_VALS = [0.00008, 0.00012, 0.00015]
LONG_STOP_VALS = [0.004, 0.005]
LONG_TARGET_VALS = [0.008, 0.010, 0.012]

SHORT_SESSION_END_VALS = ["11:30", "12:00"]
SHORT_ACCUM_BARS_VALS = [8, 10]
SHORT_MAX_RANGE_VALS = [0.0025, 0.0030]
SHORT_STOP_BUFFER_VALS = [0.0003, 0.0004]
SHORT_R_MULTIPLE_VALS = [1.5, 2.0]

# total = 3*3*3*2*3*2*2*2*2*2 = 5184 combos
param_grid = list(product(
    LONG_SESSION_END_VALS,
    LONG_ATR_FILTER_VALS,
    LONG_EMA_SEP_VALS,
    LONG_STOP_VALS,
    LONG_TARGET_VALS,
    SHORT_SESSION_END_VALS,
    SHORT_ACCUM_BARS_VALS,
    SHORT_MAX_RANGE_VALS,
    SHORT_STOP_BUFFER_VALS,
    SHORT_R_MULTIPLE_VALS,
))

print(f"Total parameter combinations: {len(param_grid)}")

# =========================
# HELPERS
# =========================
def make_time_mask(start_str: str, end_str: str):
    start_t = pd.to_datetime(start_str).time()
    end_t = pd.to_datetime(end_str).time()
    return (time_arr >= start_t) & (time_arr < end_t), end_t

def run_backtest(
    long_session_end_str,
    long_atr_filter,
    long_ema_sep,
    long_stop_pct,
    long_target_pct,
    short_session_end_str,
    short_accum_bars,
    short_max_range_pct,
    short_stop_buffer_pct,
    short_r_multiple,
):
    long_mask, long_end_t = make_time_mask(LONG_SESSION_START, long_session_end_str)
    short_mask, short_end_t = make_time_mask(SHORT_SESSION_START, short_session_end_str)

    # short accumulation features for this combo
    range_high = pd.Series(high_arr).rolling(short_accum_bars).max().shift(1).to_numpy()
    range_low = pd.Series(low_arr).rolling(short_accum_bars).min().shift(1).to_numpy()
    range_size = range_high - range_low
    range_pct = range_size / close_arr

    long_signal = (
        (close_arr > ema_trend_arr) &
        (sep_arr > long_ema_sep) &
        (atr_pct_arr > long_atr_filter) &
        long_mask
    )

    short_signal = (
        short_mask &
        (range_pct < short_max_range_pct) &
        (high_arr > (range_high * (1.0 + SWEEP_BUFFER_PCT))) &
        (close_arr < range_high)
    )

    position = 0
    entry_price = 0.0
    entry_date = None
    entry_side = ""

    short_stop = 0.0
    short_target = 0.0

    daily_long = 0
    daily_short = 0
    current_day = None

    returns = np.zeros(len(close_arr), dtype=float)
    trade_log = []

    for i in range(len(close_arr)):
        if i == 0:
            current_day = date_arr[i]
            continue

        price = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        day = date_arr[i]
        t = time_arr[i]
        ts = ts_arr[i]
        prev_price = close_arr[i - 1]

        if day != current_day:
            current_day = day
            daily_long = 0
            daily_short = 0

        exited_this_bar = False
        ret = 0.0

        # mark to market
        if position == 1:
            ret = (price / prev_price) - 1.0
        elif position == -1:
            ret = (prev_price / price) - 1.0

        # forced flat on day change
        if position != 0 and entry_date is not None and day != entry_date:
            ret = -EXIT_COST
            trade_log.append({
                "side": entry_side,
                "entry_price": entry_price,
                "exit_price": prev_price,
                "exit_reason": "DAY_CHANGE_FORCE_EXIT",
            })
            position = 0
            entry_price = 0.0
            entry_date = None
            entry_side = ""
            short_stop = 0.0
            short_target = 0.0
            exited_this_bar = True

        # long exit
        if position == 1:
            stop_price = entry_price * (1.0 - long_stop_pct)
            target_price = entry_price * (1.0 + long_target_pct)

            if low <= stop_price:
                ret = (stop_price / prev_price) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "LONG",
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "exit_reason": "LONG_STOP",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                exited_this_bar = True

            elif high >= target_price:
                ret = (target_price / prev_price) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "LONG",
                    "entry_price": entry_price,
                    "exit_price": target_price,
                    "exit_reason": "LONG_TARGET",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                exited_this_bar = True

            elif t >= long_end_t:
                ret = (price / prev_price) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "LONG",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "exit_reason": "LONG_TIME",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                exited_this_bar = True

        # short exit
        elif position == -1:
            if high >= short_stop:
                ret = (prev_price / short_stop) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": short_stop,
                    "exit_reason": "SHORT_STOP",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                short_stop = 0.0
                short_target = 0.0
                exited_this_bar = True

            elif low <= short_target:
                ret = (prev_price / short_target) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": short_target,
                    "exit_reason": "SHORT_TARGET",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                short_stop = 0.0
                short_target = 0.0
                exited_this_bar = True

            elif t >= short_end_t:
                ret = (prev_price / price) - 1.0 - EXIT_COST
                trade_log.append({
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "exit_reason": "SHORT_TIME",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_side = ""
                short_stop = 0.0
                short_target = 0.0
                exited_this_bar = True

        # entries
        if position == 0 and (not exited_this_bar):
            if long_signal[i] and daily_long < MAX_LONG_TRADES_PER_DAY:
                position = 1
                entry_price = price
                entry_date = day
                entry_side = "LONG"
                daily_long += 1
                ret -= ENTRY_COST

            elif short_signal[i] and daily_short < MAX_SHORT_TRADES_PER_DAY:
                raw_stop = high * (1.0 + short_stop_buffer_pct)
                risk = raw_stop - price

                if risk > 0:
                    position = -1
                    entry_price = price
                    entry_date = day
                    entry_side = "SHORT"
                    short_stop = raw_stop
                    short_target = price - (risk * short_r_multiple)
                    daily_short += 1
                    ret -= ENTRY_COST

        returns[i] = ret

    equity_curve = np.cumprod(1 + returns)
    final_equity = float(equity_curve[-1])

    trade_df = pd.DataFrame(trade_log)
    if trade_df.empty:
        return {
            "equity": final_equity,
            "closed_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    trade_df["pnl_pct"] = np.where(
        trade_df["side"] == "LONG",
        (trade_df["exit_price"] / trade_df["entry_price"]) - 1.0,
        (trade_df["entry_price"] / trade_df["exit_price"]) - 1.0,
    )

    closed_trades = len(trade_df)
    wins = (trade_df["pnl_pct"] > 0).sum()
    losses = (trade_df["pnl_pct"] <= 0).sum()
    win_rate = wins / closed_trades if closed_trades else 0.0

    gross_win = trade_df.loc[trade_df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_loss = abs(trade_df.loc[trade_df["pnl_pct"] <= 0, "pnl_pct"].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    avg_win = trade_df.loc[trade_df["pnl_pct"] > 0, "pnl_pct"].mean() if wins > 0 else 0.0
    avg_loss = trade_df.loc[trade_df["pnl_pct"] <= 0, "pnl_pct"].mean() if losses > 0 else 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min())

    long_trades = int((trade_df["side"] == "LONG").sum())
    short_trades = int((trade_df["side"] == "SHORT").sum())

    return {
        "equity": final_equity,
        "closed_trades": int(closed_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": max_drawdown,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "long_trades": long_trades,
        "short_trades": short_trades,
    }

# =========================
# OPT LOOP
# =========================
results = []
best_equity = -999
tested = 0

for params in param_grid:
    (
        long_session_end,
        long_atr_filter,
        long_ema_sep,
        long_stop_pct,
        long_target_pct,
        short_session_end,
        short_accum_bars,
        short_max_range_pct,
        short_stop_buffer_pct,
        short_r_multiple,
    ) = params

    stats = run_backtest(
        long_session_end,
        long_atr_filter,
        long_ema_sep,
        long_stop_pct,
        long_target_pct,
        short_session_end,
        short_accum_bars,
        short_max_range_pct,
        short_stop_buffer_pct,
        short_r_multiple,
    )

    results.append({
        "long_session_end": long_session_end,
        "long_atr_filter": long_atr_filter,
        "long_ema_sep": long_ema_sep,
        "long_stop_pct": long_stop_pct,
        "long_target_pct": long_target_pct,
        "short_session_end": short_session_end,
        "short_accum_bars": short_accum_bars,
        "short_max_range_pct": short_max_range_pct,
        "short_stop_buffer_pct": short_stop_buffer_pct,
        "short_r_multiple": short_r_multiple,
        **stats,
    })

    tested += 1
    if stats["equity"] > best_equity:
        best_equity = stats["equity"]

    if tested % 100 == 0:
        print(f"Tested {tested} combos... best equity so far: {best_equity:.4f}")

results_df = pd.DataFrame(results)

# quality filters
results_df = results_df[results_df["closed_trades"] >= 20].copy()

# rank by actual usefulness
results_df["score"] = (
    (results_df["equity"] * 100)
    + (results_df["win_rate"] * 10)
    + (results_df["profit_factor"].clip(upper=5) * 8)
    + (results_df["max_drawdown"] * 100)
)

results_df = results_df.sort_values(
    by=["score", "equity", "profit_factor", "closed_trades"],
    ascending=[False, False, False, False]
).reset_index(drop=True)

print("\nTOP 20 FAST COMBINED RESULTS\n")
if len(results_df) == 0:
    print("No strategies passed the filters.")
else:
    print(results_df.head(20).to_string(index=False))

results_df.to_csv("fast_combined_optimization_results.csv", index=False)
print("\nSaved fast_combined_optimization_results.csv")
print(f"Buy & Hold Equity on dataset: {buy_hold_equity:.4f}")
