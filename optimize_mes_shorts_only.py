import pandas as pd
import numpy as np
from itertools import product

# =========================
# LOAD MES DATA
# =========================
df = pd.read_csv("mes_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")

df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp_ny"].dt.date
df["time"] = df["timestamp_ny"].dt.time

df = df[["timestamp_ny", "date", "time", "close", "high", "low"]].copy()

# =========================
# SMALLER, FOCUSED GRID
# =========================
FAST_VALS = [18, 20, 22]
SLOW_VALS = [25, 27, 30]
TREND_VALS = [200, 250, 300]

SESSION_START_VALS = ["09:35", "09:40", "09:45"]
SESSION_END_VALS = ["11:30", "12:00"]

ATR_FILTER_VALS = [0.0010, 0.0012]
EMA_SEP_VALS = [0.00010, 0.00015, 0.00020]

STOP_VALS = [0.004, 0.005]
TP_VALS = [0.006, 0.008]

# =========================
# COSTS
# =========================
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# PRECOMPUTE ATR
# =========================
prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()

df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()
df["atr_pct"] = df["atr"] / df["close"]

# =========================
# PRECOMPUTE EMAS
# =========================
ema_cache = {}
for span in sorted(set(FAST_VALS + SLOW_VALS + TREND_VALS)):
    ema_cache[span] = df["close"].ewm(span=span, adjust=False).mean()

# =========================
# SHORT-ONLY BACKTEST
# =========================
def run_short_backtest(
    fast: int,
    slow: int,
    trend: int,
    session_start_str: str,
    session_end_str: str,
    atr_filter: float,
    ema_sep: float,
    stop_pct: float,
    tp_pct: float,
) -> dict:
    session_start = pd.to_datetime(session_start_str).time()
    session_end = pd.to_datetime(session_end_str).time()

    temp = df.copy()
    temp["ema_fast"] = ema_cache[fast]
    temp["ema_slow"] = ema_cache[slow]
    temp["ema_trend"] = ema_cache[trend]

    sep_series = (temp["ema_fast"] - temp["ema_slow"]) / temp["close"]

    short_entry_signal = (
        (temp["close"] < temp["ema_trend"]) &
        ((-sep_series) > ema_sep) &
        (temp["atr_pct"] > atr_filter) &
        (temp["time"] >= session_start) &
        (temp["time"] < session_end)
    )

    position = 0
    entry_price = 0.0
    entry_date = None
    entry_time = None

    returns = []
    trade_log = []

    for i in range(len(temp)):
        ret = 0.0

        current_close = temp.at[i, "close"]
        current_high = temp.at[i, "high"]
        current_low = temp.at[i, "low"]
        current_time = temp.at[i, "time"]
        current_date = temp.at[i, "date"]
        current_ts = temp.at[i, "timestamp_ny"]

        if i == 0:
            returns.append(ret)
            continue

        prev_close_bar = temp.at[i - 1, "close"]

        # mark-to-market while short
        if position == -1:
            ret = (prev_close_bar / current_close) - 1.0

        # force flat at day change using prior bar close
        if position == -1 and current_date != entry_date:
            ret = -EXIT_COST_PCT
            trade_log.append({
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": temp.at[i - 1, "timestamp_ny"],
                "exit_price": prev_close_bar,
                "exit_reason": "DAY_CHANGE_FORCE_EXIT",
            })
            position = 0
            entry_price = 0.0
            entry_date = None
            entry_time = None

        # normal exits
        if position == -1:
            stop_price = entry_price * (1.0 + stop_pct)
            target_price = entry_price * (1.0 - tp_pct)
            momentum_exit = sep_series.iloc[i] > 0
            session_exit = current_time >= session_end

            if current_high >= stop_price:
                ret = (prev_close_bar / stop_price) - 1.0 - EXIT_COST_PCT
                trade_log.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_ts,
                    "exit_price": stop_price,
                    "exit_reason": "STOP",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_time = None

            elif current_low <= target_price:
                ret = (prev_close_bar / target_price) - 1.0 - EXIT_COST_PCT
                trade_log.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_ts,
                    "exit_price": target_price,
                    "exit_reason": "TARGET",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_time = None

            elif session_exit:
                ret = (prev_close_bar / current_close) - 1.0 - EXIT_COST_PCT
                trade_log.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_ts,
                    "exit_price": current_close,
                    "exit_reason": "TIME",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_time = None

            elif momentum_exit:
                ret = (prev_close_bar / current_close) - 1.0 - EXIT_COST_PCT
                trade_log.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_ts,
                    "exit_price": current_close,
                    "exit_reason": "MOMENTUM_EXIT",
                })
                position = 0
                entry_price = 0.0
                entry_date = None
                entry_time = None

        # entry
        if position == 0 and short_entry_signal.iloc[i]:
            position = -1
            entry_price = current_close
            entry_date = current_date
            entry_time = current_ts
            ret -= ENTRY_COST_PCT

        returns.append(ret)

    equity_curve = (1 + pd.Series(returns).fillna(0)).cumprod()
    trade_df = pd.DataFrame(trade_log)

    if not trade_df.empty:
        trade_df["pnl_pct"] = (trade_df["entry_price"] / trade_df["exit_price"]) - 1.0
    else:
        trade_df = pd.DataFrame(columns=["entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "pnl_pct"])

    closed_trades = len(trade_df)
    wins = (trade_df["pnl_pct"] > 0).sum() if closed_trades > 0 else 0
    losses = (trade_df["pnl_pct"] <= 0).sum() if closed_trades > 0 else 0
    win_rate = wins / closed_trades if closed_trades > 0 else 0.0

    gross_win = trade_df.loc[trade_df["pnl_pct"] > 0, "pnl_pct"].sum() if wins > 0 else 0.0
    gross_loss = abs(trade_df.loc[trade_df["pnl_pct"] <= 0, "pnl_pct"].sum()) if losses > 0 else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    score = (
        (equity_curve.iloc[-1] * 100.0)
        + (min(profit_factor, 5.0) * 5.0)
        + (win_rate * 10.0)
        + (max_drawdown * 50.0)
    )

    return {
        "equity": float(equity_curve.iloc[-1]),
        "closed_trades": int(closed_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "score": float(score),
    }

# =========================
# OPTIMIZATION LOOP
# =========================
results = []

combos = list(product(
    FAST_VALS,
    SLOW_VALS,
    TREND_VALS,
    SESSION_START_VALS,
    SESSION_END_VALS,
    ATR_FILTER_VALS,
    EMA_SEP_VALS,
    STOP_VALS,
    TP_VALS,
))

print(f"Total raw combinations: {len(combos)}")

tested = 0

for fast, slow, trend, session_start, session_end, atr_filter, ema_sep, stop_pct, tp_pct in combos:
    if fast >= slow:
        continue
    if tp_pct <= stop_pct:
        continue
    if pd.to_datetime(session_start).time() >= pd.to_datetime(session_end).time():
        continue

    stats = run_short_backtest(
        fast=fast,
        slow=slow,
        trend=trend,
        session_start_str=session_start,
        session_end_str=session_end,
        atr_filter=atr_filter,
        ema_sep=ema_sep,
        stop_pct=stop_pct,
        tp_pct=tp_pct,
    )

    results.append({
        "fast": fast,
        "slow": slow,
        "trend": trend,
        "session_start": session_start,
        "session_end": session_end,
        "atr_filter": atr_filter,
        "ema_sep": ema_sep,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "equity": stats["equity"],
        "closed_trades": stats["closed_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown": stats["max_drawdown"],
        "score": stats["score"],
    })

    tested += 1
    if tested % 50 == 0:
        best_equity = max(r["equity"] for r in results)
        print(f"Tested {tested} valid short strategies... best equity so far: {best_equity:.4f}")

# =========================
# RESULTS
# =========================
results_df = pd.DataFrame(results)

results_df = results_df[results_df["closed_trades"] >= 8].copy()
results_df = results_df[results_df["equity"] > 0.99].copy()
results_df["profit_factor"] = results_df["profit_factor"].replace([np.inf, -np.inf], np.nan)

results_df = results_df.sort_values(
    by=["score", "equity", "profit_factor", "max_drawdown"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

print("\nTOP 20 MES SHORT-ONLY RESULTS\n")
if not results_df.empty:
    print(results_df.head(20).to_string(index=False))
else:
    print("No short-only setups passed the filters.")

results_df.to_csv("mes_short_only_optimization_results.csv", index=False)
print("\nSaved mes_short_only_optimization_results.csv")
