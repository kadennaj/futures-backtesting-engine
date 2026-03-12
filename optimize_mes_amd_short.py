import pandas as pd
import numpy as np
from itertools import product

print("Optimizing MES AMD Short Strategy")

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
# COSTS
# =========================
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# PARAM GRID
# =========================
SESSION_START_VALS = ["09:35", "09:40", "09:45"]
SESSION_END_VALS = ["11:30", "12:00"]

ACCUM_BARS_VALS = [6, 8, 10]
MAX_RANGE_PCT_VALS = [0.0030, 0.0040, 0.0050]
SWEEP_BUFFER_PCT_VALS = [0.0001, 0.0002, 0.0003]
STOP_BUFFER_PCT_VALS = [0.0002, 0.0003, 0.0004]
R_MULTIPLE_VALS = [1.2, 1.5, 2.0]

# =========================
# BACKTEST FUNCTION
# =========================
def run_backtest(
    session_start_str: str,
    session_end_str: str,
    accum_bars: int,
    max_range_pct: float,
    sweep_buffer_pct: float,
    stop_buffer_pct: float,
    r_multiple: float,
):
    temp = df.copy()

    session_start = pd.to_datetime(session_start_str).time()
    session_end = pd.to_datetime(session_end_str).time()

    temp["in_session"] = (temp["time"] >= session_start) & (temp["time"] < session_end)

    # Build accumulation box
    temp["accum_high"] = temp["high"].rolling(accum_bars).max().shift(1)
    temp["accum_low"] = temp["low"].rolling(accum_bars).min().shift(1)
    temp["accum_range"] = temp["accum_high"] - temp["accum_low"]
    temp["accum_range_pct"] = temp["accum_range"] / temp["close"]

    temp["is_accumulation"] = temp["accum_range_pct"] < max_range_pct

    # Rejection bar logic
    temp["rejection_bar"] = (
        (temp["high"] > (temp["accum_high"] * (1.0 + sweep_buffer_pct))) &
        (temp["close"] < temp["accum_high"])
    )

    temp["short_entry_signal"] = (
        temp["in_session"] &
        temp["is_accumulation"] &
        temp["rejection_bar"]
    ).astype(int)

    position = 0
    entry_price = 0.0
    entry_time = None
    entry_date = None
    stop_price = 0.0
    target_price = 0.0

    returns = []
    trade_log = []

    for i in range(len(temp)):
        ret = 0.0

        current_close = temp.at[i, "close"]
        current_high = temp.at[i, "high"]
        current_low = temp.at[i, "low"]
        current_time = temp.at[i, "time"]
        current_ts = temp.at[i, "timestamp_ny"]
        current_date = temp.at[i, "date"]

        if i == 0:
            returns.append(ret)
            continue

        prev_close_bar = temp.at[i - 1, "close"]

        if position == -1:
            ret = (prev_close_bar / current_close) - 1.0

        # Force flat on day change
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
            entry_time = None
            entry_date = None
            stop_price = 0.0
            target_price = 0.0

        if position == -1:
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
                entry_time = None
                entry_date = None
                stop_price = 0.0
                target_price = 0.0

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
                entry_time = None
                entry_date = None
                stop_price = 0.0
                target_price = 0.0

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
                entry_time = None
                entry_date = None
                stop_price = 0.0
                target_price = 0.0

        # Entry
        if position == 0 and temp.at[i, "short_entry_signal"] == 1:
            entry_price = current_close
            entry_time = current_ts
            entry_date = current_date

            raw_stop = current_high * (1.0 + stop_buffer_pct)
            risk = raw_stop - entry_price

            if risk > 0:
                stop_price = raw_stop
                target_price = entry_price - (risk * r_multiple)
                position = -1
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

    exit_counts = trade_df["exit_reason"].value_counts().to_dict() if not trade_df.empty else {}

    return {
        "equity": float(equity_curve.iloc[-1]),
        "closed_trades": int(closed_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "score": float(score),
        "time_exits": int(exit_counts.get("TIME", 0)),
        "stop_exits": int(exit_counts.get("STOP", 0)),
        "target_exits": int(exit_counts.get("TARGET", 0)),
        "day_change_exits": int(exit_counts.get("DAY_CHANGE_FORCE_EXIT", 0)),
    }

# =========================
# OPTIMIZATION LOOP
# =========================
results = []

combos = list(product(
    SESSION_START_VALS,
    SESSION_END_VALS,
    ACCUM_BARS_VALS,
    MAX_RANGE_PCT_VALS,
    SWEEP_BUFFER_PCT_VALS,
    STOP_BUFFER_PCT_VALS,
    R_MULTIPLE_VALS,
))

print(f"Total raw combinations: {len(combos)}")

tested = 0

for session_start, session_end, accum_bars, max_range_pct, sweep_buffer_pct, stop_buffer_pct, r_multiple in combos:
    if pd.to_datetime(session_start).time() >= pd.to_datetime(session_end).time():
        continue

    stats = run_backtest(
        session_start_str=session_start,
        session_end_str=session_end,
        accum_bars=accum_bars,
        max_range_pct=max_range_pct,
        sweep_buffer_pct=sweep_buffer_pct,
        stop_buffer_pct=stop_buffer_pct,
        r_multiple=r_multiple,
    )

    results.append({
        "session_start": session_start,
        "session_end": session_end,
        "accum_bars": accum_bars,
        "max_range_pct": max_range_pct,
        "sweep_buffer_pct": sweep_buffer_pct,
        "stop_buffer_pct": stop_buffer_pct,
        "r_multiple": r_multiple,
        "equity": stats["equity"],
        "closed_trades": stats["closed_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown": stats["max_drawdown"],
        "score": stats["score"],
        "time_exits": stats["time_exits"],
        "stop_exits": stats["stop_exits"],
        "target_exits": stats["target_exits"],
        "day_change_exits": stats["day_change_exits"],
    })

    tested += 1
    if tested % 20 == 0:
        best_equity = max(r["equity"] for r in results)
        print(f"Tested {tested} AMD configs... best equity so far: {best_equity:.4f}")

# =========================
# RESULTS
# =========================
results_df = pd.DataFrame(results)
results_df = results_df[results_df["closed_trades"] >= 8].copy()
results_df["profit_factor"] = results_df["profit_factor"].replace([np.inf, -np.inf], np.nan)

results_df = results_df.sort_values(
    by=["score", "equity", "profit_factor", "max_drawdown"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

print("\nTOP 20 MES AMD SHORT RESULTS\n")
if not results_df.empty:
    print(results_df.head(20).to_string(index=False))
else:
    print("No AMD setups passed the filters.")

results_df.to_csv("mes_amd_short_optimization_results.csv", index=False)
print("\nSaved mes_amd_short_optimization_results.csv")
