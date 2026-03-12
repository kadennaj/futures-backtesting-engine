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

# Keep only needed columns
df = df[["close", "high", "low", "time"]].copy()

# =========================
# COMMON ARRAYS
# =========================
close = df["close"].to_numpy(dtype=float)
high = df["high"].to_numpy(dtype=float)
low = df["low"].to_numpy(dtype=float)
time_arr = df["time"].to_numpy()

prev_close = np.roll(close, 1)
prev_close[0] = close[0]

tr1 = high - low
tr2 = np.abs(high - prev_close)
tr3 = np.abs(low - prev_close)
tr = np.maximum.reduce([tr1, tr2, tr3])

atr = pd.Series(tr).rolling(14).mean().to_numpy()
atr_pct = atr / close

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
FAST_VALS = [12, 16, 20]
SLOW_VALS = [21, 25, 30]
TREND_VALS = [200, 250, 300]
SESSION_END_VALS = ["11:30", "12:00", "12:30"]
ATR_FILTER_VALS = [0.0007, 0.0009, 0.0011]
EMA_SEP_VALS = [0.00015, 0.0002, 0.00025]
STOP_VALS = [0.004, 0.005, 0.006]
TP_VALS = [0.006, 0.008, 0.010]

# =========================
# PRECOMPUTE EMAS
# =========================
ema_cache = {}
for span in sorted(set(FAST_VALS + SLOW_VALS + TREND_VALS)):
    ema_cache[span] = pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()

# =========================
# BACKTEST FUNCTION
# =========================
def run_backtest(
    fast: int,
    slow: int,
    trend: int,
    session_end_str: str,
    atr_filter: float,
    ema_sep: float,
    stop_pct: float,
    tp_pct: float,
) -> dict:
    ema_fast = ema_cache[fast]
    ema_slow = ema_cache[slow]
    ema_trend = ema_cache[trend]

    session_start = pd.to_datetime("09:35").time()
    session_end = pd.to_datetime(session_end_str).time()

    trend_filter = close > ema_trend
    momentum_filter = ((ema_fast - ema_slow) / close) > ema_sep
    vol_filter = atr_pct > atr_filter
    time_filter = np.array(
        [(t >= session_start) and (t < session_end) for t in time_arr],
        dtype=bool,
    )

    entry_signal = trend_filter & momentum_filter & vol_filter & time_filter
    exit_signal = ((ema_fast - ema_slow) / close) < 0

    position = 0
    entry_price = 0.0

    equity = 1.0
    equity_peak = 1.0
    max_dd = 0.0

    entries = 0
    closed_trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    current_trade_return = 1.0

    for i in range(1, len(close)):
        price = close[i]
        prev_price = close[i - 1]
        hi = high[i]
        lo = low[i]
        now_time = time_arr[i]

        bar_ret = 0.0
        trade_closed_this_bar = False

        if position == 1:
            bar_ret = price / prev_price - 1.0

            stop_price = entry_price * (1.0 - stop_pct)
            target_price = entry_price * (1.0 + tp_pct)

            hit_stop = lo <= stop_price
            hit_target = hi >= target_price
            session_exit = now_time >= session_end
            custom_exit = bool(exit_signal[i])

            if hit_stop:
                bar_ret = stop_price / prev_price - 1.0 - EXIT_COST_PCT
                position = 0
                trade_closed_this_bar = True
            elif hit_target:
                bar_ret = target_price / prev_price - 1.0 - EXIT_COST_PCT
                position = 0
                trade_closed_this_bar = True
            elif session_exit or custom_exit:
                bar_ret = price / prev_price - 1.0 - EXIT_COST_PCT
                position = 0
                trade_closed_this_bar = True

        if position == 0 and entry_signal[i]:
            position = 1
            entry_price = price
            bar_ret -= ENTRY_COST_PCT
            entries += 1
            current_trade_return = 1.0

        equity *= (1.0 + bar_ret)

        if position == 1 or trade_closed_this_bar:
            current_trade_return *= (1.0 + bar_ret)

        if trade_closed_this_bar:
            pnl = current_trade_return - 1.0
            closed_trades += 1
            if pnl > 0:
                wins += 1
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
            current_trade_return = 1.0

        if equity > equity_peak:
            equity_peak = equity
        dd = equity / equity_peak - 1.0
        if dd < max_dd:
            max_dd = dd

    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
    win_rate = wins / closed_trades if closed_trades > 0 else 0.0

    return {
        "equity": float(equity),
        "entries": int(entries),
        "closed_trades": int(closed_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_dd),
    }

# =========================
# OPTIMIZATION LOOP
# =========================
results = []

combos = list(product(
    FAST_VALS,
    SLOW_VALS,
    TREND_VALS,
    SESSION_END_VALS,
    ATR_FILTER_VALS,
    EMA_SEP_VALS,
    STOP_VALS,
    TP_VALS,
))

print(f"Total raw combinations: {len(combos)}")

tested = 0

for fast, slow, trend, session_end, atr_filter, ema_sep, stop_pct, tp_pct in combos:
    if fast >= slow:
        continue
    if tp_pct <= stop_pct:
        continue

    stats = run_backtest(
        fast=fast,
        slow=slow,
        trend=trend,
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
        "session_end": session_end,
        "atr_filter": atr_filter,
        "ema_sep": ema_sep,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "equity": stats["equity"],
        "entries": stats["entries"],
        "closed_trades": stats["closed_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown": stats["max_drawdown"],
    })

    tested += 1
    if tested % 100 == 0:
        best_equity = max(r["equity"] for r in results)
        print(f"Tested {tested} valid strategies... best equity so far: {best_equity:.4f}")

# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

# remove dead strategies
results_df = results_df[results_df["entries"] >= 8].copy()

# clean inf PF
results_df["profit_factor"] = results_df["profit_factor"].replace([np.inf, -np.inf], np.nan)

results_df = results_df.sort_values(
    by=["equity", "profit_factor", "max_drawdown", "entries"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

print("\nTOP 15 MES ORIGINAL RESULTS\n")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("mes_original_optimization_results.csv", index=False)
print("\nSaved mes_original_optimization_results.csv")
