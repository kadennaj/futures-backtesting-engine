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

# =========================
# COSTS
# =========================
COMMISSION_PCT = 0.00025
SLIPPAGE_PCT = 0.00025
ENTRY_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT
EXIT_COST_PCT = COMMISSION_PCT + SLIPPAGE_PCT

# =========================
# PARAMETER GRID
# centered around your winners
# =========================
FAST_VALS = [12, 16, 20]
SLOW_VALS = [21, 25, 30]
TREND_VALS = [200, 250]
SESSION_END_VALS = ["11:30", "12:00", "12:30"]
ATR_FILTER_VALS = [0.0009, 0.0011]
EMA_SEP_VALS = [0.0002, 0.00025]
STOP_VALS = [0.004, 0.005]
TP_VALS = [0.008, 0.010]

# =========================
# BACKTEST FUNCTION
# =========================
def run_ema_backtest(df_in, fast, slow, trend, session_end_str, atr_filter, ema_sep, stop, tp):
    df_bt = df_in.copy()

    df_bt["ema_fast"] = df_bt["close"].ewm(span=fast, adjust=False).mean()
    df_bt["ema_slow"] = df_bt["close"].ewm(span=slow, adjust=False).mean()
    df_bt["ema_trend"] = df_bt["close"].ewm(span=trend, adjust=False).mean()

    session_start = pd.to_datetime("09:35").time()
    session_end = pd.to_datetime(session_end_str).time()

    trend_filter = df_bt["close"] > df_bt["ema_trend"]
    momentum_filter = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) > ema_sep
    vol_filter = df_bt["atr_pct"] > atr_filter
    time_filter = (df_bt["time"] >= session_start) & (df_bt["time"] < session_end)

    entry_signal = trend_filter & momentum_filter & vol_filter & time_filter
    exit_signal = ((df_bt["ema_fast"] - df_bt["ema_slow"]) / df_bt["close"]) < 0

    position = 0
    entry_price = 0.0
    signals = []
    trade_returns = []

    for i in range(len(df_bt)):
        signal = 0
        ret = 0.0

        current_close = df_bt.iloc[i]["close"]
        current_high = df_bt.iloc[i]["high"]
        current_low = df_bt.iloc[i]["low"]
        current_time = df_bt.iloc[i]["time"]

        if i == 0:
            signals.append(signal)
            trade_returns.append(ret)
            continue

        prev_close_bar = df_bt.iloc[i - 1]["close"]

        if position == 1:
            ret = (current_close / prev_close_bar) - 1

        if position == 1:
            stop_price = entry_price * (1 - stop)
            target_price = entry_price * (1 + tp)

            hit_stop = current_low <= stop_price
            hit_target = current_high >= target_price
            session_exit = current_time >= session_end
            custom_exit = bool(exit_signal.iloc[i])

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

    running_max = df_bt["strategy_equity"].cummax()
    drawdown = df_bt["strategy_equity"] / running_max - 1
    max_drawdown = drawdown.min()

    trade_list = []
    in_trade = False
    entry_eq = None

    for i in range(len(df_bt)):
        curr_sig = df_bt.iloc[i]["signal"]
        prev_sig = df_bt.iloc[i - 1]["signal"] if i > 0 else 0

        if curr_sig == 1 and prev_sig == 0:
            in_trade = True
            entry_eq = df_bt.iloc[i]["strategy_equity"]

        if curr_sig == 0 and prev_sig == 1 and in_trade:
            exit_eq = df_bt.iloc[i]["strategy_equity"]
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
        "closed_trades": len(trade_list),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
    }

# =========================
# WALK-FORWARD SETUP
# use unique trading days
# =========================
unique_days = sorted(df["date"].unique())

TRAIN_DAYS = 30
TEST_DAYS = 10
STEP_DAYS = 10

wf_results = []

start_idx = 0
window_num = 1

while start_idx + TRAIN_DAYS + TEST_DAYS <= len(unique_days):
    train_days = unique_days[start_idx:start_idx + TRAIN_DAYS]
    test_days = unique_days[start_idx + TRAIN_DAYS:start_idx + TRAIN_DAYS + TEST_DAYS]

    df_train = df[df["date"].isin(train_days)].copy().reset_index(drop=True)
    df_test = df[df["date"].isin(test_days)].copy().reset_index(drop=True)

    print(f"\nWindow {window_num}")
    print(f"Train: {train_days[0]} to {train_days[-1]} ({len(train_days)} days)")
    print(f"Test:  {test_days[0]} to {test_days[-1]} ({len(test_days)} days)")

    best_train = None
    best_params = None

    tested = 0

    for fast, slow, trend, session_end, atr_filter, ema_sep, stop, tp in product(
        FAST_VALS, SLOW_VALS, TREND_VALS, SESSION_END_VALS,
        ATR_FILTER_VALS, EMA_SEP_VALS, STOP_VALS, TP_VALS
    ):
        if fast >= slow:
            continue

        train_stats = run_ema_backtest(
            df_train, fast, slow, trend, session_end, atr_filter, ema_sep, stop, tp
        )

        tested += 1

        score = (
            train_stats["equity"],
            train_stats["profit_factor"],
            -abs(train_stats["max_drawdown"]),
            train_stats["closed_trades"]
        )

        if best_train is None or score > best_train:
            best_train = score
            best_params = {
                "fast": fast,
                "slow": slow,
                "trend": trend,
                "session_end": session_end,
                "atr_filter": atr_filter,
                "ema_sep": ema_sep,
                "stop": stop,
                "tp": tp,
                "train_equity": train_stats["equity"],
                "train_profit_factor": train_stats["profit_factor"],
                "train_drawdown": train_stats["max_drawdown"],
                "train_trades": train_stats["closed_trades"],
            }

    test_stats = run_ema_backtest(
        df_test,
        best_params["fast"],
        best_params["slow"],
        best_params["trend"],
        best_params["session_end"],
        best_params["atr_filter"],
        best_params["ema_sep"],
        best_params["stop"],
        best_params["tp"],
    )

    row = {
        "window": window_num,
        "train_start": str(train_days[0]),
        "train_end": str(train_days[-1]),
        "test_start": str(test_days[0]),
        "test_end": str(test_days[-1]),

        "fast": best_params["fast"],
        "slow": best_params["slow"],
        "trend": best_params["trend"],
        "session_end": best_params["session_end"],
        "atr_filter": best_params["atr_filter"],
        "ema_sep": best_params["ema_sep"],
        "stop": best_params["stop"],
        "tp": best_params["tp"],

        "train_equity": best_params["train_equity"],
        "train_profit_factor": best_params["train_profit_factor"],
        "train_drawdown": best_params["train_drawdown"],
        "train_trades": best_params["train_trades"],

        "test_equity": test_stats["equity"],
        "test_buy_hold_equity": test_stats["buy_hold_equity"],
        "test_profit_factor": test_stats["profit_factor"],
        "test_drawdown": test_stats["max_drawdown"],
        "test_trades": test_stats["closed_trades"],
        "test_win_rate": test_stats["win_rate"],
    }

    wf_results.append(row)

    print("Best Params:", best_params)
    print("Test Stats:", {
        "equity": round(test_stats["equity"], 4),
        "pf": round(test_stats["profit_factor"], 3),
        "dd": round(test_stats["max_drawdown"], 4),
        "trades": test_stats["closed_trades"],
        "win_rate": round(test_stats["win_rate"] * 100, 2)
    })

    start_idx += STEP_DAYS
    window_num += 1

# =========================
# FINAL SUMMARY
# =========================
wf_df = pd.DataFrame(wf_results)

print("\nWALK-FORWARD RESULTS\n")
print(wf_df.to_string(index=False))

if len(wf_df) > 0:
    compounded_test_equity = wf_df["test_equity"].prod()
    avg_test_equity = wf_df["test_equity"].mean()
    avg_test_pf = wf_df["test_profit_factor"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    avg_test_dd = wf_df["test_drawdown"].mean()
    total_test_trades = wf_df["test_trades"].sum()
    avg_test_win_rate = wf_df["test_win_rate"].mean()

    print("\nSUMMARY")
    print("Compounded Test Equity:", round(compounded_test_equity, 4))
    print("Average Test Equity:", round(avg_test_equity, 4))
    print("Average Test Profit Factor:", round(avg_test_pf, 3) if pd.notna(avg_test_pf) else "N/A")
    print("Average Test Drawdown:", round(avg_test_dd, 4))
    print("Total Test Trades:", int(total_test_trades))
    print("Average Test Win Rate:", round(avg_test_win_rate * 100, 2), "%")

wf_df.to_csv("walk_forward_results.csv", index=False)
print("\nSaved walk_forward_results.csv")
