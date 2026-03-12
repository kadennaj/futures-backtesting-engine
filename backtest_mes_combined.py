import pandas as pd
import numpy as np

print("MES Combined Strategy - Prop Firm Version (Large Dataset)")

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
# GLOBAL SETTINGS
# =========================
MAX_LONG_TRADES_PER_DAY = 3
MAX_SHORT_TRADES_PER_DAY = 1

COMMISSION = 0.00025
SLIPPAGE = 0.00025

ENTRY_COST = COMMISSION + SLIPPAGE
EXIT_COST = COMMISSION + SLIPPAGE

# =========================
# LONG SETTINGS
# =========================
FAST_EMA = 20
SLOW_EMA = 27
TREND_EMA = 250

ATR_PERIOD = 14
ATR_FILTER = 0.0012

LONG_SESSION_START = "09:35"
LONG_SESSION_END = "15:30"

LONG_STOP = 0.005
LONG_TARGET = 0.012

# =========================
# SHORT SETTINGS
# =========================
SHORT_SESSION_START = "09:40"
SHORT_SESSION_END = "11:30"

ACCUM_BARS = 10
MAX_RANGE = 0.002

SWEEP_BUFFER = 0.0002
STOP_BUFFER = 0.0004

R_MULTIPLE = 2.0

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

sep = (df["ema_fast"] - df["ema_slow"]) / df["close"]

# =========================
# LONG ENTRY SIGNAL
# =========================
long_start = pd.to_datetime(LONG_SESSION_START).time()
long_end = pd.to_datetime(LONG_SESSION_END).time()

long_time = (df["time"] >= long_start) & (df["time"] < long_end)

long_signal = (
    (df["close"] > df["ema_trend"]) &
    (sep > 0) &
    (df["atr_pct"] > ATR_FILTER) &
    long_time
)

# =========================
# SHORT ENTRY SIGNAL
# =========================
short_start = pd.to_datetime(SHORT_SESSION_START).time()
short_end = pd.to_datetime(SHORT_SESSION_END).time()

short_time = (df["time"] >= short_start) & (df["time"] < short_end)

df["range_high"] = df["high"].rolling(ACCUM_BARS).max().shift(1)
df["range_low"] = df["low"].rolling(ACCUM_BARS).min().shift(1)

range_size = df["range_high"] - df["range_low"]
range_pct = range_size / df["close"]

accum = range_pct < MAX_RANGE

sweep = (
    (df["high"] > df["range_high"] * (1 + SWEEP_BUFFER)) &
    (df["close"] < df["range_high"])
)

short_signal = (
    short_time &
    accum &
    sweep &
    (df["close"] < df["ema_trend"])
)

# =========================
# TRADE ENGINE
# =========================
position = 0

entry_price = 0
entry_time = None
entry_date = None

short_stop = 0
short_target = 0

returns = []
trade_log = []

current_day = None
daily_long = 0
daily_short = 0

long_entries = 0
short_entries = 0

for i in range(len(df)):

    price = df.at[i,"close"]
    high = df.at[i,"high"]
    low = df.at[i,"low"]
    time = df.at[i,"time"]
    date = df.at[i,"date"]
    ts = df.at[i,"timestamp_ny"]

    if i == 0:
        current_day = date
        returns.append(0)
        continue

    prev_price = df.at[i-1,"close"]

    if date != current_day:
        current_day = date
        daily_long = 0
        daily_short = 0

    ret = 0

    if position == 1:
        ret = (price/prev_price)-1

    elif position == -1:
        ret = (prev_price/price)-1

    # =====================
    # LONG EXIT
    # =====================
    if position == 1:

        stop = entry_price*(1-LONG_STOP)
        target = entry_price*(1+LONG_TARGET)

        if low <= stop:

            ret = (stop/prev_price)-1-EXIT_COST

            trade_log.append({
            "side":"LONG",
            "entry_price":entry_price,
            "exit_price":stop})

            position=0

        elif high >= target:

            ret=(target/prev_price)-1-EXIT_COST

            trade_log.append({
            "side":"LONG",
            "entry_price":entry_price,
            "exit_price":target})

            position=0

        elif time>=long_end:

            ret=(price/prev_price)-1-EXIT_COST

            trade_log.append({
            "side":"LONG",
            "entry_price":entry_price,
            "exit_price":price})

            position=0

    # =====================
    # SHORT EXIT
    # =====================
    elif position==-1:

        if high>=short_stop:

            ret=(prev_price/short_stop)-1-EXIT_COST

            trade_log.append({
            "side":"SHORT",
            "entry_price":entry_price,
            "exit_price":short_stop})

            position=0

        elif low<=short_target:

            ret=(prev_price/short_target)-1-EXIT_COST

            trade_log.append({
            "side":"SHORT",
            "entry_price":entry_price,
            "exit_price":short_target})

            position=0

        elif time>=short_end:

            ret=(prev_price/price)-1-EXIT_COST

            trade_log.append({
            "side":"SHORT",
            "entry_price":entry_price,
            "exit_price":price})

            position=0

    # =====================
    # ENTRY
    # =====================
    if position==0:

        if long_signal.iloc[i] and daily_long<MAX_LONG_TRADES_PER_DAY:

            position=1
            entry_price=price
            entry_time=ts
            entry_date=date

            ret-=ENTRY_COST

            long_entries+=1
            daily_long+=1

        elif short_signal.iloc[i] and daily_short<MAX_SHORT_TRADES_PER_DAY:

            stop=high*(1+STOP_BUFFER)
            risk=stop-price

            if risk>0:

                position=-1
                entry_price=price
                entry_time=ts
                entry_date=date

                short_stop=stop
                short_target=price-(risk*R_MULTIPLE)

                ret-=ENTRY_COST

                short_entries+=1
                daily_short+=1

    returns.append(ret)

# =========================
# RESULTS
# =========================
df["strategy_return"]=returns
df["market_return"]=df["close"].pct_change().fillna(0)

df["strategy_equity"]=(1+df["strategy_return"]).cumprod()
df["buy_hold_equity"]=(1+df["market_return"]).cumprod()

trade_df=pd.DataFrame(trade_log)

if len(trade_df)>0:

    trade_df["pnl_pct"]=np.where(
    trade_df["side"]=="LONG",
    (trade_df["exit_price"]/trade_df["entry_price"])-1,
    (trade_df["entry_price"]/trade_df["exit_price"])-1)

    wins=(trade_df["pnl_pct"]>0).sum()
    win_rate=wins/len(trade_df)

    profit_factor=(
    trade_df.loc[trade_df.pnl_pct>0,"pnl_pct"].sum()/
    abs(trade_df.loc[trade_df.pnl_pct<=0,"pnl_pct"].sum())
    )

else:

    win_rate=0
    profit_factor=0

# =========================
# DRAWDOWN
# =========================
running_max = df["strategy_equity"].cummax()
drawdown = df["strategy_equity"]/running_max - 1
max_dd = drawdown.min()

# =========================
# PRINT
# =========================
print("Buy & Hold:",round(df["buy_hold_equity"].iloc[-1],4))
print("Strategy:",round(df["strategy_equity"].iloc[-1],4))
print("Trades:",len(trade_df))
print("Long Trades:",long_entries)
print("Short Trades:",short_entries)
print("Win Rate:",round(win_rate*100,2),"%")
print("Profit Factor:",round(profit_factor,2))
print("Max Drawdown:",round(max_dd,4))
