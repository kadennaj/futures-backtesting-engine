import pandas as pd
import yfinance as yf

def get_data(ticker, period="60d", interval="5m"):
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    for candidate in ["datetime", "date", "index"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "timestamp"})
            break

    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[keep_cols]

# Try MES first
ticker = "MES=F"

try:
    df = get_data(ticker, period="60d", interval="5m")
except Exception:
    print("MES=F failed, trying ES=F instead...")
    ticker = "ES=F"
    df = get_data(ticker, period="60d", interval="5m")

print("Using ticker:", ticker)
print(df.head())
print("\nRows downloaded:", len(df))

df.to_csv("mes_data.csv", index=False)
print("Saved mes_data.csv")
