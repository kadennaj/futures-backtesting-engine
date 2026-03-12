import os
import pandas as pd
from polygon import RESTClient

api_key = os.getenv("MASSIVE_API_KEY")
if not api_key:
    raise ValueError("Missing MASSIVE_API_KEY")

client = RESTClient(api_key=api_key)

rows = list(client.get_aggs(
    ticker="SPY",
    multiplier=5,
    timespan="minute",
    from_="2023-01-01",
    to="2026-03-11",
    limit=50000,
))

if not rows:
    raise ValueError("No rows returned for SPY")

df = pd.DataFrame([{
    "timestamp": pd.to_datetime(r.timestamp, unit="ms", utc=True),
    "open": r.open,
    "high": r.high,
    "low": r.low,
    "close": r.close,
    "volume": r.volume,
} for r in rows])

print(df.head())
print("\nRows downloaded:", len(df))

df.to_csv("spy_data_massive.csv", index=False)
print("Saved spy_data_massive.csv")
