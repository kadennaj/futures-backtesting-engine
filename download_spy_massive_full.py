import os
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta

api_key = os.getenv("MASSIVE_API_KEY")
if not api_key:
    raise ValueError("Missing MASSIVE_API_KEY")

client = RESTClient(api_key=api_key)

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2026, 3, 11)

CHUNK_DAYS = 30
ticker = "SPY"

all_rows = []
current_start = START_DATE

while current_start < END_DATE:
    current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)

    print(f"Downloading {current_start.date()} to {current_end.date()}")

    try:
        rows = list(client.get_aggs(
            ticker=ticker,
            multiplier=5,
            timespan="minute",
            from_=current_start.strftime("%Y-%m-%d"),
            to=current_end.strftime("%Y-%m-%d"),
            limit=50000,
        ))

        for r in rows:
            all_rows.append({
                "timestamp": pd.to_datetime(r.timestamp, unit="ms", utc=True),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            })

        print(f"  got {len(rows)} rows")

    except Exception as e:
        print(f"  failed: {e}")

    current_start = current_end

df = pd.DataFrame(all_rows)

if df.empty:
    raise ValueError("No data downloaded.")

df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

print("\nFinal dataset:")
print(df.head())
print(df.tail())
print("Rows:", len(df))
print("Unique dates:", df["timestamp"].dt.date.nunique())

df.to_csv("spy_data_massive_full.csv", index=False)
print("Saved spy_data_massive_full.csv")
