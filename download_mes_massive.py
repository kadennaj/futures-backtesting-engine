import os
import requests
import pandas as pd

API_KEY = os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    raise ValueError("Missing MASSIVE_API_KEY environment variable.")

BASE_URL = "https://api.massive.com"
TICKER = "MESH6"  # example contract; replace if needed

def fetch_aggs(ticker: str, start_date: str, end_date: str, multiplier: int = 5, timespan: str = "minute"):
    url = f"{BASE_URL}/futures/vX/aggs/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    all_rows = []
    next_url = url

    while next_url:
        resp = requests.get(next_url, params=params if next_url == url else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        all_rows.extend(results)

        next_url = data.get("next_url")
        if next_url and "apiKey=" not in next_url:
            next_url = f"{next_url}&apiKey={API_KEY}"

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    rename_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    df = fetch_aggs(
        ticker=TICKER,
        start_date="2025-12-01",
        end_date="2026-03-11",
        multiplier=5,
        timespan="minute",
    )

    print(df.head())
    print("\nRows downloaded:", len(df))

    df.to_csv("mes_data_massive.csv", index=False)
    print("Saved mes_data_massive.csv")
