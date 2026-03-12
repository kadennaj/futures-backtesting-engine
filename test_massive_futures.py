import os
from polygon import RESTClient

api_key = os.getenv("MASSIVE_API_KEY")
if not api_key:
    raise ValueError("Missing MASSIVE_API_KEY environment variable.")

client = RESTClient(api_key=api_key)

# Try likely MES contract symbols here one at a time
test_tickers = [
    "MESH6",
    "MESM6",
    "MESU6",
    "MESZ5",
]

for ticker in test_tickers:
    print(f"\nTesting {ticker} ...")
    try:
        aggs = list(client.get_aggs(
            ticker=ticker,
            multiplier=5,
            timespan="minute",
            from_="2025-12-01",
            to="2025-12-05",
            limit=10,
        ))
        print(f"Rows: {len(aggs)}")
        if aggs:
            print("Success:", ticker)
            print(aggs[0])
            break
    except Exception as e:
        print("Failed:", e)
