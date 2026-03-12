import os
import requests

api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY is not set")

url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2025-01-01/2025-01-31"

params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": api_key
}

r = requests.get(url, params=params, timeout=30)
r.raise_for_status()

data = r.json()

print("Status:", data.get("status"))
print("Bars returned:", len(data.get("results", [])))
print("First bar:", data.get("results", [None])[0])

