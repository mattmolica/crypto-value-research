"""Full daily price history from DefiLlama's free coins API (no key needed).

CoinGecko's free API only serves the last 365 days, but DefiLlama's coins API
returns daily prices back to a token's launch. It caps each request at 500
points, so we walk forward in 500-day chunks.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://coins.llama.fi"
MAX_SPAN = 500  # most points DefiLlama returns per request
DAY = 86_400
DEFAULT_START = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp())


def fetch_price_history(
    gecko_id: str,
    start: int = DEFAULT_START,
    end: int | None = None,
    session: requests.Session | None = None,
    pause: float = 0.3,
) -> pd.Series:
    """Daily USD prices for a CoinGecko id, indexed by UTC date (oldest first).

    Days before the token existed are simply absent. Returns an empty Series if
    DefiLlama has no prices for the id.
    """
    http = session or requests
    end = end or int(time.time())
    key = f"coingecko:{gecko_id}"
    points: dict[pd.Timestamp, float] = {}
    t = start
    while t < end:
        resp = http.get(
            f"{BASE_URL}/chart/{key}",
            params={"start": t, "span": MAX_SPAN, "period": "1d"},
            timeout=60,
        )
        resp.raise_for_status()
        for p in resp.json().get("coins", {}).get(key, {}).get("prices", []):
            day = pd.Timestamp(int(p["timestamp"]), unit="s").normalize()
            points[day] = float(p["price"])
        t += MAX_SPAN * DAY
        if pause and t < end:
            time.sleep(pause)
    return pd.Series(points, dtype=float).sort_index()
