"""CoinGecko market cap, FDV and supply.

Works without a key on the public API. If COINGECKO_API_KEY is set, it is sent
as a Demo API key, which gets higher rate limits.
"""

from __future__ import annotations

import os
import time

import requests

BASE_URL = "https://api.coingecko.com/api/v3"

FIELDS = (
    "id", "symbol", "name", "current_price", "market_cap", "fully_diluted_valuation",
    "circulating_supply", "total_supply", "max_supply",
)


BATCH = 100  # ids per request; CoinGecko returns at most 250 rows a page and long URLs fail


def fetch_markets(
    ids: list[str],
    session: requests.Session | None = None,
    retries: int = 4,
    backoff: float = 30.0,
) -> dict[str, dict]:
    """Return market data for CoinGecko ids, keyed by id. Unknown ids are left out.

    Asks for `BATCH` ids at a time, so any number of ids works. When CoinGecko
    says we're asking too often (HTTP 429, common on the free API right after
    other calls), waits `backoff` seconds and retries, up to `retries` times.
    """
    http = session or requests
    ids = list(dict.fromkeys(ids))  # drop repeats, keep order
    out: dict[str, dict] = {}
    for i in range(0, len(ids), BATCH):
        for attempt in range(retries + 1):
            resp = http.get(
                f"{BASE_URL}/coins/markets",
                params={"vs_currency": "usd", "ids": ",".join(ids[i:i + BATCH]), "per_page": 250},
                headers=_headers(),
                timeout=30,
            )
            if resp.status_code != 429 or attempt == retries:
                break
            time.sleep(backoff)
        resp.raise_for_status()
        out.update({row["id"]: {f: row.get(f) for f in FIELDS} for row in resp.json()})
    return out


def _headers() -> dict:
    key = os.environ.get("COINGECKO_API_KEY")
    return {"x-cg-demo-api-key": key} if key else {}


def fetch_market_cap_history(
    gecko_id: str,
    days: int = 365,
    session: requests.Session | None = None,
    retries: int = 4,
    backoff: float = 30.0,
) -> "pd.Series":
    """Daily circulating market cap for the last `days` days, indexed by UTC date.

    The free and Demo plans refuse anything older than 365 days. When CoinGecko
    says we're asking too often (HTTP 429), waits `backoff` seconds and retries.
    """
    import time

    import pandas as pd

    http = session or requests
    for attempt in range(retries + 1):
        resp = http.get(
            f"{BASE_URL}/coins/{gecko_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            headers=_headers(),
            timeout=60,
        )
        if resp.status_code != 429 or attempt == retries:
            break
        time.sleep(backoff)
    resp.raise_for_status()
    caps = {}
    for ms, value in resp.json().get("market_caps", []):
        caps[pd.Timestamp(int(ms), unit="ms").normalize()] = float(value)  # last value per day wins
    return pd.Series(caps, dtype=float).sort_index()


def fetch_supply_growth(
    gecko_id: str,
    days: int = 365,
    session: requests.Session | None = None,
    retries: int = 4,
    backoff: float = 30.0,
) -> float | None:
    """How much circulating supply grew over the last `days` days (0.25 = +25%).

    CoinGecko has no supply history on the free plan, but it has daily market
    cap and price, and circulating supply = market cap / price. We compare the
    first and last days. Returns None when either day is missing or zero
    (CoinGecko reports 0 market cap when it can't verify circulating supply).
    """
    http = session or requests
    for attempt in range(retries + 1):
        resp = http.get(
            f"{BASE_URL}/coins/{gecko_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            headers=_headers(),
            timeout=60,
        )
        if resp.status_code != 429 or attempt == retries:
            break
        time.sleep(backoff)
    resp.raise_for_status()
    data = resp.json()
    return supply_growth(data.get("market_caps", []), data.get("prices", []))


def supply_growth(market_caps: list[list[float]], prices: list[list[float]]) -> float | None:
    """Circulating supply growth from first to last point of CoinGecko's chart lists."""
    supply = []
    for (_, cap), (_, price) in zip(market_caps, prices):
        if cap and price and cap > 0 and price > 0:
            supply.append(cap / price)
    if len(supply) < 2 or supply[0] <= 0:
        return None
    # Skip leading days before CoinGecko started tracking circulating supply.
    return supply[-1] / supply[0] - 1
