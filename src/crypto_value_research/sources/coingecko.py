"""CoinGecko market cap, FDV and supply.

Works without a key on the public API. If COINGECKO_API_KEY is set, it is sent
as a Demo API key, which gets higher rate limits.
"""

from __future__ import annotations

import os

import requests

BASE_URL = "https://api.coingecko.com/api/v3"

FIELDS = (
    "id", "symbol", "name", "current_price", "market_cap", "fully_diluted_valuation",
    "circulating_supply", "total_supply", "max_supply",
)


def fetch_markets(ids: list[str], session: requests.Session | None = None) -> dict[str, dict]:
    """Return market data for CoinGecko ids, keyed by id. Unknown ids are left out."""
    if not ids:
        return {}
    http = session or requests
    headers = {}
    if key := os.environ.get("COINGECKO_API_KEY"):
        headers["x-cg-demo-api-key"] = key
    resp = http.get(
        f"{BASE_URL}/coins/markets",
        params={"vs_currency": "usd", "ids": ",".join(ids), "per_page": 250},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return {row["id"]: {f: row.get(f) for f in FIELDS} for row in resp.json()}
