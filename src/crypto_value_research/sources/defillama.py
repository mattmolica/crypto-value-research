"""DefiLlama fees and revenue (free API, no key needed).

Docs: https://defillama.com/docs/api
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

BASE_URL = "https://api.llama.fi"

# DefiLlama data types and the column name we give each one.
DATA_TYPES = {
    "dailyFees": "fees",  # everything users paid
    "dailyRevenue": "revenue",  # the part the protocol keeps
    "dailyHoldersRevenue": "holders_revenue",  # the part that reaches token holders
}


@dataclass
class ProtocolFlows:
    slug: str
    name: str
    gecko_id: str | None
    # Trailing 30-day totals in USD, keyed by "fees", "revenue", "holders_revenue".
    # None means DefiLlama has no series for that data type.
    total_30d: dict[str, float | None]


def fetch_summary(slug: str, data_type: str, session: requests.Session | None = None) -> dict | None:
    """Return DefiLlama's summary for one protocol and data type, or None if it has none."""
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/summary/fees/{slug}",
        params={"dataType": data_type, "excludeTotalDataChart": "true",
                "excludeTotalDataChartBreakdown": "true"},
        timeout=30,
    )
    if resp.status_code in (400, 404):
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_protocol_flows(slug: str, session: requests.Session | None = None) -> ProtocolFlows:
    """Fetch trailing 30-day fees, revenue and holders revenue for a protocol slug."""
    name, gecko_id = slug, None
    totals: dict[str, float | None] = {}
    for data_type, column in DATA_TYPES.items():
        data = fetch_summary(slug, data_type, session)
        if not data:
            totals[column] = None
            continue
        name = data.get("name") or name
        gecko_id = gecko_id or data.get("gecko_id")
        totals[column] = data.get("total30d")
    return ProtocolFlows(slug=slug, name=name, gecko_id=gecko_id, total_30d=totals)
