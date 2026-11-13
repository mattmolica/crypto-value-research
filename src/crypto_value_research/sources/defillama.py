"""DefiLlama fees and revenue (free API, no key needed).

Docs: https://defillama.com/docs/api

We pull each protocol's daily series and add them up ourselves over trailing
windows (30, 90 and 365 days), because DefiLlama's summary only offers 24h, 7d,
30d and 1y totals, and a 30-day window is too noisy: payouts that stop and
start, weekly bookings and one-off catch-up days can swing it by 15% or more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.llama.fi"

# DefiLlama data types and the column name we give each one.
DATA_TYPES = {
    "dailyFees": "fees",  # everything users paid
    "dailyRevenue": "revenue",  # the part the protocol keeps
    "dailyHoldersRevenue": "holders_revenue",  # the part that reaches token holders
}

# Trailing windows, in days, that we total every series over.
WINDOWS = (30, 90, 365)

DAY = 86_400


@dataclass
class ProtocolFlows:
    slug: str
    name: str
    gecko_id: str | None
    # Annualized USD flows per window: annual[days][column], e.g. annual[90]["fees"].
    # None means DefiLlama has no series for that data type.
    annual: dict[int, dict[str, float | None]] = field(default_factory=dict)
    # Full days of revenue history DefiLlama has (fees if there is no revenue series).
    history_days: int | None = None


def fetch_summary(slug: str, data_type: str, session: requests.Session | None = None) -> dict | None:
    """Return DefiLlama's summary (with the daily chart) for one protocol and data type.

    Returns None if DefiLlama has no such series.
    """
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/summary/fees/{slug}",
        params={"dataType": data_type, "excludeTotalDataChartBreakdown": "true"},
        timeout=60,
    )
    if resp.status_code in (400, 404):
        return None
    resp.raise_for_status()
    return resp.json()


def annualize(chart: list[list[float]], days: int, now: datetime | None = None) -> float | None:
    """Total the last `days` full days of a daily chart and scale it to one year.

    `chart` is DefiLlama's totalDataChart: [[unix_seconds, usd], ...]. Today's
    partial day is left out. If the series is younger than the window, the total
    is scaled by the days it actually covers. Returns None for an empty series.
    """
    now = now or datetime.now(timezone.utc)
    today = int(now.timestamp()) // DAY * DAY
    start = today - days * DAY
    points = [(int(t), float(v)) for t, v in chart if int(t) < today]
    if not points:
        return None
    covered = min(days, (today - points[0][0]) // DAY)
    if covered <= 0:
        return None
    total = sum(v for t, v in points if t >= start)
    return total * 365 / covered


def fetch_protocol_flows(
    slug: str, session: requests.Session | None = None, now: datetime | None = None
) -> ProtocolFlows:
    """Fetch annualized fees, revenue and holders revenue for a protocol slug, per window."""
    name, gecko_id = slug, None
    annual: dict[int, dict[str, float | None]] = {w: {} for w in WINDOWS}
    history: dict[str, int] = {}
    for data_type, column in DATA_TYPES.items():
        data = fetch_summary(slug, data_type, session)
        chart = (data or {}).get("totalDataChart") or []
        if data:
            name = data.get("name") or name
            gecko_id = gecko_id or data.get("gecko_id")
        for w in WINDOWS:
            annual[w][column] = annualize(chart, w, now) if chart else None
        if chart:
            history[column] = days_of_history(chart, now)
    return ProtocolFlows(slug=slug, name=name, gecko_id=gecko_id, annual=annual,
                         history_days=history.get("revenue", history.get("fees")))


def days_of_history(chart: list[list[float]], now: datetime | None = None) -> int:
    """Full days from a chart's first point up to (not including) today."""
    now = now or datetime.now(timezone.utc)
    today = int(now.timestamp()) // DAY * DAY
    first = min(int(t) for t, _ in chart)
    return max(0, (today - first) // DAY)


# --- Every protocol at once (used to build the all-token universe) -----------

def fetch_overview(data_type: str = "dailyRevenue", session: requests.Session | None = None) -> list[dict]:
    """One row per protocol (and chain) DefiLlama tracks for this data type.

    Each row carries slug, name, category, protocolType ("protocol" or "chain"),
    parentProtocol (e.g. "parent#uniswap") when it is one product of a larger
    project, and DefiLlama's own totals such as total30d and total1y.
    """
    http = session or requests
    resp = http.get(
        f"{BASE_URL}/overview/fees",
        params={"dataType": data_type, "excludeTotalDataChart": "true",
                "excludeTotalDataChartBreakdown": "true"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("protocols") or []


def fetch_directory(session: requests.Session | None = None) -> dict:
    """DefiLlama's protocol directory: which CoinGecko token belongs to which protocol.

    Returns {"protocols": [{defillamaId, name, geckoId, ...}],
             "parentProtocols": [{id, name, gecko_id, ...}]}.
    """
    http = session or requests
    resp = http.get(f"{BASE_URL}/lite/protocols2", timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return {"protocols": data.get("protocols") or [],
            "parentProtocols": data.get("parentProtocols") or []}


def slugify(name: str) -> str:
    """DefiLlama's URL slug for a name: lower case, spaces to dashes, apostrophes dropped."""
    return name.strip().lower().replace(" ", "-").replace("'", "")
