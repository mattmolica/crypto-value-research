"""Valuation multiples for tokens: price-to-fees, price-to-sales, FDV-to-revenue.

Flows arrive already annualized (see sources.defillama). By default the
multiples use the trailing 90 days, which tracks the current run rate without
the noise of a single month; the trailing year is kept beside it as a check.

Market value is normally CoinGecko's market cap. For tokens whose holders lock
them to earn the protocol's fees (vote-escrow tokens such as veAERO or veCRV),
CoinGecko leaves the locked tokens out of circulating supply even though those
lockers are the ones being paid. For those we value every issued token instead
(price x total supply), so the multiples aren't flattered.
"""

from __future__ import annotations

import math

import pandas as pd

FLOWS = ("fees", "revenue", "holders_revenue")

NOTE_LOCKED = "valued on all issued tokens (most supply is locked to earn fees)"
NOTE_PAYOUT_STOPPED = "no holder payouts in the last 30 days, though there were in the past year"


def _ratio(numerator: float | None, denominator: float | None) -> float:
    """numerator / denominator, or NaN when either is missing or the denominator is not positive."""
    if numerator is None or denominator is None:
        return math.nan
    if isinstance(numerator, float) and math.isnan(numerator):
        return math.nan
    if isinstance(denominator, float) and math.isnan(denominator):
        return math.nan
    if denominator <= 0:
        return math.nan
    return numerator / denominator


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def market_value(row: pd.Series) -> float | None:
    """The value we divide by: market cap, or all issued tokens for locked-supply tokens."""
    if row.get("locked_supply"):
        price, supply = row.get("price"), row.get("total_supply")
        if not _missing(price) and not _missing(supply):
            return price * supply
    return row.get("market_cap")


def add_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Add market value, valuation multiples and data-quality notes.

    Expects columns: market_cap, fdv, fees_annual, revenue_annual,
    holders_revenue_annual. Optional: price, total_supply, locked_supply,
    revenue_annual_1y, holders_revenue_annual_30d, holders_revenue_annual_1y.
    """
    out = df.copy()
    for col in ("locked_supply", "price", "total_supply", "revenue_annual_1y",
                "holders_revenue_annual_30d", "holders_revenue_annual_1y"):
        if col not in out:
            out[col] = False if col == "locked_supply" else math.nan
    for col in FLOWS:
        out[f"{col}_annual"] = out[f"{col}_annual"].astype(float)

    out["market_value"] = [market_value(r) for _, r in out.iterrows()]
    mv = out["market_value"]
    out["p_f"] = [_ratio(m, f) for m, f in zip(mv, out["fees_annual"])]
    out["p_s"] = [_ratio(m, r) for m, r in zip(mv, out["revenue_annual"])]
    out["p_s_1y"] = [_ratio(m, r) for m, r in zip(mv, out["revenue_annual_1y"])]
    out["fdv_rev"] = [_ratio(v, r) for v, r in zip(out["fdv"], out["revenue_annual"])]
    out["p_e_holders"] = [_ratio(m, h) for m, h in zip(mv, out["holders_revenue_annual"])]
    # How much of the fully diluted supply is circulating (the rest is future dilution).
    out["mcap_to_fdv"] = [_ratio(m, v) for m, v in zip(out["market_cap"], out["fdv"])]
    out["notes"] = [_notes(r) for _, r in out.iterrows()]
    return out


def _notes(row: pd.Series) -> str:
    notes = []
    if row.get("locked_supply") and row.get("market_value") != row.get("market_cap"):
        notes.append(NOTE_LOCKED)
    recent, year = row.get("holders_revenue_annual_30d"), row.get("holders_revenue_annual_1y")
    if not _missing(recent) and not _missing(year) and year > 0 and recent <= 0:
        notes.append(NOTE_PAYOUT_STOPPED)
    return "; ".join(notes)


def rank(df: pd.DataFrame, by: str = "p_s") -> pd.DataFrame:
    """Sort cheapest first on `by`; tokens with no positive denominator go last."""
    out = df.sort_values(by, ascending=True, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
