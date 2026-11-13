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
NOTE_FDV = "CoinGecko has no circulating market cap; valued on fully diluted value"

# Caution notes: something about the data makes the multiple hard to trust.
# Rows with any of these get caution=True so they can be filtered out.
NOTE_NO_MARKET = "token not found on CoinGecko"
NOTE_NO_REVENUE = "no revenue in the window"
NOTE_TINY = "market value under $10M: thin trading, price may be unreliable"
NOTE_YOUNG = "only {days} days of revenue history"
NOTE_P_S_BELOW_1 = "revenue is more than the whole market value: check the token match"

# A warning sign, not a data problem: cheap may mean shrinking.
NOTE_FALLING = "revenue run rate down {pct:.0%} vs the past year"

TINY_MARKET_VALUE = 10e6
# Revenue counts as falling when the window's run rate is under this share of the year's.
FALLING_SHARE = 0.5


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
    """The value we divide by: market cap, or all issued tokens for locked-supply tokens.

    When CoinGecko reports no circulating market cap (0 or blank, which happens
    when it can't verify circulating supply) we use fully diluted value instead.
    That can only make a token look more expensive, never cheaper, and it is noted.
    """
    if row.get("locked_supply"):
        price, supply = row.get("price"), row.get("total_supply")
        if not _missing(price) and not _missing(supply):
            return price * supply
    cap = row.get("market_cap")
    if _missing(cap) or cap <= 0:
        fdv = row.get("fdv")
        return fdv if not _missing(fdv) and fdv > 0 else None
    return cap


def add_multiples(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Add market value, valuation multiples and data-quality notes.

    Expects columns: market_cap, fdv, fees_annual, revenue_annual,
    holders_revenue_annual. Optional: price, total_supply, locked_supply,
    revenue_annual_1y, holders_revenue_annual_30d, holders_revenue_annual_1y,
    history_days, has_market_data. `window` is the days the flows cover, used
    to spot protocols with less history than that.
    """
    out = df.copy()
    for col in ("locked_supply", "price", "total_supply", "revenue_annual_1y",
                "holders_revenue_annual_30d", "holders_revenue_annual_1y", "history_days",
                "has_market_data"):
        if col not in out:
            out[col] = {"locked_supply": False, "has_market_data": True}.get(col, math.nan)
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
    cautions = [_cautions(r, window) for _, r in out.iterrows()]
    out["caution"] = [bool(c) for c in cautions]
    out["notes"] = ["; ".join(c + _notes(r)) for c, (_, r) in zip(cautions, out.iterrows())]
    return out


def _notes(row: pd.Series) -> list[str]:
    """Things worth knowing that don't make the multiple untrustworthy."""
    notes = []
    cap, mv = row.get("market_cap"), row.get("market_value")
    if row.get("locked_supply") and mv != cap:
        notes.append(NOTE_LOCKED)
    elif not _missing(mv) and (_missing(cap) or cap <= 0):
        notes.append(NOTE_FDV)
    recent, year = row.get("holders_revenue_annual_30d"), row.get("holders_revenue_annual_1y")
    if not _missing(recent) and not _missing(year) and year > 0 and recent <= 0:
        notes.append(NOTE_PAYOUT_STOPPED)
    rev, rev_1y = row.get("revenue_annual"), row.get("revenue_annual_1y")
    if not _missing(rev) and not _missing(rev_1y) and rev_1y > 0 and 0 < rev < FALLING_SHARE * rev_1y:
        notes.append(NOTE_FALLING.format(pct=1 - rev / rev_1y))
    return notes


def _cautions(row: pd.Series, window: int) -> list[str]:
    """Data problems that make the multiple hard to trust."""
    out = []
    mv, rev = row.get("market_value"), row.get("revenue_annual")
    if not row.get("has_market_data", True) or _missing(mv):
        out.append(NOTE_NO_MARKET)
    if _missing(rev) or rev <= 0:
        out.append(NOTE_NO_REVENUE)
    if not _missing(mv) and mv < TINY_MARKET_VALUE:
        out.append(NOTE_TINY)
    days = row.get("history_days")
    if not _missing(days) and days < window:
        out.append(NOTE_YOUNG.format(days=int(days)))
    if not _missing(mv) and not _missing(rev) and 0 < mv < rev:
        out.append(NOTE_P_S_BELOW_1)
    return out


def rank(df: pd.DataFrame, by: str = "p_s") -> pd.DataFrame:
    """Sort cheapest first on `by`; tokens with no positive denominator go last."""
    out = df.sort_values(by, ascending=True, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
