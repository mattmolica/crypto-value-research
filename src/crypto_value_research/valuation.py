"""Valuation multiples for tokens: price-to-fees, price-to-sales, FDV-to-revenue.

All flows are annualized from DefiLlama's trailing 30 days (x 365/30), which
reflects the current run rate rather than a year that may predate a fee switch
or a big change in usage.
"""

from __future__ import annotations

import math

import pandas as pd

ANNUALIZE_30D = 365 / 30


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


def add_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Add annualized flows and valuation multiples.

    Expects columns: market_cap, fdv, fees_30d, revenue_30d, holders_revenue_30d.
    """
    out = df.copy()
    for col in ("fees", "revenue", "holders_revenue"):
        out[f"{col}_annual"] = out[f"{col}_30d"].astype(float) * ANNUALIZE_30D
    out["p_f"] = [_ratio(m, f) for m, f in zip(out["market_cap"], out["fees_annual"])]
    out["p_s"] = [_ratio(m, r) for m, r in zip(out["market_cap"], out["revenue_annual"])]
    out["fdv_rev"] = [_ratio(v, r) for v, r in zip(out["fdv"], out["revenue_annual"])]
    out["p_e_holders"] = [
        _ratio(m, h) for m, h in zip(out["market_cap"], out["holders_revenue_annual"])
    ]
    # How much of the fully diluted supply is not yet circulating (future dilution).
    out["mcap_to_fdv"] = [_ratio(m, v) for m, v in zip(out["market_cap"], out["fdv"])]
    return out


def rank(df: pd.DataFrame, by: str = "p_s") -> pd.DataFrame:
    """Sort cheapest first on `by`; tokens with no positive denominator go last."""
    out = df.sort_values(by, ascending=True, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
