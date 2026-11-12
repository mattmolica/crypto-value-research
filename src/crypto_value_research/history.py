"""Valuation history: how expensive each token has been over its whole life.

Usage:
    python -m crypto_value_research.history
    python -m crypto_value_research.history --out-dir data/history

Free data has one gap: nobody publishes past circulating supply for free, so
circulating market cap only goes back a year (CoinGecko's free limit). Price
and total supply can be rebuilt back to launch, so the long history is valued
on fully diluted value (FDV = price x every token that exists):

- price: DefiLlama coins API, daily back to launch
- supply: today's total supply from CoinGecko; for tokens whose supply moved a
  lot we model or estimate it (see SUPPLY_RULES), and say which way it leans
- revenue and fees: DefiLlama, totalled over the trailing 90 days and scaled
  to a year, the same as the ranking

Writes two CSVs: weekly history per token, and a summary of where each token's
P/S sits today against its own past.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .rank_tokens import DEFAULT_TOKENS
from .sources import coingecko, defillama, llama_prices

WINDOW = 90  # trailing days of revenue/fees behind each day's multiple
MIN_REVENUE_SHARE = 0.02  # days with revenue under 2% of the peak don't count toward the lifetime range

SKY_LAUNCH = pd.Timestamp("2024-09-18")
MKR_TO_SKY = 24_000  # 1 MKR swaps into 24,000 SKY


@dataclass(frozen=True)
class SupplyRule:
    method: str  # "exact", "modeled" or "estimated", then a plain description
    note: str = ""  # which way the estimate leans, for the reader


EXACT = SupplyRule("exact: fixed total supply")
SUPPLY_RULES: dict[str, SupplyRule] = {
    "curve-finance": SupplyRule("modeled: CRV's published emission schedule"),
    "aerodrome": SupplyRule("estimated: straight line from 500M at launch to today's supply"),
    "uniswap": SupplyRule("estimated: today's supply",
                          "about 11% of UNI was burned in late 2025, so earlier FDV is slightly understated"),
    "jupiter": SupplyRule("estimated: today's supply",
                          "about 3B JUP was burned in 2025, so earlier FDV is understated"),
    "pancakeswap": SupplyRule("estimated: today's supply",
                              "CAKE supply was cut sharply by burns and a lower cap, so earlier FDV is understated"),
    "pendle": SupplyRule("estimated: today's supply",
                         "PENDLE supply grows each week, so earlier FDV is overstated"),
    "pump": SupplyRule("estimated: today's supply",
                       "PUMP supply has shrunk through buybacks, so earlier FDV is understated"),
}

SYMBOL_OVERRIDES = {"aerodrome": "AERO"}  # DefiLlama gives the parent protocol's "-"


# --- supply models -------------------------------------------------------------

CRV_START = pd.Timestamp("2020-08-13")
CRV_INITIAL = 1.303e9  # minted at launch to vesting contracts, team, investors and the DAO
CRV_FIRST_YEAR_RATE = 274.8e6  # emissions in the first year; the rate falls by 2^(1/4) each year


def crv_supply(day: pd.Timestamp) -> float:
    """CRV total supply on a day, from Curve's published emission schedule."""
    years = (day - CRV_START).days / 365.25
    if years < 0:
        return math.nan
    total, rate = CRV_INITIAL, CRV_FIRST_YEAR_RATE
    while years > 0:
        step = min(1.0, years)
        total += rate * step
        rate /= 2 ** 0.25
        years -= step
    return total


AERO_LAUNCH = pd.Timestamp("2023-08-28")
AERO_LAUNCH_SUPPLY = 5e8


def linear_supply(index: pd.DatetimeIndex, start: pd.Timestamp, start_supply: float, end_supply: float) -> pd.Series:
    """Supply rising in a straight line from `start_supply` at `start` to `end_supply` on the last day."""
    span = max(1, (index[-1] - start).days)
    days = np.clip((index - start).days, 0, span)
    return pd.Series(np.interp(days, [0, span], [start_supply, end_supply]), index=index)


def supply_history(slug: str, index: pd.DatetimeIndex, total_supply_now: float) -> pd.Series:
    """Estimated total supply for each day in `index`."""
    if slug == "curve-finance":
        return pd.Series([crv_supply(d) for d in index], index=index, dtype=float)
    if slug == "aerodrome":
        return linear_supply(index, AERO_LAUNCH, AERO_LAUNCH_SUPPLY, total_supply_now)
    return pd.Series(float(total_supply_now), index=index)


# --- building the history ------------------------------------------------------

def daily_series(chart: list[list[float]]) -> pd.Series:
    """DefiLlama's [[unix_seconds, usd], ...] as a daily Series (missing days are 0)."""
    if not chart:
        return pd.Series(dtype=float)
    days = pd.to_datetime([int(t) for t, _ in chart], unit="s").normalize()
    s = pd.Series([float(v) for _, v in chart], index=days)
    return s.groupby(level=0).sum().resample("D").sum()


def trailing_annual(daily: pd.Series, days: int = WINDOW) -> pd.Series:
    """Each day's total over the trailing `days`, scaled to a year (NaN until a full window exists)."""
    if daily.empty:
        return daily
    return daily.rolling(days, min_periods=days).sum() * 365 / days


def _positive(s: pd.Series) -> pd.Series:
    return s.where(s > 0)


def valuation_history(
    prices: pd.Series,
    supply: pd.Series,
    revenue_daily: pd.Series,
    fees_daily: pd.Series,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily FDV, trailing revenue and fees, and multiples, indexed by date.

    Columns: price, supply, fdv, market_cap, revenue_annual_90d,
    fees_annual_90d, ps_fdv, pf_fdv, ps_mcap. Gaps of up to 3 days in the
    price series are filled with the last price.
    """
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    idx = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    price = prices.reindex(idx).ffill(limit=3)
    df = pd.DataFrame({"price": price, "supply": supply.reindex(idx)})
    df["fdv"] = df.price * df.supply
    df["market_cap"] = market_cap.reindex(idx) if market_cap is not None else np.nan
    df["revenue_annual_90d"] = trailing_annual(revenue_daily).reindex(idx)
    df["fees_annual_90d"] = trailing_annual(fees_daily).reindex(idx)
    df["ps_fdv"] = df.fdv / _positive(df.revenue_annual_90d)
    df["pf_fdv"] = df.fdv / _positive(df.fees_annual_90d)
    df["ps_mcap"] = df.market_cap / _positive(df.revenue_annual_90d)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Where today's P/S (on FDV) sits against the token's own history.

    Days with revenue under MIN_REVENUE_SHARE of the peak are left out of the
    range, so a protocol's first weeks with almost no revenue don't set the
    "normal" level. `cheaper_than_pct` is the share of those days with a higher
    P/S than today: 0.96 means it has rarely been this cheap.
    """
    ps = df.ps_fdv.replace([np.inf, -np.inf], np.nan).dropna()
    now = float(ps.iloc[-1]) if len(ps) else math.nan
    peak = df.revenue_annual_90d.max()
    good = df[df.revenue_annual_90d > MIN_REVENUE_SHARE * peak].ps_fdv.dropna() if peak and peak > 0 else ps.iloc[:0]
    return {
        "ps_fdv_now": now,
        "ps_fdv_low": good.min() if len(good) else math.nan,
        "ps_fdv_median": good.median() if len(good) else math.nan,
        "ps_fdv_high": good.max() if len(good) else math.nan,
        "cheaper_than_pct": float((good > now).mean()) if len(good) and not math.isnan(now) else math.nan,
        "ps_history_since": good.index.min().strftime("%Y-%m-%d") if len(good) else "",
        "ps_years": round(len(good) / 365, 1),
        "price_since": df.index.min().strftime("%Y-%m-%d"),
    }


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Sundays plus the latest day with an FDV, to keep files small."""
    last = df.dropna(subset=["fdv"]).index.max()
    keep = (df.index.dayofweek == 6) | (df.index == last)
    return df[keep & df.fdv.notna()]


def build_token(
    slug: str, gecko_id: str, revenue: dict, session: requests.Session, market: dict
) -> tuple[pd.DataFrame, dict]:
    """Fetch and assemble one token's daily history and summary row.

    `revenue` is DefiLlama's dailyRevenue summary, `market` CoinGecko's
    current market data keyed by id.
    """
    fees = defillama.fetch_summary(slug, "dailyFees", session) or {}
    m = market.get(gecko_id, {})

    prices = llama_prices.fetch_price_history(gecko_id, session=session)
    if slug == "sky":  # SKY replaced MKR in Sept 2024: splice in MKR's price at the swap rate
        mkr = llama_prices.fetch_price_history("maker", session=session) / MKR_TO_SKY
        prices = pd.concat([mkr[mkr.index < SKY_LAUNCH], prices[prices.index >= SKY_LAUNCH]])

    idx = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    supply = supply_history(slug, idx, m.get("total_supply") or math.nan)
    try:
        mcap = coingecko.fetch_market_cap_history(gecko_id, session=session)
    except requests.HTTPError as err:  # the long history doesn't need it, so carry on without
        print(f"warning: no market cap history for {slug} ({err}); ps_mcap left blank", file=sys.stderr)
        mcap = None
    df = valuation_history(prices, supply, daily_series(revenue.get("totalDataChart") or []),
                           daily_series(fees.get("totalDataChart") or []), mcap)

    rule = SUPPLY_RULES.get(slug, EXACT)
    note = rule.note
    if slug == "sky":
        note = "before Sept 2024 uses MKR price divided by 24,000 (the swap rate to SKY)"
    fdv_now = m.get("fully_diluted_valuation")
    row = {
        "symbol": SYMBOL_OVERRIDES.get(slug, (m.get("symbol") or revenue.get("symbol") or slug).upper()),
        "name": revenue.get("name") or slug,
        **summarize(df),
        "mcap_to_fdv_now": (m["market_cap"] / fdv_now) if fdv_now and m.get("market_cap") else math.nan,
        "supply_method": rule.method,
        "note": note,
    }
    return df, row


def build_all(tokens: dict[str, str | None], pause: float = 6.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly history and a summary row for each token. `tokens` maps DefiLlama slug to CoinGecko id (None: DefiLlama's)."""
    session = requests.Session()
    revenue = {slug: defillama.fetch_summary(slug, "dailyRevenue", session) or {} for slug in tokens}
    ids = {slug: gid or revenue[slug].get("gecko_id") for slug, gid in tokens.items()}
    market = coingecko.fetch_markets([i for i in ids.values() if i], session)
    rows, frames = [], []
    for slug in tokens:
        if not ids[slug]:
            print(f"warning: no CoinGecko id for {slug}, skipped", file=sys.stderr)
            continue
        df, row = build_token(slug, ids[slug], revenue[slug], session, market)
        w = weekly(df).copy()
        w.insert(0, "symbol", row["symbol"])
        w.insert(0, "date", w.index.strftime("%Y-%m-%d"))
        frames.append(w.reset_index(drop=True))
        rows.append(row)
        print(f"{row['symbol']:>7}: P/S {row['ps_fdv_now']:,.1f}x, cheaper than {row['cheaper_than_pct']:.0%} of its history",
              file=sys.stderr)
        time.sleep(pause)  # stay under CoinGecko's free rate limit
    summary = pd.DataFrame(rows).sort_values("cheaper_than_pct", ascending=False)
    return pd.concat(frames, ignore_index=True), summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/history", help="folder for the two CSVs (default: data/history)")
    args = p.parse_args(argv)

    history, summary = build_all(DEFAULT_TOKENS)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "full_history.csv", index=False)
    summary.to_csv(out / "full_summary.csv", index=False)
    print(summary.drop(columns=["note"]).round(2).to_string(index=False))
    print(f"\nWrote {out / 'full_history.csv'} and {out / 'full_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
