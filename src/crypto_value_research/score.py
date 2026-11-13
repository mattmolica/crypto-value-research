"""Value score: is a token cheap for a good reason or a bad one?

A low price-to-sales ratio alone can't tell a bargain from a project that is
shrinking ("value trap"). The score puts cheapness next to business health and
dilution, and every piece is shown so you can see why a token scored what it did.

How it works
------------
1. Red-flag gate. Rows the ranking marks `caution` (market value under $10M,
   no revenue, too little history, no CoinGecko price, revenue bigger than the
   whole market value) get no score: they are "Unrated".

2. Each metric becomes a percentile among the rated tokens: 100 is best of the
   group, 0 worst, 50 the middle. A metric we don't know counts as 50.
   Percentiles need no hand-tuned thresholds, which keeps the score from being
   fitted to make today's list look good.

3. Score (0-100) = weighted average of the percentiles, using WEIGHTS below:
     Cheap (40)        P/S on the last 90 days (lower is better)          25
                       holder yield: payouts to holders / market value    15
     Healthy (40)      revenue trend: last 90 days vs last year           20
                       share of revenue that reaches holders              10
                       size of revenue (bigger lasts longer)              10
     Low dilution (20) share of total supply already circulating          10
                       circulating supply growth over the past year        10
   If holder payouts stopped in the last 30 days, the holder metrics use the
   last 30 days, so they score at the bottom.

4. Label from the Cheap and Healthy parts (each rescaled to 0-100):
     Value candidate    cheap >= 60 and healthy >= 60
     Value trap risk    cheap >= 60 and healthy < 40
     Good, priced in    healthy >= 60, not cheap
     Avoid              healthy < 40, not cheap
     Mixed              everything else

Usage:
    python -m crypto_value_research.score --ranking data/ranking.csv --out data/scores.csv
    python -m crypto_value_research.score --ranking data/ranking.csv --no-supply-growth
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import pandas as pd
import requests

from .sources import coingecko

# metric -> (pillar, weight, higher_is_better)
WEIGHTS: dict[str, tuple[str, float, bool]] = {
    "p_s": ("cheap", 25, False),
    "holder_yield": ("cheap", 15, True),
    "revenue_trend": ("healthy", 20, True),
    "holder_share": ("healthy", 10, True),
    "revenue_size": ("healthy", 10, True),
    "circulating_share": ("dilution", 10, True),
    "supply_growth": ("dilution", 10, False),
}
PILLARS = ("cheap", "healthy", "dilution")

HIGH, LOW = 60, 40
VALUE = "Value candidate"
TRAP = "Value trap risk"
PRICED_IN = "Good, priced in"
AVOID = "Avoid"
MIXED = "Mixed"
UNRATED = "Unrated"


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    """The raw metrics the score uses, one column each (see WEIGHTS).

    Expects the ranking table's columns (market_value, revenue_annual,
    revenue_annual_1y, holders_revenue_annual, holders_revenue_annual_30d,
    holders_revenue_annual_1y, market_cap, fdv, p_s) and optionally
    supply_growth.
    """
    out = pd.DataFrame(index=df.index)
    mv = df["market_value"].map(_num)
    rev = df["revenue_annual"].map(_num)
    rev_1y = df.get("revenue_annual_1y", pd.Series(math.nan, index=df.index)).map(_num)
    holders = df["holders_revenue_annual"].map(_num).fillna(0.0)
    h30 = df.get("holders_revenue_annual_30d", pd.Series(math.nan, index=df.index)).map(_num)
    h1y = df.get("holders_revenue_annual_1y", pd.Series(math.nan, index=df.index)).map(_num)
    stopped = (h1y > 0) & (h30 <= 0)
    holders = holders.where(~stopped, 0.0).clip(lower=0)

    out["p_s"] = df["p_s"].map(_num)
    out["holder_yield"] = (holders / mv).where(mv > 0)
    out["revenue_trend"] = (rev / rev_1y).where(rev_1y > 0)
    out["holder_share"] = (holders / rev).where(rev > 0).clip(upper=1.0)
    out["revenue_size"] = rev.where(rev > 0)
    cap, fdv = df["market_cap"].map(_num), df["fdv"].map(_num)
    out["circulating_share"] = (cap / fdv).where((cap > 0) & (fdv > 0)).clip(upper=1.0)
    out["supply_growth"] = df.get("supply_growth", pd.Series(math.nan, index=df.index)).map(_num)
    out["payout_stopped"] = stopped
    return out


def percentile(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """0-100 rank among the known values (ties share a rank); unknown values get 50."""
    known = values.dropna()
    if len(known) <= 1:
        return pd.Series(50.0, index=values.index)
    ranks = known.rank(method="average", ascending=higher_is_better)
    pct = (ranks - 1) / (len(known) - 1) * 100
    return pct.reindex(values.index).fillna(50.0)


def label(cheap: float, healthy: float) -> str:
    if cheap >= HIGH and healthy >= HIGH:
        return VALUE
    if cheap >= HIGH and healthy < LOW:
        return TRAP
    if healthy >= HIGH:
        return PRICED_IN
    if healthy < LOW:
        return AVOID
    return MIXED


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add value score columns to a ranking table and sort best first.

    Adds the raw metrics, a `<metric>_pct` percentile for each, a 0-100 score
    per pillar (`cheap_score`, `healthy_score`, `dilution_score`), the overall
    `value_score`, and `verdict`. Rows with caution=True get verdict "Unrated"
    and no score, and don't count toward the other tokens' percentiles.
    """
    out = df.copy().reset_index(drop=True)
    m = metrics(out)
    for c in m:
        out[c] = m[c]
    caution = out.get("caution", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    rated = ~caution

    total = pd.Series(0.0, index=out.index)
    pillar_sum = {p: pd.Series(0.0, index=out.index) for p in PILLARS}
    pillar_weight = dict.fromkeys(PILLARS, 0.0)
    for metric, (pillar, weight, higher) in WEIGHTS.items():
        pct = pd.Series(math.nan, index=out.index)
        pct[rated] = percentile(m.loc[rated, metric], higher)
        out[f"{metric}_pct"] = pct
        total += weight * pct
        pillar_sum[pillar] += weight * pct
        pillar_weight[pillar] += weight
    for p in PILLARS:
        out[f"{p}_score"] = pillar_sum[p] / pillar_weight[p]
    out["value_score"] = total / sum(w for _, w, _ in WEIGHTS.values())
    out["verdict"] = [
        label(c, h) if r else UNRATED
        for c, h, r in zip(out["cheap_score"], out["healthy_score"], rated)
    ]
    out = out.sort_values("value_score", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "score_rank", [i + 1 if r else None for i, r in enumerate(out["verdict"] != UNRATED)])
    out["score_rank"] = out["score_rank"].astype("Int64")
    return out


def add_supply_growth(
    df: pd.DataFrame, session: requests.Session | None = None, pause: float = 2.5
) -> pd.DataFrame:
    """Fetch each token's one-year circulating supply growth from CoinGecko.

    One request per token; the free API allows roughly 30 a minute, hence the
    pause. Tokens that fail get NaN, which the score treats as average.
    """
    session = session or requests.Session()
    growth = []
    for gid in df["gecko_id"]:
        g = None
        if isinstance(gid, str) and gid:
            try:
                g = coingecko.fetch_supply_growth(gid, session=session)
            except requests.RequestException as e:
                print(f"warning: no supply history for {gid}: {e}", file=sys.stderr)
            time.sleep(pause)
        growth.append(math.nan if g is None else g)
    out = df.copy()
    out["supply_growth"] = growth
    return out


DISPLAY = ["score_rank", "symbol", "name", "value_score", "cheap_score", "healthy_score",
           "dilution_score", "verdict", "p_s", "revenue_trend", "holder_share", "supply_growth"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranking", required=True,
                   help="CSV written by `rank_tokens --out` (the input to score)")
    p.add_argument("--no-supply-growth", action="store_true",
                   help="skip the CoinGecko supply history calls (supply growth counts as average)")
    p.add_argument("--out", help="write the scored table to this CSV path")
    args = p.parse_args(argv)

    table = pd.read_csv(args.ranking)
    if not args.no_supply_growth and "supply_growth" not in table:
        print(f"Fetching a year of supply history for {len(table)} tokens (about 2.5s each)...",
              file=sys.stderr)
        table = add_supply_growth(table)
    scored = score_table(table)
    shown = scored[[c for c in DISPLAY if c in scored]]
    print(shown.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n" + scored["verdict"].value_counts().to_string())
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
