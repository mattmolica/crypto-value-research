"""Build a ranked valuation table for tokens.

By default it ranks every DefiLlama project that earned at least $1M of revenue
over the past year and has a token on CoinGecko (see universe.py). Use
`--tokens default` for the original hand-picked list.

Usage:
    python -m crypto_value_research.rank_tokens
    python -m crypto_value_research.rank_tokens --sort p_f --out data/ranking.csv
    python -m crypto_value_research.rank_tokens --window 365 --min-revenue 5e6
    python -m crypto_value_research.rank_tokens --tokens default
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import universe as universe_mod
from .sources import coingecko, defillama
from .valuation import add_multiples, rank

# DefiLlama protocol slug -> CoinGecko id. None means use DefiLlama's own gecko_id.
DEFAULT_TOKENS: dict[str, str | None] = {
    "uniswap": None,
    "aave": None,
    "lido": None,
    "hyperliquid": None,
    "sky": None,
    "pump": None,
    "ethena": None,
    "jupiter": None,
    "pancakeswap": None,
    "raydium": None,
    "aerodrome": "aerodrome-finance",
    "curve-finance": None,
    "gmx": None,
    "pendle": None,
}

# Vote-escrow tokens: holders lock them to earn the protocol's fees, and CoinGecko
# leaves locked tokens out of market cap. These are valued on all issued tokens.
# Listed by DefiLlama slug and by CoinGecko id, so either way of naming matches.
LOCKED_SUPPLY = {
    "aerodrome", "aero", "curve-finance",
    "aerodrome-finance", "curve-dao-token",
}

WINDOW_LABELS = {30: "30 days", 90: "90 days", 365: "year"}

DISPLAY_COLUMNS = [
    "rank", "symbol", "name", "market_cap", "fdv", "mcap_to_fdv",
    "fees_annual", "revenue_annual", "holders_revenue_annual",
    "p_f", "p_s", "p_s_1y", "fdv_rev", "p_e_holders", "caution", "notes",
]



def build_table(
    tokens: dict[str, str | None],
    pause: float = 0.3,
    window: int = 90,
    locked: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Join DefiLlama flows with CoinGecko market data and add multiples.

    `window` (30, 90 or 365 days) picks the flows the main multiples use; the
    trailing year is always added as p_s_1y. Slugs in `locked` are valued on all
    issued tokens rather than circulating market cap.
    """
    session = requests.Session()
    flows = []
    for slug, gecko_override in tokens.items():
        try:
            f = defillama.fetch_protocol_flows(slug, session)
        except requests.RequestException as e:
            # One bad protocol shouldn't sink a run over hundreds of them.
            print(f"warning: skipped {slug}, DefiLlama request failed: {e}", file=sys.stderr)
            continue
        f.gecko_id = gecko_override or f.gecko_id
        flows.append(f)
        time.sleep(pause)

    ids = [f.gecko_id for f in flows if f.gecko_id]
    markets = coingecko.fetch_markets(ids, session)

    rows = []
    for f in flows:
        m = markets.get(f.gecko_id or "", {})
        if not m:
            print(f"warning: no CoinGecko market data for {f.slug} ({f.gecko_id})", file=sys.stderr)
        main, year, month = f.annual[window], f.annual[365], f.annual[30]
        rows.append({
            "slug": f.slug,
            "gecko_id": f.gecko_id,
            "has_market_data": bool(m),
            "symbol": (m.get("symbol") or "").upper(),
            "name": f.name,
            "price": m.get("current_price"),
            "market_cap": m.get("market_cap"),
            "fdv": m.get("fully_diluted_valuation"),
            "circulating_supply": m.get("circulating_supply"),
            "total_supply": m.get("total_supply"),
            "max_supply": m.get("max_supply"),
            "locked_supply": f.slug in locked or (f.gecko_id or "") in locked,
            "history_days": getattr(f, "history_days", None),
            "fees_annual": main.get("fees"),
            "revenue_annual": main.get("revenue"),
            "holders_revenue_annual": main.get("holders_revenue"),
            "revenue_annual_1y": year.get("revenue"),
            "holders_revenue_annual_30d": month.get("holders_revenue"),
            "holders_revenue_annual_1y": year.get("holders_revenue"),
        })
    return add_multiples(pd.DataFrame(rows), window=window)


def format_table(df: pd.DataFrame) -> str:
    def usd(x):
        return "n/a" if pd.isna(x) else f"${x / 1e6:,.0f}M"

    def mult(x):
        return "n/a" if pd.isna(x) else f"{x:,.1f}x"

    shown = df[DISPLAY_COLUMNS].copy()
    for c in ("market_cap", "fdv", "fees_annual", "revenue_annual", "holders_revenue_annual"):
        shown[c] = shown[c].map(usd)
    for c in ("p_f", "p_s", "p_s_1y", "fdv_rev", "p_e_holders"):
        shown[c] = shown[c].map(mult)
    shown["mcap_to_fdv"] = shown["mcap_to_fdv"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")
    return shown.to_markdown(index=False) if _has_tabulate() else shown.to_string(index=False)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sort", default="p_s", choices=["p_s", "p_f", "fdv_rev", "p_e_holders"],
                   help="multiple to rank by, cheapest first (default: p_s)")
    p.add_argument("--window", type=int, default=90, choices=sorted(WINDOW_LABELS),
                   help="trailing days of fees/revenue the multiples use (default: 90)")
    p.add_argument("--tokens", default="all", choices=["all", "default"],
                   help="all: every DefiLlama project with revenue and a token (default); "
                        "default: the hand-picked list in DEFAULT_TOKENS")
    p.add_argument("--min-revenue", type=float, default=universe_mod.MIN_REVENUE_1Y,
                   help="with --tokens all: trailing-year revenue floor in USD (default: 1,000,000)")
    p.add_argument("--out", help="also write the full table to this CSV path")
    p.add_argument("--universe-out",
                   help="with --tokens all: write every project and why it was or wasn't ranked")
    args = p.parse_args(argv)

    meta = None
    if args.tokens == "all":
        uni = universe_mod.build_universe(args.min_revenue)
        tokens = universe_mod.revenue_tokens(uni)
        meta = uni[uni["status"] == universe_mod.RANKED][["slug", "category", "products"]]
        print(f"Universe: {len(tokens)} projects with ${args.min_revenue:,.0f}+ revenue in the past year "
              f"and a CoinGecko token (out of {len(uni)} on DefiLlama).", file=sys.stderr)
        if args.universe_out:
            uni.to_csv(args.universe_out, index=False)
    else:
        tokens = DEFAULT_TOKENS

    table = build_table(tokens, window=args.window, locked=LOCKED_SUPPLY)
    if meta is not None:
        table = table.merge(meta, on="slug", how="left")
    table = rank(table, by=args.sort)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Token valuation ranking, sorted by {args.sort} (cheapest first), {stamp}")
    print(f"Flows are DefiLlama's trailing {WINDOW_LABELS[args.window]}, scaled to a year; "
          "p_s_1y uses the trailing year. Market data from CoinGecko. "
          "Rows marked caution have a data problem worth checking (see notes).\n")
    print(format_table(table))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
