"""Build a ranked valuation table for a list of tokens.

Usage:
    python -m crypto_value_research.rank_tokens
    python -m crypto_value_research.rank_tokens --sort p_f --out data/ranking.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

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

DISPLAY_COLUMNS = [
    "rank", "symbol", "name", "market_cap", "fdv", "mcap_to_fdv",
    "fees_annual", "revenue_annual", "holders_revenue_annual",
    "p_f", "p_s", "fdv_rev", "p_e_holders",
]


def build_table(tokens: dict[str, str | None], pause: float = 0.3) -> pd.DataFrame:
    session = requests.Session()
    flows = []
    for slug, gecko_override in tokens.items():
        f = defillama.fetch_protocol_flows(slug, session)
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
        rows.append({
            "slug": f.slug,
            "gecko_id": f.gecko_id,
            "symbol": (m.get("symbol") or "").upper(),
            "name": f.name,
            "price": m.get("current_price"),
            "market_cap": m.get("market_cap"),
            "fdv": m.get("fully_diluted_valuation"),
            "circulating_supply": m.get("circulating_supply"),
            "total_supply": m.get("total_supply"),
            "max_supply": m.get("max_supply"),
            "fees_30d": f.total_30d.get("fees"),
            "revenue_30d": f.total_30d.get("revenue"),
            "holders_revenue_30d": f.total_30d.get("holders_revenue"),
        })
    return add_multiples(pd.DataFrame(rows))


def format_table(df: pd.DataFrame) -> str:
    def usd(x):
        return "n/a" if pd.isna(x) else f"${x / 1e6:,.0f}M"

    def mult(x):
        return "n/a" if pd.isna(x) else f"{x:,.1f}x"

    shown = df[DISPLAY_COLUMNS].copy()
    for c in ("market_cap", "fdv", "fees_annual", "revenue_annual", "holders_revenue_annual"):
        shown[c] = shown[c].map(usd)
    for c in ("p_f", "p_s", "fdv_rev", "p_e_holders"):
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
    p.add_argument("--out", help="also write the full table to this CSV path")
    args = p.parse_args(argv)

    table = rank(build_table(DEFAULT_TOKENS), by=args.sort)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Token valuation ranking, sorted by {args.sort} (cheapest first), {stamp}")
    print("Flows are DefiLlama trailing 30 days x 365/30. Market data from CoinGecko.\n")
    print(format_table(table))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
