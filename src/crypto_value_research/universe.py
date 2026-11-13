"""Find every DefiLlama protocol that earns revenue and has a tradable token.

DefiLlama tracks a few thousand protocols. Many are one product of a larger
project (Uniswap V2, V3 and V4 all belong to "parent#uniswap"), and the token
belongs to the project, not the product. So we:

1. pull every protocol's revenue totals in one call (the "overview"),
2. fold products into their parent project and add up their revenue,
3. look up each project's CoinGecko token in DefiLlama's directory,
4. keep projects above a revenue floor, and sort the rest into reasons they
   were left out (no token, too little revenue, token shared with a bigger
   project).

Chains (Ethereum, Solana, ...) are left out on purpose: their "revenue" is
burned gas, and their tokens are valued on very different grounds.

Usage:
    python -m crypto_value_research.universe              # print the token list
    python -m crypto_value_research.universe --out data/universe.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd
import requests

from .sources import defillama

# Revenue floor: projects earning less than this over the trailing year are left out.
MIN_REVENUE_1Y = 1_000_000

# Statuses for each project. Only "ranked" projects go on to the valuation table.
RANKED = "ranked"
NO_TOKEN = "no token"
TOO_SMALL = "revenue under floor"
SHARED_TOKEN = "token shared with a bigger project"
EXCLUDED = "excluded"

# Tokens whose price follows an asset they hold, so revenue says little about
# their value (a gold token is worth its gold). Reason shown in the output.
EXCLUDED_TOKENS = {
    "tether-gold": "gold-backed token: price tracks gold, not revenue",
}


def group_projects(overview: list[dict], directory: dict) -> pd.DataFrame:
    """Fold DefiLlama's protocol rows into projects and attach their token.

    Pure function (no network), so it is easy to test. Returns one row per
    project with: key, slug (for DefiLlama's per-protocol API), name, gecko_id,
    category (the one earning the most), revenue_1y, revenue_30d, products.
    """
    child_token = {str(p.get("defillamaId")): p.get("geckoId") for p in directory.get("protocols", [])}
    parents = {p["id"]: p for p in directory.get("parentProtocols", [])}

    groups: dict[str, dict] = defaultdict(lambda: {
        "revenue_1y": 0.0, "revenue_30d": 0.0, "products": [], "gecko_ids": [],
        "categories": defaultdict(float),
    })
    for row in overview:
        if row.get("protocolType", "protocol") != "protocol":
            continue
        key = row.get("parentProtocol") or row["slug"]
        g = groups[key]
        year = float(row.get("total1y") or 0)
        g["revenue_1y"] += year
        g["revenue_30d"] += float(row.get("total30d") or 0)
        g["products"].append(row["slug"])
        g["categories"][row.get("category") or "?"] += year
        if gid := child_token.get(str(row.get("defillamaId", row.get("id")))):
            g["gecko_ids"].append(gid)
        g.setdefault("first_name", row.get("name") or row["slug"])

    rows = []
    for key, g in groups.items():
        if key.startswith("parent#"):
            parent = parents.get(key, {})
            name = parent.get("name") or key.split("#", 1)[1]
            # The project's own token; failing that, one its products agree on.
            gecko_id = parent.get("gecko_id") or _only(g["gecko_ids"])
            slug = defillama.slugify(name)
        else:
            name = g["first_name"]
            gecko_id = _only(g["gecko_ids"])
            slug = key
        rows.append({
            "key": key,
            "slug": slug,
            "name": name,
            "gecko_id": gecko_id or None,
            "category": max(g["categories"], key=g["categories"].get),
            "revenue_1y": g["revenue_1y"],
            "revenue_30d": g["revenue_30d"],
            "products": len(g["products"]),
        })
    cols = ["key", "slug", "name", "gecko_id", "category", "revenue_1y", "revenue_30d", "products"]
    return pd.DataFrame(rows, columns=cols)


def _only(ids: list[str]) -> str | None:
    """The single distinct id in `ids`, or None if there are none or they disagree."""
    distinct = {i for i in ids if i}
    return distinct.pop() if len(distinct) == 1 else None


def classify(projects: pd.DataFrame, min_revenue: float = MIN_REVENUE_1Y) -> pd.DataFrame:
    """Add a `status` column saying whether each project is ranked, and if not, why.

    Sorted by trailing-year revenue, biggest first.
    """
    out = projects.sort_values("revenue_1y", ascending=False).reset_index(drop=True)
    status, seen = [], {}
    for _, r in out.iterrows():
        gid = r["gecko_id"]
        if r["revenue_1y"] < min_revenue:
            status.append(TOO_SMALL)
        elif pd.isna(gid) or not gid:
            status.append(NO_TOKEN)
        elif gid in EXCLUDED_TOKENS:
            status.append(f"{EXCLUDED}: {EXCLUDED_TOKENS[gid]}")
        elif gid in seen:
            status.append(f"{SHARED_TOKEN} ({seen[gid]})")
        else:
            seen[gid] = r["name"]
            status.append(RANKED)
    out["status"] = status
    return out


def build_universe(
    min_revenue: float = MIN_REVENUE_1Y, session: requests.Session | None = None
) -> pd.DataFrame:
    """Every DefiLlama project with its token and whether it makes the ranking."""
    session = session or requests.Session()
    overview = defillama.fetch_overview("dailyRevenue", session)
    directory = defillama.fetch_directory(session)
    return classify(group_projects(overview, directory), min_revenue)


def revenue_tokens(universe: pd.DataFrame) -> dict[str, str]:
    """{defillama_slug: gecko_id} for the ranked projects, ready for rank_tokens/history."""
    ranked = universe[universe["status"] == RANKED]
    return dict(zip(ranked["slug"], ranked["gecko_id"]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-revenue", type=float, default=MIN_REVENUE_1Y,
                   help="trailing-year revenue floor in USD (default: 1,000,000)")
    p.add_argument("--out", help="write every project and its status to this CSV path")
    args = p.parse_args(argv)

    uni = build_universe(args.min_revenue)
    counts = uni["status"].str.split(":").str[0].str.replace(r" \(.*", "", regex=True).value_counts()
    print(f"{len(uni)} DefiLlama projects; revenue floor ${args.min_revenue:,.0f} a year")
    for s, n in counts.items():
        print(f"  {s}: {n}")
    big_no_token = uni[uni["status"] == NO_TOKEN].head(10)
    if len(big_no_token):
        print("\nBiggest earners with no token: " + ", ".join(big_no_token["name"]))
    if args.out:
        uni.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
