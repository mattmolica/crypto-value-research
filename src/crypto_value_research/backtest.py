"""Backtest: did tokens that were cheap on price-to-sales go on to do better?

The full value score can't be rebuilt for past dates (holder payouts, supply
growth and circulating supply have no free history), so this tests its biggest
single ingredient, P/S, on the tokens whose P/S history `history` built.

Two questions, both at 13, 26 and 52 weeks ahead:

1. Cheap vs peers. Each week, sort the tokens by P/S. Did the cheapest third
   beat the most expensive third over the next N weeks? Also shown as a
   growth-of-$1 line: every 13 weeks buy the cheapest third (equal amounts),
   hold 13 weeks, repeat; same for the most expensive third and for all.
2. Cheap vs its own past. Each week, is a token's P/S lower than most of its
   own history so far (only past weeks count, so no peeking ahead)? Did tokens
   in their cheapest third of history beat the average token afterwards?

Read the results with the limits in mind:
- few tokens (6 to 14 at a time) and a handful of years, so one token's run
  can swing everything
- survivorship: these are today's large revenue earners; tokens that died
  never made the list
- P/S here is on fully diluted value, with total supply estimated for tokens
  whose supply changed a lot (see `history`)
- overlapping windows: weekly starts with 52-week holds share most of their
  days, so the count of truly separate periods is far smaller than the count
  of weeks (shown as `independent_periods`)
- price only: holder payouts are not added back

Usage:
    python -m crypto_value_research.backtest --history data/history/full_history.csv
    python -m crypto_value_research.backtest --history full_history.csv --out data/backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (13, 26, 52)  # weeks ahead
MIN_TOKENS = 6  # fewest tokens with a P/S on a date for it to count
MIN_OWN_HISTORY = 26  # weeks of a token's own P/S needed before judging it against its past
REBALANCE = 13  # weeks between rebalances in the growth-of-$1 line


def spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return math.nan
    return float(a.rank().corr(b.rank()))


def load(path: Path | str) -> pd.DataFrame:
    """Weekly (Sunday) price and P/S per token from `history`'s full_history.csv."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"].dt.dayofweek == 6]
    return df[["date", "symbol", "price", "ps_fdv"]].sort_values(["symbol", "date"]).reset_index(drop=True)


def add_forward_returns(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add `fwd_<weeks>`: price return from each week to `weeks` later (NaN if not there yet)."""
    out = df.copy()
    prices = out.set_index(["symbol", "date"])["price"]
    for h in horizons:
        later = pd.MultiIndex.from_arrays([out["symbol"], out["date"] + pd.Timedelta(weeks=h)])
        out[f"fwd_{h}"] = prices.reindex(later).to_numpy() / out["price"].to_numpy() - 1
    return out


def thirds(values: pd.Series) -> pd.Series:
    """Label each value 'cheap', 'middle' or 'expensive' by rank (lowest P/S = cheap)."""
    n = len(values)
    k = max(1, n // 3)
    order = values.rank(method="first")
    return pd.Series(np.where(order <= k, "cheap", np.where(order > n - k, "expensive", "middle")),
                     index=values.index)


# ---------------------------------------------------------------- 1. cheap vs peers

def cross_section(df: pd.DataFrame, horizon: int, min_tokens: int = MIN_TOKENS) -> pd.DataFrame:
    """Per week: average forward return of the cheapest and dearest third, and the rank correlation."""
    col = f"fwd_{horizon}"
    rows = []
    for date, g in df.dropna(subset=["ps_fdv", col]).groupby("date"):
        g = g[g["ps_fdv"] > 0]
        if len(g) < min_tokens:
            continue
        t = thirds(g["ps_fdv"])
        rows.append({
            "date": date,
            "tokens": len(g),
            "cheap": g.loc[t == "cheap", col].mean(),
            "expensive": g.loc[t == "expensive", col].mean(),
            "all": g[col].mean(),
            "cheap_symbols": " ".join(g.loc[t == "cheap", "symbol"]),
            # positive = lower P/S went with higher later return
            "rank_corr": spearman(-g["ps_fdv"], g[col]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["spread"] = out["cheap"] - out["expensive"]
    return out


def summarize_cross(cs: pd.DataFrame, horizon: int) -> dict:
    if cs.empty:
        return {"horizon_weeks": horizon, "weeks": 0}
    return {
        "horizon_weeks": horizon,
        "weeks": int(len(cs)),
        "independent_periods": int(len(cs) // horizon) or 1,
        "first": cs["date"].min().strftime("%Y-%m-%d"),
        "last": cs["date"].max().strftime("%Y-%m-%d"),
        "avg_cheap": float(cs["cheap"].mean()),
        "avg_expensive": float(cs["expensive"].mean()),
        "avg_all": float(cs["all"].mean()),
        "median_spread": float(cs["spread"].median()),
        "cheap_won_pct": float((cs["spread"] > 0).mean()),
        "avg_rank_corr": float(cs["rank_corr"].mean()),
    }


def growth_of_dollar(df: pd.DataFrame, step: int = REBALANCE, min_tokens: int = MIN_TOKENS) -> pd.DataFrame:
    """Value of $1 rebalanced every `step` weeks into the cheapest third, dearest third, and all tokens."""
    col = f"fwd_{step}"
    if col not in df:
        df = add_forward_returns(df, (step,))
    cs = cross_section(df, step, min_tokens)
    if cs.empty:
        return pd.DataFrame()
    dates = sorted(cs["date"])
    picks, d = [], dates[0]
    by_date = cs.set_index("date")
    while d in by_date.index:
        picks.append(d)
        d = d + pd.Timedelta(weeks=step)
    rows = [{"date": picks[0], "cheap": 1.0, "expensive": 1.0, "all": 1.0}]
    for d in picks:
        r = by_date.loc[d]
        last = rows[-1]
        rows.append({"date": d + pd.Timedelta(weeks=step),
                     **{k: last[k] * (1 + r[k]) for k in ("cheap", "expensive", "all")}})
    return pd.DataFrame(rows)


def leave_one_out(df: pd.DataFrame, step: int = REBALANCE) -> list[dict]:
    """Rerun the growth-of-$1 line without each token in turn.

    If dropping one token flips the result, the result was that token, not a
    pattern. Sorted with the token whose removal hurts "cheap" most first.
    """
    rows = []
    for sym in sorted(df["symbol"].unique()):
        g = growth_of_dollar(df[df["symbol"] != sym], step)
        if g.empty:
            continue
        last = g.iloc[-1]
        rows.append({"without": sym, "cheap": float(last["cheap"]),
                     "expensive": float(last["expensive"]), "all": float(last["all"])})
    return sorted(rows, key=lambda r: r["cheap"] / r["expensive"])


# ---------------------------------------------------------------- 2. cheap vs own past

def own_history_rank(df: pd.DataFrame, min_weeks: int = MIN_OWN_HISTORY) -> pd.Series:
    """Share of a token's earlier weeks with a higher P/S than this week (1 = cheapest ever so far).

    Only weeks before the current one count, so the label uses no future data.
    NaN until the token has `min_weeks` of P/S history.
    """
    out = pd.Series(math.nan, index=df.index)
    for _, g in df.groupby("symbol"):
        past: list[float] = []
        for i, ps in zip(g.index, g["ps_fdv"]):
            if not (ps > 0):
                continue
            if len(past) >= min_weeks:
                out[i] = float(np.mean(np.array(past) > ps))
            past.append(ps)
    return out


def own_history(df: pd.DataFrame, horizons=HORIZONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward returns when a token was cheap / middle / expensive against its own past.

    Returns (pooled, per_token). `excess` is the token's return minus the average
    return of all tokens with data that week, so a market-wide rally or crash
    doesn't count as the cheapness signal working.
    """
    d = df.copy()
    d["own_rank"] = own_history_rank(d)
    d["zone"] = pd.cut(d["own_rank"], [-0.01, 1 / 3, 2 / 3, 1.0],
                       labels=["expensive", "middle", "cheap"]).astype(object)
    pooled, per_token = [], []
    for h in horizons:
        col = f"fwd_{h}"
        x = d.dropna(subset=["zone", col]).copy()
        x[f"ex_{h}"] = x[col] - x.groupby("date")[col].transform("mean")
        for zone in ("cheap", "middle", "expensive"):
            z = x[x["zone"] == zone]
            pooled.append({
                "horizon_weeks": h, "zone": zone, "observations": len(z),
                "tokens": z["symbol"].nunique(),
                "avg_return": z[col].mean(), "median_return": z[col].median(),
                "avg_excess": z[f"ex_{h}"].mean(), "median_excess": z[f"ex_{h}"].median(),
                "beat_avg_pct": (z[f"ex_{h}"] > 0).mean(),
            })
        if h == 26:
            for sym, g in x.groupby("symbol"):
                c, e = g[g["zone"] == "cheap"], g[g["zone"] == "expensive"]
                per_token.append({
                    "symbol": sym, "weeks": len(g),
                    "cheap_weeks": len(c), "expensive_weeks": len(e),
                    "excess_when_cheap": c[f"ex_{h}"].mean() if len(c) else math.nan,
                    "excess_when_expensive": e[f"ex_{h}"].mean() if len(e) else math.nan,
                })
    return pd.DataFrame(pooled), pd.DataFrame(per_token)


# ---------------------------------------------------------------- run

def run(history: pd.DataFrame) -> dict:
    df = add_forward_returns(history)
    cross = {h: cross_section(df, h) for h in HORIZONS}
    pooled, per_token = own_history(df)
    growth = growth_of_dollar(df)
    loo = leave_one_out(df)
    return {
        "tokens": sorted(history["symbol"].unique().tolist()),
        "history_through": history["date"].max().strftime("%Y-%m-%d"),
        "cheap_vs_peers": [summarize_cross(cross[h], h) for h in HORIZONS],
        "cheap_vs_peers_weekly_26": _records(cross[26].assign(date=cross[26]["date"].dt.strftime("%Y-%m-%d"))
                                             if not cross[26].empty else cross[26]),
        "growth_of_dollar": _records(growth.assign(date=growth["date"].dt.strftime("%Y-%m-%d"))
                                     if not growth.empty else growth),
        "growth_leave_one_out": loo,
        "cheap_vs_own_past": _records(pooled),
        "own_past_by_token_26w": _records(per_token),
    }


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", double_precision=6))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", default="data/history/full_history.csv",
                   help="full_history.csv written by `python -m crypto_value_research.history`")
    p.add_argument("--out", help="write the full results as JSON to this path")
    args = p.parse_args(argv)

    result = run(load(args.history))
    print(f"Tokens: {' '.join(result['tokens'])}\n")
    print("1. Cheapest third vs most expensive third by P/S, average return over the next N weeks")
    for s in result["cheap_vs_peers"]:
        if not s["weeks"]:
            continue
        print(f"  {s['horizon_weeks']:>2}w: cheap {s['avg_cheap']:+.0%}  expensive {s['avg_expensive']:+.0%}  "
              f"all {s['avg_all']:+.0%}  cheap won {s['cheap_won_pct']:.0%} of {s['weeks']} weeks "
              f"(~{s['independent_periods']} separate periods), rank corr {s['avg_rank_corr']:+.2f}")
    g = result["growth_of_dollar"]
    if g:
        last = g[-1]
        print(f"\n  $1 rebalanced every {REBALANCE} weeks from {g[0]['date']} to {last['date']}: "
              f"cheap ${last['cheap']:.2f}, expensive ${last['expensive']:.2f}, all ${last['all']:.2f}")
    loo = result["growth_leave_one_out"]
    if loo:
        w = loo[0]
        print(f"  Weakest with one token left out (without {w['without']}): "
              f"cheap ${w['cheap']:.2f}, expensive ${w['expensive']:.2f}, all ${w['all']:.2f}")
    print("\n2. Cheap vs the token's own past, return vs the average token afterwards")
    for r in result["cheap_vs_own_past"]:
        print(f"  {r['horizon_weeks']:>2}w {r['zone']:>9}: {r['observations']:>4} token-weeks, "
              f"excess {r['avg_excess']:+.1%}, beat average {r['beat_avg_pct']:.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, allow_nan=False))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
