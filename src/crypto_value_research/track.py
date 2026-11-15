"""Track record: do the value scores pick tokens that go on to do well?

Every time the value score runs, its table can be saved as a dated snapshot
(`snapshots/value-scores-YYYY-MM-DD.csv`). This module goes back to each
snapshot and asks what has happened to the price of every token since:

- each token's return from the moment it was scored until now
- the average and median return of each label (Value candidate, Value trap
  risk, Good priced in, Avoid, Mixed, Unrated) and of each score fifth
  (top 20% of value scores, next 20%, ...)
- the same over the same dates for Bitcoin, Ether, and an equal-weight basket
  of every token in the snapshot (the "do nothing clever" benchmark)
- the rank correlation between value score and return: +1 would mean higher
  scores always did better, 0 means the score told us nothing

Prices come from DefiLlama's free coins API: the price at the snapshot's
`scored_at` time and today's price, for every token and both benchmarks, so
everything is measured the same way. Price only: holder payouts (buybacks,
fee shares) are left out, which slightly understates high-yield tokens.

A few days of returns are noise. The results start to mean something after
several months and several snapshots.

Usage:
    python -m crypto_value_research.track save --scores data/scores.csv          # store today's run
    python -m crypto_value_research.track                                       # measure every snapshot
    python -m crypto_value_research.track --out-dir data/tracking --no-paths
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .score import AVOID, MIXED, PRICED_IN, TRAP, UNRATED, VALUE
from .sources import llama_prices

SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_COLUMNS = [
    "scored_at", "score_rank", "slug", "gecko_id", "symbol", "name", "category", "price",
    "market_value", "p_s", "value_score", "cheap_score", "healthy_score", "dilution_score",
    "verdict", "caution", "notes",
]
LABELS = [VALUE, TRAP, PRICED_IN, MIXED, AVOID, UNRATED]
BUCKETS = ["Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%"]
BENCHMARKS = {"BTC": "bitcoin", "ETH": "ethereum"}
# A token's DefiLlama start price this far from the price the score saw means
# the two sources disagree about which coin it is; such rows are flagged.
PRICE_CHECK_TOLERANCE = 0.15


# ---------------------------------------------------------------- snapshots

def slim_snapshot(scores: pd.DataFrame, scored_at: str) -> pd.DataFrame:
    """Keep the columns tracking needs and stamp the time the prices were taken."""
    out = scores.copy()
    out["scored_at"] = scored_at
    return out[[c for c in SNAPSHOT_COLUMNS if c in out]]


def save_snapshot(scores: pd.DataFrame, out_dir: Path = SNAPSHOT_DIR, scored_at: str | None = None) -> Path:
    """Write a score table as `value-scores-<date>.csv` in `out_dir` and return the path.

    `scored_at` is the UTC time the prices in the table were fetched
    (ISO format); it defaults to now.
    """
    scored_at = scored_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"value-scores-{scored_at[:10]}.csv"
    slim_snapshot(scores, scored_at).to_csv(path, index=False)
    return path


def load_snapshots(folder: Path = SNAPSHOT_DIR) -> list[pd.DataFrame]:
    """Every saved snapshot, oldest first."""
    return [pd.read_csv(p) for p in sorted(folder.glob("value-scores-*.csv"))]


def to_timestamp(scored_at: str) -> int:
    ts = pd.Timestamp(scored_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


# ---------------------------------------------------------------- returns

def score_buckets(snapshot: pd.DataFrame) -> pd.Series:
    """Split rated tokens into fifths by value score (Top 20% ... Bottom 20%)."""
    out = pd.Series(pd.NA, index=snapshot.index, dtype="object")
    rated = snapshot["verdict"].ne(UNRATED) & snapshot["value_score"].notna()
    if rated.sum() < len(BUCKETS):
        return out
    rank = snapshot.loc[rated, "value_score"].rank(ascending=False, method="first")
    out[rated] = pd.qcut(rank, len(BUCKETS), labels=BUCKETS).astype(str)
    return out


def token_returns(snapshot: pd.DataFrame, start: dict[str, float], end: dict[str, float]) -> pd.DataFrame:
    """One row per token: start and end price and the return between them.

    `start` and `end` map CoinGecko id to price. A token missing from `start`
    falls back to the price saved in the snapshot. `price_check` is False when
    DefiLlama's start price is more than 15% from the snapshot's price.
    """
    out = snapshot.copy()
    out["score_bucket"] = score_buckets(out)
    llama_start = out["gecko_id"].map(start)
    out["start_price"] = llama_start.fillna(out["price"])
    out["end_price"] = out["gecko_id"].map(end)
    out["return"] = out["end_price"] / out["start_price"] - 1
    out.loc[~(out["start_price"] > 0), "return"] = math.nan
    gap = (llama_start / out["price"] - 1).abs()
    out["price_check"] = ~(gap > PRICE_CHECK_TOLERANCE)
    return out


def group_table(perf: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    """Average, median and hit rates of returns for each value of `column`.

    `vs_basket` is the group's average return minus the equal-weight average of
    every token in the snapshot; `beat_basket` the share of the group's tokens
    that did better than that average.
    """
    priced = perf[perf["return"].notna() & perf["price_check"]]
    basket = priced["return"].mean()
    rows = []
    for name in order:
        g = perf[perf[column] == name]
        r = priced.loc[priced[column] == name, "return"]
        if g.empty:
            continue
        rows.append({
            "group": name,
            "tokens": len(g),
            "priced": len(r),
            "avg_return": r.mean() if len(r) else math.nan,
            "median_return": r.median() if len(r) else math.nan,
            "vs_basket": r.mean() - basket if len(r) else math.nan,
            "pct_up": (r > 0).mean() if len(r) else math.nan,
            "beat_basket": (r > basket).mean() if len(r) else math.nan,
            "best": _extreme(priced[priced[column] == name], max),
            "worst": _extreme(priced[priced[column] == name], min),
        })
    return pd.DataFrame(rows)


def _extreme(g: pd.DataFrame, pick) -> str:
    if g.empty:
        return ""
    i = pick(g.index, key=lambda k: g.at[k, "return"])
    return f"{g.at[i, 'symbol']} {g.at[i, 'return']:+.0%}"


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy: Pearson correlation of the ranks."""
    if a.nunique() < 2 or b.nunique() < 2:
        return math.nan
    return float(a.rank().corr(b.rank()))


def rank_correlation(perf: pd.DataFrame) -> float:
    """Spearman correlation of value score with return among rated, priced tokens."""
    r = perf[perf["verdict"].ne(UNRATED) & perf["return"].notna() & perf["price_check"]]
    if len(r) < 5:
        return math.nan
    return spearman(r["value_score"], r["return"])


def summarize(perf: pd.DataFrame, bench: dict[str, float], scored_at: str, as_of: str) -> dict:
    """The headline numbers for one snapshot, ready for JSON."""
    priced = perf[perf["return"].notna() & perf["price_check"]]
    rated = priced[priced["verdict"].ne(UNRATED)]
    days = (pd.Timestamp(as_of) - pd.Timestamp(scored_at)).total_seconds() / 86_400
    return {
        "scored_at": scored_at,
        "as_of": as_of,
        "days": round(days, 1),
        "tokens": int(len(perf)),
        "priced": int(len(priced)),
        "unpriced": perf.loc[perf["return"].isna(), "symbol"].fillna(perf["slug"]).astype(str).tolist(),
        "price_mismatch": perf.loc[~perf["price_check"], "symbol"].astype(str).tolist(),
        "benchmarks": {
            **bench,
            "All tokens (equal weight)": _f(priced["return"].mean()),
            "Rated tokens (equal weight)": _f(rated["return"].mean()),
        },
        "rank_correlation": _f(rank_correlation(perf)),
    }


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ---------------------------------------------------------------- paths over time

def group_paths(
    perf: pd.DataFrame, daily: dict[str, pd.Series], start_time: pd.Timestamp, column: str = "verdict",
    benchmarks: dict[str, tuple[pd.Series, float]] | None = None,
) -> pd.DataFrame:
    """Day-by-day return of each group since the snapshot, as if you had put an
    equal amount into every token in the group at scoring time and held.

    `daily` maps CoinGecko id to daily prices after `start_time`. Each token's
    path is price / start price - 1, carried forward over missing days; a
    group's path is the average of its tokens' paths. `benchmarks` maps a name
    to (daily prices, start price). The first row, at `start_time`, is all zero.
    """
    priced = perf[perf["return"].notna() & perf["price_check"]]
    series = {}
    for _, row in priced.iterrows():
        s = daily.get(row["gecko_id"])
        if s is not None and len(s):
            series[row.name] = s / row["start_price"] - 1
    if not series:
        return pd.DataFrame()
    paths = pd.DataFrame(series)
    paths.loc[start_time] = 0.0
    paths = paths.sort_index().ffill()
    out = pd.DataFrame(index=paths.index)
    for name in LABELS:
        cols = [i for i in priced.index[priced[column] == name] if i in paths]
        if cols:
            out[name] = paths[cols].mean(axis=1)
    out["All tokens (equal weight)"] = paths.mean(axis=1)
    for name, (s, start_price) in (benchmarks or {}).items():
        if len(s) and start_price:
            out[name] = (s / start_price - 1).reindex(out.index).ffill()
            out.loc[start_time, name] = 0.0
    return out


def fetch_daily(gecko_ids: list[str], since: int, session: requests.Session, pause: float = 0.3) -> dict[str, pd.Series]:
    out = {}
    for gid in dict.fromkeys(i for i in gecko_ids if isinstance(i, str) and i):
        try:
            s = llama_prices.fetch_price_history(gid, start=since, session=session, pause=0)
        except requests.RequestException as err:
            print(f"warning: no daily prices for {gid} ({err})", file=sys.stderr)
            continue
        if len(s):
            out[gid] = s[s.index > pd.Timestamp(since, unit="s")]
        if pause:
            time.sleep(pause)
    return out


# ---------------------------------------------------------------- run

def track_snapshot(
    snapshot: pd.DataFrame, session: requests.Session, now: int | None = None, paths: bool = True,
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    """Measure one snapshot: per-token returns, summary, and group tables (+ daily paths)."""
    scored_at = str(snapshot["scored_at"].iloc[0])
    t0 = to_timestamp(scored_at)
    ids = snapshot["gecko_id"].dropna().tolist() + list(BENCHMARKS.values())
    start = llama_prices.fetch_prices_at(ids, t0, session=session)
    end = llama_prices.fetch_prices_at(ids, now, session=session)
    as_of = datetime.fromtimestamp(now or time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    perf = token_returns(snapshot, start, end)
    bench = {sym: _f(end[g] / start[g] - 1) if start.get(g) and end.get(g) else None
             for sym, g in BENCHMARKS.items()}
    summary = summarize(perf, bench, scored_at, as_of)
    tables = {
        "labels": group_table(perf, "verdict", LABELS),
        "buckets": group_table(perf, "score_bucket", BUCKETS),
    }
    if paths:
        daily = fetch_daily(ids, t0, session)
        t_end = pd.Timestamp(as_of).tz_localize(None)
        for g, price in end.items():  # finish every path at the latest price, not last midnight
            if g in daily:
                daily[g] = pd.concat([daily[g][daily[g].index < t_end], pd.Series({t_end: price})])
        bench_daily = {sym: (daily.get(g, pd.Series(dtype=float)), start.get(g)) for sym, g in BENCHMARKS.items()}
        tables["paths"] = group_paths(perf, daily, pd.Timestamp(t0, unit="s"), "verdict", bench_daily)
    return perf, summary, tables


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", double_precision=6))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")
    s = sub.add_parser("save", help="store a value score table as a dated snapshot")
    s.add_argument("--scores", required=True, help="CSV written by `score --out`")
    s.add_argument("--scored-at", help="UTC time the prices were fetched, e.g. 2026-11-13T18:55:00Z (default: now)")
    s.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR, help="folder of saved snapshots")
    p.add_argument("--out-dir", type=Path, default=Path("data/tracking"), help="where to write results")
    p.add_argument("--no-paths", action="store_true", help="skip the day-by-day chart data (one request per token)")
    args = p.parse_args(argv)

    if args.command == "save":
        path = save_snapshot(pd.read_csv(args.scores), args.snapshot_dir, args.scored_at)
        print(f"Saved {path}")
        return 0

    snapshots = load_snapshots(args.snapshot_dir)
    if not snapshots:
        print(f"No snapshots in {args.snapshot_dir}; save one with `track save --scores ...`", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    report = []
    for snap in snapshots:
        perf, summary, tables = track_snapshot(snap, session, paths=not args.no_paths)
        day = summary["scored_at"][:10]
        perf.to_csv(args.out_dir / f"tokens-{day}.csv", index=False)
        for name, t in tables.items():
            t.to_csv(args.out_dir / f"{name}-{day}.csv", index=name == "paths")
        entry = {**summary, **{k: _records(v) for k, v in tables.items() if k != "paths"}}
        if "paths" in tables and not tables["paths"].empty:
            pth = tables["paths"]
            entry["paths"] = {"dates": [d.strftime("%Y-%m-%dT%H:%MZ") for d in pth.index],
                              **{c: [_f(x) for x in pth[c]] for c in pth}}
        entry["tokens_detail"] = _records(perf[[
            "symbol", "name", "verdict", "score_bucket", "value_score", "start_price", "end_price",
            "return", "price_check"]])
        report.append(entry)

        print(f"\nSnapshot {summary['scored_at']}  ({summary['days']} days, {summary['priced']} tokens priced)")
        print("Benchmarks: " + ", ".join(f"{k} {v:+.1%}" for k, v in summary["benchmarks"].items() if v is not None))
        print(tables["labels"].drop(columns=["best", "worst"]).to_string(index=False, float_format=lambda x: f"{x:+.1%}"))
        print(f"Score vs return rank correlation: {summary['rank_correlation']}")
    (args.out_dir / "tracking.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    print(f"\nWrote {args.out_dir}/tracking.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
