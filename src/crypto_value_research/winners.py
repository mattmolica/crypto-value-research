"""What did the tokens that beat Bitcoin have in common, and could we spot them ahead of time?

The earlier backtest asked "do cheap tokens do better?" (answer: not reliably).
This module flips the question. It cuts history into windows (13 or 26 weeks),
labels every token in every window as a winner (beat BTC) or a loser, and
checks which facts known at the *start* of the window separate the two.

Every feature only uses data up to the window's start date, so nothing peeks
ahead:

    ps_peers      P/S (on FDV) rank vs other tokens that week (high = pricier)
    ps_own        P/S vs the token's own past (high = pricier than usual)
    rev_growth_13 revenue (90d annualised) now vs 13 weeks earlier, log
    rev_growth_52 same vs 52 weeks earlier
    revenue       revenue size, log
    size          fully diluted value, log
    rev_share     revenue / fees (how much of the fees the protocol keeps)
    mom_13        price change vs BTC over the last 13 weeks, log
    mom_52        same over the last 52 weeks
    volatility    std of weekly log returns over the last 13 weeks
    drawdown      price / highest price so far (1 = at its high)
    age           weeks since the token's first price

Three tests, each built to be hard to fool:

1. One feature at a time. In each window, rank tokens by the feature and
   correlate with how much they beat or trailed BTC (Spearman). Average across
   windows that do not overlap, with a t-stat. Also the share of top-third vs
   bottom-third tokens that beat BTC, and the same with the single best token
   of each window removed (so one moonshot can't make a feature look good).
2. A simple model. Logistic regression on the features (as within-window
   ranks), trained only on windows that had already finished, tested on the
   next one (walk-forward). Scored by AUC within each test window
   (0.5 = coin flip) and by how its top-third picks did vs BTC.
3. Timing. Whether *any* alts beat BTC in a window is mostly the market tide,
   so the share of tokens beating BTC is compared with what was visible at the
   window's start (BTC's own momentum, alts' momentum vs BTC).

Limits: today's revenue earners only (tokens that died are missing, which
flatters every result); FDV uses estimated past supply for about half the
tokens; price only (no payouts); 2021-2026 is roughly one and a half crypto
cycles, so "independent periods" are few.

Usage:
    python -m crypto_value_research.winners --history data/history/full_history.csv \
        --summary data/history/full_summary.csv --out results/winners.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "ps_peers", "ps_own", "rev_growth_13", "rev_growth_52", "revenue", "size",
    "rev_share", "mom_13", "mom_52", "volatility", "drawdown", "age",
)
LABELS = {
    "ps_peers": "P/S vs other tokens (pricier)",
    "ps_own": "P/S vs own past (pricier)",
    "rev_growth_13": "Revenue growth, last 13 weeks",
    "rev_growth_52": "Revenue growth, last 52 weeks",
    "revenue": "Revenue size",
    "size": "Token value (FDV)",
    "rev_share": "Share of fees kept as revenue",
    "mom_13": "Price vs BTC, last 13 weeks",
    "mom_52": "Price vs BTC, last 52 weeks",
    "volatility": "Volatility, last 13 weeks",
    "drawdown": "Price vs its high so far (near high)",
    "age": "Token age",
}
MIN_FDV = 10e6
MIN_TOKENS = 10
MIN_OWN_HISTORY = 26


# ---------------------------------------------------------------- data

def load_history(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"].dt.dayofweek == 6]
    cols = ["date", "symbol", "price", "fdv", "revenue_annual_90d", "fees_annual_90d", "ps_fdv"]
    return df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def weekly_btc(daily: pd.Series, dates) -> pd.Series:
    """BTC price on each date (last known daily price on or before it)."""
    daily = daily.sort_index()
    daily.index = pd.to_datetime(daily.index).normalize()
    return daily.reindex(pd.DatetimeIndex(sorted(set(dates))), method="ffill")


def own_rank(ps: pd.Series, min_weeks: int = MIN_OWN_HISTORY) -> pd.Series:
    """Share of a token's earlier weeks with a lower P/S (1 = priciest ever so far)."""
    out = np.full(len(ps), math.nan)
    past: list[float] = []
    for i, v in enumerate(ps.to_numpy()):
        if not (v > 0):
            continue
        if len(past) >= min_weeks:
            out[i] = float(np.mean(np.array(past) < v))
        past.append(v)
    return pd.Series(out, index=ps.index)


def build_panel(hist: pd.DataFrame, btc: pd.Series, horizon: int) -> pd.DataFrame:
    """One row per token per week with start-of-window features and the result vs BTC."""
    btc_w = weekly_btc(btc, hist["date"])
    parts = []
    for sym, g in hist.groupby("symbol"):
        g = g.set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="7D")
        g = g.reindex(full)
        g["symbol"] = sym
        b = btc_w.reindex(full)
        lp, lb = np.log(g["price"]), np.log(b)
        rev = g["revenue_annual_90d"].where(g["revenue_annual_90d"] > 0)
        f = pd.DataFrame(index=full)
        f["symbol"] = sym
        f["fdv"] = g["fdv"]
        f["ps_fdv"] = g["ps_fdv"].where(g["ps_fdv"] > 0)
        f["ps_own"] = own_rank(f["ps_fdv"])
        f["rev_growth_13"] = np.log(rev / rev.shift(13))
        f["rev_growth_52"] = np.log(rev / rev.shift(52))
        f["revenue"] = np.log(rev)
        f["size"] = np.log(g["fdv"])
        f["rev_share"] = (rev / g["fees_annual_90d"].where(g["fees_annual_90d"] > 0)).clip(upper=1)
        f["mom_13"] = (lp - lp.shift(13)) - (lb - lb.shift(13))
        f["mom_52"] = (lp - lp.shift(52)) - (lb - lb.shift(52))
        f["volatility"] = lp.diff().rolling(13, min_periods=10).std()
        f["drawdown"] = g["price"] / g["price"].cummax()
        f["age"] = np.arange(len(full), dtype=float)
        f["ret"] = g["price"].shift(-horizon) / g["price"] - 1
        f["btc_ret"] = b.shift(-horizon) / b - 1
        f["excess"] = (lp.shift(-horizon) - lp) - (lb.shift(-horizon) - lb)
        parts.append(f.rename_axis("date").reset_index())
    p = pd.concat(parts, ignore_index=True)
    p = p[(p["fdv"] >= MIN_FDV) & p["ps_fdv"].notna() & p["excess"].notna()].copy()
    p["beat"] = (p["excess"] > 0).astype(int)
    p["ps_peers"] = p.groupby("date")["ps_fdv"].rank(pct=True)
    counts = p.groupby("date")["symbol"].transform("size")
    return p[counts >= MIN_TOKENS].reset_index(drop=True)


def window_starts(panel: pd.DataFrame, horizon: int, offset: int = 0) -> list[pd.Timestamp]:
    """Non-overlapping window start dates, every `horizon` weeks, shifted by `offset` weeks."""
    dates = sorted(panel["date"].unique())
    return [pd.Timestamp(d) for d in dates[offset::horizon]]


# ---------------------------------------------------------------- 1. one feature at a time

def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 5 or a.nunique() < 2:
        return math.nan
    return float(a.rank().corr(b.rank()))


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return math.nan
    return float(x.mean() / x.std() * math.sqrt(len(x)))


def feature_tests(panel: pd.DataFrame, starts: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    sub = panel[panel["date"].isin(starts)]
    for feat in FEATURES:
        corrs, top_beat, bot_beat, top_ex, bot_ex, top_ex_trim, bot_ex_trim = [], [], [], [], [], [], []
        for _, g in sub.groupby("date"):
            g = g.dropna(subset=[feat])
            if len(g) < MIN_TOKENS:
                continue
            corrs.append(spearman(g[feat], g["excess"]))
            r = g[feat].rank(pct=True)
            top, bot = g[r > 2 / 3], g[r <= 1 / 3]
            top_beat.append(top["beat"].mean())
            bot_beat.append(bot["beat"].mean())
            top_ex.append(top["excess"].median())
            bot_ex.append(bot["excess"].median())
            # without the window's single best token
            best = g["excess"].idxmax()
            top_ex_trim.append(np.expm1(top.drop(index=best, errors="ignore")["excess"]).mean())
            bot_ex_trim.append(np.expm1(bot.drop(index=best, errors="ignore")["excess"]).mean())
        c = pd.Series(corrs)
        rows.append({
            "feature": feat,
            "label": LABELS[feat],
            "windows": int(c.notna().sum()),
            "rank_corr": float(c.mean()),
            "t": tstat(c),
            "same_sign_share": float((np.sign(c) == np.sign(c.mean())).mean()),
            "top_third_beat_btc": float(np.mean(top_beat)),
            "bottom_third_beat_btc": float(np.mean(bot_beat)),
            "top_third_median_vs_btc": float(np.expm1(np.mean(top_ex))),
            "bottom_third_median_vs_btc": float(np.expm1(np.mean(bot_ex))),
            "top_minus_bottom_avg_no_best": float(np.mean(np.array(top_ex_trim) - np.array(bot_ex_trim))),
        })
    return pd.DataFrame(rows)


def offset_stability(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Re-run the per-feature correlation for every possible start offset of the window grid."""
    rows = []
    for off in range(horizon):
        ft = feature_tests(panel, window_starts(panel, horizon, off))
        rows.append(ft.set_index("feature")["rank_corr"].rename(off))
    m = pd.concat(rows, axis=1)
    return pd.DataFrame({"min": m.min(axis=1), "max": m.max(axis=1),
                         "same_sign_offsets": (np.sign(m).eq(np.sign(m.mean(axis=1)), axis=0)).mean(axis=1)})


def leave_one_token_out(panel: pd.DataFrame, starts, features=FEATURES) -> pd.DataFrame:
    """Biggest swing in each feature's average correlation from dropping any one token."""
    base = feature_tests(panel, starts).set_index("feature")["rank_corr"]
    lo, hi = base.copy(), base.copy()
    for sym in panel["symbol"].unique():
        r = feature_tests(panel[panel["symbol"] != sym], starts).set_index("feature")["rank_corr"]
        lo, hi = np.minimum(lo, r), np.maximum(hi, r)
    return pd.DataFrame({"loo_min": lo, "loo_max": hi})


# ---------------------------------------------------------------- 2. walk-forward model

def ranked(df: pd.DataFrame, feats) -> np.ndarray:
    """Within-date percentile ranks centred on 0; missing = 0 (the middle)."""
    x = df.groupby("date")[list(feats)].rank(pct=True) - 0.5
    return x.fillna(0.0).to_numpy()


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Plain L2-regularised logistic regression (Newton steps). Returns [intercept, *coefs]."""
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + reg @ w
        hess = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + x @ w[1:])))


def auc(score, y) -> float:
    score, y = np.asarray(score), np.asarray(y)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def walk_forward(panel: pd.DataFrame, horizon: int, starts, feats=FEATURES,
                 min_train_windows: int = 4, target: str = "beat_median") -> pd.DataFrame:
    """Train on finished windows only, test on the next window start.

    target "beat_median" asks "who will beat the other tokens?" (relative), which
    the features can speak to; "beat" asks "who will beat BTC?", which also
    depends on the overall market tide. Both are scored by AUC within the test
    window on the beat-BTC outcome.
    """
    p = panel.copy()
    p["beat_median"] = (p["excess"] > p.groupby("date")["excess"].transform("median")).astype(int)
    rows = []
    for i, d in enumerate(starts):
        train = p[p["date"] + pd.Timedelta(weeks=horizon) <= d]
        train = train[train["date"].isin(starts)]
        if train["date"].nunique() < min_train_windows:
            continue
        test = p[p["date"] == d]
        w = fit_logistic(ranked(train, feats), train[target].to_numpy())
        s = predict(w, ranked(test, feats))
        top = test[s >= np.quantile(s, 2 / 3)]
        bot = test[s <= np.quantile(s, 1 / 3)]
        rows.append({
            "date": d, "tokens": len(test), "train_windows": train["date"].nunique(),
            "auc_beat_btc": auc(s, test["beat"]),
            "auc_beat_median": auc(s, test["beat_median"]),
            "top_beat_btc": top["beat"].mean(), "all_beat_btc": test["beat"].mean(),
            "top_avg_vs_btc": float(np.expm1(top["excess"]).mean()),
            "bottom_avg_vs_btc": float(np.expm1(bot["excess"]).mean()),
            "all_avg_vs_btc": float(np.expm1(test["excess"]).mean()),
            "top_symbols": " ".join(top.sort_values("excess", ascending=False)["symbol"]),
            "coefs": dict(zip(feats, np.round(w[1:], 3))),
        })
    return pd.DataFrame(rows)


def summarize_model(wf: pd.DataFrame) -> dict:
    if wf.empty:
        return {"test_windows": 0}
    gap = wf["top_avg_vs_btc"] - wf["bottom_avg_vs_btc"]
    return {
        "test_windows": len(wf),
        "auc_beat_btc_mean": float(wf["auc_beat_btc"].mean()),
        "auc_beat_median_mean": float(wf["auc_beat_median"].mean()),
        "auc_above_half_share": float((wf["auc_beat_median"] > 0.5).mean()),
        "auc_t": tstat(wf["auc_beat_median"] - 0.5),
        "top_beat_btc": float(wf["top_beat_btc"].mean()),
        "all_beat_btc": float(wf["all_beat_btc"].mean()),
        "top_avg_vs_btc": float(wf["top_avg_vs_btc"].mean()),
        "bottom_avg_vs_btc": float(wf["bottom_avg_vs_btc"].mean()),
        "all_avg_vs_btc": float(wf["all_avg_vs_btc"].mean()),
        "top_beat_bottom_share": float((gap > 0).mean()),
        "top_minus_bottom_t": tstat(gap),
    }


# ---------------------------------------------------------------- 3. timing

def tide(panel: pd.DataFrame, btc: pd.Series, starts) -> pd.DataFrame:
    """Per window: share of tokens beating BTC, next to what was visible at its start."""
    btc_w = weekly_btc(btc, panel["date"])
    lb = np.log(btc_w)
    rows = []
    for d in starts:
        g = panel[panel["date"] == d]
        rows.append({
            "date": d, "tokens": len(g),
            "share_beat_btc": g["beat"].mean(),
            "median_vs_btc": float(np.expm1(g["excess"].median())),
            "btc_ret": float(g["btc_ret"].iloc[0]),
            "btc_mom_13": float(lb.get(d, np.nan) - lb.get(d - pd.Timedelta(weeks=13), np.nan)),
            "alts_mom_13_vs_btc": float(g["mom_13"].median()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver

def _rec(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def run(hist: pd.DataFrame, btc: pd.Series, horizons=(13, 26), loo: bool = True) -> dict:
    out = {}
    for h in horizons:
        panel = build_panel(hist, btc, h)
        starts = window_starts(panel, h)
        ft = feature_tests(panel, starts)
        st = offset_stability(panel, h).reset_index()
        ft = ft.merge(st, on="feature")
        if loo:
            ft = ft.merge(leave_one_token_out(panel, starts).reset_index(), on="feature")
        wf_rel = walk_forward(panel, h, starts, target="beat_median")
        wf_btc = walk_forward(panel, h, starts, target="beat")
        td = tide(panel, btc, starts)
        out[str(h)] = {
            "horizon_weeks": h,
            "windows": len(starts),
            "token_windows": int(panel["date"].isin(starts).sum()),
            "tokens": int(panel["symbol"].nunique()),
            "share_beat_btc": float(panel[panel["date"].isin(starts)]["beat"].mean()),
            "features": _rec(ft),
            "model_relative": summarize_model(wf_rel),
            "model_vs_btc": summarize_model(wf_btc),
            "model_ps_own_only": summarize_model(walk_forward(panel, h, starts, feats=("ps_own",))),
            "model_windows": _rec(wf_rel.drop(columns=["coefs"])) if not wf_rel.empty else [],
            "final_coefs": wf_rel["coefs"].iloc[-1] if not wf_rel.empty else {},
            "tide": _rec(td),
            "tide_corr": {
                "btc_mom_13": spearman(td["btc_mom_13"], td["share_beat_btc"]),
                "alts_mom_13_vs_btc": spearman(td["alts_mom_13_vs_btc"], td["share_beat_btc"]),
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", required=True, help="full_history.csv from `history`")
    ap.add_argument("--btc", help="CSV with date,price for BTC (default: fetch from DefiLlama)")
    ap.add_argument("--out", default="winners.json")
    ap.add_argument("--no-loo", action="store_true", help="skip the slow leave-one-token-out check")
    args = ap.parse_args(argv)
    hist = load_history(args.history)
    if args.btc:
        b = pd.read_csv(args.btc, parse_dates=["date"])
        btc = b.set_index("date")["price"]
    else:
        from .sources import llama_prices
        btc = llama_prices.fetch_price_history("bitcoin")
    res = run(hist, btc, loo=not args.no_loo)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    for h, r in res.items():
        print(f"\n== {h}-week windows: {r['windows']} windows, {r['token_windows']} token-windows, "
              f"{r['share_beat_btc']:.0%} beat BTC")
        cols = ["feature", "rank_corr", "t", "same_sign_share", "top_third_beat_btc", "bottom_third_beat_btc",
                "top_minus_bottom_avg_no_best"] + (["loo_min", "loo_max"] if not args.no_loo else []) + ["same_sign_offsets"]
        print(pd.DataFrame(r["features"])[cols].round(3).to_string(index=False))
        print("model (relative target):", {k: round(v, 3) for k, v in r["model_relative"].items()})
        print("model (P/S vs own past only):", {k: round(v, 3) for k, v in r["model_ps_own_only"].items()})
        print("model (beat-BTC target):", {k: round(v, 3) for k, v in r["model_vs_btc"].items()})
        print("timing corr:", {k: round(v, 2) for k, v in r["tide_corr"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
