import math

import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from crypto_value_research import track
from crypto_value_research.sources import llama_prices


def snapshot(n=10):
    verdicts = ["Value candidate", "Avoid", "Mixed", "Good, priced in", "Value trap risk"]
    rows = []
    for i in range(n):
        rows.append({
            "scored_at": "2026-11-13T18:55:00Z", "slug": f"t{i}", "gecko_id": f"g{i}",
            "symbol": f"T{i}", "name": f"Token {i}", "price": 10.0 + i,
            "value_score": 90 - i * 5, "verdict": verdicts[i % len(verdicts)],
        })
    rows.append({"scored_at": "2026-11-13T18:55:00Z", "slug": "u", "gecko_id": "gu", "symbol": "U",
                 "name": "Unrated", "price": 1.0, "value_score": math.nan, "verdict": "Unrated"})
    return pd.DataFrame(rows)


def test_fetch_prices_at_uses_historical_endpoint_and_batches():
    ids = [f"g{i}" for i in range(60)]

    def handler(url, params, headers):
        coins = url.rsplit("/", 1)[1].split(",")
        return FakeResponse({"coins": {c: {"price": 2.0} for c in coins}})

    s = FakeSession(handler)
    out = llama_prices.fetch_prices_at(ids, 1794600000, session=s, pause=0)
    assert len(out) == 60 and out["g0"] == 2.0
    assert len(s.calls) == 2
    assert "/prices/historical/1794600000/coingecko:g0," in s.calls[0]["url"]
    llama_prices.fetch_prices_at(["g1"], None, session=s, pause=0)
    assert "/prices/current/coingecko:g1" in s.calls[-1]["url"]


def test_save_and_load_snapshot(tmp_path):
    scores = snapshot().drop(columns="scored_at").assign(extra_col=1)
    path = track.save_snapshot(scores, tmp_path, "2026-11-13T18:55:00Z")
    assert path.name == "value-scores-2026-11-13.csv"
    loaded = track.load_snapshots(tmp_path)
    assert len(loaded) == 1
    assert "extra_col" not in loaded[0] and loaded[0]["scored_at"].iloc[0] == "2026-11-13T18:55:00Z"


def test_token_returns_fallback_and_price_check():
    snap = snapshot()
    start = {f"g{i}": 10.0 + i for i in range(10)}
    start.pop("g3")            # missing at DefiLlama: falls back to snapshot price
    start["g4"] = 100.0        # DefiLlama disagrees with the score's price: flagged
    end = {f"g{i}": (10.0 + i) * 1.1 for i in range(10)}
    perf = track.token_returns(snap, start, end)
    assert perf.loc[perf.slug == "t3", "return"].iloc[0] == pytest.approx(0.1)
    assert not perf.loc[perf.slug == "t4", "price_check"].iloc[0]
    assert math.isnan(perf.loc[perf.slug == "u", "return"].iloc[0])  # no end price
    assert set(perf["score_bucket"].dropna()) == set(track.BUCKETS)
    assert perf.loc[perf.slug == "t0", "score_bucket"].iloc[0] == "Top 20%"


def test_group_table_and_correlation():
    snap = snapshot()
    start = {f"g{i}": 1.0 for i in range(10)} | {"gu": 1.0}
    # higher score -> higher return
    end = {f"g{i}": 1.0 + (9 - i) / 10 for i in range(10)} | {"gu": 1.0}
    perf = track.token_returns(snap.assign(price=1.0), start, end)
    labels = track.group_table(perf, "verdict", track.LABELS)
    assert list(labels["group"]) == [g for g in track.LABELS if g in set(snap.verdict)]
    vc = labels.set_index("group").loc["Value candidate"]
    assert vc["tokens"] == 2 and vc["avg_return"] == pytest.approx((0.9 + 0.4) / 2)
    basket = perf["return"].mean()
    assert vc["vs_basket"] == pytest.approx(vc["avg_return"] - basket)
    assert track.rank_correlation(perf) == pytest.approx(1.0)


def test_group_paths_starts_at_zero_and_averages():
    snap = snapshot(5)
    perf = track.token_returns(snap.assign(price=1.0), {f"g{i}": 1.0 for i in range(5)},
                               {f"g{i}": 2.0 for i in range(5)})
    t0 = pd.Timestamp("2026-11-13 18:55")
    days = pd.to_datetime(["2026-11-14", "2026-11-15"])
    daily = {f"g{i}": pd.Series([1.5, 2.0], index=days) for i in range(5)}
    btc = (pd.Series([90.0, 110.0], index=days), 100.0)
    paths = track.group_paths(perf, daily, t0, benchmarks={"BTC": btc})
    assert (paths.loc[t0] == 0).all()
    assert paths.loc[days[1], "All tokens (equal weight)"] == pytest.approx(1.0)
    assert paths.loc[days[0], "BTC"] == pytest.approx(-0.1)


def test_track_snapshot_end_to_end():
    snap = snapshot(5)

    def handler(url, params, headers):
        if "/prices/" in url:
            coins = url.rsplit("/", 1)[1].split(",")
            price = 1.2 if "/1794790000/" in url else 1.0
            return FakeResponse({"coins": {c: {"price": price} for c in coins}})
        key = url.rsplit("/", 1)[1]
        return FakeResponse({"coins": {key: {"prices": [{"timestamp": 1794700800, "price": 1.1}]}}})

    perf, summary, tables = track.track_snapshot(snap.assign(price=1.0), FakeSession(handler),
                                                 now=1794790000)
    assert summary["benchmarks"]["BTC"] == pytest.approx(0.2)
    assert summary["priced"] == 6 and summary["unpriced"] == []
    assert set(tables) == {"labels", "buckets", "paths"}
    assert tables["paths"].iloc[-1]["All tokens (equal weight)"] == pytest.approx(0.2)
