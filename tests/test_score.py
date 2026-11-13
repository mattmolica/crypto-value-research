import math

import pandas as pd
import pytest

from crypto_value_research import score
from crypto_value_research.sources import coingecko
from conftest import FakeResponse, FakeSession


def _row(name, mv, rev, rev_1y, holders, cap=None, fdv=None, h30=None, h1y=None,
         growth=math.nan, caution=False):
    cap = mv if cap is None else cap
    return {
        "name": name, "gecko_id": name, "market_value": mv, "market_cap": cap,
        "fdv": fdv if fdv is not None else cap, "revenue_annual": rev,
        "revenue_annual_1y": rev_1y, "holders_revenue_annual": holders,
        "holders_revenue_annual_30d": holders if h30 is None else h30,
        "holders_revenue_annual_1y": holders if h1y is None else h1y,
        "p_s": mv / rev if rev else math.nan, "supply_growth": growth, "caution": caution,
    }


def _frame():
    return pd.DataFrame([
        # cheap, growing, pays holders, no dilution
        _row("gem", 1e9, 200e6, 150e6, 150e6, growth=0.0),
        # just as cheap and still paying out, but revenue has collapsed to a quarter
        # of last year's and supply is ballooning: a value trap
        _row("trap", 1e9, 200e6, 800e6, 100e6, cap=2e8, fdv=1e9, growth=0.8),
        # healthy but expensive
        _row("pricey", 5e9, 100e6, 60e6, 80e6, growth=0.01),
        # middling everything
        _row("meh", 2e9, 100e6, 100e6, 20e6, cap=1.5e9, fdv=2e9, growth=0.1),
        # flagged by the ranking: no score
        _row("junk", 5e6, 10e6, 10e6, 0.0, caution=True),
    ])


def test_percentile_ranks_and_neutral_for_unknown():
    s = pd.Series([3.0, 1.0, 2.0, math.nan])
    assert list(score.percentile(s, higher_is_better=True)) == [100.0, 0.0, 50.0, 50.0]
    assert list(score.percentile(s, higher_is_better=False)) == [0.0, 100.0, 50.0, 50.0]
    assert list(score.percentile(pd.Series([math.nan, 4.0]), True)) == [50.0, 50.0]


def test_weights_add_to_100():
    assert sum(w for _, w, _ in score.WEIGHTS.values()) == 100
    assert {p for p, _, _ in score.WEIGHTS.values()} == set(score.PILLARS)


def test_score_separates_value_from_trap():
    out = score.score_table(_frame()).set_index("name")
    assert out.loc["gem", "verdict"] == score.VALUE
    assert out.loc["trap", "verdict"] == score.TRAP
    assert out.loc["pricey", "verdict"] == score.PRICED_IN
    assert out.loc["junk", "verdict"] == score.UNRATED
    assert math.isnan(out.loc["junk", "value_score"])
    assert pd.isna(out.loc["junk", "score_rank"])
    # same P/S, so equally cheap on that metric, but the trap scores far lower overall
    assert out.loc["gem", "p_s_pct"] == out.loc["trap", "p_s_pct"]
    assert out.loc["gem", "value_score"] - out.loc["trap", "value_score"] > 15
    assert out.index[0] == "gem" and out.loc["gem", "score_rank"] == 1
    for c in ("value_score", "cheap_score", "healthy_score", "dilution_score"):
        assert out[c].dropna().between(0, 100).all()


def test_score_is_weighted_average_of_percentiles():
    out = score.score_table(_frame())
    for _, r in out[out["verdict"] != score.UNRATED].iterrows():
        expected = sum(w * r[f"{m}_pct"] for m, (_, w, _) in score.WEIGHTS.items()) / 100
        assert r["value_score"] == pytest.approx(expected)


def test_unrated_rows_do_not_move_others():
    base = score.score_table(_frame()[lambda d: d["name"] != "junk"]).set_index("name")
    full = score.score_table(_frame()).set_index("name")
    assert base["value_score"].to_dict() == pytest.approx(full["value_score"].drop("junk").to_dict())


def test_metrics_and_stopped_payouts():
    df = pd.DataFrame([_row("x", 1e9, 100e6, 50e6, 40e6, cap=5e8, fdv=1e9, h30=0.0, h1y=30e6)])
    m = score.metrics(df).iloc[0]
    assert bool(m["payout_stopped"])
    assert m["holder_yield"] == 0 and m["holder_share"] == 0
    assert m["revenue_trend"] == pytest.approx(2.0)
    assert m["circulating_share"] == pytest.approx(0.5)


@pytest.mark.parametrize("cheap,healthy,expected", [
    (80, 80, score.VALUE), (80, 20, score.TRAP), (30, 70, score.PRICED_IN),
    (30, 20, score.AVOID), (50, 50, score.MIXED), (80, 50, score.MIXED),
])
def test_label(cheap, healthy, expected):
    assert score.label(cheap, healthy) == expected


def test_supply_growth_from_chart():
    caps = [[0, 0.0], [1, 100.0], [2, 300.0]]  # first day untracked (0 cap)
    prices = [[0, 1.0], [1, 1.0], [2, 2.0]]    # supply 100 -> 150
    assert coingecko.supply_growth(caps, prices) == pytest.approx(0.5)
    assert coingecko.supply_growth([[0, 0.0]], [[0, 1.0]]) is None


def test_add_supply_growth_uses_fake_session():
    def handler(url, params, headers):
        if "/coins/bad/" in url:
            return FakeResponse({}, status_code=404)
        return FakeResponse({"market_caps": [[0, 100.0], [1, 110.0]], "prices": [[0, 1.0], [1, 1.0]]})

    df = pd.DataFrame({"gecko_id": ["good", "bad", None]})
    out = score.add_supply_growth(df, session=FakeSession(handler), pause=0)
    assert out["supply_growth"][0] == pytest.approx(0.1)
    assert math.isnan(out["supply_growth"][1]) and math.isnan(out["supply_growth"][2])
