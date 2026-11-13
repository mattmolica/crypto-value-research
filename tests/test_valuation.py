import math

import pandas as pd

from crypto_value_research.valuation import (
    NOTE_FDV, NOTE_LOCKED, NOTE_NO_MARKET, NOTE_NO_REVENUE, NOTE_P_S_BELOW_1,
    NOTE_PAYOUT_STOPPED, NOTE_TINY, add_multiples, rank,
)


def _frame():
    return pd.DataFrame([
        # cheap: $1B cap, $120M/yr revenue
        {"name": "cheap", "market_cap": 1e9, "fdv": 2e9, "fees_annual": 240e6,
         "revenue_annual": 120e6, "holders_revenue_annual": 60e6},
        # expensive: $1B cap, $12M/yr revenue
        {"name": "pricey", "market_cap": 1e9, "fdv": 1e9, "fees_annual": 60e6,
         "revenue_annual": 12e6, "holders_revenue_annual": None},
        # no revenue at all
        {"name": "none", "market_cap": 5e8, "fdv": None, "fees_annual": 36e6,
         "revenue_annual": 0.0, "holders_revenue_annual": None},
    ])


def test_multiples():
    df = add_multiples(_frame()).set_index("name")
    assert math.isclose(df.loc["cheap", "p_s"], 1e9 / 120e6)
    assert math.isclose(df.loc["cheap", "p_f"], 1e9 / 240e6)
    assert math.isclose(df.loc["cheap", "fdv_rev"], 2e9 / 120e6)
    assert math.isclose(df.loc["cheap", "mcap_to_fdv"], 0.5)
    assert df.loc["cheap", "market_value"] == 1e9
    assert math.isnan(df.loc["cheap", "p_s_1y"])  # no 1y column given
    assert math.isnan(df.loc["pricey", "p_e_holders"])
    assert math.isnan(df.loc["none", "p_s"])  # zero revenue -> no multiple
    assert math.isnan(df.loc["none", "fdv_rev"])
    assert df.loc["cheap", "notes"] == df.loc["pricey", "notes"] == ""
    assert df.loc["none", "notes"] == NOTE_NO_REVENUE
    assert list(df["caution"]) == [False, False, True]


def test_locked_supply_valued_on_issued_tokens():
    df = pd.DataFrame([
        {"name": "ve", "price": 0.5, "market_cap": 100e6, "total_supply": 1e9, "fdv": 500e6,
         "locked_supply": True, "fees_annual": 100e6, "revenue_annual": 50e6,
         "holders_revenue_annual": 50e6},
        # locked but no supply data -> fall back to market cap, no note
        {"name": "ve-nodata", "price": None, "market_cap": 100e6, "total_supply": None,
         "fdv": None, "locked_supply": True, "fees_annual": 1e6, "revenue_annual": 1e6,
         "holders_revenue_annual": 1e6},
    ])
    out = add_multiples(df).set_index("name")
    assert out.loc["ve", "market_value"] == 500e6
    assert math.isclose(out.loc["ve", "p_e_holders"], 10.0)
    assert math.isclose(out.loc["ve", "mcap_to_fdv"], 0.2)  # still on circulating cap
    assert out.loc["ve", "notes"] == NOTE_LOCKED
    assert out.loc["ve-nodata", "market_value"] == 100e6
    assert out.loc["ve-nodata", "notes"] == ""


def test_payout_stopped_note():
    df = _frame()
    df["holders_revenue_annual_30d"] = [0.0, 0.0, None]
    df["holders_revenue_annual_1y"] = [5e6, 0.0, 5e6]
    df["revenue_annual_1y"] = [60e6, 12e6, 0.0]
    out = add_multiples(df).set_index("name")
    assert out.loc["cheap", "notes"] == NOTE_PAYOUT_STOPPED  # paid this year, not lately
    assert out.loc["pricey", "notes"] == ""  # never paid
    assert out.loc["none", "notes"] == NOTE_NO_REVENUE  # no payout note without recent data
    assert math.isclose(out.loc["cheap", "p_s_1y"], 1e9 / 60e6)


def test_rank_puts_cheapest_first_and_missing_last():
    ranked = rank(add_multiples(_frame()), by="p_s")
    assert list(ranked["name"]) == ["cheap", "pricey", "none"]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_ratio_edge_cases():
    from crypto_value_research.valuation import _ratio

    assert _ratio(10.0, 2.0) == 5.0
    assert math.isnan(_ratio(None, 2.0))
    assert math.isnan(_ratio(10.0, None))
    assert math.isnan(_ratio(math.nan, 2.0))
    assert math.isnan(_ratio(10.0, math.nan))
    assert math.isnan(_ratio(10.0, 0))
    assert math.isnan(_ratio(10.0, -5.0))  # negative revenue -> no multiple


def test_add_multiples_does_not_mutate_input():
    df = _frame()
    before = df.copy()
    add_multiples(df)
    pd.testing.assert_frame_equal(df, before)


def test_rank_by_other_column():
    ranked = rank(add_multiples(_frame()), by="p_f")
    # p_f: cheap 1e9/240e6, pricey 1e9/60e6, none 5e8/36e6
    assert list(ranked["name"]) == ["cheap", "none", "pricey"]


def _one(**kw):
    row = {"name": "x", "market_cap": 1e9, "fdv": 1e9, "fees_annual": 100e6,
           "revenue_annual": 50e6, "holders_revenue_annual": None}
    row.update(kw)
    return add_multiples(pd.DataFrame([row])).iloc[0]


def test_clean_row_has_no_caution():
    r = _one()
    assert not r["caution"]
    assert r["notes"] == ""


def test_missing_circulating_cap_falls_back_to_fdv():
    for cap in (None, 0.0):
        r = _one(market_cap=cap, fdv=2e9)
        assert r["market_value"] == 2e9
        assert math.isclose(r["p_s"], 2e9 / 50e6)
        assert r["notes"] == NOTE_FDV
        assert not r["caution"]


def test_no_market_data_is_a_caution():
    r = _one(market_cap=None, fdv=None, has_market_data=False)
    assert math.isnan(r["p_s"])
    assert r["caution"]
    assert NOTE_NO_MARKET in r["notes"]


def test_tiny_market_value_is_a_caution():
    r = _one(market_cap=5e6, fdv=5e6, revenue_annual=1e6)
    assert r["caution"]
    assert r["notes"] == NOTE_TINY


def test_revenue_above_market_value_is_a_caution():
    r = _one(market_cap=40e6, fdv=40e6, revenue_annual=50e6)
    assert r["caution"]
    assert NOTE_P_S_BELOW_1 in r["notes"]


def test_young_protocol_is_a_caution():
    r = add_multiples(pd.DataFrame([{
        "market_cap": 1e9, "fdv": 1e9, "fees_annual": 1e8, "revenue_annual": 5e7,
        "holders_revenue_annual": None, "history_days": 40}]), window=90).iloc[0]
    assert r["caution"]
    assert r["notes"] == "only 40 days of revenue history"
    old = add_multiples(pd.DataFrame([{
        "market_cap": 1e9, "fdv": 1e9, "fees_annual": 1e8, "revenue_annual": 5e7,
        "holders_revenue_annual": None, "history_days": 400}]), window=90).iloc[0]
    assert not old["caution"]


def test_falling_revenue_is_noted_but_not_a_caution():
    r = _one(revenue_annual=20e6, revenue_annual_1y=100e6)
    assert not r["caution"]
    assert r["notes"] == "revenue run rate down 80% vs the past year"
    assert _one(revenue_annual=60e6, revenue_annual_1y=100e6)["notes"] == ""
