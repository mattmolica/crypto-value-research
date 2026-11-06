import math

import pandas as pd

from crypto_value_research.valuation import ANNUALIZE_30D, add_multiples, rank


def _frame():
    return pd.DataFrame([
        # cheap: $1B cap, $10M/30d revenue
        {"name": "cheap", "market_cap": 1e9, "fdv": 2e9, "fees_30d": 20e6,
         "revenue_30d": 10e6, "holders_revenue_30d": 5e6},
        # expensive: $1B cap, $1M/30d revenue
        {"name": "pricey", "market_cap": 1e9, "fdv": 1e9, "fees_30d": 5e6,
         "revenue_30d": 1e6, "holders_revenue_30d": None},
        # no revenue at all
        {"name": "none", "market_cap": 5e8, "fdv": None, "fees_30d": 3e6,
         "revenue_30d": 0.0, "holders_revenue_30d": None},
    ])


def test_multiples():
    df = add_multiples(_frame()).set_index("name")
    assert math.isclose(df.loc["cheap", "p_s"], 1e9 / (10e6 * ANNUALIZE_30D))
    assert math.isclose(df.loc["cheap", "p_f"], 1e9 / (20e6 * ANNUALIZE_30D))
    assert math.isclose(df.loc["cheap", "fdv_rev"], 2e9 / (10e6 * ANNUALIZE_30D))
    assert math.isclose(df.loc["cheap", "mcap_to_fdv"], 0.5)
    assert math.isnan(df.loc["pricey", "p_e_holders"])
    assert math.isnan(df.loc["none", "p_s"])  # zero revenue -> no multiple
    assert math.isnan(df.loc["none", "fdv_rev"])


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
    # p_f: cheap 1e9/(20e6*k), pricey 1e9/(5e6*k), none 5e8/(3e6*k)
    assert list(ranked["name"]) == ["cheap", "none", "pricey"]
