import math

import pandas as pd
import pytest

from crypto_value_research import backtest


def history(weeks=80, tokens=6):
    """Synthetic weekly data: token i has P/S (i+1)*10 and grows (tokens-i)% a week,
    so the cheapest tokens always do best."""
    dates = pd.date_range("2022-01-02", periods=weeks, freq="W-SUN")
    rows = []
    for i in range(tokens):
        for w, d in enumerate(dates):
            rows.append({"date": d, "symbol": f"T{i}", "price": (1 + (tokens - i) / 100) ** w,
                         "ps_fdv": (i + 1) * 10.0})
    return pd.DataFrame(rows)


def test_load_keeps_sundays(tmp_path):
    df = history(5, 2)
    extra = df.iloc[[0]].assign(date=pd.Timestamp("2022-01-05"))
    path = tmp_path / "h.csv"
    pd.concat([df, extra]).to_csv(path, index=False)
    assert len(backtest.load(path)) == len(df)


def test_forward_returns():
    df = backtest.add_forward_returns(history(30, 1), (13,))
    assert df["fwd_13"].iloc[0] == pytest.approx(1.01 ** 13 - 1)
    assert df["fwd_13"].iloc[-1] != df["fwd_13"].iloc[-1]  # NaN: future not known


def test_thirds():
    t = backtest.thirds(pd.Series([5.0, 1.0, 3.0, 2.0, 4.0, 6.0]))
    assert list(t) == ["expensive", "cheap", "middle", "cheap", "middle", "expensive"]


def test_cross_section_finds_cheap_winning():
    df = backtest.add_forward_returns(history())
    cs = backtest.cross_section(df, 13)
    assert (cs["spread"] > 0).all()
    assert cs["rank_corr"].iloc[0] == pytest.approx(1.0)
    s = backtest.summarize_cross(cs, 13)
    assert s["cheap_won_pct"] == 1.0 and s["independent_periods"] == len(cs) // 13


def test_too_few_tokens_skipped():
    df = backtest.add_forward_returns(history(tokens=4))
    assert backtest.cross_section(df, 13).empty


def test_growth_of_dollar_compounds():
    g = backtest.growth_of_dollar(history())
    assert g.iloc[0]["cheap"] == 1.0
    assert g.iloc[-1]["cheap"] > g.iloc[-1]["all"] > g.iloc[-1]["expensive"]
    # cheapest two grow 6% and 5% a week
    step = ((1.06 ** 13) + (1.05 ** 13)) / 2
    assert g.iloc[1]["cheap"] == pytest.approx(step)


def test_own_history_rank_uses_only_past():
    df = pd.DataFrame({"date": pd.date_range("2022-01-02", periods=4, freq="W-SUN"),
                       "symbol": "A", "price": 1.0, "ps_fdv": [10.0, 20.0, 5.0, 30.0]})
    r = backtest.own_history_rank(df, min_weeks=2)
    assert math.isnan(r.iloc[1])
    assert r.iloc[2] == 1.0          # cheaper than both earlier weeks
    assert r.iloc[3] == 0.0          # dearer than all earlier weeks


def test_run_produces_all_sections():
    out = backtest.run(history(tokens=7))
    assert {"cheap_vs_peers", "growth_of_dollar", "growth_leave_one_out", "cheap_vs_own_past"} <= set(out)
    assert len(out["growth_leave_one_out"]) == 7


def test_load_filters_small_fdv_and_symbols(tmp_path):
    df = history(5, 2)
    df["fdv"] = [1e6] * 5 + [1e9] * 5
    path = tmp_path / "h.csv"
    df.to_csv(path, index=False)
    out = backtest.load(path, min_fdv=1e7)
    assert out.loc[out.symbol == "T0", "ps_fdv"].isna().all()
    assert out.loc[out.symbol == "T1", "ps_fdv"].notna().all()
    assert set(backtest.load(path, symbols={"T1"})["symbol"]) == {"T1"}


def test_exact_supply_symbols(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"symbol": ["A", "B"], "supply_method": ["exact: fixed total supply", "estimated: x"]}).to_csv(p, index=False)
    assert backtest.exact_supply_symbols(p) == {"A"}


def test_medians_reported():
    s = backtest.summarize_cross(backtest.cross_section(backtest.add_forward_returns(history()), 13), 13)
    assert s["cheap_median_won_pct"] == 1.0 and s["avg_tokens"] == 6


def test_checks_rows(tmp_path):
    df = history(80, 7)
    df["fdv"] = 1e9
    hp, sp = tmp_path / "h.csv", tmp_path / "s.csv"
    df.to_csv(hp, index=False)
    pd.DataFrame({"symbol": [f"T{i}" for i in range(7)], "supply_method": "exact: fixed"}).to_csv(sp, index=False)
    main = backtest.run(backtest.load(hp, 1e7))
    rows = backtest.checks(hp, sp, 1e7, main)
    assert [r["variant"] for r in rows][0].startswith("Main")
    assert len(rows) == 3 and rows[0]["growth_cheap"] > rows[0]["growth_expensive"]
