import math

import numpy as np
import pandas as pd
import pytest

from conftest import FakeResponse
from crypto_value_research import history
from crypto_value_research.sources import coingecko, llama_prices


# --- DefiLlama price history ---------------------------------------------------

def test_price_history_walks_in_500_day_chunks(fake_session):
    start = 1_600_000_000
    end = start + 1200 * llama_prices.DAY

    def handler(url, params, headers):
        t = params["start"]
        return FakeResponse({"coins": {"coingecko:uni": {"prices": [
            {"timestamp": t, "price": 1.0}, {"timestamp": t + llama_prices.DAY, "price": 2.0}]}}})

    s = fake_session(handler)
    prices = llama_prices.fetch_price_history("uni", start=start, end=end, session=s, pause=0)
    assert [c["params"]["start"] for c in s.calls] == [start, start + 500 * 86400, start + 1000 * 86400]
    assert all(c["params"]["span"] == 500 for c in s.calls)
    assert s.calls[0]["url"] == f"{llama_prices.BASE_URL}/chart/coingecko:uni"
    assert len(prices) == 6 and prices.index.is_monotonic_increasing


def test_price_history_empty_when_unknown(fake_session):
    s = fake_session(lambda *a: FakeResponse({"coins": {}}))
    assert llama_prices.fetch_price_history("nope", start=0, end=10, session=s, pause=0).empty


def test_market_cap_history_keeps_last_value_per_day(fake_session):
    ms = int(pd.Timestamp("2026-01-01").timestamp() * 1000)
    s = fake_session(lambda *a: FakeResponse({"market_caps": [[ms, 1.0], [ms + 3_600_000, 2.0], [ms + 86_400_000, 3.0]]}))
    caps = coingecko.fetch_market_cap_history("uni", session=s)
    assert list(caps) == [2.0, 3.0]
    assert s.calls[0]["params"]["days"] == 365


def test_market_cap_history_retries_when_rate_limited(fake_session):
    answers = iter([FakeResponse(status_code=429), FakeResponse({"market_caps": [[0, 5.0]]})])
    s = fake_session(lambda *a: next(answers))
    caps = coingecko.fetch_market_cap_history("uni", session=s, backoff=0)
    assert list(caps) == [5.0] and len(s.calls) == 2


# --- supply models ---------------------------------------------------------------

def test_crv_supply_follows_emission_schedule():
    assert math.isnan(history.crv_supply(pd.Timestamp("2020-01-01")))
    assert history.crv_supply(history.CRV_START) == pytest.approx(history.CRV_INITIAL)
    one_year = history.CRV_START + pd.Timedelta(days=365.25)
    assert history.crv_supply(one_year) == pytest.approx(history.CRV_INITIAL + history.CRV_FIRST_YEAR_RATE, rel=1e-3)
    # the schedule lands near the ~2.43B CoinGecko reported in Nov 2026
    assert history.crv_supply(pd.Timestamp("2026-11-10")) == pytest.approx(2.43e9, rel=0.02)


def test_linear_supply_runs_from_start_to_today():
    idx = pd.date_range("2023-08-28", periods=11, freq="D")
    s = history.linear_supply(idx, idx[0], 100.0, 200.0)
    assert s.iloc[0] == 100.0 and s.iloc[-1] == 200.0 and s.iloc[5] == pytest.approx(150.0)


def test_fixed_supply_is_flat():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    assert list(history.supply_history("lido", idx, 1e9)) == [1e9] * 3


# --- valuation history and summary ----------------------------------------------

def _flat(start, n, value):
    return pd.Series(value, index=pd.date_range(start, periods=n, freq="D"), dtype=float)


def test_trailing_annual_needs_a_full_window():
    s = history.trailing_annual(_flat("2024-01-01", 100, 1.0), days=90)
    assert s.iloc[:89].isna().all()
    assert s.iloc[89] == pytest.approx(365.0)


def test_daily_series_sums_per_day_and_fills_gaps():
    t0 = int(pd.Timestamp("2024-01-01").timestamp())
    s = history.daily_series([[t0, 1.0], [t0 + 3600, 2.0], [t0 + 2 * 86400, 5.0]])
    assert list(s) == [3.0, 0.0, 5.0]


def test_valuation_history_multiples():
    n = 200
    prices = _flat("2024-01-01", n, 2.0)
    supply = _flat("2024-01-01", n, 1000.0)
    revenue = _flat("2024-01-01", n, 1.0)  # 365 a year
    fees = _flat("2024-01-01", n, 2.0)
    mcap = _flat("2024-01-01", n, 500.0)
    df = history.valuation_history(prices, supply, revenue, fees, mcap)
    last = df.iloc[-1]
    assert last.fdv == 2000.0
    assert last.ps_fdv == pytest.approx(2000 / 365)
    assert last.pf_fdv == pytest.approx(2000 / 730)
    assert last.ps_mcap == pytest.approx(500 / 365)
    assert df.ps_fdv.iloc[:89].isna().all()  # no multiple before a full 90-day window


def test_valuation_history_zero_revenue_gives_no_multiple():
    n = 120
    df = history.valuation_history(_flat("2024-01-01", n, 1.0), _flat("2024-01-01", n, 1.0),
                                   _flat("2024-01-01", n, 0.0), _flat("2024-01-01", n, 0.0))
    assert df.ps_fdv.isna().all()


def test_summarize_ranks_today_against_history():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"ps_fdv": [10.0, 20.0, 30.0, 40.0, 15.0], "revenue_annual_90d": [100.0] * 5}, index=idx)
    s = history.summarize(df)
    assert s["ps_fdv_now"] == 15.0
    assert s["ps_fdv_low"] == 10.0 and s["ps_fdv_high"] == 40.0 and s["ps_fdv_median"] == 20.0
    assert s["cheaper_than_pct"] == pytest.approx(0.6)  # 20, 30, 40 were pricier


def test_summarize_ignores_days_with_tiny_revenue():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"ps_fdv": [5000.0, 10.0, 20.0, 15.0], "revenue_annual_90d": [1.0, 100.0, 100.0, 100.0]}, index=idx)
    s = history.summarize(df)
    assert s["ps_fdv_high"] == 20.0
    assert s["ps_history_since"] == "2024-01-02"


def test_weekly_keeps_sundays_and_latest_day():
    idx = pd.date_range("2024-01-01", "2024-01-17", freq="D")  # Mon .. Wed
    df = pd.DataFrame({"fdv": np.arange(len(idx), dtype=float)}, index=idx)
    w = history.weekly(df)
    assert list(w.index.strftime("%Y-%m-%d")) == ["2024-01-07", "2024-01-14", "2024-01-17"]
