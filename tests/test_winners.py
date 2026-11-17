import math

import numpy as np
import pandas as pd
import pytest

from crypto_value_research import winners


def synthetic(weeks=120, tokens=12, seed=0):
    """Weekly history where tokens whose P/S rises above their own past then fall behind.

    Token i's P/S swings on its own cycle; its price drifts down when P/S is
    high vs its past and up when low. BTC is flat, so beating BTC = going up.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-03", periods=weeks, freq="W-SUN")
    rows = []
    for i in range(tokens):
        price = 1.0
        for w, d in enumerate(dates):
            ps = 10 * (1.5 + np.sin(w / 6 + i))
            drift = -0.03 * np.sin(w / 6 + i)  # pricey -> drifts down next weeks
            price *= math.exp(drift + rng.normal(0, 0.01))
            rev = 1e7 * (1 + i)
            rows.append({"date": d, "symbol": f"T{i}", "price": price, "fdv": ps * rev,
                         "revenue_annual_90d": rev, "fees_annual_90d": 2 * rev, "ps_fdv": ps})
    hist = pd.DataFrame(rows)
    btc = pd.Series(100.0, index=pd.date_range(dates[0], dates[-1], freq="D"))
    return hist, btc


def test_own_rank_uses_only_the_past():
    s = pd.Series([1.0] * 30 + [5.0, 0.5])
    r = winners.own_rank(s, min_weeks=26)
    assert math.isnan(r.iloc[0])
    assert r.iloc[30] == 1.0  # pricier than every earlier week
    assert r.iloc[31] == 0.0  # cheaper than every earlier week


def test_panel_has_no_lookahead_in_features():
    hist, btc = synthetic(80, 10)
    p = winners.build_panel(hist, btc, 13)
    # changing prices after a date must not change features on that date
    cut = p["date"].min() + pd.Timedelta(weeks=20)
    hist2 = hist.copy()
    hist2.loc[hist2["date"] > cut, "price"] *= 3
    p2 = winners.build_panel(hist2, btc, 13)
    a = p[p["date"] == cut].set_index("symbol")
    b = p2[p2["date"] == cut].set_index("symbol")
    for f in ("mom_13", "volatility", "drawdown", "ps_own"):
        pd.testing.assert_series_equal(a[f], b[f])


def test_excess_is_vs_btc():
    hist, btc = synthetic(60, 10)
    p = winners.build_panel(hist, btc, 13)
    row = p.iloc[0]
    assert row["excess"] == pytest.approx(math.log1p(row["ret"]))  # BTC flat
    assert row["beat"] == int(row["ret"] > 0)


def test_feature_tests_spot_planted_signal():
    hist, btc = synthetic()
    p = winners.build_panel(hist, btc, 4)
    ft = winners.feature_tests(p, winners.window_starts(p, 4)).set_index("feature")
    assert ft.loc["ps_own", "rank_corr"] < -0.3
    assert ft.loc["ps_own", "bottom_third_beat_btc"] > ft.loc["ps_own", "top_third_beat_btc"]


def test_walk_forward_trains_on_finished_windows_only():
    hist, btc = synthetic()
    p = winners.build_panel(hist, btc, 4)
    starts = winners.window_starts(p, 4)
    wf = winners.walk_forward(p, 4, starts, feats=("ps_own",))
    assert not wf.empty
    first = wf.iloc[0]
    finished = [d for d in starts if d + pd.Timedelta(weeks=4) <= first["date"]]
    assert first["train_windows"] == len(finished)
    assert wf["auc_beat_median"].mean() > 0.6


def test_logistic_and_auc():
    x = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0, 0, 1, 1])
    w = winners.fit_logistic(x, y, l2=0.1)
    assert w[1] > 0
    assert winners.auc(winners.predict(w, x), y) == 1.0
    assert winners.auc([0.1, 0.2], [1, 1]) != winners.auc([0.1, 0.2], [1, 1])  # NaN with one class
