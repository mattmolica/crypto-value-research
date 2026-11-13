import pandas as pd
import pytest

from crypto_value_research import rank_tokens
from crypto_value_research.sources.defillama import ProtocolFlows

def _annual(fees, revenue, holders, year_revenue=None):
    """Same flows in every window, except the trailing year's revenue if given."""
    base = {"fees": fees, "revenue": revenue, "holders_revenue": holders}
    year = dict(base, revenue=year_revenue if year_revenue is not None else revenue)
    return {30: dict(base), 90: dict(base), 365: year}


FLOWS = {
    "alpha": ProtocolFlows("alpha", "Alpha", "alpha-token", _annual(240e6, 120e6, 60e6, 240e6)),
    "beta": ProtocolFlows("beta", "Beta", "beta-wrong-id", _annual(48e6, 12e6, None)),
    "gamma": ProtocolFlows("gamma", "Gamma", None, _annual(None, None, None)),
}

MARKETS = {
    "alpha-token": {"symbol": "alp", "current_price": 2.0, "market_cap": 1e9,
                    "fully_diluted_valuation": 2e9, "total_supply": 1e9},
    "beta-token": {"symbol": "bet", "current_price": 1.0, "market_cap": 1e9,
                   "fully_diluted_valuation": 1e9},
}


@pytest.fixture
def fake_sources(monkeypatch):
    requested = {}

    def fetch_protocol_flows(slug, session=None):
        f = FLOWS[slug]
        return ProtocolFlows(f.slug, f.name, f.gecko_id, {w: dict(v) for w, v in f.annual.items()})

    def fetch_markets(ids, session=None):
        requested["ids"] = list(ids)
        return {i: MARKETS[i] for i in ids if i in MARKETS}

    monkeypatch.setattr(rank_tokens.defillama, "fetch_protocol_flows", fetch_protocol_flows)
    monkeypatch.setattr(rank_tokens.coingecko, "fetch_markets", fetch_markets)
    return requested


TOKENS = {"alpha": None, "beta": "beta-token", "gamma": None}


def test_build_table_joins_sources(fake_sources, capsys):
    df = rank_tokens.build_table(TOKENS, pause=0).set_index("slug")

    # The override replaces DefiLlama's gecko_id; tokens without an id are not requested.
    assert fake_sources["ids"] == ["alpha-token", "beta-token"]
    assert df.loc["beta", "gecko_id"] == "beta-token"
    assert df.loc["alpha", "symbol"] == "ALP"
    assert df.loc["alpha", "fdv"] == 2e9
    assert df.loc["alpha", "p_s"] == pytest.approx(1e9 / 120e6)
    assert df.loc["alpha", "p_s_1y"] == pytest.approx(1e9 / 240e6)
    assert pd.isna(df.loc["gamma", "market_cap"])
    assert "no CoinGecko market data for gamma" in capsys.readouterr().err


def test_build_table_values_locked_tokens_on_issued_supply(fake_sources):
    df = rank_tokens.build_table(TOKENS, pause=0, locked={"alpha"}).set_index("slug")
    # alpha: price $2 x 1B issued tokens = $2B, not the $1B circulating cap
    assert df.loc["alpha", "market_value"] == 2e9
    assert df.loc["alpha", "p_s"] == pytest.approx(2e9 / 120e6)
    assert "issued tokens" in df.loc["alpha", "notes"]
    assert df.loc["beta", "market_value"] == 1e9


def test_build_table_window_picks_flows(fake_sources):
    df = rank_tokens.build_table(TOKENS, pause=0, window=365).set_index("slug")
    assert df.loc["alpha", "revenue_annual"] == 240e6


def test_format_table_renders_units(fake_sources):
    table = rank_tokens.rank(rank_tokens.build_table(TOKENS, pause=0), by="p_s")
    text = rank_tokens.format_table(table)
    assert "$1,000M" in text  # market cap in millions
    assert "50%" in text  # alpha mcap/fdv
    assert "n/a" in text  # gamma has no data
    assert text.index("ALP") < text.index("BET")  # cheapest first


def test_main_writes_csv(fake_sources, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rank_tokens, "DEFAULT_TOKENS", TOKENS)
    monkeypatch.setattr(rank_tokens.time, "sleep", lambda s: None)
    out = tmp_path / "ranking.csv"

    assert rank_tokens.main(["--tokens", "default", "--sort", "p_f", "--out", str(out)]) == 0

    stdout = capsys.readouterr().out
    assert "sorted by p_f" in stdout
    assert "trailing 90 days" in stdout
    saved = pd.read_csv(out)
    assert list(saved["slug"]) == ["alpha", "beta", "gamma"]
    assert list(saved["rank"]) == [1, 2, 3]


def test_main_all_tokens_uses_universe(fake_sources, monkeypatch, tmp_path, capsys):
    from crypto_value_research import universe

    uni = pd.DataFrame([
        {"key": "alpha", "slug": "alpha", "name": "Alpha", "gecko_id": "alpha-token",
         "category": "Dexs", "revenue_1y": 2e8, "revenue_30d": 1e7, "products": 3,
         "status": universe.RANKED},
        {"key": "beta", "slug": "beta", "name": "Beta", "gecko_id": "beta-token",
         "category": "Lending", "revenue_1y": 5e7, "revenue_30d": 1e6, "products": 1,
         "status": universe.RANKED},
        {"key": "corp", "slug": "corp", "name": "Corp", "gecko_id": None,
         "category": "Stablecoin Issuer", "revenue_1y": 9e9, "revenue_30d": 8e8, "products": 1,
         "status": universe.NO_TOKEN},
    ])
    seen = {}

    def build_universe(min_revenue, session=None):
        seen["min_revenue"] = min_revenue
        return uni

    monkeypatch.setattr(rank_tokens.universe_mod, "build_universe", build_universe)
    monkeypatch.setattr(rank_tokens.time, "sleep", lambda s: None)
    out, uni_out = tmp_path / "ranking.csv", tmp_path / "universe.csv"

    assert rank_tokens.main(["--min-revenue", "5e6", "--out", str(out),
                             "--universe-out", str(uni_out)]) == 0

    assert seen["min_revenue"] == 5e6
    saved = pd.read_csv(out)
    assert list(saved["slug"]) == ["alpha", "beta"]  # corp has no token, so isn't ranked
    assert list(saved["category"]) == ["Dexs", "Lending"]
    assert "caution" in saved.columns
    assert len(pd.read_csv(uni_out)) == 3
    assert "Universe: 2 projects" in capsys.readouterr().err


def test_build_table_skips_protocols_that_error(fake_sources, monkeypatch, capsys):
    import requests

    original = rank_tokens.defillama.fetch_protocol_flows

    def flaky(slug, session=None):
        if slug == "beta":
            raise requests.HTTPError("HTTP 500")
        return original(slug, session)

    monkeypatch.setattr(rank_tokens.defillama, "fetch_protocol_flows", flaky)
    df = rank_tokens.build_table(TOKENS, pause=0)
    assert list(df["slug"]) == ["alpha", "gamma"]
    assert "skipped beta" in capsys.readouterr().err


def test_locked_tokens_match_by_coingecko_id(fake_sources):
    df = rank_tokens.build_table({"alpha": None}, pause=0, locked={"alpha-token"}).set_index("slug")
    assert df.loc["alpha", "market_value"] == 2e9


def test_main_rejects_unknown_sort():
    with pytest.raises(SystemExit):
        rank_tokens.main(["--sort", "nonsense"])
