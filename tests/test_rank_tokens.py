import pandas as pd
import pytest

from crypto_value_research import rank_tokens
from crypto_value_research.sources.defillama import ProtocolFlows

FLOWS = {
    "alpha": ProtocolFlows("alpha", "Alpha", "alpha-token",
                           {"fees": 20e6, "revenue": 10e6, "holders_revenue": 5e6}),
    "beta": ProtocolFlows("beta", "Beta", "beta-wrong-id",
                          {"fees": 4e6, "revenue": 1e6, "holders_revenue": None}),
    "gamma": ProtocolFlows("gamma", "Gamma", None,
                           {"fees": None, "revenue": None, "holders_revenue": None}),
}

MARKETS = {
    "alpha-token": {"symbol": "alp", "current_price": 2.0, "market_cap": 1e9,
                    "fully_diluted_valuation": 2e9},
    "beta-token": {"symbol": "bet", "current_price": 1.0, "market_cap": 1e9,
                   "fully_diluted_valuation": 1e9},
}


@pytest.fixture
def fake_sources(monkeypatch):
    requested = {}

    def fetch_protocol_flows(slug, session=None):
        f = FLOWS[slug]
        return ProtocolFlows(f.slug, f.name, f.gecko_id, dict(f.total_30d))

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
    assert df.loc["alpha", "p_s"] == pytest.approx(1e9 / (10e6 * 365 / 30))
    assert pd.isna(df.loc["gamma", "market_cap"])
    assert "no CoinGecko market data for gamma" in capsys.readouterr().err


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

    assert rank_tokens.main(["--sort", "p_f", "--out", str(out)]) == 0

    stdout = capsys.readouterr().out
    assert "sorted by p_f" in stdout
    saved = pd.read_csv(out)
    assert list(saved["slug"]) == ["alpha", "beta", "gamma"]
    assert list(saved["rank"]) == [1, 2, 3]


def test_main_rejects_unknown_sort():
    with pytest.raises(SystemExit):
        rank_tokens.main(["--sort", "nonsense"])
