import pandas as pd
import pytest

from conftest import FakeResponse
from crypto_value_research import universe
from crypto_value_research.sources import defillama

# A small slice shaped like DefiLlama's overview and directory responses.
OVERVIEW = [
    # Two products of one project, which owns the token.
    {"slug": "swap-v2", "name": "Swap V2", "defillamaId": "1", "parentProtocol": "parent#swap",
     "category": "Dexs", "protocolType": "protocol", "total1y": 3e6, "total30d": 2e5},
    {"slug": "swap-v3", "name": "Swap V3", "defillamaId": "2", "parentProtocol": "parent#swap",
     "category": "Dexs", "protocolType": "protocol", "total1y": 9e6, "total30d": 1e6},
    # Standalone protocol with its own token.
    {"slug": "lendy", "name": "Lendy", "defillamaId": "3", "category": "Lending",
     "protocolType": "protocol", "total1y": 20e6, "total30d": 1.5e6},
    # Big earner, no token.
    {"slug": "stableco", "name": "StableCo", "defillamaId": "4", "category": "Stablecoin Issuer",
     "protocolType": "protocol", "total1y": 5e9, "total30d": 4e8},
    # Too small.
    {"slug": "tiny", "name": "Tiny", "defillamaId": "5", "category": "Yield",
     "protocolType": "protocol", "total1y": 2e5, "total30d": 1e4},
    # Chains are left out entirely.
    {"slug": "somechain", "name": "SomeChain", "defillamaId": "6", "category": "Chain",
     "protocolType": "chain", "total1y": 9e8, "total30d": 7e7},
    # Smaller project pointing at Lendy's token.
    {"slug": "lendy-lite", "name": "Lendy Lite", "defillamaId": "7", "category": "Lending",
     "protocolType": "protocol", "total1y": 2e6, "total30d": 1e5},
    # Gold-backed token.
    {"slug": "goldco", "name": "GoldCo", "defillamaId": "8", "category": "RWA",
     "protocolType": "protocol", "total1y": 7e6, "total30d": 6e5},
    # Missing totals must not crash.
    {"slug": "blank", "name": "Blank", "defillamaId": "9", "category": "Dexs",
     "protocolType": "protocol"},
]

DIRECTORY = {
    "protocols": [
        {"defillamaId": "1", "name": "Swap V2", "geckoId": None},
        {"defillamaId": "2", "name": "Swap V3", "geckoId": None},
        {"defillamaId": "3", "name": "Lendy", "geckoId": "lendy-token"},
        {"defillamaId": "4", "name": "StableCo", "geckoId": None},
        {"defillamaId": "5", "name": "Tiny", "geckoId": "tiny-token"},
        {"defillamaId": "6", "name": "SomeChain", "geckoId": "somechain"},
        {"defillamaId": "7", "name": "Lendy Lite", "geckoId": "lendy-token"},
        {"defillamaId": "8", "name": "GoldCo", "geckoId": "tether-gold"},
    ],
    "parentProtocols": [{"id": "parent#swap", "name": "Swap Pro", "gecko_id": "swap-token"}],
}


@pytest.fixture
def uni():
    return universe.classify(universe.group_projects(OVERVIEW, DIRECTORY)).set_index("key")


def test_products_fold_into_their_parent(uni):
    swap = uni.loc["parent#swap"]
    assert swap["revenue_1y"] == 12e6
    assert swap["revenue_30d"] == 1.2e6
    assert swap["products"] == 2
    assert swap["gecko_id"] == "swap-token"
    assert swap["name"] == "Swap Pro"
    assert swap["slug"] == "swap-pro"  # DefiLlama's slug for the parent's name


def test_chains_are_left_out(uni):
    assert "somechain" not in uni.index


def test_statuses(uni):
    assert uni.loc["parent#swap", "status"] == universe.RANKED
    assert uni.loc["lendy", "status"] == universe.RANKED
    assert uni.loc["stableco", "status"] == universe.NO_TOKEN
    assert uni.loc["tiny", "status"] == universe.TOO_SMALL
    assert uni.loc["lendy-lite", "status"] == f"{universe.SHARED_TOKEN} (Lendy)"
    assert uni.loc["goldco", "status"].startswith(universe.EXCLUDED)
    assert uni.loc["blank", "status"] == universe.TOO_SMALL


def test_sorted_biggest_first(uni):
    assert list(uni.index[:2]) == ["stableco", "lendy"]


def test_revenue_floor_is_adjustable():
    out = universe.classify(universe.group_projects(OVERVIEW, DIRECTORY), min_revenue=15e6)
    assert universe.revenue_tokens(out) == {"lendy": "lendy-token"}


def test_revenue_tokens_dict(uni):
    assert universe.revenue_tokens(uni.reset_index()) == {
        "lendy": "lendy-token", "swap-pro": "swap-token"}


def test_parent_without_token_uses_products_if_they_agree():
    directory = {
        "protocols": [{"defillamaId": "1", "geckoId": "swap-token"},
                      {"defillamaId": "2", "geckoId": "swap-token"}],
        "parentProtocols": [{"id": "parent#swap", "name": "Swap", "gecko_id": None}],
    }
    g = universe.group_projects(OVERVIEW[:2], directory).iloc[0]
    assert g["gecko_id"] == "swap-token"
    directory["protocols"][1]["geckoId"] = "other-token"  # products disagree -> no guess
    g = universe.group_projects(OVERVIEW[:2], directory).iloc[0]
    assert g["gecko_id"] is None


def test_build_universe_fetches_both_endpoints(fake_session):
    def handler(url, params, headers):
        if url.endswith("/overview/fees"):
            assert params["dataType"] == "dailyRevenue"
            return FakeResponse({"protocols": OVERVIEW})
        if url.endswith("/lite/protocols2"):
            return FakeResponse(dict(DIRECTORY, chains=[]))
        raise AssertionError(url)

    out = universe.build_universe(session=fake_session(handler))
    assert set(universe.revenue_tokens(out)) == {"lendy", "swap-pro"}


def test_main_prints_counts(monkeypatch, tmp_path, capsys):
    frame = universe.classify(universe.group_projects(OVERVIEW, DIRECTORY))
    monkeypatch.setattr(universe, "build_universe", lambda min_revenue: frame)
    out = tmp_path / "u.csv"
    assert universe.main(["--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ranked: 2" in text
    assert "StableCo" in text
    assert len(pd.read_csv(out)) == len(frame)


def test_slugify():
    assert defillama.slugify("Sky") == "sky"
    assert defillama.slugify("ether.fi") == "ether.fi"
    assert defillama.slugify("Curve Finance") == "curve-finance"
    assert defillama.slugify("Kelp's DAO") == "kelps-dao"
