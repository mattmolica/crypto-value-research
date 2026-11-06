import pytest
import requests

from conftest import FakeResponse
from crypto_value_research.sources import coingecko, defillama


# --- DefiLlama ---------------------------------------------------------------

def test_fetch_summary_returns_none_on_missing(fake_session):
    for status in (400, 404):
        s = fake_session(lambda url, params, headers, st=status: FakeResponse(status_code=st))
        assert defillama.fetch_summary("nope", "dailyFees", s) is None


def test_fetch_summary_raises_on_server_error(fake_session):
    s = fake_session(lambda *a: FakeResponse(status_code=500))
    with pytest.raises(requests.HTTPError):
        defillama.fetch_summary("aave", "dailyFees", s)


def test_fetch_summary_request_shape(fake_session):
    s = fake_session(lambda *a: FakeResponse({"total30d": 1}))
    defillama.fetch_summary("aave", "dailyRevenue", s)
    call = s.calls[0]
    assert call["url"] == f"{defillama.BASE_URL}/summary/fees/aave"
    assert call["params"]["dataType"] == "dailyRevenue"
    assert call["params"]["excludeTotalDataChart"] == "true"


def test_fetch_protocol_flows_collects_all_types(fake_session):
    totals = {"dailyFees": 30e6, "dailyRevenue": 9e6}  # no holders revenue series

    def handler(url, params, headers):
        dt = params["dataType"]
        if dt not in totals:
            return FakeResponse(status_code=404)
        return FakeResponse({"name": "Aave", "gecko_id": "aave", "total30d": totals[dt]})

    flows = defillama.fetch_protocol_flows("aave", fake_session(handler))
    assert flows.slug == "aave"
    assert flows.name == "Aave"
    assert flows.gecko_id == "aave"
    assert flows.total_30d == {"fees": 30e6, "revenue": 9e6, "holders_revenue": None}


def test_fetch_protocol_flows_unknown_slug_keeps_defaults(fake_session):
    s = fake_session(lambda *a: FakeResponse(status_code=404))
    flows = defillama.fetch_protocol_flows("ghost", s)
    assert flows.name == "ghost"
    assert flows.gecko_id is None
    assert set(flows.total_30d.values()) == {None}
    assert len(s.calls) == len(defillama.DATA_TYPES)


# --- CoinGecko ---------------------------------------------------------------

ROW = {
    "id": "aave", "symbol": "aave", "name": "Aave", "current_price": 100.0,
    "market_cap": 1.5e9, "fully_diluted_valuation": 1.6e9, "circulating_supply": 15e6,
    "total_supply": 16e6, "max_supply": 16e6, "extra_field": "dropped",
}


def test_fetch_markets_empty_ids_makes_no_request(fake_session):
    s = fake_session(lambda *a: pytest.fail("should not be called"))
    assert coingecko.fetch_markets([], s) == {}
    assert s.calls == []


def test_fetch_markets_keys_by_id_and_keeps_known_fields(fake_session, monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    s = fake_session(lambda *a: FakeResponse([ROW]))
    out = coingecko.fetch_markets(["aave", "missing"], s)
    assert list(out) == ["aave"]
    assert set(out["aave"]) == set(coingecko.FIELDS)
    assert out["aave"]["market_cap"] == 1.5e9
    call = s.calls[0]
    assert call["params"]["ids"] == "aave,missing"
    assert call["params"]["vs_currency"] == "usd"
    assert call["headers"] == {}


def test_fetch_markets_sends_api_key_when_set(fake_session, monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "test-key")
    s = fake_session(lambda *a: FakeResponse([]))
    coingecko.fetch_markets(["aave"], s)
    assert s.calls[0]["headers"] == {"x-cg-demo-api-key": "test-key"}


def test_fetch_markets_raises_on_http_error(fake_session):
    s = fake_session(lambda *a: FakeResponse(status_code=429))
    with pytest.raises(requests.HTTPError):
        coingecko.fetch_markets(["aave"], s)
