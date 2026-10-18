"""The Google Geolocation spend guard: a monthly cap counted before every call, a shared
cache so a familiar spot is looked up once, and no call whenever the budget can't be checked.
BeaconDB (free, no key) answers without a key, past the budget, and when Google finds nothing."""

import asyncio

import httpx
import pytest

import app.geo as geo
from app.config import Settings
from app.geo import AccessPoint, Geolocator

APS = [AccessPoint("a8:bb:cc:dd:ee:01", -50), AccessPoint("a8:bb:cc:dd:ee:02", -60), AccessPoint("a8:bb:cc:dd:ee:03", -70)]


class BudgetDB:
    def __init__(self, limit_left=10, broken=False):
        self.left = limit_left
        self.broken = broken
        self.cache = {}
        self.reserved = 0

    async def cached_position(self, key, days):
        if self.broken:
            raise ConnectionError("db down")
        return self.cache.get(key)

    async def cache_position(self, key, lat, lng, acc):
        self.cache[key] = (lat, lng, acc)

    async def reserve_geolocation_call(self, month, limit):
        if self.left <= 0:
            return False
        self.left -= 1
        self.reserved += 1
        return True


@pytest.fixture
def google(monkeypatch):
    """Counts requests to the (mocked) Google endpoint."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"location": {"lat": 29.65, "lng": -82.32}, "accuracy": 30})

    real = httpx.AsyncClient
    monkeypatch.setattr(geo.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return calls


def locator(db, **over):
    """Google only, unless a test turns BeaconDB back on."""
    return Geolocator(Settings(_env_file=None, google_geolocation_api_key="k", **{"beacondb_url": "", **over}), db)


@pytest.fixture
def providers(monkeypatch):
    """Mocked Google and BeaconDB; `answers` sets each one's reply, `calls` records hosts in order."""
    state = {"calls": [], "answers": {"www.googleapis.com": 200, "api.beacondb.net": 200}}

    def handler(request):
        host = request.url.host
        state["calls"].append(host)
        if state["answers"][host] != 200:
            return httpx.Response(state["answers"][host], json={"error": {"code": state["answers"][host]}})
        lat = 29.65 if host == "www.googleapis.com" else 29.64
        return httpx.Response(200, json={"location": {"lat": lat, "lng": -82.34}, "accuracy": 40})

    real = httpx.AsyncClient
    monkeypatch.setattr(geo.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return state


def both(db, **over):
    return Geolocator(Settings(_env_file=None, **over), db)


def test_calls_google_once_then_reuses_the_shared_cache(google):
    db = BudgetDB()
    loc = asyncio.run(locator(db).locate(APS))
    assert (loc.lat, loc.lng, loc.accuracy_m) == (29.65, -82.32, 30)
    # A fresh instance (another serverless invocation) finds it in the database.
    assert asyncio.run(locator(db).locate(APS)).lat == 29.65
    assert len(google) == 1 and db.reserved == 1


def test_no_call_once_the_month_is_spent(google):
    db = BudgetDB(limit_left=0)
    assert asyncio.run(locator(db).locate(APS)) is None
    assert google == []


def test_no_call_without_a_working_budget_check(google):
    assert asyncio.run(locator(None).locate(APS)) is None  # no database, no budget
    assert asyncio.run(locator(BudgetDB(broken=True)).locate(APS)) is None  # database down
    assert asyncio.run(locator(BudgetDB(), geolocation_monthly_limit=0).locate(APS)) is None  # turned off
    assert google == []


def test_beacondb_answers_without_a_google_key(providers):
    db = BudgetDB()
    loc = asyncio.run(both(db).locate(APS))
    assert (loc.lat, loc.source) == (29.64, "wifi")
    assert providers["calls"] == ["api.beacondb.net"] and db.reserved == 0  # no Google budget spent
    assert asyncio.run(both(db).locate(APS)).lat == 29.64  # cached for the next instance
    assert providers["calls"] == ["api.beacondb.net"]


def test_beacondb_works_without_a_database(providers):
    assert asyncio.run(both(None).locate(APS)).lat == 29.64  # free, so no budget is needed
    assert asyncio.run(both(BudgetDB(broken=True)).locate(APS)).lat == 29.64


def test_google_first_then_beacondb_when_google_has_no_answer(providers):
    assert asyncio.run(both(BudgetDB(), google_geolocation_api_key="k").locate(APS)).lat == 29.65
    assert providers["calls"] == ["www.googleapis.com"]
    providers["calls"].clear()
    providers["answers"]["www.googleapis.com"] = 404
    assert asyncio.run(both(BudgetDB(), google_geolocation_api_key="k").locate(APS)).lat == 29.64
    assert providers["calls"] == ["www.googleapis.com", "api.beacondb.net"]


def test_beacondb_past_the_google_budget(providers):
    loc = asyncio.run(both(BudgetDB(limit_left=0), google_geolocation_api_key="k").locate(APS))
    assert loc.lat == 29.64 and providers["calls"] == ["api.beacondb.net"]


def test_no_position_when_nobody_knows_the_spot(providers):
    providers["answers"]["api.beacondb.net"] = 404
    assert asyncio.run(both(BudgetDB()).locate(APS)) is None
    assert asyncio.run(both(BudgetDB(), beacondb_url="").locate(APS)) is None  # every provider off


def test_beacondb_requests_identify_the_app(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(404, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(geo.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    asyncio.run(both(None).locate(APS))
    assert seen[0].headers["user-agent"].startswith("LastScene/")
    assert "key" not in seen[0].url.params  # BeaconDB never gets the Google key
    assert b'"considerIp":false' in seen[0].content.replace(b" ", b"")
