"""The Google Geolocation spend guard: a monthly cap counted before every call, a shared
cache so a familiar spot is looked up once, and no call whenever the budget can't be checked."""

import asyncio

import httpx
import pytest

import app.geo as geo
from app.config import Settings
from app.geo import AccessPoint, Geolocator

APS = [AccessPoint("aa:bb:cc:dd:ee:01", -50), AccessPoint("aa:bb:cc:dd:ee:02", -60), AccessPoint("aa:bb:cc:dd:ee:03", -70)]


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
    return Geolocator(Settings(_env_file=None, google_geolocation_api_key="k", **over), db)


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
