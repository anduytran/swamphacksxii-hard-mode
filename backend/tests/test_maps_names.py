"""Public place names from Grounding with Google Maps: opt-in, public places only, cached,
capped per day, fail-soft, and the Interactions API call shape on a fake client."""

import asyncio
from types import SimpleNamespace

from google.genai import interactions as gi

from app.config import Settings
from app.db_maps import maps_key
from app.gemini import GeminiAI, parse_place_name
from app.maps_names import name_memory_place

from .test_api import client  # noqa: F401  (fixture)

STARBUCKS = {"name": "Starbucks, Reitz Union", "sources": [{"title": "Starbucks", "url": "https://maps.google.com/?cid=1"}]}
REITZ, HOME = (29.64601, -82.34772), (29.6516, -82.3248)


class MapsDB:
    """In-memory stand-in for db_maps.MapsNamesMixin."""

    def __init__(self, calls_left=50):
        self.memories = {}  # memory id -> (lat, lng, place kind)
        self.cache = {}
        self.links = {}
        self.calls_left = calls_left
        self.reserved = 0

    def add(self, memory_id, position, kind):
        self.memories[memory_id] = (*position, kind)

    async def public_memory_position(self, memory_id):
        lat, lng, kind = self.memories[memory_id]
        return (lat, lng) if kind == "public" else None

    async def cached_place_name(self, lat, lng, max_age_days):
        key = maps_key(lat, lng)
        return (key in self.cache), self.cache.get(key)

    async def cache_place_name(self, lat, lng, named):
        self.cache[maps_key(lat, lng)] = named

    async def link_memory_place_name(self, memory_id, lat, lng):
        self.links[memory_id] = maps_key(lat, lng)

    async def reserve_maps_call(self, day, limit):
        if self.calls_left <= 0 or self.reserved >= limit:
            return False
        self.calls_left -= 1
        self.reserved += 1
        return True


class MapsAI:
    def __init__(self, answer=STARBUCKS, fail=False):
        self.answer, self.fail = answer, fail
        self.asked = []

    async def name_place(self, lat, lng):
        self.asked.append((lat, lng))
        if self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return self.answer


ON = Settings(_env_file=None, maps_grounding=True)


def name(db, ai, memory_id, settings=ON):
    return asyncio.run(name_memory_place(db, ai, settings, memory_id))


def test_off_by_default():
    settings = Settings(_env_file=None)
    assert settings.maps_grounding is False and settings.maps_daily_limit == 50
    db, ai = MapsDB(), MapsAI()
    db.add("m1", REITZ, "public")
    assert name(db, ai, "m1", settings) is None
    assert ai.asked == [] and db.reserved == 0


def test_private_places_are_never_sent():
    db, ai = MapsDB(), MapsAI()
    db.add("home", HOME, "private")
    assert name(db, ai, "home") is None
    assert ai.asked == [] and db.reserved == 0 and db.cache == {}


def test_names_a_public_place_once_then_uses_the_cache():
    db, ai = MapsDB(), MapsAI()
    db.add("m1", REITZ, "public")
    db.add("m2", (REITZ[0] + 0.00001, REITZ[1]), "public")  # ~1 m away: same rounded spot
    assert name(db, ai, "m1") == STARBUCKS
    assert name(db, ai, "m2") == STARBUCKS
    assert ai.asked == [REITZ] and db.reserved == 1
    assert db.links == {"m1": maps_key(*REITZ), "m2": maps_key(*REITZ)}


def test_a_miss_is_cached_too():
    db, ai = MapsDB(), MapsAI(answer=None)
    db.add("m1", REITZ, "public")
    db.add("m2", REITZ, "public")
    assert name(db, ai, "m1") is None and name(db, ai, "m2") is None
    assert len(ai.asked) == 1 and db.links == {}


def test_daily_limit_stops_calls():
    db, ai = MapsDB(), MapsAI()
    for i in range(3):
        db.add(f"m{i}", (29.64 + i * 0.01, -82.34), "public")
    settings = Settings(_env_file=None, maps_grounding=True, maps_daily_limit=2)
    results = [name(db, ai, f"m{i}", settings) for i in range(3)]
    assert results == [STARBUCKS, STARBUCKS, None]
    assert len(ai.asked) == 2
    off = Settings(_env_file=None, maps_grounding=True, maps_daily_limit=0)
    db.add("m9", (29.7, -82.3), "public")
    assert name(db, ai, "m9", off) is None and len(ai.asked) == 2


def test_errors_leave_the_memory_unnamed():
    db, ai = MapsDB(), MapsAI(fail=True)
    db.add("m1", REITZ, "public")
    assert name(db, ai, "m1") is None
    assert db.cache == {}  # not cached, so a later day can try again


# ---------------------------------------------------------------- the Gemini call


def maps_interaction(text="Starbucks, Reitz Union", url="https://maps.google.com/?cid=1"):
    """A real SDK Interaction, as the API returns it for a google_maps tool call."""
    return gi.Interaction.model_validate({
        "id": "i1",
        "status": "completed",
        "steps": [
            {"type": "google_maps_call", "id": "c1", "arguments": {}},
            {"type": "google_maps_result", "call_id": "c1",
             "result": [{"places": [{"name": "Starbucks", "url": url, "place_id": "places/1"}]}]},
            {"type": "model_output", "content": [{"type": "text", "text": text, "annotations": [
                {"type": "place_citation", "name": "Starbucks", "url": url, "start_index": 0, "end_index": 9}]}]},
        ],
    })


class FakeInteractions:
    def __init__(self, response):
        self.response = response
        self.created = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return self.response


def test_gemini_call_shape():
    settings = Settings(_env_file=None, gemini_api_key="test")
    gemini = GeminiAI(settings)
    fake = FakeInteractions(maps_interaction())
    gemini.client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    assert asyncio.run(gemini.name_place(*REITZ)) == STARBUCKS
    (call,) = fake.created
    assert call["model"] == settings.gemini_model
    assert call["tools"] == [{"type": "google_maps", "latitude": REITZ[0], "longitude": REITZ[1]}]
    assert call["store"] is False
    assert isinstance(call["input"], str)


def test_parse_needs_sources_and_only_keeps_web_links():
    assert parse_place_name(maps_interaction(url="javascript:alert(1)")) is None
    no_steps = gi.Interaction.model_validate({"id": "i2", "status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "Somewhere"}]}]})
    assert parse_place_name(no_steps) is None  # a name without its Google Maps source can't be shown
    assert parse_place_name(maps_interaction(text="  ")) is None


# ---------------------------------------------------------------- dashboard


def test_history_carries_the_name_with_its_sources(client):
    from .test_api import CAREGIVER, log

    log(client)
    (row,) = client.app_state.db.rows

    async def place_names_for(ids):
        return {row.id: STARBUCKS} if row.id in ids else {}

    client.app_state.db.place_names_for = place_names_for
    (entry,) = client.get("/api/v1/dashboard/history", headers=CAREGIVER).json()
    assert entry["place_name"] == STARBUCKS


def test_history_without_names_still_works(client):
    from .test_api import CAREGIVER, log

    log(client)
    (entry,) = client.get("/api/v1/dashboard/history", headers=CAREGIVER).json()
    assert entry["place_name"] is None
