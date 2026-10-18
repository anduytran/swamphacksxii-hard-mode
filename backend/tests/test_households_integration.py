"""Household setup, places and the public map against the real database.

Opt-in (writes to DATABASE_URL, cleans up after itself):
    RUN_DB_TESTS=1 .venv/bin/python -m pytest tests/test_households_integration.py
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth import Principal, require_caregiver, require_device
from app.config import get_settings
from app.db_places import household_hash
from app.geo import Location
from app.main import create_app

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DB_TESTS"), reason="set RUN_DB_TESTS=1 to run against DATABASE_URL")
RUN = uuid.uuid4().hex[:8]
HOME, STORE = (29.6516, -82.3248), (29.6550, -82.3400)


class FixedGeo:
    """Pretend Wi-Fi positioning: the scan's first BSSID picks a known spot."""

    spots = {"aa:aa:aa:aa:aa:01": HOME, "aa:aa:aa:aa:aa:02": STORE}

    async def locate(self, aps):
        spot = self.spots.get(aps[0].bssid) if aps else None
        return Location(*spot, 20, "wifi") if spot else None


class OfflineAI:
    speech_media_type = "audio/wav"

    async def transcribe(self, f, audio, ct, hints=None):
        return audio.decode()

    async def embed(self, t, *, query=False):
        return [1.0] + [0.0] * 1535

    async def embed_many(self, texts):
        return [[1.0] + [0.0] * 1535 for _ in texts]

    async def extract_items(self, text):
        from app.ai import ItemMention
        return [ItemMention("umbrella", "")] if "umbrella" in text else []

    async def describe_scene(self, *a):
        return None

    async def answer(self, q, m, language="en"):
        return "ok"

    async def speak(self, t):
        return b"RIFF"


@pytest.fixture
def setup():
    app = create_app(use_lifespan=False)
    app.state.ai, app.state.geo = OfflineAI(), FixedGeo()
    who = {"user": f"it-{RUN}-owner", "device": f"it-{RUN}-dev0@clients"}
    app.dependency_overrides[require_caregiver] = lambda: Principal(sub=who["user"], claims={})
    app.dependency_overrides[require_device] = lambda: Principal(sub=who["device"], claims={"gty": "client-credentials"})
    with TestClient(app) as client:
        client.who = who
        yield client
        db, run = app.state.db, client.portal.call
        run(db.pool.execute, "DELETE FROM query_logs WHERE device_id LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM memory_logs WHERE device_id LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM device_pairing WHERE device_id LIKE $1", f"it-{RUN}-%")
        ids = [str(r["id"]) for r in run(db.pool.fetch, "SELECT id FROM households WHERE created_by LIKE $1", f"it-{RUN}-%")]
        run(db.pool.execute, "DELETE FROM public_lost_stats WHERE household_hash = ANY($1::text[])", [household_hash(i) for i in ids])
        run(db.pool.execute, "DELETE FROM households WHERE created_by LIKE $1", f"it-{RUN}-%")


def wifi(bssid):
    return {"wifi": f'[{{"bssid": "{bssid}", "signal": -50}}, {{"bssid": "bb:bb:bb:bb:bb:bb", "signal": -80}}]'}


def log(client, text, bssid, code="PAIR-1"):
    files = {"audio": ("a.wav", text.encode(), "audio/wav"), "image": ("i.jpg", b"\xff\xd8", "image/jpeg")}
    r = client.post("/api/v1/memory/log", files=files, data=wifi(bssid), headers={"X-Pairing-Code": code})
    assert r.status_code == 200, r.text


def test_setup_for_someone_else_then_places_and_public_map(setup):
    c = setup
    # The wearable uploads before anyone has claimed it: located, but in no household yet.
    log(c, "putting my keys down", "aa:aa:aa:aa:aa:01")
    hh = c.post("/api/v1/households", json={"name": f"Mom ({RUN})", "display_name": "Dylan"}).json()["id"]
    assert c.post(f"/api/v1/households/{hh}/devices", json={"client_id": f"it-{RUN}-dev0", "pairing_code": "nope"}).status_code == 400
    claimed = c.post(f"/api/v1/households/{hh}/devices", json={"client_id": f"it-{RUN}-dev0", "pairing_code": "pair-1"}).json()
    assert claimed["memories_placed"] == 1

    for _ in range(2):
        log(c, "putting my keys down", "aa:aa:aa:aa:aa:01")
    log(c, "leaving my umbrella here", "aa:aa:aa:aa:aa:02")
    me = c.get("/api/v1/households/me").json()["households"][0]
    labels = {p["label"]: p for p in me["places"]}
    assert set(labels) == {"Home", "Place 2"} and labels["Home"]["visits"] == 3
    assert all(p["kind"] == "private" for p in me["places"])

    # Family member joins with an invite; another user can't see the household.
    code = c.post(f"/api/v1/households/{hh}/invites").json()["code"]
    c.who["user"] = f"it-{RUN}-stranger"
    assert c.get("/api/v1/households/me").json()["households"] == []
    assert c.get("/api/v1/dashboard/history").json() == []
    assert c.patch(f"/api/v1/households/{hh}", json={"share_public": True}).status_code == 404
    c.who["user"] = f"it-{RUN}-sister"
    assert c.post("/api/v1/invites/accept", json={"code": code, "display_name": "Sis"}).status_code == 200
    assert len(c.get("/api/v1/dashboard/history").json()) == 4
    assert c.post("/api/v1/invites/accept", json={"code": code}).status_code == 400  # one use

    # Mark the store public and opt in; new umbrella memories there count publicly.
    store = labels["Place 2"]["id"]
    assert c.patch(f"/api/v1/households/{hh}/places/{store}", json={"name": "Publix", "kind": "public"}).status_code == 200
    assert c.patch(f"/api/v1/households/{hh}", json={"share_public": True}).status_code == 403  # sister isn't owner
    c.who["user"] = f"it-{RUN}-owner"
    assert c.patch(f"/api/v1/households/{hh}", json={"share_public": True}).status_code == 200
    assert c.get("/api/v1/households/me").json()["households"][0]["language"] == "en"  # the default
    assert c.patch(f"/api/v1/households/{hh}", json={"language": "es"}).status_code == 200
    assert c.get("/api/v1/households/me").json()["households"][0]["language"] == "es"
    log(c, "leaving my umbrella here", "aa:aa:aa:aa:aa:02")

    ours = c.get("/api/v1/dashboard/places").json()
    assert {p["label"] for p in ours["households"][0]["places"]} == {"Home", "Publix"}
    assert len(ours["memories"]) == 5

    # One household isn't enough for the public map (k-anonymity: 3).
    cells = c.get("/api/v1/public/lost-map").json()["cells"]
    assert not any(abs(x["lat"] - STORE[0]) < 0.002 and abs(x["lng"] - STORE[1]) < 0.002 for x in cells)
