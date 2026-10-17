"""Observations and navigation against the real database.

Opt-in (writes to DATABASE_URL, cleans up after itself):
    RUN_DB_TESTS=1 .venv/bin/python -m pytest tests/test_observations_integration.py
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.auth import Principal, require_caregiver, require_device
from app.config import get_settings
from app.main import create_app

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DB_TESTS"), reason="set RUN_DB_TESTS=1 to run against DATABASE_URL")
RUN = uuid.uuid4().hex[:8]
DEVICE, OTHER, SECOND = (f"it-{RUN}-{n}@clients" for n in ("cam", "other", "wear"))


class OfflineAI:
    speech_media_type = "audio/wav"

    async def transcribe(self, f, audio, ct):
        return audio.decode()

    async def embed(self, t, *, query=False):
        return [1.0] + [0.0] * 1535

    async def embed_many(self, texts):
        return [[1.0] + [0.0] * 1535 for _ in texts]

    async def extract_items(self, text):
        return []

    async def answer(self, q, memories):
        self.answered = memories
        return "ok"

    async def speak(self, t):
        return b"RIFF"


@pytest.fixture
def c():
    settings = get_settings().model_copy(update={"observation_min_interval_seconds": 0})
    app = create_app(use_lifespan=False)
    app.state.ai = OfflineAI()
    who = {"device": DEVICE}
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[require_device] = lambda: Principal(sub=who["device"], claims={"gty": "client-credentials"})
    app.dependency_overrides[require_caregiver] = lambda: Principal(sub=f"it-{RUN}-owner", claims={})
    with TestClient(app) as client:
        client.who, client.settings = who, settings
        yield client
        db, run = app.state.db, client.portal.call
        run(db.pool.execute, "DELETE FROM item_last_seen WHERE scope LIKE $1 OR scope IN (SELECT id::text FROM households WHERE created_by LIKE $1)", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM query_logs WHERE device_id LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM memory_logs WHERE device_id LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM device_pairing WHERE device_id LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM households WHERE created_by LIKE $1", f"it-{RUN}-%")


def obs(minutes_ago=0, label="keys", scene="kitchen counter", event="placed", conf=0.71, bbox=(0.42, 0.55, 0.08, 0.06), oid=None):
    return {
        "observation_id": oid or str(uuid.uuid4()),
        "captured_at": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(),
        "source": "auto",
        "pipeline": {"name": "lastscene-vision", "version": "0.3.0", "detector": "yolo-world", "matcher": "orb"},
        "image_size": {"width": 1280, "height": 720},
        "scene": {"id": scene.replace(" ", "-"), "label": scene, "match_score": 0.8},
        "objects": [{"label": label, "confidence": conf, "bbox": list(bbox), "track_id": f"{label}-1", "event": event}],
    }


def post(c, body, image=b"\xff\xd8frame", headers=None):
    return c.post("/api/v1/observations", files={"image": ("f.jpg", image, "image/jpeg")},
                  data={"observation": json.dumps(body)}, headers=headers or {"X-Pairing-Code": "CAM-1"})


def test_observation_lifecycle(c):
    # Ingest, then a retry with the same id is a no-op.
    first = obs(minutes_ago=30)
    r = post(c, first, image=b"\xff\xd8older-frame")
    assert r.status_code == 201, r.text
    assert r.json()["items"] == ["keys"]
    again = post(c, first)
    assert again.status_code == 200 and again.json()["duplicate"] and again.json()["memory_id"] == r.json()["memory_id"]

    # Validation and limits.
    assert post(c, {**obs(), "source": "manual"}).status_code == 422
    assert post(c, obs(conf=0.2)).status_code == 422  # below min_detection_confidence
    future = obs()
    future["captured_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    assert post(c, future).status_code == 422
    c.who["device"] = OTHER
    assert post(c, first).status_code == 409  # someone else's observation_id
    c.who["device"] = DEVICE

    # A newer sighting moves last-seen; the frame and box come back for navigation.
    newer = obs(minutes_ago=5, scene="hallway table", bbox=(0.1, 0.2, 0.05, 0.05))
    assert post(c, newer, image=b"\xff\xd8newer-frame").status_code == 201
    ls = c.get("/api/v1/items/Keys/last-seen").json()
    assert ls["scene"]["label"] == "hallway table" and ls["bbox"] == [0.1, 0.2, 0.05, 0.05]
    assert ls["image_size"] == {"width": 1280, "height": 720} and ls["maybe"] is False
    frame = c.get("/" + ls["image_url"].split("/", 3)[3])  # absolute link → path for the test client
    assert frame.status_code == 200 and frame.content == b"\xff\xd8newer-frame"

    # A late upload of an older frame does not override the newer last-seen.
    assert post(c, obs(minutes_ago=60, scene="garage shelf")).status_code == 201
    assert c.get("/api/v1/items/keys/last-seen").json()["scene"]["label"] == "hallway table"

    # Rate limit: back-to-back uploads from one device.
    c.settings.observation_min_interval_seconds = 30
    assert post(c, obs()).status_code == 429
    c.settings.observation_min_interval_seconds = 0

    # Delete the wrong (newest) detection: last-seen rolls back to the previous sighting.
    gone = c.delete(f"/api/v1/observations/{newer['observation_id']}")
    assert gone.status_code == 200 and gone.json()["items"] == ["keys"]
    assert c.get("/api/v1/items/keys/last-seen").json()["scene"]["label"] == "kitchen counter"
    assert c.delete(f"/api/v1/observations/{newer['observation_id']}").status_code == 404

    # Found, and the full list.
    assert c.post("/api/v1/items/keys/found", json={}).status_code == 200
    assert c.get("/api/v1/items/keys/last-seen").json()["found_at"] is not None
    assert c.post("/api/v1/items/umbrella/found", json={}).status_code == 404
    assert [i["label"] for i in c.get("/api/v1/items/last-seen").json()["items"]] == ["keys"]

    # Voice answers can use automatic memories.
    c.post("/api/v1/memory/query", files={"audio": ("q.wav", b"where are my keys", "audio/wav")})
    assert any("keys" in " ".join(ch.content for ch in m.chunks) for m in c.app.state.ai.answered)


def test_household_shares_last_seen_and_dashboard_shows_boxes(c):
    hh = c.post("/api/v1/households", json={"name": f"Obs {RUN}"}).json()["id"]
    for device, code in ((DEVICE, "CAM-1"), (SECOND, "WEAR-1")):
        c.who["device"] = device
        assert post(c, obs(minutes_ago=50, label="wallet"), headers={"X-Pairing-Code": code}).status_code == 201
        assert c.post(f"/api/v1/households/{hh}/devices", json={"client_id": device, "pairing_code": code}).status_code == 200

    # The camera's sighting is visible to the other device in the same household.
    c.who["device"] = DEVICE
    assert post(c, obs(minutes_ago=1, label="wallet", scene="dining table")).status_code == 201
    c.who["device"] = SECOND
    assert c.get("/api/v1/items/wallet/last-seen").json()["scene"]["label"] == "dining table"

    history = c.get("/api/v1/dashboard/history").json()
    auto = [e for e in history if e["source"] == "auto"]
    assert auto and auto[0]["detections"][0]["bbox"] and auto[0]["summary"].startswith("wallet")


def ago(minutes):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def last_seen_keys(c):
    return c.get("/api/v1/items/keys/last-seen").json()


def test_late_upload_from_before_found_does_not_reopen(c):
    # Sighting A (t0); "found it" at t2; a queued sighting B captured at t1 (t0 < t1 < t2) arrives late.
    assert post(c, obs(minutes_ago=30)).status_code == 201
    found_at = ago(10)
    assert c.post("/api/v1/items/keys/found", json={"found_at": found_at.isoformat()}).status_code == 200
    assert post(c, obs(minutes_ago=20, scene="hallway table")).status_code == 201
    ls = last_seen_keys(c)
    assert ls["found_at"] is not None, "a sighting from before 'found it' reopened the item"
    assert datetime.fromisoformat(ls["found_at"]) == found_at

    # A sighting after found_at does reopen it.
    assert post(c, obs(minutes_ago=1, scene="dining table")).status_code == 201
    ls = last_seen_keys(c)
    assert ls["scene"]["label"] == "dining table" and ls["found_at"] is None


def test_delete_keeps_found_when_remaining_sighting_is_older(c):
    z, a, b = obs(minutes_ago=40, scene="garage shelf"), obs(minutes_ago=30), obs(minutes_ago=20, scene="hallway table")
    for o in (z, a, b):
        assert post(c, o).status_code == 201
    found_at = ago(10)
    assert c.post("/api/v1/items/keys/found", json={"found_at": found_at.isoformat()}).status_code == 200

    # Deleting an older sighting rebuilds last-seen from B, which is still before found_at.
    assert c.delete(f"/api/v1/observations/{z['observation_id']}").status_code == 200
    ls = last_seen_keys(c)
    assert ls["scene"]["label"] == "hallway table" and ls["found_at"] is not None
    assert datetime.fromisoformat(ls["found_at"]) == found_at

    # Deleting the current last-seen rolls back to A, also before found_at.
    assert c.delete(f"/api/v1/observations/{b['observation_id']}").status_code == 200
    ls = last_seen_keys(c)
    assert ls["scene"]["label"] == "kitchen counter" and ls["found_at"] is not None
    assert datetime.fromisoformat(ls["found_at"]) == found_at


def test_delete_removes_last_seen_left_in_pre_household_scope(c):
    # Sent before the device was claimed: last-seen stays scoped to the device itself.
    early = obs(minutes_ago=30, label="wallet")
    assert post(c, early).status_code == 201
    hh = c.post("/api/v1/households", json={"name": f"Scope {RUN}"}).json()["id"]
    assert c.post(f"/api/v1/households/{hh}/devices", json={"client_id": DEVICE, "pairing_code": "CAM-1"}).status_code == 200
    assert c.delete(f"/api/v1/observations/{early['observation_id']}").status_code == 200
    left = c.portal.call(c.app.state.db.pool.fetchval, "SELECT count(*) FROM item_last_seen WHERE scope = $1", DEVICE)
    assert left == 0
