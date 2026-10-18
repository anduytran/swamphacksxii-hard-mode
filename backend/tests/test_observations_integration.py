"""Observations and navigation against the real database.

Opt-in (writes to DATABASE_URL, cleans up after itself):
    RUN_DB_TESTS=1 .venv/bin/python -m pytest tests/test_observations_integration.py
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import httpx
import pytest
from fastapi.testclient import TestClient

from app.auth import Principal, require_caregiver, require_device
from app.config import get_settings
from app.main import create_app
from app.raia import Raia

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DB_TESTS"), reason="set RUN_DB_TESTS=1 to run against DATABASE_URL")
RUN = uuid.uuid4().hex[:8]
DEVICE, OTHER, SECOND = (f"it-{RUN}-{n}@clients" for n in ("cam", "other", "wear"))


class OfflineAI:
    speech_media_type = "audio/wav"

    async def transcribe(self, f, audio, ct, hints=None):
        return audio.decode()

    async def embed(self, t, *, query=False):
        return [1.0] + [0.0] * 1535

    async def embed_many(self, texts):
        return [[1.0] + [0.0] * 1535 for _ in texts]

    async def extract_items(self, text):
        return []

    async def answer(self, q, memories, language="en"):
        self.answered = memories
        return "ok"

    async def speak(self, t):
        return b"RIFF"


@pytest.fixture
def c():
    # Merging off by default: these tests post repeat sightings on purpose (one test turns it on).
    settings = get_settings().model_copy(update={"observation_min_interval_seconds": 0, "passive_merge_minutes": 0})
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
        run(db.pool.execute, "DELETE FROM raia_contexts WHERE scope LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM wifi_places WHERE scope LIKE $1", f"it-{RUN}-%")
        run(db.pool.execute, "DELETE FROM item_history WHERE scope LIKE $1 OR scope IN (SELECT id::text FROM households WHERE created_by LIKE $1)", f"it-{RUN}-%")
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
    resp = c.post("/api/v1/memory/query", files={"audio": ("q.wav", b"where are my keys", "audio/wav")})
    # Answered from the camera's sighting (by the template when it's clear-cut, else the model).
    assert resp.status_code == 200 and "keys" in unquote(resp.headers["x-answer"]).lower()


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


def test_raia_reads_a_logged_item_once_per_cooldown(c):
    """raia frame context (app/raia.py) on the real tables: logged items, the cooldown claim, the saved chunk."""
    sent = []

    def api(req: httpx.Request) -> httpx.Response:
        path = req.url.path.removeprefix("/external")
        sent.append(path)
        if req.url.host == "storage.example":
            return httpx.Response(200)
        return {
            "/users/search": httpx.Response(404),
            "/users": httpx.Response(201, json={"id": "u-1"}),
            "/conversations": httpx.Response(201, json={"id": "conv-1"}),
            "/conversations/conv-1/message-files/upload-url": httpx.Response(
                200, json={"signedUrl": "https://storage.example/put", "storagePath": "conversations/conv-1/f.jpg"}),
            "/conversations/conv-1/messages": httpx.Response(201, json={"message": json.dumps({
                "visible": True, "where": "on the hallway table", "surface": "table", "nearby": ["bowl"],
                "room": "hallway", "visible_text": "", "summary": "The keys are on the hallway table."})}),
        }[path]

    settings = c.settings.model_copy(update={"raia_agent_key": "k"})
    c.app.dependency_overrides[get_settings] = lambda: settings
    c.app.state.raia = Raia(settings, http=httpx.AsyncClient(transport=httpx.MockTransport(api)))
    db, run = c.app.state.db, c.portal.call

    # Not one of the user's items yet: nothing is sent.
    assert post(c, obs(label="keys")).status_code == 201
    assert sent == []

    # A spoken memory names the keys; then a sighting is read, and the next one within 30 min isn't.
    spoken = run(db.insert, DEVICE, "I put my keys on the hallway table", None, None)
    run(db.add_items, spoken.id, [("Keys", "hallway table")])
    first = post(c, {**obs(label="keys", conf=0.71), "location": {"lat": 29.6465, "lng": -82.3475, "accuracy_m": 20}})
    assert first.status_code == 201 and sent.count("/conversations/conv-1/messages") == 1
    assert post(c, obs(label="keys", conf=0.9)).status_code == 201
    assert sent.count("/conversations/conv-1/messages") == 1

    # Saved as a memory of its own, "LastScene: ...", with the item and place indexed.
    row = run(db.raia_context, DEVICE, "keys")
    assert json.loads(row["context"])["surface"] == "table" and row["error"] is None
    memory_id = str(row["memory_id"])
    saved = run(db.pool.fetchrow, "SELECT transcript, image IS NOT NULL AS has_image FROM memory_logs WHERE id = $1::uuid", memory_id)
    assert saved["transcript"] == "LastScene: The keys are on the hallway table." and saved["has_image"]
    chunks = run(db.chunks_for, [memory_id])[memory_id]
    assert ("item", "keys: on the hallway table") in [(ch.kind, ch.content) for ch in chunks]
    assert "hallway table" in run(db.pool.fetchval, "SELECT location FROM memory_items WHERE memory_id = $1::uuid", memory_id)
    # The sighting raia read carries its context (the item drawer and Live show it).
    read = run(db.pool.fetchval, "SELECT context FROM item_history WHERE observation_id = $1::uuid",
               run(db.pool.fetchval, "SELECT observation_id FROM memory_logs WHERE id = $1::uuid", first.json()["memory_id"]))
    assert read and (json.loads(read) if isinstance(read, str) else read)["where"] == "on the hallway table"
    # The raia memory has the sighting's location.
    where = run(db.pool.fetchrow, "SELECT lat, lng FROM memory_logs WHERE id = $1::uuid", memory_id)
    assert (round(where["lat"], 4), round(where["lng"], 4)) == (29.6465, -82.3475)

    # Past the cooldown it's read again.
    run(db.pool.execute, "UPDATE raia_contexts SET claimed_at = now() - interval '31 minutes' WHERE scope = $1", DEVICE)
    assert post(c, obs(label="keys", conf=0.9)).status_code == 201
    assert sent.count("/conversations/conv-1/messages") == 2


def test_repeat_passive_sightings_refresh_one_memory(c):
    """Within PASSIVE_MERGE_MINUTES, the same item seen again updates its passive memory: one timeline entry."""
    c.settings.passive_merge_minutes = 30
    first = post(c, obs(minutes_ago=3, conf=0.5), image=b"\xff\xd8first")
    assert first.status_code == 201 and first.json()["merged"] is False
    again = post(c, obs(minutes_ago=1, conf=0.9, scene="hallway table"), image=b"\xff\xd8second")
    assert again.status_code == 200 and again.json()["merged"] is True
    assert again.json()["memory_id"] == first.json()["memory_id"]
    late = post(c, obs(minutes_ago=5, conf=0.7), image=b"\xff\xd8older")  # an older frame arriving late
    assert late.status_code == 201 and late.json()["merged"] is False

    db, run = c.app.state.db, c.portal.call
    memory_id = first.json()["memory_id"]
    row = run(db.pool.fetchrow, "SELECT image FROM memory_logs WHERE id = $1::uuid", memory_id)
    assert bytes(row["image"]) == b"\xff\xd8second"
    dets = run(db.detections_for, [memory_id])[memory_id]
    assert [(d["label"], d["confidence"]) for d in dets] == [("keys", 0.9)]
    assert c.get("/api/v1/items/keys/last-seen").json()["scene"]["label"] == "hallway table"


def test_every_passive_sighting_gets_a_location(c):
    """Its own position, else the wearer's last known one (both Pis are worn together)."""
    db, run = c.app.state.db, c.portal.call
    own = post(c, {**obs(minutes_ago=2), "location": {"lat": 29.6465, "lng": -82.3475, "accuracy_m": 20}})
    borrowed = post(c, obs(minutes_ago=1, label="wallet"))
    assert own.status_code == 201 and borrowed.status_code == 201
    rows = {r["id"]: r for r in run(db.pool.fetch, "SELECT id::text, lat, lng, accuracy_m FROM memory_logs WHERE id = ANY($1::uuid[])",
                                   [own.json()["memory_id"], borrowed.json()["memory_id"]])}
    for memory in (own, borrowed):
        row = rows[memory.json()["memory_id"]]
        assert (round(row["lat"], 4), round(row["lng"], 4)) == (29.6465, -82.3475)

    # The item's LastScene position: its newest located sighting.
    import time as _time
    for _ in range(10):  # located after the response
        found = run(db.lastscene_position, DEVICE, [DEVICE], "wallet")
        if found:
            break
        _time.sleep(0.5)
    assert found and (round(found["lat"], 4), round(found["lng"], 4)) == (29.6465, -82.3475)
    assert run(db.lastscene_position, DEVICE, [DEVICE], "umbrella") is None


def test_an_items_last_five_sightings_can_be_trashed(c):
    """Every passive sighting is kept (newest 5 per item) even when memories merge; trashing a wrong one
    removes it and its memory, and the item's last-seen falls back to the newest left."""
    c.settings.passive_merge_minutes = 30
    hh = c.post("/api/v1/households", json={"name": f"Hist {RUN}"}).json()["id"]
    assert post(c, obs(minutes_ago=30, label="banana"), headers={"X-Pairing-Code": "CAM-H"}).status_code == 201
    assert c.post(f"/api/v1/households/{hh}/devices", json={"client_id": DEVICE, "pairing_code": "CAM-H"}).status_code == 200
    scenes = ["desk", "chair", "shelf", "sink", "bed", "floor"]
    for n, scene in enumerate(scenes):  # 6 sightings, 5 to 0 minutes ago
        assert post(c, obs(minutes_ago=5 - n, label="banana", scene=scene, conf=0.6 + n / 20)).status_code in (200, 201)

    listed = c.get("/api/v1/dashboard/items/banana/sightings").json()["sightings"]
    assert [s["scene"] for s in listed] == ["floor", "bed", "sink", "shelf", "chair"]  # newest 5
    assert listed[0]["last_seen"] and not any(s["last_seen"] for s in listed[1:])
    assert all(s["image_url"].startswith("/api/v1/images/ih-") for s in listed)
    live = [s for s in c.get("/api/v1/dashboard/live").json()["sightings"] if s["item"] == "banana"]
    assert [s["id"] for s in live][:5] == [s["id"] for s in listed] and live[0]["last_seen"]  # the Live page's too
    photo = c.get(listed[1]["image_url"])
    assert photo.status_code == 200 and photo.content.startswith(b"\xff\xd8")

    # The newest was a mistake: trash it. Last-seen goes back to "bed", with that sighting's photo.
    gone = c.delete(f"/api/v1/dashboard/items/banana/sightings/{listed[0]['id']}")
    assert gone.status_code == 200 and gone.json()["remaining"] == 5
    assert c.get("/api/v1/items/banana/last-seen").json()["scene"]["label"] == "bed"
    after = c.get("/api/v1/dashboard/items/banana/sightings").json()["sightings"]
    # The sighting from before the camera joined the household (kept under the device) is next in line.
    assert [s["scene"] for s in after] == ["bed", "sink", "shelf", "chair", "kitchen counter"] and after[0]["last_seen"]
    assert c.delete(f"/api/v1/dashboard/items/banana/sightings/{listed[0]['id']}").status_code == 404
    assert c.delete(f"/api/v1/dashboard/items/orange/sightings/{after[0]['id']}").status_code == 404  # wrong item


def test_wifi_fingerprints_are_kept_and_places_can_be_named(c):
    """A sighting keeps its Wi-Fi scan (memory and sighting); the LastScene lookup returns it; named places save."""
    db, run = c.app.state.db, c.portal.call
    scan = [{"bssid": f"a8:00:00:00:00:0{i}", "signal": -60 - 3 * i} for i in range(1, 5)]
    sent = post(c, {**obs(minutes_ago=1, label="orange"), "location": {"wifi": scan}})
    assert sent.status_code == 201
    import time as _time
    for _ in range(10):
        found = run(db.lastscene_position, DEVICE, [DEVICE], "orange")
        if found and found["wifi"]:
            break
        _time.sleep(0.5)
    assert found["wifi"] == {a["bssid"]: a["signal"] for a in scan}
    stored = run(db.pool.fetchval, "SELECT wifi FROM memory_logs WHERE id = $1::uuid", sent.json()["memory_id"])
    assert stored and len(json.loads(stored) if isinstance(stored, str) else stored) == 4

    saved = run(db.save_wifi_place, DEVICE, "Reitz Union", found["wifi"])
    again = run(db.save_wifi_place, DEVICE, "reitz union", found["wifi"])  # same place, other capitalisation
    assert again["id"] == saved["id"] and again["samples"] == 2
    [place] = run(db.wifi_places, DEVICE)
    assert place["name"] == "Reitz Union" and place["aps"] == found["wifi"]
    assert run(db.delete_wifi_place, [DEVICE], saved["id"]) and run(db.wifi_places, DEVICE) == []

    # A named place that is a floor-plan room: surveyed, and later sightings there land in it.
    rion = run(db.save_wifi_place, DEVICE, "Rion Ballroom", found["wifi"], None, "rion")
    run(db.set_wifi_place_position, rion["id"], 29.6465, -82.3475, 15.0)
    [named] = run(db.wifi_places, DEVICE)
    assert (named["lat"], named["lng"], named["accuracy_m"]) == (29.6465, -82.3475, 15.0)
    assert run(db.surveyed_rooms, [DEVICE]) == {"rion": 1}
    again = post(c, {**obs(minutes_ago=0, label="banana"), "location": {"wifi": scan}})
    for _ in range(10):
        room = run(db.pool.fetchval, "SELECT wifi_room FROM memory_logs WHERE id = $1::uuid", again.json()["memory_id"])
        if room:
            break
        _time.sleep(0.5)
    assert room == "rion"
    assert any(s.wifi_room == "rion" for s in run(db.item_sightings, 1, [DEVICE]) if s.name == "banana")
