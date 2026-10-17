"""LastScene bridge: contract conversion, event selection, uploader behavior against a local mock.

The mock (tests/mock_lastscene.py) follows the integration contract; it is not Dylan's server.
"""

import json
import shutil
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from lastscene_bridge.auth import Auth0ClientCredentials, token_provider_from_env
from lastscene_bridge.bridge import LastSceneBridge
from lastscene_bridge.capture_time import recording_time_ms
from lastscene_bridge.observation import (NoCaptureTime, build_observation, captured_at_iso, event_for, item_label,
                                          normalize_bbox, observation_id, prepare_image, scene_block)
from lastscene_bridge.selector import EventSelector
from lastscene_bridge.uploader import RedirectError, Uploader, normalize_api_base
from memory_engine.api import LastSeen
from memory_engine.detector import Detection
from tests import mock_lastscene as mock

FIX = Path(__file__).parent / "fixtures"
BOX = (223, 406, 344, 860)          # pixel box on bus.jpg (810x1080)
T0 = 1_792_220_000_000              # a real-looking UTC epoch ms for observed_ms


@pytest.fixture
def server():
    srv, state, url = mock.serve(0, rate_limit_s=0.2)
    yield state, url
    srv.shutdown()


def env_for(url):
    return {"LASTSCENE_API_BASE": url, "LASTSCENE_PAIRING_CODE": mock.PAIRING_CODE, "AUTH0_DOMAIN": url,
            "AUTH0_CLIENT_ID": mock.CLIENT_ID, "AUTH0_CLIENT_SECRET": mock.CLIENT_SECRET, "AUTH0_AUDIENCE": mock.AUDIENCE}


def make_uploader(url, queue, min_interval_s=0.25):
    e = {**env_for(url), "LASTSCENE_TOKEN_CACHE": str(Path(queue).parent / f"{Path(queue).name}_auth0.json")}
    return Uploader(e["LASTSCENE_API_BASE"], e["LASTSCENE_PAIRING_CODE"], token_provider_from_env(e), queue,
                    min_interval_s=min_interval_s, max_backoff_s=2)


class Scripted:
    def __init__(self):
        self.box = None

    def detect(self, image):
        return [Detection("bunch of keys", 0.148, self.box)] if self.box else []


def observe_keys(tmp_path, image, observed=True, seconds=3, start_ms=0):
    """Real LastSeen API with a scripted detector: the keys sit in view for `seconds`, then leave."""
    det = Scripted()
    ls = LastSeen(out_dir=str(tmp_path / "mem"), detector=det, search_detector=det, source="test-cam")
    events = []
    for i in range(int(seconds * 2) + 6):
        det.box = BOX if i < seconds * 2 else None
        t = start_ms + i * 500
        events.append(ls.process_frame(image, t, wall_ms=T0 + t, observed_ms=(T0 + t) if observed else None))
    events.append(ls.finish())
    return ls, events


@pytest.fixture
def bus():
    return cv2.imread(str(FIX / "bus.jpg"))


# --- conversion ----------------------------------------------------------------------------
def test_bbox_is_normalized_xywh_by_the_uploaded_image_size():
    assert normalize_bbox((192, 108, 384, 324), 1920, 1080) == [0.1, 0.1, 0.1, 0.2]
    assert normalize_bbox((-10, 0, 2000, 1080), 1920, 1080) == [0.0, 0.0, 1.0, 1.0]  # clipped to the image


def test_downscaled_upload_keeps_the_same_normalized_box(bus):
    raw = cv2.imencode(".png", bus)[1].tobytes()             # PNG, so it gets re-encoded
    jpeg, w, h, scale = prepare_image(raw, max_bytes=60_000)  # force a downscale
    assert len(jpeg) <= 60_000 and scale < 1 and (w, h) != (810, 1080)
    small = normalize_bbox([v * scale for v in BOX], w, h)
    full = normalize_bbox(BOX, 810, 1080)
    assert all(abs(a - b) < 0.01 for a, b in zip(small, full))


def test_timestamps_ids_labels():
    assert captured_at_iso(1_000_000_000_123) == "2001-09-09T01:46:40.123Z"  # 10^9 s after the epoch
    assert captured_at_iso(0) == "1970-01-01T00:00:00.000Z"
    assert observation_id("cam", 1) == observation_id("cam", 1) != observation_id("cam", 2)
    assert item_label("bunch of keys") == "keys" and item_label("Reading Glasses") == "reading glasses"


def test_observation_matches_contract_and_is_conservative(tmp_path, bus):
    ls, batches = observe_keys(tmp_path, bus)
    created = next(e for b in batches for e in b if e["event"] == "created")
    image = (Path(ls.store.out) / created["image_path"]).read_bytes()
    obs, jpeg = build_observation([created], image, {"name": "lastseen-ml", "version": "t", "detector": "d", "matcher": "orb"})
    assert set(obs) == {"observation_id", "captured_at", "source", "pipeline", "image_size", "objects"}  # no scene
    assert obs["source"] == "auto" and obs["image_size"] == {"width": 810, "height": 1080}
    (o,) = obs["objects"]
    assert o == {"label": "keys", "confidence": 0.148, "bbox": normalize_bbox(BOX, 810, 1080),
                 "track_id": created["memory_id"], "event": "placed"}  # first confirmed sighting
    assert mock.validate(obs, jpeg) is None
    scene = scene_block("Grand Ballroom table")
    with_scene, _ = build_observation([created], image, obs["pipeline"], scene)
    assert with_scene["scene"] == {"id": "grand-ballroom-table", "label": "Grand Ballroom table"}
    assert mock.validate(with_scene, jpeg) is None


def test_event_names_follow_what_the_engine_established():
    assert event_for({"event": "created", "place_status": "first_sighting"}) == "placed"
    assert event_for({"event": "created", "place_status": None}) == "placed"          # no scene matcher
    assert event_for({"event": "created", "place_status": "unverified"}) == "placed"  # no proof it moved
    assert event_for({"event": "created", "place_status": "likely_new_place"}) == "moved"
    assert event_for({"event": "created", "place_status": "moved_in_scene"}) == "moved"
    assert event_for({"event": "image_updated", "_reason": "seen"}) == "seen"
    assert event_for({"event": "image_updated", "_reason": "moved"}) == "moved"
    assert "picked_up" not in {event_for({"event": k, "place_status": p, "_reason": r})
                               for k in ("created", "image_updated") for p in (None, "likely_new_place")
                               for r in ("seen", "moved")}


def test_unknown_capture_time_means_no_observation(tmp_path, bus):
    ls, batches = observe_keys(tmp_path, bus, observed=False)
    bridge = LastSceneBridge(ls, outbox_dir=tmp_path / "outbox")
    for b in batches:
        bridge.handle(b)
    assert bridge.stats["observations"] == 0 and bridge.stats["skipped_no_capture_time"] >= 1
    with pytest.raises(NoCaptureTime):
        created = next(e for b in batches for e in b if e["event"] == "created")
        build_observation([created], (Path(ls.store.out) / created["image_path"]).read_bytes(), {})


def test_low_confidence_is_sent_and_reported_not_dropped(tmp_path, bus):
    ls, batches = observe_keys(tmp_path, bus)
    bridge = LastSceneBridge(ls, outbox_dir=tmp_path / "outbox")
    for b in batches:
        bridge.handle(b)
    assert bridge.stats["observations"] == 1 and bridge.stats["below_contract_min_confidence"] == 1
    strict = LastSceneBridge(ls, outbox_dir=tmp_path / "outbox2", enforce_contract_min_confidence=True)
    ls2, batches2 = observe_keys(tmp_path / "b", bus)
    strict.ls = ls2
    for b in batches2:
        strict.handle(b)
    assert strict.stats["observations"] == 0 and strict.stats["dropped_by_contract_min_confidence"] == 1


# --- selection --------------------------------------------------------------------------------
def ev(kind, mid="m1", t=0, bbox=(100, 100, 200, 200), conf=0.2):
    return {"event": kind, "memory_id": mid, "observed_ms": T0 + t, "video_ms": t, "bbox": list(bbox),
            "confidence": conf, "source": "cam", "label": "bunch of keys", "_image_size": (1000, 1000)}


def test_selector_sends_events_not_frames():
    s = EventSelector()
    assert len(s.select([ev("created")])) == 1
    assert s.select([ev("image_updated", t=500)]) == []                        # mid-sighting: clearer frame only
    assert s.select([ev("image_updated", t=900), ev("sighting_ended")]) == []  # end, but barely moved
    moved = s.select([ev("image_updated", t=2000, bbox=(500, 500, 600, 600)), ev("sighting_ended")])
    assert len(moved) == 1 and moved[0][0]["_reason"] == "moved"                # end, evidence moved > 10% of frame
    assert s.select([ev("resighted", t=5000)]) == []                           # old evidence frame: never sent
    later = s.select([ev("image_updated", t=11 * 60 * 1000, bbox=(500, 500, 600, 600)), ev("sighting_ended")])
    assert len(later) == 1 and later[0][0]["_reason"] == "seen"                 # seen again after > 10 minutes


def test_selector_batches_objects_from_one_frame():
    groups = EventSelector().select([ev("created", "a"), ev("created", "b", bbox=(300, 300, 400, 400))])
    assert len(groups) == 1 and len(groups[0]) == 2


# --- uploader against the mock ------------------------------------------------------------------
def test_end_to_end_upload_last_seen_and_found(tmp_path, bus, server):
    state, url = server
    ls, batches = observe_keys(tmp_path, bus)
    up = make_uploader(url, tmp_path / "queue")
    bridge = LastSceneBridge(ls, uploader=up)
    for b in batches:
        bridge.handle(b)
    assert [s for _, s in up.flush()] == ["sent"]
    assert len(state.observations) == 1
    last = up.last_seen("keys").json()
    assert last["bbox"] == normalize_bbox(BOX, 810, 1080) and last["source"] == "auto"
    # the frame the memory was confirmed on (2nd look, 500 ms in); the end-of-sighting update at the
    # same spot is not re-sent (selector: no move, < 10 min)
    assert last["captured_at"] == captured_at_iso(T0 + 500)
    assert requests.get(url + last["image_url"]).content[:2] == b"\xff\xd8"
    result = bridge.found("keys", found_at_ms=T0 + 60_000)
    assert result["backend_status"] == 200 and not result["queued"] and result["observation_id"] == last["observation_id"]
    assert state.found["keys"] == {"found_at": captured_at_iso(T0 + 60_000)}
    assert up.last_seen("keys").json()["found_at"] is not None
    assert ls.latest_memory("bunch of keys") is None  # local evidence deleted


def test_contract_rate_limit_one_post_per_2s(tmp_path, bus):
    srv, state, url = mock.serve(0)  # real contract limit: 2 s
    try:
        up = make_uploader(url, tmp_path / "q", min_interval_s=2.05)
        img = cv2.imencode(".jpg", bus)[1].tobytes()
        for i in range(2):
            obs, jpeg = build_observation([{**ev("created", f"m{i}", t=i), "bbox": list(BOX)}], img,
                                          {"name": "n", "version": "v", "detector": "d", "matcher": "orb"})
            up.enqueue(obs, jpeg)
        assert [s for _, s in up.flush()] == ["sent", "sent"]
        gap = state.post_times[1] - state.post_times[0]
        assert gap >= 2.0, gap
    finally:
        srv.shutdown()


def _queue_one(up, bus, mid="m1", t=0, event="seen"):
    img = cv2.imencode(".jpg", bus)[1].tobytes()
    obs, jpeg = build_observation([{**ev("created", mid, t=t), "bbox": list(BOX)}], img,
                                  {"name": "n", "version": "v", "detector": "d", "matcher": "orb"})
    obs["objects"][0]["event"] = event
    return up.enqueue(obs, jpeg), obs


def test_retry_after_500_reuses_the_same_id(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    state.forced = [500]
    assert up.flush() == [(oid, "retry")] and len(up.pending()) == 1
    time.sleep(0.3)
    assert up.flush() == [(oid, "sent")] and list(state.observations) == [oid]
    assert up.enqueue(json.loads((next((tmp_path / "q" / "sent").iterdir()) / "observation.json").read_text()), b"x") == oid
    assert up.pending() == []  # already sent: not queued twice


def test_duplicate_post_is_a_200_no_op(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    up.flush()
    sent = next((tmp_path / "q" / "sent").iterdir())
    shutil.move(str(sent), str(tmp_path / "q" / "pending" / sent.name))  # simulate a lost response
    time.sleep(0.3)
    assert up.flush() == [(oid, "duplicate")] and len(state.observations) == 1


def test_429_waits_and_succeeds(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    state.forced = [429]
    assert up.flush() == [(oid, "sent")]
    assert any("429" in s for _, _, s in up.log)


def test_offline_queue_survives_and_flushes_in_capture_order(tmp_path, bus):
    up = make_uploader("http://127.0.0.1:9", tmp_path / "q")  # nothing listens there
    ids = [_queue_one(up, bus, mid=f"m{i}", t=t)[0] for i, t in enumerate([3000, 1000, 2000])]
    assert up.flush()[0][1] == "retry" and len(up.pending()) == 3
    srv, state, url = mock.serve(0, rate_limit_s=0.2)
    try:
        restarted = make_uploader(url, tmp_path / "q")     # e.g. after a reboot: same queue folder
        assert [s for _, s in restarted.flush()] == ["sent"] * 3
        order = [state.observations[i]["obs"]["captured_at"] for i in state.observations]
        assert order == sorted(order) and set(state.observations) == set(ids)
    finally:
        srv.shutdown()


def test_expired_token_is_refreshed_once(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    up.tokens.get()
    state.tokens.clear()            # server forgot our token
    oid, _ = _queue_one(up, bus)
    assert up.flush() == [(oid, "sent")]
    assert state.token_requests == 2


def test_403_surfaces_without_a_new_token(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    up.tokens.get()
    oid, _ = _queue_one(up, bus)
    state.forced = [403]            # valid token, missing scope: a new token would not help
    assert up.flush() == [(oid, "auth")] and up.pending()
    assert state.token_requests == 1
    assert any("403" in s for _, _, s in up.log)


def test_rejected_fresh_token_is_not_refetched_on_every_retry(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    state.forced = [401, 401, 401]  # the API keeps rejecting even a fresh token (e.g. wrong audience)
    assert up.flush() == [(oid, "auth")] and state.token_requests == 2  # one refetch, as before
    assert up.flush() == [(oid, "auth")] and state.token_requests == 2  # retry: no new Auth0 token
    time.sleep(0.3)
    assert up.flush() == [(oid, "sent")] and state.token_requests == 2


def test_other_request_errors_are_transient(tmp_path, bus, server, monkeypatch):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    real = requests.request
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("truncated body")
        return real(*a, **k)
    monkeypatch.setattr("lastscene_bridge.uploader.requests.request", flaky)
    assert up.flush() == [(oid, "retry")] and up.pending()
    time.sleep(0.3)
    assert up.flush() == [(oid, "sent")]


def test_apex_domain_is_normalized_to_www(caplog):
    with caplog.at_level("WARNING", logger="lastscene_bridge.uploader"):
        assert normalize_api_base("https://lastscene.photo/") == "https://www.lastscene.photo"
    assert "www.lastscene.photo" in caplog.text
    assert normalize_api_base("https://www.lastscene.photo") == "https://www.lastscene.photo"
    assert normalize_api_base("http://127.0.0.1:8765/") == "http://127.0.0.1:8765"


def test_redirect_is_a_configuration_error_not_a_401(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    state.redirect = "https://elsewhere.example"
    assert up.flush() == [(oid, "config")]
    assert "https://elsewhere.example/api/v1/observations" in up.config_error
    assert up.pending() and not state.observations     # kept queued for after the fix, nothing sent
    assert state.token_requests == 1                    # not treated as an auth problem
    with pytest.raises(RedirectError, match="308"):
        up.last_seen("keys")
    state.redirect = None
    assert up.flush() == [(oid, "sent")]


def test_bad_field_goes_to_failed_with_the_server_message(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus, event="dropped")
    assert up.flush() == [(oid, "failed")]
    err = json.loads((next((tmp_path / "q" / "failed").iterdir()) / "error.json").read_text())
    assert err["status"] == 422 and "event" in err["body"]


def test_too_large_image_is_reencoded(tmp_path, bus, server):
    state, url = server
    state.max_image_bytes = 60_000
    up = make_uploader(url, tmp_path / "q")
    oid, _ = _queue_one(up, bus)
    assert up.flush() == [(oid, "sent")]
    stored = state.observations[oid]
    assert len(stored["image"]) <= 60_000 and mock.validate(stored["obs"], stored["image"]) is None


def test_no_endpoint_no_upload_and_secrets_are_redacted(tmp_path):
    with pytest.raises(RuntimeError):
        Uploader("", "code", None, tmp_path / "q")
    assert "s3cret" not in repr(Auth0ClientCredentials("t.auth0.com", "id", "s3cret", "aud"))


# --- capture time from file metadata -------------------------------------------------------------
def test_capture_time_from_mvhd(tmp_path):
    secs = 3_843_610_554  # 2025-10-17 05:35:54 UTC in seconds since 1904
    mvhd = b"\x00\x00\x00\x00" + struct.pack(">I", secs) + b"\x00" * 92
    moov = struct.pack(">I4s", 8 + 8 + len(mvhd), b"moov") + struct.pack(">I4s", 8 + len(mvhd), b"mvhd") + mvhd
    f = tmp_path / "x.mov"
    f.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  \x00\x00\x00\x00" + moov)
    assert recording_time_ms(f) == (secs - 2_082_844_800) * 1000
    empty = tmp_path / "y.mov"
    empty.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  \x00\x00\x00\x00")
    assert recording_time_ms(empty) is None
