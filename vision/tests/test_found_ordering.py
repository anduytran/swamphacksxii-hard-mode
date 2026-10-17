""""Found it" vs the upload queue, against a mock that reproduces the real backend's reopen bug.

Real backend (db_observations.upsert_last_seen): any sighting with captured_at >= the stored one
replaces last-seen and clears found_at, even if it was captured before the found action. So the
bridge must never upload, after found, a sighting of that item captured before found.
"""

import time

import cv2

from lastscene_bridge.bridge import LastSceneBridge
from lastscene_bridge.observation import build_observation, captured_at_iso
from memory_engine.api import LastSeen
from memory_engine.detector import Detection
from tests.test_bridge import BOX, T0, _queue_one, bus, ev, make_uploader, observe_keys, server  # noqa: F401

FOUND_MS = T0 + 60_000


def _sent_once(tmp_path, bus, url):
    """The keys were seen (T0 + 500 ms) and that sighting reached the server."""
    ls, batches = observe_keys(tmp_path, bus)
    up = make_uploader(url, tmp_path / "q")
    bridge = LastSceneBridge(ls, uploader=up)
    for b in batches:
        bridge.handle(b)
    assert [s for _, s in up.flush()] == ["sent"]
    return ls, up, bridge


def test_mock_reproduces_the_backend_reopen_bug(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    assert bridge.found("keys", found_at_ms=FOUND_MS)["backend_status"] == 200
    other = make_uploader(url, tmp_path / "other_queue")   # knows nothing about the found action
    stale, _ = _queue_one(other, bus, mid="m2", t=30_000)  # captured BEFORE found
    assert other.flush() == [(stale, "sent")]
    assert other.last_seen("keys").json()["found_at"] is None  # reopened with an older frame


def test_sighting_queued_before_found_is_superseded_and_item_stays_found(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    first = up.last_seen("keys").json()
    late, _ = _queue_one(up, bus, mid="m2", t=30_000)     # a later sighting, captured before found...
    state.forced = [503]
    assert up.flush() == [(late, "retry")]                 # ...stuck in the queue by a server hiccup
    result = bridge.found("keys", found_at_ms=FOUND_MS)
    assert result["superseded"] == [late] and result["backend_status"] == 200 and not result["queued"]
    assert ls.latest_memory("bunch of keys") is None
    time.sleep(0.3)
    assert up.flush() == [] and late not in state.observations
    assert [p.name.split("_", 1)[1] for p in (tmp_path / "q" / "superseded").iterdir()] == [late]  # kept for audit
    rec = up.last_seen("keys").json()
    assert rec["found_at"] is not None and rec["observation_id"] == first["observation_id"]
    assert any("superseded" in s for _, _, s in up.log)


def test_found_is_retried_and_sent_in_queue_order(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    img = cv2.imencode(".jpg", bus)[1].tobytes()
    both = [{**ev("created", "m4", t=50_000), "bbox": list(BOX)},
            {**ev("created", "m5", t=50_000), "bbox": [10, 10, 100, 100], "label": "wallet"}]
    mixed, jpeg = build_observation(both, img, {"name": "n", "version": "v", "detector": "d", "matcher": "orb"})
    up.enqueue(mixed, jpeg)                                # keys + wallet, captured before found: stays queued
    state.forced = [503, 503]
    assert up.flush() == [(mixed["observation_id"], "retry")]
    result = bridge.found("keys", found_at_ms=FOUND_MS)     # no thread: flushes here, mixed fails again
    assert result["superseded"] == [] and result["queued"]  # so found waits behind it
    assert result["backend_status"] is None
    after, _ = _queue_one(up, bus, mid="m6", t=90_000)    # captured after found
    time.sleep(0.3)
    state.forced_found = [503]
    assert up.flush() == [(mixed["observation_id"], "sent"), (result["found_id"], "retry")]
    assert up.is_pending(result["found_id"]) and up.is_pending(after)  # found keeps its place
    time.sleep(0.3)
    assert up.flush() == [(result["found_id"], "sent"), (after, "sent")]
    assert state.arrivals[-3:] == [("observation", mixed["observation_id"]), ("found", "keys"), ("observation", after)]
    assert state.found["keys"] == {"found_at": captured_at_iso(FOUND_MS)}


def test_sighting_captured_after_found_still_uploads_and_reopens(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    assert bridge.found("keys", found_at_ms=FOUND_MS)["backend_status"] == 200
    assert up.last_seen("keys").json()["found_at"] is not None
    time.sleep(0.3)
    after, _ = _queue_one(up, bus, mid="m2", t=90_000)    # keys put down again after being found
    assert up.flush() == [(after, "sent")]
    rec = up.last_seen("keys").json()
    assert rec["observation_id"] == after and rec["found_at"] is None  # expected: a real new sighting


def test_stale_sighting_enqueued_after_found_was_sent_is_not_uploaded(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    assert bridge.found("keys", found_at_ms=FOUND_MS)["backend_status"] == 200
    stale, _ = _queue_one(up, bus, mid="m2", t=30_000)    # e.g. a frame still in inference when "found it" was pressed
    assert up.pending() == [] and stale not in state.observations
    assert up.last_seen("keys").json()["found_at"] is not None


def test_found_without_a_server_sighting_is_terminal(tmp_path, bus, server):
    state, url = server
    ls, _ = observe_keys(tmp_path, bus)                   # memories exist locally, nothing was uploaded
    up = make_uploader(url, tmp_path / "q")
    result = LastSceneBridge(ls, uploader=up).found("keys", found_at_ms=FOUND_MS)
    assert result["backend_status"] == 404 and not result["queued"]
    assert ls.latest_memory("bunch of keys") is None      # local deletion still happened
    assert [p.name.split("_", 1)[1] for p in (tmp_path / "q" / "failed").iterdir()] == [result["found_id"]]
    time.sleep(0.3)
    assert up.flush() == [] and "keys" not in state.found


def test_background_thread_sends_the_found_action(tmp_path, bus, server):
    state, url = server
    ls, up, bridge = _sent_once(tmp_path, bus, url)
    up.start(poll_s=0.05)
    try:
        result = bridge.found("keys", found_at_ms=FOUND_MS)  # returns without waiting for the network
        deadline = time.monotonic() + 5
        while up.is_pending(result["found_id"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "keys" in state.found and up.http_status[result["found_id"]] == 200
    finally:
        up.stop()


def test_found_deletes_memories_of_every_prompt_for_that_item(tmp_path, bus):
    class Two:  # two prompts that both mean "keys"
        def detect(self, image):
            return [Detection("bunch of keys", 0.148, BOX), Detection("keychain", 0.12, (500, 100, 700, 300))]
    det = Two()
    ls = LastSeen(out_dir=str(tmp_path / "mem"), detector=det, search_detector=det, source="test-cam")
    for i in range(6):
        ls.process_frame(bus, i * 500, wall_ms=T0 + i * 500, observed_ms=T0 + i * 500)
    assert {r["label"] for r in ls.memories()} == {"bunch of keys", "keychain"}
    result = LastSceneBridge(ls).found("keys", found_at_ms=FOUND_MS)
    assert len(result["removed_locally"]) == 2 and ls.memories() == []
