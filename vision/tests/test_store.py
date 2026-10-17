import json

import numpy as np

from memory_engine.store import MemoryStore, crop


def event(kind, bbox=(10, 10, 30, 30), conf=0.5):
    return {"event": kind, "memory_id": "mem_keys_abc12345", "label": "keys", "bbox": list(bbox),
            "confidence": conf, "frame_id": 3, "video_ms": 300, "wall_ms": 1, "first_seen_video_ms": 0,
            "last_seen_video_ms": 300, "sightings": 1}


def test_created_saves_images_and_snapshot(tmp_path):
    store = MemoryStore(str(tmp_path), "clip.mp4")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = store.handle(event("created"), image)
    store.close()

    assert (tmp_path / out["image_path"]).exists()
    assert (tmp_path / out["crop_path"]).exists()
    snapshot = json.loads((tmp_path / "memories.json").read_text())
    assert snapshot["source"] == "clip.mp4"
    assert snapshot["memories"][0]["memory_id"] == "mem_keys_abc12345"
    assert "event" not in snapshot["memories"][0]
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["event"] == "created"


def test_non_image_events_keep_previous_paths(tmp_path):
    store = MemoryStore(str(tmp_path), "clip.mp4")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    store.handle(event("created"), image)
    ended = store.handle(event("sighting_ended"), None)
    store.close()
    assert ended["image_path"] == "images/mem_keys_abc12345.jpg"
    assert len(json.loads((tmp_path / "memories.json").read_text())["memories"]) == 1


def test_last_seen_time_reaches_memories_json(tmp_path):
    """The saved image is from early on; last_seen must keep moving and end up in the file."""
    from memory_engine.detector import Detection
    from memory_engine.memory import MemoryMaker

    maker, store = MemoryMaker(), MemoryStore(str(tmp_path), "clip.mp4")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keys = [Detection("keys", 0.5, (10, 10, 30, 30))]
    for i in range(20):                    # seen every 100 ms from 0 to 1900 ms
        for e in maker.process(i, i * 100, 0, keys):
            store.handle(e, image)
    for i in range(20, 40):                # then out of view until the sighting ends
        for e in maker.process(i, i * 100, 0, []):
            store.handle(e, image)
    store.close()

    record = json.loads((tmp_path / "memories.json").read_text())["memories"][0]
    assert record["first_seen_video_ms"] == 0
    assert record["last_seen_video_ms"] == 1900  # the latest sighting, not the first
    assert record["video_ms"] == 1900            # and the saved image is from that latest sighting


def test_session_fields_are_added_without_touching_existing_ones(tmp_path):
    store = MemoryStore(str(tmp_path), "clip.mp4", session_id="sess_test", source_kind="file")
    out = store.handle(event("created"), np.zeros((100, 100, 3), dtype=np.uint8))
    store.close()
    original_fields = {"memory_id", "label", "bbox", "confidence", "frame_id", "video_ms", "wall_ms",
                       "first_seen_video_ms", "last_seen_video_ms", "sightings", "image_path",
                       "crop_path", "source"}
    assert original_fields <= out.keys()
    assert out["session_id"] == "sess_test" and out["source_kind"] == "file"
    assert "_image" not in out
    snapshot = json.loads((tmp_path / "memories.json").read_text())
    assert snapshot["session_id"] == "sess_test"


def test_attached_image_wins_over_fallback(tmp_path):
    store = MemoryStore(str(tmp_path), "clip.mp4")
    evidence = np.full((100, 100, 3), 255, dtype=np.uint8)   # white: the frame the event is about
    current = np.zeros((100, 100, 3), dtype=np.uint8)        # black: whatever frame is current now
    e = event("image_updated")
    e["_image"] = evidence
    out = store.handle(e, current)
    store.close()
    import cv2
    assert cv2.imread(str(tmp_path / out["image_path"])).mean() > 250


def test_delete_removes_images_record_and_history(tmp_path):
    store = MemoryStore(str(tmp_path), "clip.mp4")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = store.handle(event("created"), image)
    store.handle(event("sighting_ended"), image)
    other = event("created")
    other["memory_id"] = "mem_wallet_00000001"
    store.handle(other, image)

    removed = store.delete("mem_keys_abc12345")
    store.close()
    assert removed["files_removed"] == [out["image_path"], out["crop_path"]]
    assert not (tmp_path / out["image_path"]).exists() and not (tmp_path / out["crop_path"]).exists()
    ids = [m["memory_id"] for m in json.loads((tmp_path / "memories.json").read_text())["memories"]]
    assert ids == ["mem_wallet_00000001"]
    log = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    keys_lines = [e for e in log if e["memory_id"] == "mem_keys_abc12345"]
    assert keys_lines == [{"event": "deleted", "memory_id": "mem_keys_abc12345", "label": "keys",
                           "wall_ms": keys_lines[0]["wall_ms"]}]   # only a tombstone, no bbox/times/paths
    assert store.delete("mem_keys_abc12345") is None


def test_memories_survive_a_restart(tmp_path):
    store = MemoryStore(str(tmp_path), "first_run")
    store.handle(event("created"), np.zeros((100, 100, 3), dtype=np.uint8))
    store.close()
    again = MemoryStore(str(tmp_path), "second_run")
    assert "mem_keys_abc12345" in again.records
    again.close()


def test_crop_stays_inside_image():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    assert crop(image, (0, 0, 20, 20)).shape[:2] == (30, 30)  # padded right/bottom, clipped left/top
    assert crop(image, (40, 40, 50, 50)).shape[:2] == (15, 15)
