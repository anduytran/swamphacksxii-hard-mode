"""Memory logic tests. Detections here are hand-written test inputs, not model output."""

from memory_engine.detector import Detection
from memory_engine.memory import MemoryConfig, MemoryMaker, iou

DESK = (100, 100, 160, 140)
SHELF = (400, 50, 460, 90)


def feed(maker, frames, fps=10, start_ms=0, start_id=0):
    """frames: list of detection lists, one per frame, spaced 1000/fps ms apart."""
    events = []
    for i, dets in enumerate(frames):
        t = start_ms + int(i * 1000 / fps)
        events += maker.process(start_id + i, t, 1_700_000_000_000 + t, dets)
    return events


def kinds(events):
    return [e["event"] for e in events]


def keys(bbox=DESK, conf=0.5):
    return Detection("keys", conf, bbox)


def test_iou_basics():
    assert iou(DESK, DESK) == 1.0
    assert iou(DESK, SHELF) == 0.0
    assert 0 < iou((0, 0, 10, 10), (5, 0, 15, 10)) < 1


def test_steady_object_creates_exactly_one_memory():
    maker = MemoryMaker()
    events = feed(maker, [[keys()]] * 20)  # 2 s at 10 fps
    assert kinds(events) == ["created"]
    e = events[0]
    assert e["label"] == "keys" and e["bbox"] == list(DESK)
    assert e["memory_id"].startswith("mem_keys_")
    assert e["video_ms"] == 500  # first frame meeting both 3 hits and 500 ms


def test_one_frame_fluke_is_ignored():
    maker = MemoryMaker()
    events = feed(maker, [[keys()], [], [], [keys()], [], [], [], [], [], []])
    assert events == []
    assert maker.memories == {}


def test_weak_detections_are_ignored():
    maker = MemoryMaker()
    assert feed(maker, [[keys(conf=0.1)]] * 20) == []


def test_same_place_after_looking_away_is_not_a_duplicate():
    maker = MemoryMaker()
    first = feed(maker, [[keys()]] * 10)               # 0-0.9 s on the desk
    gap = feed(maker, [[]] * 30, start_ms=1000)          # 3 s looking away: sighting ends
    again = feed(maker, [[keys()]] * 10, start_ms=4000)  # back on the same desk spot
    assert kinds(first) == ["created"]
    assert kinds(gap) == ["image_updated", "sighting_ended"]  # evidence moves to the last frame seen
    assert kinds(again) == ["resighted"]
    assert len(maker.memories) == 1
    assert again[0]["sightings"] == 2
    assert again[0]["memory_id"] == first[0]["memory_id"]


def test_object_in_a_new_place_gets_a_new_memory():
    maker = MemoryMaker()
    feed(maker, [[keys(DESK)]] * 10)
    feed(maker, [[]] * 30, start_ms=1000)
    moved = feed(maker, [[keys(SHELF)]] * 10, start_ms=4000)
    assert kinds(moved) == ["created"]
    assert len(maker.memories) == 2


def test_two_objects_with_same_label_get_two_memories():
    maker = MemoryMaker()
    events = feed(maker, [[keys(DESK), keys(SHELF)]] * 10)
    assert kinds(events) == ["created", "created"]
    assert {tuple(e["bbox"]) for e in events} == {DESK, SHELF}


def test_clearer_frame_replaces_the_image():
    maker = MemoryMaker()
    events = feed(maker, [[keys(conf=0.4)]] * 6 + [[keys(conf=0.8)]])
    assert kinds(events) == ["created", "image_updated"]
    assert events[1]["confidence"] == 0.8
    assert events[1]["frame_id"] == 6


def test_small_confidence_wobble_does_not_spam_updates():
    maker = MemoryMaker()
    events = feed(maker, [[keys(conf=c)] for c in [0.5, 0.52, 0.51, 0.53, 0.5, 0.52, 0.54, 0.51]])
    assert kinds(events) == ["created"]


def test_same_behavior_at_5fps_and_30fps():
    # thresholds are in milliseconds, so camera frame rate shouldn't change the outcome
    for fps in (5, 30):
        maker = MemoryMaker()
        events = feed(maker, [[keys()]] * (2 * fps), fps=fps)
        assert kinds(events) == ["created"], fps


def test_finish_closes_open_sightings():
    maker = MemoryMaker()
    feed(maker, [[keys()]] * 10)
    assert kinds(maker.finish()) == ["image_updated", "sighting_ended"]


def test_evidence_follows_the_latest_position_not_the_clearest_frame():
    # very clear on the left of the desk, then nudged right and less clear, then gone
    maker = MemoryMaker()
    left, right = (100, 100, 160, 140), (120, 100, 180, 140)  # overlapping, so one track
    feed(maker, [[keys(left, 0.9)]] * 8 + [[keys(right, 0.6)]] * 4)
    end = maker.finish()
    assert kinds(end) == ["image_updated", "sighting_ended"]
    assert end[0]["bbox"] == list(right)
    assert end[0]["video_ms"] == 1100   # the last frame, not the clearest one at the start


def test_weak_final_frames_are_not_used_as_evidence():
    # 0.2 < 0.6 x 0.9: too unsure to be evidence, so the last confident frame is kept
    maker = MemoryMaker(MemoryConfig(min_conf=0.1))
    feed(maker, [[keys(conf=0.9)]] * 8 + [[keys(conf=0.2)]] * 4)
    end = maker.finish()
    assert end[0]["event"] == "image_updated"
    assert end[0]["video_ms"] == 700 and end[0]["confidence"] == 0.9
    assert end[-1]["last_seen_video_ms"] == 1100  # but it was still *seen* until 1.1 s


def test_events_carry_the_image_they_describe():
    maker = MemoryMaker()
    frames = [f"frame{i}" for i in range(12)]  # stand-ins for images, to check which one travels
    events = []
    for i in range(12):
        events += maker.process(i, i * 100, 0, [keys()], image=frames[i])
    events += maker.finish()
    created, updated = events[0], events[1]
    assert created["_image"] == frames[created["frame_id"]]
    assert updated["_image"] == frames[11]


def test_observed_time_is_kept_separately_and_may_be_unknown():
    maker = MemoryMaker()
    events = []
    for i in range(10):
        events += maker.process(i, i * 100, 5_000 + i, [keys()], observed_ms=None)
    assert events[0]["observed_ms"] is None and events[0]["wall_ms"] == 5_005
    maker2 = MemoryMaker()
    events2 = []
    for i in range(10):
        events2 += maker2.process(i, i * 100, 0, [keys()], observed_ms=1_000_000 + i * 100)
    assert events2[0]["observed_ms"] == 1_000_500
    assert events2[0]["last_seen_observed_ms"] == 1_000_500


def test_config_is_adjustable():
    maker = MemoryMaker(MemoryConfig(min_hits=2, min_seen_ms=0))
    assert kinds(feed(maker, [[keys()]] * 2)) == ["created"]
