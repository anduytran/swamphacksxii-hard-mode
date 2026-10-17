"""Scene-matching integration (the feature flag) on real photos with hand-placed boxes.

bus.jpg / zidane.jpg are real photos; the "keys" box is a fixed region we pretend the
detector reported. Each scenario also runs with the flag off to show what the original
image-position rule does in the same situation.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from memory_engine.detector import Detection
from memory_engine.memory import MemoryConfig, MemoryMaker
from memory_engine.scene import SceneMatcher

FIX = Path(__file__).parent / "fixtures"
BOX = (223, 406, 344, 860)
CFG = MemoryConfig(min_conf=0.05, min_hits=2)


@pytest.fixture(scope="module")
def bus():
    return cv2.imread(str(FIX / "bus.jpg"))


@pytest.fixture(scope="module")
def zidane_as_bus_size(bus):
    return cv2.resize(cv2.imread(str(FIX / "zidane.jpg")), (bus.shape[1], bus.shape[0]))


def shift(img, dx):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), (w, h))


def moved(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3])


def run(maker, segments, fps=2):
    """segments: list of (image, box or None, seconds). A None box = object not seen."""
    events, t, i = [], 0, 0
    for image, box, seconds in segments:
        for _ in range(int(seconds * fps)):
            dets = [Detection("keys", 0.5, box)] if box else []
            events += maker.process(i, t, t, dets, image=image)
            t += 1000 // fps
            i += 1
    return events + maker.finish()


def kinds(events, kind="created"):
    return [e for e in events if e["event"] == kind]


def both(segments):
    on = run(MemoryMaker(CFG, scene_matcher=SceneMatcher()), segments)
    off = run(MemoryMaker(CFG), segments)
    return on, off


def test_first_memory_is_labeled_first_sighting(bus):
    on, _ = both([(bus, BOX, 2)])
    assert [e["place_status"] for e in kinds(on)] == ["first_sighting"]


def test_camera_turned_same_spot_is_a_resighting(bus):
    # look at the keys, look away (black), look back with the camera turned (scene shifted 250 px)
    segments = [(bus, BOX, 2), (np.zeros_like(bus), None, 3), (shift(bus, 250), moved(BOX, 250), 2)]
    on, off = both(segments)
    assert len(kinds(on)) == 1 and len(kinds(on, "resighted")) + len(kinds(on, "image_updated")) >= 1
    assert len(kinds(off)) == 2  # original rule: image moved -> duplicate memory


def test_different_place_at_same_image_position_is_not_a_resighting(bus, zidane_as_bus_size):
    segments = [(bus, BOX, 2), (np.zeros_like(bus), None, 3), (zidane_as_bus_size, BOX, 2)]
    on, off = both(segments)
    created = kinds(on)
    assert len(created) == 2
    assert created[1]["place_status"] in ("likely_new_place", "unverified")
    assert created[1]["compared_with"] == created[0]["memory_id"]
    assert len(kinds(off)) == 1  # original rule: same pixels -> wrongly "same place"


def test_object_moved_within_the_same_view(bus):
    far_spot = (600, 150, 700, 300)
    segments = [(bus, BOX, 2), (np.zeros_like(bus), None, 3), (bus, far_spot, 2)]
    on, _ = both(segments)
    assert [e["place_status"] for e in kinds(on)] == ["first_sighting", "moved_in_scene"]


def test_blank_view_is_unverified_not_a_new_place(bus):
    blank = np.full_like(bus, 200)
    segments = [(bus, BOX, 2), (np.zeros_like(bus), None, 3), (blank, (10, 10, 60, 60), 2)]
    on, _ = both(segments)
    assert kinds(on)[1]["place_status"] == "unverified"


def test_two_key_rings_in_view_at_once_are_two_objects(bus):
    other = (600, 150, 700, 300)
    maker = MemoryMaker(CFG, scene_matcher=SceneMatcher())
    events = []
    for i in range(4):
        events += maker.process(i, i * 500, 0, [Detection("keys", 0.5, BOX), Detection("keys", 0.4, other)],
                                image=bus)
    statuses = sorted(e["place_status"] for e in kinds(events))
    assert statuses == ["another_instance", "first_sighting"]


def test_disappearing_keeps_the_last_confirmed_sighting(bus):
    maker = MemoryMaker(CFG, scene_matcher=SceneMatcher())
    events = run(maker, [(bus, BOX, 2), (np.zeros_like(bus), None, 5)])
    (memory,) = maker.memories.values()
    assert memory.place_status == "first_sighting"
    assert memory.last_seen_video_ms == 1500          # last frame it was actually seen
    assert kinds(events, "sighting_ended")             # sighting closed, memory kept


def test_forget_removes_memory_and_pauses_that_label(bus):
    maker = MemoryMaker(CFG, scene_matcher=SceneMatcher())
    run(maker, [(bus, BOX, 2)])
    (mid,) = maker.memories
    assert maker.forget(mid, cooldown_ms=2000)
    assert maker.memories == {} and maker._features == {}
    t0 = maker._now_ms
    during = [e for i in range(1, 4) for e in maker.process(100 + i, t0 + i * 500, 0, [Detection("keys", 0.5, BOX)], image=bus)]
    assert kinds(during) == []                         # still in hand / in view right after "found it"
    later = [e for i in range(5, 9) for e in maker.process(100 + i, t0 + i * 500, 0, [Detection("keys", 0.5, BOX)], image=bus)]
    assert len(kinds(later)) == 1                      # after the cooldown, memories resume
