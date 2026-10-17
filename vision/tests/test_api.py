"""LastSeen API tests with real photos and a stand-in detector (scripted boxes)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from memory_engine.api import LastSeen, direction, resolve_label
from memory_engine.detector import Detection

FIX = Path(__file__).parent / "fixtures"
BOX = (223, 406, 344, 860)
LABEL = "bunch of keys"


class ScriptedDetector:
    """Returns whatever box is set in .box (None = nothing)."""

    def __init__(self):
        self.box = None

    def detect(self, image):
        return [Detection(LABEL, 0.5, self.box)] if self.box else []


@pytest.fixture
def bus():
    return cv2.imread(str(FIX / "bus.jpg"))


@pytest.fixture
def ls(tmp_path):
    det = ScriptedDetector()
    api = LastSeen(out_dir=str(tmp_path), detector=det, search_detector=det)
    api.det = det
    yield api
    api.close()


def feed(ls, image, box, seconds, start_ms):
    ls.det.box = box
    for i in range(int(seconds * 2)):
        ls.process_frame(image, start_ms + i * 500, wall_ms=start_ms + i * 500)
    return start_ms + int(seconds * 1000)


def test_resolve_label():
    assert resolve_label("My Keys") == LABEL
    assert resolve_label("wallet") == "wallet"


def test_direction():
    assert direction((0, 40, 20, 60), (100, 100))["horizontal"] == "left"
    assert direction((80, 40, 100, 60), (100, 100))["horizontal"] == "right"
    d = direction((45, 45, 55, 55), (100, 100))
    assert (d["horizontal"], d["vertical"], d["in_frame"]) == ("center", "middle", True)
    assert direction((150, 40, 170, 60), (100, 100))["in_frame"] is False


def test_latest_memory_is_the_most_recent_place(ls, bus, tmp_path):
    zidane = cv2.resize(cv2.imread(str(FIX / "zidane.jpg")), (bus.shape[1], bus.shape[0]))
    t = feed(ls, bus, BOX, 2, 0)                        # keys at place A
    t = feed(ls, np.zeros_like(bus), None, 3, t)        # walk away
    feed(ls, zidane, (600, 300, 700, 400), 2, t)        # keys at place B
    ls.finish()
    mem = ls.latest_memory(LABEL)
    assert mem["place_status"] in ("likely_new_place", "unverified")
    assert mem["bbox"] == [600, 300, 700, 400]
    assert Path(mem["image_file"]).exists() and len(mem["alternatives"]) == 1
    saved = cv2.imread(mem["image_file"])
    assert np.abs(saved.astype(int) - zidane.astype(int)).mean() < 3   # evidence is place B's frame


def test_guide_maps_the_remembered_spot_into_a_turned_view(ls, bus):
    feed(ls, bus, BOX, 2, 0)
    ls.finish()
    mem = ls.latest_memory(LABEL)
    turned = cv2.warpAffine(bus, np.float32([[1, 0, -150], [0, 1, 0]]), (bus.shape[1], bus.shape[0]))
    ls.det.box = None                                   # detector misses the keys in the live frame
    g = ls.guide(turned, mem["memory_id"])
    assert g["scene_verdict"] == "same_scene"
    assert g["status"] == "remembered_spot_in_view"
    assert abs(g["target_box"][0] - (BOX[0] - 150)) < 5  # "it was here", even without a detection
    ls.det.box = (BOX[0] - 150, BOX[1], BOX[2] - 150, BOX[3])
    assert ls.guide(turned, mem["memory_id"])["status"] == "object_at_remembered_spot"


def test_guide_in_an_unrecognized_place(ls, bus):
    feed(ls, bus, BOX, 2, 0)
    ls.finish()
    mem = ls.latest_memory(LABEL)
    ls.det.box = None
    g = ls.guide(np.full_like(bus, 120), mem["memory_id"])
    assert g["status"] == "remembered_place_not_recognized" and g["direction"] is None


def test_found_deletes_everything_and_pauses_new_memories(ls, bus, tmp_path):
    t = feed(ls, bus, BOX, 2, 0)
    mem = ls.latest_memory(LABEL)
    removed = ls.found(mem["memory_id"])
    assert removed[0]["memory_id"] == mem["memory_id"]
    assert not Path(mem["image_file"]).exists() and not Path(mem["crop_file"]).exists()
    assert ls.latest_memory(LABEL) is None
    import json
    log = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log if e["memory_id"] == mem["memory_id"]] == ["deleted"]
    feed(ls, bus, BOX, 2, t)                            # keys still in view right after "found it"
    assert ls.latest_memory(LABEL) is None              # cooldown: no instant new memory
