"""Live demo loop (lastscene_bridge/live_view.py): commands, guidance labels, one inference per frame.

Real LastSeen + dry-run bridge with a scripted detector; no GPU, no window (cv2 window calls
are faked where the --show path is tested).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lastscene_bridge import live_view
from lastscene_bridge.bridge import LastSceneBridge
from lastscene_bridge.live_view import (FrameDetections, LiveSession, guidance_cue, guidance_overlay,
                                        key_command, parse_command)
from memory_engine.api import LastSeen
from memory_engine.detector import Detection
from memory_engine.frames import Frame

FIX = Path(__file__).parent / "fixtures"
BOX = (223, 406, 344, 860)          # keys box on bus.jpg (same as test_bridge.py)
T0 = 1_792_220_000_000


class Scripted:
    def __init__(self):
        self.box = None
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection("bunch of keys", 0.148, self.box)] if self.box else []


class SpyBridge:
    """The real dry-run bridge, recording found() calls."""

    def __init__(self, bridge):
        self.inner = bridge
        self.found_calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def found(self, label):
        self.found_calls.append(label)
        return self.inner.found(label)


@pytest.fixture
def bus():
    return cv2.imread(str(FIX / "bus.jpg"))


@pytest.fixture
def setup(tmp_path):
    det = Scripted()
    ls = LastSeen(out_dir=str(tmp_path / "mem"), detector=FrameDetections(det), search_detector=det,
                  source="test-cam")
    bridge = SpyBridge(LastSceneBridge(ls, outbox_dir=tmp_path / "outbox", preset_name="m960d"))
    log = []
    session = LiveSession(ls, bridge, log=log.append, now_ms=lambda: T0 + 60_000)
    yield session, det, bridge, log
    ls.close()


def frame(i, image):
    t = i * 500
    return Frame(i, t, T0 + t, image.copy(), T0 + t)  # a new array per frame, like a live camera


def turned(bus):
    """Same place, camera turned: the keys box moves 150 px left."""
    return cv2.warpAffine(bus, np.float32([[1, 0, -150], [0, 1, 0]]), (bus.shape[1], bus.shape[0]))


def put_keys_down(session, det, bus, n=6):
    det.box = BOX
    for i in range(n):
        session.step(frame(i, bus))
    det.box = None
    return n


def test_commands_from_text_and_keys():
    assert [parse_command(t) for t in ("where\n", " W ", "where are my keys?", "found", "found it!", "f",
                                        "quit", "q", "exit", "hello", "")] == \
        ["where", "where", "where", "found", "found", "found", "quit", "quit", "quit", None, None]
    assert [key_command(k) for k in (ord("w"), ord("f"), ord("q"), 27, -1, ord("x"))] == \
        ["where", "found", "quit", "quit", None, None]


def test_where_guides_then_found_clears_once_with_keys(setup, bus):
    session, det, bridge, log = setup
    n = put_keys_down(session, det, bus)
    assert session.ls.latest_memory("bunch of keys") is not None
    assert bridge.stats["observations"] >= 1          # the same loop fed the bridge
    session.submit("where")
    session.step(frame(n, turned(bus)))
    assert session.guiding
    assert any(line.startswith("  Last seen") for line in log)
    assert session.last_guide["status"] == "remembered_spot_in_view"
    assert any("guide: Not detected right now. Memory: last seen here" in line for line in log)

    session.submit("found")
    session.step(frame(n + 1, turned(bus)))
    assert bridge.found_calls == ["keys"]              # exactly once, with the contract label
    assert not session.guiding and session.last_guide is None
    assert session.ls.memories("bunch of keys") == []  # all local memories of keys are gone
    assert "local memor" in session.notice and "dry run" in session.notice


def test_where_without_a_memory_does_not_guide(setup, bus):
    session, det, bridge, log = setup
    session.submit("where")
    session.step(frame(0, bus))
    assert not session.guiding and session.last_guide is None
    assert any("No memory of keys yet" in line for line in log)


def test_quit_stops_the_loop_before_the_next_frame(setup, bus):
    session, det, bridge, log = setup
    seen = []

    def frames():
        for i in range(10):
            seen.append(i)
            if i == 2:
                session.submit("quit")
            yield frame(i, bus)

    session.run(frames())
    assert seen == [0, 1, 2] and session.stopped
    assert det.calls == 2                              # the frame after "quit" was not processed


def test_found_errors_are_reported_and_the_loop_keeps_running(setup, bus):
    session, det, bridge, log = setup

    def broken(label):
        raise ConnectionError("offline")

    bridge.found = broken
    session.submit("found")
    session.step(frame(0, bus))
    assert any("found failed: ConnectionError" in line for line in log)
    assert not session.stopped


def test_guidance_reuses_the_frames_detection(setup, bus):
    session, det, bridge, log = setup
    n = put_keys_down(session, det, bus)
    session.submit("where")
    before = det.calls
    for i in range(3):
        session.step(frame(n + i, turned(bus)))
    assert session.guiding and det.calls - before == 3  # one inference per frame, not two


def test_projected_box_is_never_labelled_visible(setup, bus):
    session, det, bridge, log = setup
    n = put_keys_down(session, det, bus)
    session.submit("where")
    view = turned(bus)
    session.step(frame(n, view))
    g = session.last_guide
    assert g["remembered_box_in_view"] and not g["detection"]["visible"]
    items = guidance_overlay(g, view.shape)
    boxes = [i for i in items if i["kind"] != "arrow"]
    assert [(i["kind"], i["style"], i["label"]) for i in boxes] == [("memory", "dashed", "memory: last seen here")]
    assert "visible" not in guidance_cue(g).lower()

    det.box = (BOX[0] - 150, BOX[1], BOX[2] - 150, BOX[3])  # now the detector really sees the keys
    session.step(frame(n + 1, view))
    g = session.last_guide
    labels = {i["kind"]: i["label"] for i in guidance_overlay(g, view.shape) if i["kind"] != "arrow"}
    assert labels["memory"] == "memory: last seen here"
    assert labels["live"].startswith("visible now")
    assert guidance_cue(g).startswith("KEYS VISIBLE NOW")


def test_cue_and_overlay_follow_the_detection_not_the_status():
    projected_only = {"status": "remembered_spot_out_of_frame", "detection": {"visible": False},
                      "remembered_box_in_view": [900, 10, 1000, 80], "target_box": [900, 10, 1000, 80],
                      "direction": {"horizontal": "right", "vertical": "up", "in_frame": False}}
    items = guidance_overlay(projected_only, (480, 640, 3))
    assert all("visible" not in i.get("label", "") for i in items)
    assert any(i["kind"] == "arrow" and i["to"][0] <= 620 for i in items)  # arrow clipped to the frame
    assert guidance_cue(projected_only) == "Memory: last-seen spot is out of view, turn up and right"
    # a status that would imply a detection, without one, never yields "visible"
    assert "visible" not in guidance_cue({"status": "object_detected", "detection": {"visible": False}}).lower()


def test_console_lines_become_commands(setup):
    session, *_ , log = setup
    t = session.start_console(lines=iter(["where\n", "bogus\n", "found\n", "quit\n"]))
    t.join(timeout=5)
    queued = []
    while not session.commands.empty():
        queued.append(session.commands.get_nowait())
    assert queued == ["where", "found", "quit"]
    assert any("unknown command 'bogus'" in line for line in log)


def test_window_keys_drive_the_same_commands(setup, bus, monkeypatch):
    session, det, bridge, log = setup
    keys = iter([-1] * 6 + [ord("w"), -1, ord("f"), ord("q")])
    shown = []
    monkeypatch.setattr(live_view.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(live_view.cv2, "waitKey", lambda ms: next(keys, -1))
    monkeypatch.setattr(live_view.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(live_view.cv2, "destroyAllWindows", lambda: None)
    session.show = True
    det.box = BOX
    session.run(frame(i, bus if i < 6 else turned(bus)) for i in range(20))
    session.close()
    assert bridge.found_calls == ["keys"] and session.stopped
    assert any(line.startswith("> where are my keys") for line in log)
    assert len(shown) == 10 and shown[0] == bus.shape  # rendered each processed frame, then quit


def test_found_typed_after_the_last_frame_is_still_handled(setup, bus):
    session, det, bridge, log = setup
    n = put_keys_down(session, det, bus)

    def frames():
        yield frame(n, bus)
        session.submit("found")  # e.g. typed while the stream was down, then it gave up

    session.run(frames())
    assert bridge.found_calls == ["keys"] and session.ls.memories("bunch of keys") == []


def test_console_without_a_real_stdin_is_skipped(setup, monkeypatch):
    import io

    session, *_ = setup
    monkeypatch.setattr(live_view.sys, "stdin", io.StringIO(""))  # no fileno(), like some IDE consoles
    assert session.start_console() is None
    monkeypatch.setattr(live_view.sys, "stdin", None)  # pythonw
    assert session.start_console() is None
