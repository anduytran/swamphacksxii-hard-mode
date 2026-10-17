"""record_stream.py's <clip>.capture.json sidecar and its precedence in capture_time.py."""

import json
import struct
import time

import numpy as np

from lastscene_bridge.capture_time import capture_time_ms, sidecar_path
from scripts import record_stream

FIRST = 1_792_222_824_123  # a first-frame time in ms


def mp4_with_mvhd(path, secs):
    """Minimal file with a moov/mvhd creation time (seconds since 1904), like a phone clip."""
    mvhd = b"\x00\x00\x00\x00" + struct.pack(">I", secs) + b"\x00" * 92  # version 0, flags, creation time
    moov = struct.pack(">I4s", 16 + len(mvhd), b"moov") + struct.pack(">I4s", 8 + len(mvhd), b"mvhd") + mvhd
    path.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  \x00\x00\x00\x00" + moov)
    return path


def test_sidecar_is_used_when_metadata_is_absent(tmp_path):
    clip = tmp_path / "01_placement.mp4"
    clip.write_bytes(b"\0" * 64)  # no moov/mvhd: like OpenCV's mp4v output
    assert capture_time_ms(clip) == (None, "unknown")  # never "now"
    sidecar_path(clip).write_text(json.dumps({"first_frame_unix_ms": FIRST}), encoding="utf-8")
    assert sidecar_path(clip).name == "01_placement.capture.json"
    assert capture_time_ms(clip) == (FIRST, "sidecar")


def test_file_metadata_wins_over_the_sidecar(tmp_path):
    secs = 3_875_000_000
    clip = mp4_with_mvhd(tmp_path / "phone.mov", secs)
    sidecar_path(clip).write_text(json.dumps({"first_frame_unix_ms": FIRST}), encoding="utf-8")
    assert capture_time_ms(clip) == ((secs - 2_082_844_800) * 1000, "metadata")


def test_broken_or_implausible_sidecars_are_ignored(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    for content in ("not json", "[]", json.dumps({"first_frame_unix_ms": "1792222824123"}),
                    json.dumps({"first_frame_unix_ms": 0}), json.dumps({"first_frame_unix_ms": True}),
                    json.dumps({"first_frame_utc": "2026-10-17T07:20:24Z"})):
        sidecar_path(clip).write_text(content, encoding="utf-8")
        assert capture_time_ms(clip) == (None, "unknown"), content


class FakeCamera:
    """A webcam returning 64x48 frames at ~20 fps."""

    def __init__(self, *_):
        self.n = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 0

    def read(self):
        time.sleep(0.05)
        self.n += 1
        return True, np.full((48, 64, 3), self.n % 255, np.uint8)

    def release(self):
        pass


def test_live_recording_writes_first_frame_time(tmp_path, monkeypatch):
    monkeypatch.setattr(record_stream.cv2, "VideoCapture", FakeCamera)
    before = int(time.time() * 1000)
    info = record_stream.record("0", str(tmp_path), "cam", seconds=0.6)
    after = int(time.time() * 1000)
    side = json.loads((tmp_path / "cam.capture.json").read_text(encoding="utf-8"))
    assert info["capture_sidecar"] == "cam.capture.json"
    assert before <= side["first_frame_unix_ms"] <= before + 500 < after  # the FIRST frame, not the end
    assert side["first_frame_utc"].endswith("Z") and side["frames"] == info["frames"]
    assert abs(side["fps"] - info["measured_fps"]) < 0.01 and side["timing_error_ms"] >= 0
    assert capture_time_ms(tmp_path / "cam.mp4") == (side["first_frame_unix_ms"], "sidecar")


def test_replaying_a_file_gets_no_sidecar(tmp_path):
    import cv2

    src = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    info = record_stream.record(str(src), str(tmp_path / "out"), "replay", seconds=2)
    assert info["capture_sidecar"] is None
    assert not (tmp_path / "out" / "replay.capture.json").exists()  # "now" is not when it was filmed


def test_replay_removes_a_stale_sidecar_with_the_same_name(tmp_path):
    import cv2

    src = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip.capture.json").write_text(json.dumps({"first_frame_unix_ms": FIRST}), encoding="utf-8")
    record_stream.record(str(src), str(out), "clip", seconds=2)
    assert capture_time_ms(out / "clip.mp4") == (None, "unknown")  # not dated by the old live recording
