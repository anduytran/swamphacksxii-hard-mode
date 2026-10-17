import cv2
import numpy as np
import pytest

from memory_engine.frames import FrameSource


@pytest.fixture
def tiny_video(tmp_path):
    """A 10-frame, 10 fps test clip (solid gray frames; only timing matters here)."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return str(path)


def test_file_timestamps_follow_video_position(tiny_video):
    frames = list(FrameSource(tiny_video))
    assert len(frames) == 10
    assert [f.frame_id for f in frames] == list(range(10))
    assert [f.video_ms for f in frames] == [i * 100 for i in range(10)]


def test_stride_keeps_every_nth_frame_with_true_times(tiny_video):
    frames = list(FrameSource(tiny_video, stride=3))
    assert [f.video_ms for f in frames] == [0, 300, 600, 900]
    assert [f.frame_id for f in frames] == [0, 1, 2, 3]


def test_max_fps_limits_frames_by_video_time(tiny_video):
    # 10 fps clip, limited to 2 fps -> one frame every 500 ms of video
    frames = list(FrameSource(tiny_video, max_fps=2))
    assert [f.video_ms for f in frames] == [0, 500]


def test_realtime_pacing_waits_between_frames(tiny_video):
    import time

    start = time.monotonic()
    frames = list(FrameSource(tiny_video, max_fps=2, realtime=True))
    elapsed = time.monotonic() - start
    assert [f.video_ms for f in frames] == [0, 500]
    assert elapsed >= 0.45  # waited for the 500 ms frame instead of racing through


def test_missing_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameSource(str(tmp_path / "nope.mp4"))


# --- live reconnects ------------------------------------------------------------------------
class FakeStream:
    """A capture that delivers `frames` frames, then fails (stream dropped / stalled)."""

    def __init__(self, frames, opened=True):
        self.left, self.opened, self.released = frames, opened, False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 0

    def grab(self):
        if not self.opened or self.left <= 0:
            return False
        self.left -= 1
        return True

    def retrieve(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True


def live(captures, reconnect_seconds=60.0):
    """FrameSource on a fake URL whose opener hands out `captures` in order; time is simulated."""
    pending = list(captures)
    logs, clock = [], [0.0]
    src = FrameSource("http://cam.local:81/stream", reconnect_seconds=reconnect_seconds,
                      opener=lambda s: pending.pop(0) if pending else FakeStream(0, opened=False), log=logs.append)
    src._clock = lambda: clock[0]
    src._sleep = lambda s: clock.__setitem__(0, clock[0] + s)
    return src, logs, clock


def test_live_url_reconnects_after_a_drop_and_keeps_going():
    # 2 frames, drop, camera unreachable once, then 3 more frames, then gone for good
    src, logs, clock = live([FakeStream(2), FakeStream(0, opened=False), FakeStream(3)], reconnect_seconds=10)
    frames = list(src)
    assert len(frames) == 5 and [f.frame_id for f in frames] == list(range(5))
    assert all(f.observed_ms == f.wall_ms for f in frames)        # live: decode-time host clock
    assert src.reconnects >= 2
    assert any("frames again" in m for m in logs) and any("gave up" in m for m in logs)
    assert 10 <= clock[0] <= 18.5                                  # gave up after ~reconnect_seconds


def test_backoff_doubles_up_to_8s():
    src, logs, clock = live([FakeStream(1)], reconnect_seconds=30)
    waits = []
    src._sleep = lambda s: (waits.append(s), clock.__setitem__(0, clock[0] + s))
    assert len(list(src)) == 1
    assert waits[:6] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0] and abs(sum(waits) - 30) < 1e-6


def test_reconnect_seconds_zero_ends_at_the_first_failure():
    src, logs, clock = live([FakeStream(2), FakeStream(5)], reconnect_seconds=0)
    assert len(list(src)) == 2 and src.reconnects == 0 and clock[0] == 0


def test_files_never_reconnect(tiny_video):
    src = FrameSource(tiny_video, reconnect_seconds=60)
    assert src.reconnect_seconds == 0 and len(list(src)) == 10 and src.reconnects == 0


def test_webcam_index_may_retry():
    src = FrameSource("0", opener=lambda s: FakeStream(1), log=lambda m: None)
    assert src.reconnect_seconds == 60


def test_local_file_with_another_suffix_is_still_a_file(tiny_video, tmp_path):
    from pathlib import Path

    mts = Path(tiny_video).rename(tmp_path / "camcorder.mts")
    src = FrameSource(str(mts), reconnect_seconds=60)
    frames = list(src)
    assert src.is_file and src.reconnect_seconds == 0 and len(frames) == 10
    assert all(f.observed_ms is None for f in frames)  # recorded footage never gets the wall clock
