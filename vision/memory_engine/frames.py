"""One way to read frames, whether they come from a video file or a live camera.

Every frame carries three clocks:
- video_ms: milliseconds since the start of the source. For a file this is the
  position in the video (from OpenCV); for a live camera it's time since we opened it.
  Only comparable within one source/session.
- wall_ms: real-world time (Unix epoch ms) when we *processed* the frame.
- observed_ms: real-world time when the scene was actually *seen*. For a live camera
  that's wall_ms. For a file it's recorded_at_ms + video_ms if you tell us when the
  clip was recorded, otherwise None (we don't guess).

Tracking logic uses video_ms, so a recorded clip and a live stream behave the same.
Comparing sightings across sessions needs observed_ms.

Live sources (stream URLs and webcams) reconnect: when a read fails or a URL stream stalls
for LIVE_TIMEOUT_MS, the capture is reopened with backoff for up to reconnect_seconds before
the iteration ends. Files still end at their first failed read.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
LIVE_TIMEOUT_MS = 5000  # URL open/read timeout: a stalled stream fails fast instead of hanging ~30 s


@dataclass
class Frame:
    frame_id: int
    video_ms: int
    wall_ms: int
    image: np.ndarray  # BGR, as OpenCV gives it
    observed_ms: int | None = None


def is_file_source(source: str) -> bool:
    """A video suffix, or any existing local file (.mts, .mpg, ...): recorded footage never gets
    the wall clock as observed_ms."""
    if Path(source).suffix.lower() in VIDEO_SUFFIXES:
        return True
    return "://" not in source and not source.isdigit() and Path(source).is_file()


def is_url_source(source: str) -> bool:
    """http://, rtsp://, tcp:// ... (a URL ending in a video suffix is still treated as a file)."""
    return "://" in source and not is_file_source(source)


def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if is_url_source(source):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, LIVE_TIMEOUT_MS,
                                                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, LIVE_TIMEOUT_MS])
    return cv2.VideoCapture(source)


class FrameSource:
    """Iterate over Frames from a video file, webcam index ("0") or stream URL.

    stride=N keeps every Nth frame (useful to test lower processing rates on
    recorded clips: a 30 fps clip with stride=6 behaves like a 5 fps camera).

    max_fps=F yields at most F frames per second of *video time*, whatever the camera's
    capture rate. This is the inference-rate limit: the detector only ever sees these frames.
      - Live sources: frames keep being grabbed (cheap, no decoding) so the buffer never
        goes stale, and only the newest frame is decoded when one is due.
      - Files with realtime=True: playback is paced to the video clock, so the loop waits
        between frames like it would on a live camera instead of racing through the clip.

    reconnect_seconds: how long a live source (URL or webcam) may go without a frame
    before giving up; 0 = end at the first failure. Each frame's clocks are still taken
    when it is decoded, so a reconnect shows up as a gap in video_ms, never as invented frames.
    opener(source) -> capture can be injected (tests).
    """

    def __init__(self, source: str, stride: int = 1, recorded_at_ms: int | None = None,
                 max_fps: float | None = None, realtime: bool = False, reconnect_seconds: float = 60.0,
                 opener=None, log=None):
        self.source = source
        self.stride = max(1, stride)
        self.is_file = is_file_source(source)
        self.kind = "file" if self.is_file else "live"
        self.recorded_at_ms = recorded_at_ms
        self.max_fps = max_fps
        self.realtime = realtime
        self.reconnect_seconds = reconnect_seconds if source.isdigit() or is_url_source(source) else 0
        self.reconnects = 0
        self._open = opener or open_capture
        self._log = log or (lambda msg: print(msg, flush=True))
        self._sleep, self._clock = time.sleep, time.monotonic  # replaceable in tests
        self._outage_since = None
        self._backoff = 0.5
        if self.is_file and not Path(source).exists():
            raise FileNotFoundError(source)
        self._cap = self._open(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or None

    def __iter__(self):
        start = time.monotonic()
        interval_ms = 1000 / self.max_fps if self.max_fps else 0
        next_due_ms = 0.0
        read_index = 0
        frame_id = 0
        try:
            while True:
                if not self._cap.grab():
                    if self._reopen():
                        continue
                    return  # end of file, or the stream stayed down
                read_index += 1
                if (read_index - 1) % self.stride:
                    continue
                if self.is_file:
                    video_ms = int(round(self._cap.get(cv2.CAP_PROP_POS_MSEC)))
                else:
                    video_ms = int((time.monotonic() - start) * 1000)
                if video_ms + 0.5 < next_due_ms:
                    continue  # not due yet: skip without decoding
                next_due_ms = max(next_due_ms + interval_ms, video_ms) if interval_ms else 0
                ok, image = self._cap.retrieve()
                if not ok:
                    if self._reopen():
                        continue
                    return
                if self._outage_since is not None:
                    self._log(f"[frames] {self.source}: frames again after "
                              f"{self._clock() - self._outage_since:.1f}s ({self.reconnects} reconnects so far)")
                    self._outage_since, self._backoff = None, 0.5
                if self.is_file and self.realtime:
                    wait_s = start + video_ms / 1000 - time.monotonic()
                    if wait_s > 0:
                        time.sleep(wait_s)
                wall_ms = int(time.time() * 1000)
                if self.is_file:
                    observed_ms = self.recorded_at_ms + video_ms if self.recorded_at_ms is not None else None
                else:
                    observed_ms = wall_ms
                yield Frame(frame_id, video_ms, wall_ms, image, observed_ms)
                frame_id += 1
        finally:
            self._cap.release()

    def _reopen(self) -> bool:
        """Live source without a frame: wait (backoff 0.5 s .. 8 s) and reopen, until reconnect_seconds."""
        if not self.reconnect_seconds:
            return False
        now = self._clock()
        if self._outage_since is None:
            self._outage_since = now
            self._log(f"[frames] {self.source}: no frame, reconnecting for up to {self.reconnect_seconds:g}s")
        waited = now - self._outage_since
        if waited >= self.reconnect_seconds:
            self._log(f"[frames] {self.source}: gave up after {waited:.1f}s without a frame")
            return False
        self._sleep(min(self._backoff, self.reconnect_seconds - waited))
        self._backoff = min(self._backoff * 2, 8.0)
        self._cap.release()
        self._cap = self._open(self.source)
        self.reconnects += 1
        self._log(f"[frames] {self.source}: reconnect {self.reconnects} "
                  f"({'opened' if self._cap.isOpened() else 'not reachable'})")
        return True
