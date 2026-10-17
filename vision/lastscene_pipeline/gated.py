"""Frames for an always-on wearable: grab one only when it's time, and only pass it on when it's worth it.

    GatedFrames(grab, every=2.0)    grab() -> BGR image or None, called at most once per `every` seconds

A detector on the Pi Zero 2 W takes about 0.6 s and 2 CPU-seconds per frame, so running it on a scene that
hasn't changed wastes battery, heats the board and slows speech-to-text in the same process. Each grabbed
frame is compared with the last one passed on (64x48 grayscale, blurred; mean absolute difference, 0-255):
  * changed (>= `threshold`, default 4.0; a still scene measured 0.3-1.4 on the wearable's USB camera):
    passed on, and so are the frames for the next `settle` seconds, so an item that was just put down is
    seen at rest (memories need 2 sightings at least 0.5 s apart);
  * otherwise skipped, except once every `heartbeat` seconds so a slow change (light, a still item) isn't
    missed for ever.
While `paused()` is true (someone is talking to the wearable) nothing is grabbed or passed on.
`stats` counts grabbed, passed and skipped frames.
"""

import time

import cv2
import numpy as np

from memory_engine.frames import Frame


def thumbnail(image) -> np.ndarray:
    small = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)


def change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


class GatedFrames:
    def __init__(self, grab, every: float = 2.0, threshold: float = 4.0, settle: float = 6.0, heartbeat: float = 60.0,
                 paused=lambda: False, stop=lambda: False, clock=time.monotonic, sleep=time.sleep, wall=time.time):
        self.grab, self.every, self.threshold, self.settle, self.heartbeat = grab, every, threshold, settle, heartbeat
        self.paused, self.stop, self.clock, self.sleep, self.wall = paused, stop, clock, sleep, wall
        self.stats = {"grabbed": 0, "passed": 0, "skipped": 0, "paused_s": 0.0}

    def __iter__(self):
        start = self.clock()
        last_thumb, last_pass, settle_until, frame_id = None, -1e9, -1e9, 0
        next_at = start
        while not self.stop():
            if self.paused():
                self.sleep(0.1)
                self.stats["paused_s"] += 0.1
                next_at = self.clock() + self.every  # resume a full interval after the conversation
                continue
            wait = next_at - self.clock()
            if wait > 0:
                self.sleep(min(wait, 0.1))
                continue
            next_at = self.clock() + self.every
            image = self.grab()
            if image is None:
                continue
            self.stats["grabbed"] += 1
            now = self.clock()
            thumb = thumbnail(image)
            moved = last_thumb is None or change(thumb, last_thumb) >= self.threshold
            if moved:
                settle_until = now + self.settle
            if not (moved or now < settle_until or now - last_pass >= self.heartbeat):
                self.stats["skipped"] += 1
                continue
            last_thumb, last_pass = thumb, now
            self.stats["passed"] += 1
            wall_ms = int(self.wall() * 1000)
            yield Frame(frame_id, int((now - start) * 1000), wall_ms, image, wall_ms)
            frame_id += 1
