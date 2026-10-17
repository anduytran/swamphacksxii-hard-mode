"""Real recording time of a video file, read from its QuickTime/MP4 metadata (moov/mvhd).

This is how captured_at stays honest for recorded clips: we use the time the camera wrote
into the file, never the time we happened to process it. Returns None when the file has
no creation time (e.g. re-encoded clips); those clips' memories are then not uploaded.

Accuracy: Apple doesn't document whether this marks the start or the end of recording,
so treat it as accurate to within the clip's duration.

Clips written by scripts/record_stream.py have no such metadata (OpenCV's MP4 writer leaves
it at 0). For those, record_stream writes a sidecar "<clip>.capture.json" next to the clip:

    {"first_frame_unix_ms": 1792222824123, "first_frame_utc": "2026-10-17T07:20:24.123Z",
     "clock": "recording host wall clock when the first frame arrived", "fps": 14.9, "frames": 447, ...}

capture_time_ms() uses the file metadata when present, else that sidecar, else None.
It never falls back to the current time.
"""

import datetime
import json
import struct
from pathlib import Path

_EPOCH_1904 = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)


def _atoms(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        yield kind, pos + header, pos + size
        pos += size


def recording_time_ms(path) -> int | None:
    """Unix epoch ms of the file's recorded creation time, or None if absent."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            for kind, body, stop in _atoms(f, 0, end):
                if kind == b"moov":
                    for kind2, body2, _ in _atoms(f, body, stop):
                        if kind2 == b"mvhd":
                            f.seek(body2)
                            version = f.read(1)[0]
                            f.read(3)
                            secs = struct.unpack(">Q" if version == 1 else ">I", f.read(8 if version == 1 else 4))[0]
                            if secs == 0:
                                return None
                            return int((_EPOCH_1904 + datetime.timedelta(seconds=secs)).timestamp() * 1000)
    except (OSError, struct.error, IndexError):
        return None
    return None


def sidecar_path(path) -> Path:
    """clip.mp4 -> clip.capture.json (written by scripts/record_stream.py)."""
    p = Path(path)
    return p.with_name(p.stem + ".capture.json")


def sidecar_time_ms(path) -> int | None:
    """Unix epoch ms of the clip's first frame from its .capture.json sidecar, or None."""
    try:
        data = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
        ms = data.get("first_frame_unix_ms")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(ms, bool) or not isinstance(ms, int) or ms <= 0:
        return None
    return ms


def capture_time_ms(path) -> tuple[int | None, str]:
    """(recording start ms, basis): "metadata" wins over "sidecar"; (None, "unknown") otherwise."""
    ms = recording_time_ms(path)
    if ms is not None:
        return ms, "metadata"
    ms = sidecar_time_ms(path)
    if ms is not None:
        return ms, "sidecar"
    return None, "unknown"
