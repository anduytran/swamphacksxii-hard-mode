r"""Run every LastSeen API call on the bundled sample, so you can see real inputs and outputs.

    python examples\quickstart.py --no-model     # no GPU or model download: detections are scripted
    python examples\quickstart.py                # real detector (downloads the model on first run)

Works on a temporary copy of samples/file_run, so the sample itself is never modified
(found() deletes files). The "live frame" is the sample's own evidence photo, shifted
120 px as if the wearer turned slightly.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_engine.api import LastSeen, resolve_label  # noqa: E402
from memory_engine.detector import Detection  # noqa: E402


class NoDetector:
    """Stand-in for --no-model: never detects anything (guide() then relies on scene matching)."""

    def detect(self, image):
        return []


def show(title, value):
    print(f"\n=== {title}\n{json.dumps(value, indent=2, default=str)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-model", action="store_true", help="skip the detector (no GPU/model needed)")
    p.add_argument("--preset", default="m960d", help="m960d suits this 1080p sample; the project default is 'efficient'")
    args = p.parse_args()

    work = Path(tempfile.mkdtemp(prefix="lastseen_quickstart_"))
    shutil.copytree(ROOT / "samples" / "file_run", work, dirs_exist_ok=True)
    det = NoDetector() if args.no_model else None
    ls = LastSeen(out_dir=str(work), preset=args.preset, detector=det, search_detector=det)

    # 1. "Where are my keys?"
    mem = ls.latest_memory(resolve_label("my keys"))
    show("latest_memory(resolve_label('my keys'))", mem)

    # 2. Live frame while searching: the same view, turned a little
    frame = cv2.imread(mem["image_file"])
    h, w = frame.shape[:2]
    turned = cv2.warpAffine(frame, np.float32([[1, 0, -120], [0, 1, 0]]), (w, h))
    show("guide(turned_frame, memory_id)", ls.guide(turned, mem["memory_id"]))
    show("locate(turned_frame, label)", ls.locate(turned, mem["label"]))

    # 3. "Found it": clears every memory of that object
    show("found(label=...)", ls.found(label=mem["label"]))
    show("latest_memory after found", ls.latest_memory(mem["label"]))

    # 4. Memory creation from frames (what the camera loop does, 2x per second)
    start_ms = 10_000  # after found()'s 5 s pause on this label
    events = []
    for i in range(6):
        scripted = [Detection(mem["label"], 0.3, tuple(mem["bbox"]))] if args.no_model else None
        events += ls.process_frame(frame, start_ms + i * 500, observed_ms=int(time.time() * 1000), detections=scripted)
    events += ls.finish()
    show("process_frame events (3 s of the same view)", [{k: e[k] for k in ("event", "memory_id", "confidence",
                                                                        "video_ms", "place_status")} for e in events])
    show("latest_memory after process_frame", ls.latest_memory(mem["label"]))
    ls.close()
    print(f"\n(worked in {work}; delete it whenever)")


if __name__ == "__main__":
    main()
