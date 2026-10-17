r"""A/B test: memory creation with scene matching on vs. off, on sessions built from real frames.

    python scripts\ab_scene.py

Each session strings together segments of our phone clips (all resized to 360x640,
sampled at 2 fps like the default preset), so the memory engine sees what a wearable would:
keys, looking away, keys again. Detections are computed once per frame and shared by
both modes, so any difference comes from the "same place?" decision alone.

S3 is deliberately constructed: the table frames are shifted so the keys sit at the same
pixels as on the chair. It's real imagery, edited to force the coincidence that the
image-position rule can't handle; it is labeled as such in the output.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.detector import YoloWorldDetector  # noqa: E402
from memory_engine.memory import MemoryConfig, MemoryMaker  # noqa: E402
from memory_engine.run import PRESETS  # noqa: E402
from memory_engine.scene import SceneMatcher  # noqa: E402

SIZE = (360, 640)
SESSIONS = {
    "S1 same spot, looked away, came back from another position": (
        [("IMG_7687.MP4", 0.0, 2.0, 0), ("desk_a.mov", 1.9, 3.5, 0), ("IMG_7687.MP4", 4.5, 7.8, 0)], 1, 2),
    "S2 keys moved to a different place": (
        [("desk_a.mov", 0.0, 1.7, 0), ("desk_a.mov", 1.9, 3.5, 0), ("desk_b.mov", 0.0, 6.7, 0)], 2, 2),
    "S3 different place, same image position (constructed shift)": (
        [("desk_a.mov", 0.0, 1.7, 0), ("desk_a.mov", 1.9, 3.5, 0), ("desk_b.mov", 0.0, 6.7, 48)], 2, 2),
    "S4 chair: close-up, away, far return": (
        [("desk_a.mov", 0.0, 5.3, 0)], 1, 0),
}
# (segments [(clip, start_s, end_s, shift_x_px)], expected memories, index of segment the latest memory must come from)


def frames_for(clip, start, end, shift):
    """Returns [(image shown to the memory engine, image the detector runs on)].

    For shifted segments the detector runs on the *unshifted* frame and its boxes are moved
    by the same amount (see detect_for), so a detector miss on the black-bordered frame
    can't hide what the scene-matching decision does."""
    cap = cv2.VideoCapture(f"data/clips/{clip}")
    out, t = [], start
    while t <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, SIZE)
        shown = cv2.warpAffine(img, np.float32([[1, 0, shift], [0, 1, 0]]), SIZE) if shift else img
        out.append((shown, img, shift))
        t += 0.5
    cap.release()
    return out


def detect_for(det, shown, original, shift):
    from memory_engine.detector import Detection

    if not shift:
        return det.detect(shown)
    return [Detection(d.label, d.conf, (min(d.bbox[0] + shift, SIZE[0] - 1), d.bbox[1],
                                        min(d.bbox[2] + shift, SIZE[0]), d.bbox[3]))
            for d in det.detect(original)]


def run(frames, detections, scene_on):
    p = PRESETS["efficient"]
    maker = MemoryMaker(MemoryConfig(min_conf=p["min_conf"], min_hits=2),
                        scene_matcher=SceneMatcher() if scene_on else None)
    events = []
    for i, (img, dets) in enumerate(zip(frames, detections)):
        events += maker.process(i, i * 500, i * 500, dets, image=img)
    events += maker.finish()
    return maker, events


def main():
    p = PRESETS["efficient"]
    det = YoloWorldDetector(["bunch of keys"], model=p["model"], conf=0.05, imgsz=p["imgsz"], half=p["half"])
    out = Path("data/ab")
    out.mkdir(parents=True, exist_ok=True)
    totals = {True: 0, False: 0}
    for name, (segments, expected, latest_segment) in SESSIONS.items():
        triples, owner = [], []
        for k, seg in enumerate(segments):
            f = frames_for(*seg)
            triples += f
            owner += [k] * len(f)
        frames = [shown for shown, _, _ in triples]
        detections = [detect_for(det, *t) for t in triples]
        print(f"\n{name}  ({len(frames)} frames; keys detected in {sum(bool(d) for d in detections)})")
        for scene_on in (False, True):
            maker, events = run(frames, detections, scene_on)
            mems = sorted(maker.memories.values(), key=lambda m: m.last_seen_video_ms)
            latest = mems[-1] if mems else None
            right_count = len(mems) == expected
            right_latest = latest is not None and owner[latest.frame_id] == latest_segment
            totals[scene_on] += right_count and right_latest
            statuses = [m.place_status for m in mems]
            resights = sum(e["event"] == "resighted" for e in events)
            print(f"  scene {'ON ' if scene_on else 'OFF'}: memories={len(mems)} (expected {expected}) "
                  f"{'OK ' if right_count else 'BAD'} | latest evidence from segment "
                  f"{owner[latest.frame_id] if latest else '-'} (expected {latest_segment}) "
                  f"{'OK ' if right_latest else 'BAD'} | resighted={resights} | place_status={statuses}")
            if latest is not None:
                img = frames[latest.frame_id].copy()
                cv2.rectangle(img, latest.bbox[:2], latest.bbox[2:], (0, 0, 255), 2)
                cv2.imwrite(str(out / f"{name[:2]}_{'on' if scene_on else 'off'}_latest.jpg"), img)
    print(f"\nsessions fully correct: scene OFF {totals[False]}/{len(SESSIONS)}, scene ON {totals[True]}/{len(SESSIONS)}")


if __name__ == "__main__":
    main()
