r"""Benchmark detector configurations on the real phone clips.

    python scripts\bench_detector.py                 # all configs
    python scripts\bench_detector.py --only s640h x640h

Ground truth is coarse and was labeled by eye from the frames (see LABELS):
for each clip, time windows where the keys are close up / far away / out of view.
Transition frames are left unlabeled and not scored.

Per config it reports:
  recall@0.05 / @0.15   share of labeled frames where a box was found at that confidence
  false alarms           frames with a detection while the keys were out of view
  dropped_big            boxes the >50%-of-frame filter removed (by category)
  latency                median ms per detect() call, run back-to-back (peak load)
  heavy%                 cascade only: share of frames that ran the big model
  gpu                    mean util / mean power / max temp while that config ran (peak load)
  memories               key memories created per clip when the full memory engine runs at 2 fps

Small data: 3 clips, 1 room, 2 key sets. Treat differences of a few frames as noise.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.detector import CascadeDetector, YoloWorldDetector, drop_oversized  # noqa: E402
from memory_engine.frames import FrameSource  # noqa: E402
from memory_engine.gpu_monitor import GpuMonitor  # noqa: E402
from memory_engine.memory import MemoryConfig, MemoryMaker  # noqa: E402

CLIPS = "data/clips"
PROMPT = "bunch of keys"
# (clip, start_s, end_s, category). Labeled by eye; gaps between windows are unscored.
LABELS = [
    ("desk_a.mov", 0.0, 1.7, "near"),
    ("desk_a.mov", 1.9, 3.5, "absent"),
    ("desk_a.mov", 3.8, 9.9, "far"),
    ("desk_b.mov", 0.0, 9.9, "table"),
    ("IMG_7687.MP4", 0.0, 9.9, "carpet"),
]
CATEGORIES = ["near", "far", "table", "carpet"]

CONFIGS = {  # name: (model, imgsz, half)
    "s640": ("yolov8s-worldv2.pt", 640, False),
    "s640h": ("yolov8s-worldv2.pt", 640, True),
    "s480h": ("yolov8s-worldv2.pt", 480, True),
    "s320h": ("yolov8s-worldv2.pt", 320, True),
    "m640h": ("yolov8m-worldv2.pt", 640, True),
    "m480h": ("yolov8m-worldv2.pt", 480, True),
    "x640": ("yolov8x-worldv2.pt", 640, False),
    "x640h": ("yolov8x-worldv2.pt", 640, True),
    "x480h": ("yolov8x-worldv2.pt", 480, True),
    "cascade": ("s640h>x640h", None, None),
    "cascade_p4": ("s640h>x640h every 4th", None, None),
}


def category(clip, t):
    for name, start, end, cat in LABELS:
        if name == clip and start <= t <= end:
            return cat
    return None


def make(name, conf=0.01):
    if name.startswith("cascade"):
        light, heavy = make("s640h", conf), make("x640h", conf)
        return CascadeDetector(light, heavy, accept_conf=0.25, escalate_conf=0.03,
                               periodic_every=4 if name == "cascade_p4" else 0)
    model, imgsz, half = CONFIGS[name]
    # filter disabled here so the benchmark can count what it would drop
    return YoloWorldDetector([PROMPT], model=model, conf=conf, imgsz=imgsz, half=half, max_area_frac=1.0)


def detect_filtered(det, image):
    raw = det.detect(image)
    kept = drop_oversized(raw, image.shape, 0.5)
    if isinstance(det, CascadeDetector):  # cascade members already got max_area_frac=1.0
        pass
    return raw, kept


def memories_at_2fps(det, clip):
    maker = MemoryMaker(MemoryConfig(min_conf=0.05, min_hits=2))
    created = 0
    for f in FrameSource(f"{CLIPS}/{clip}", max_fps=2):
        _, kept = detect_filtered(det, f.image)
        created += sum(e["event"] == "created" for e in maker.process(f.frame_id, f.video_ms, f.wall_ms, kept))
    return created


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="config names to run")
    p.add_argument("--cooldown", type=float, default=8.0, help="seconds to idle between configs")
    args = p.parse_args()
    names = args.only or list(CONFIGS)

    frames = []  # (clip, t, category, image), every 3rd frame, decoded once
    for clip in sorted({c for c, *_ in LABELS}):
        for f in FrameSource(f"{CLIPS}/{clip}", stride=3):
            cat = category(clip, f.video_ms / 1000)
            if cat:
                frames.append((clip, f.video_ms / 1000, cat, f.image))
    counts = {c: sum(1 for fr in frames if fr[2] == c) for c in CATEGORIES + ["absent"]}
    print(f"scored frames: {counts}\n")

    header = (f"{'config':<11} {'near':>9} {'far':>9} {'table':>9} {'carpet':>9}  {'false':>5} {'dropped_big':>12} "
              f"{'ms':>6} {'heavy%':>6}  {'util%':>5} {'W':>5} {'maxC':>5}  memories(a/b/carpet)")
    print("recall shown as @0.05/@0.15 in %\n" + header)
    rows = []
    for name in names:
        det = make(name)
        for _, _, _, img in frames[:5]:  # warm-up (CUDA init, kernel selection)
            det.detect(img)
        if isinstance(det, CascadeDetector):
            det.frames = det.heavy_calls = 0
        hits = {c: [0, 0] for c in CATEGORIES}
        false_alarms, dropped, times = 0, {}, []
        with GpuMonitor(interval_s=0.5, log_path=f"data/bench/gpu_{name}.csv") as gpu:
            for clip, t, cat, img in frames:
                start = time.perf_counter()
                raw, kept = detect_filtered(det, img)
                times.append((time.perf_counter() - start) * 1000)
                if len(raw) > len(kept):
                    dropped[cat] = dropped.get(cat, 0) + len(raw) - len(kept)
                best = max((d.conf for d in kept), default=0.0)
                if cat == "absent":
                    false_alarms += best >= 0.05
                else:
                    hits[cat][0] += best >= 0.05
                    hits[cat][1] += best >= 0.15
        heavy = f"{100 * det.heavy_fraction:.0f}" if isinstance(det, CascadeDetector) else "-"
        mems = [memories_at_2fps(det, c) for c in ("desk_a.mov", "desk_b.mov", "IMG_7687.MP4")]
        g = gpu.summary()
        rec = {c: f"{100 * hits[c][0] // counts[c]}/{100 * hits[c][1] // counts[c]}" for c in CATEGORIES}
        drop_txt = ",".join(f"{k}:{v}" for k, v in dropped.items()) or "0"
        line = (f"{name:<11} {rec['near']:>9} {rec['far']:>9} {rec['table']:>9} {rec['carpet']:>9}  "
                f"{false_alarms:>5} {drop_txt:>12} {statistics.median(times):>6.1f} {heavy:>6}  "
                f"{g.get('util_pct_mean', '-'):>5} {g.get('power_w_mean', '-'):>5} {g.get('temp_c_max', '-'):>5}  "
                f"{'/'.join(map(str, mems))}")
        print(line, flush=True)
        rows.append(line)
        del det
        time.sleep(args.cooldown)
    Path("data/bench").mkdir(parents=True, exist_ok=True)
    Path("data/bench/detector_results.txt").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    Path("data/bench").mkdir(parents=True, exist_ok=True)
    main()
