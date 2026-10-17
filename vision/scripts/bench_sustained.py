r"""Sustained, paced load test: what the GPU does at the inference rate we'd actually run.

    python scripts\bench_sustained.py --fps 2 --seconds 60

Replays real clip frames on a fixed schedule (it sleeps between inferences), first
idle, then one config at a time, with GPU telemetry. This is the number that matters
for an always-on wearable, not back-to-back peak throughput.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.detector import YoloWorldDetector  # noqa: E402
from memory_engine.frames import FrameSource  # noqa: E402
from memory_engine.gpu_monitor import GpuMonitor  # noqa: E402

CONFIGS = {"s640h": ("yolov8s-worldv2.pt", 640), "m640h": ("yolov8m-worldv2.pt", 640),
           "m480h": ("yolov8m-worldv2.pt", 480), "x640h": ("yolov8x-worldv2.pt", 640)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--only", nargs="*", default=["s640h", "m640h", "x640h"])
    p.add_argument("--cooldown", type=float, default=20.0)
    args = p.parse_args()

    images = [f.image for c in ("desk_a.mov", "desk_b.mov", "IMG_7687.MP4")
              for f in FrameSource(f"data/clips/{c}", stride=6)]
    out = Path("data/bench")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'config':<8} {'inferences':>10} {'ms/inf':>7} {'duty%':>6} {'util% mean':>10} {'W mean':>7} {'W max':>6} {'temp start':>10} {'temp max':>8}")

    with GpuMonitor(interval_s=1.0) as idle:
        time.sleep(min(30, args.seconds))
    s = idle.summary()
    print(f"{'idle':<8} {'-':>10} {'-':>7} {'-':>6} {s.get('util_pct_mean', '-'):>10} {s.get('power_w_mean', '-'):>7} "
          f"{s.get('power_w_max', '-'):>6} {'-':>10} {s.get('temp_c_max', '-'):>8}", flush=True)

    for name in args.only:
        model, imgsz = CONFIGS[name]
        det = YoloWorldDetector(["bunch of keys"], model=model, conf=0.05, imgsz=imgsz, half=True)
        for img in images[:5]:
            det.detect(img)
        times, n = [], 0
        interval = 1 / args.fps
        with GpuMonitor(interval_s=1.0, log_path=str(out / f"sustained_{name}_{args.fps:g}fps.csv")) as gpu:
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                due = start + n * interval
                if due > time.monotonic():
                    time.sleep(due - time.monotonic())
                t = time.perf_counter()
                det.detect(images[n % len(images)])
                times.append((time.perf_counter() - t) * 1000)
                n += 1
        g = gpu.summary()
        temps = [x["temp_c"] for x in gpu.samples if x["temp_c"] is not None]
        ms = statistics.median(times)
        print(f"{name:<8} {n:>10} {ms:>7.1f} {100 * ms / 1000 * args.fps:>6.1f} {g.get('util_pct_mean', '-'):>10} "
              f"{g.get('power_w_mean', '-'):>7} {g.get('power_w_max', '-'):>6} {temps[0] if temps else '-':>10} "
              f"{g.get('temp_c_max', '-'):>8}", flush=True)
        del det
        time.sleep(args.cooldown)


if __name__ == "__main__":
    main()
