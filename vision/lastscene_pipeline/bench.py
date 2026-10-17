"""Measure the detector where it will run (e.g. on the Raspberry Pi): load time, per-frame latency, peak memory.

    python -m lastscene_pipeline.bench --model lastscene-data/models/current [--image frame.jpg] [--runs 20]
                                       [--threads 4] [--runtime onnx|onnx-fp32]

Uses the same OnnxYoloDetector as detect (letterbox + inference + decode). Without --image it uses a
synthetic 1280x720 frame, which times the model but says nothing about accuracy. Prints one JSON line.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from lastscene_pipeline import EXIT_CONFIG, EXIT_OK
from lastscene_pipeline.common import PipelineError, resolve_model_dir, run_main


def peak_rss_mb() -> float | None:
    try:
        import resource  # Linux / macOS

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except ImportError:  # Windows
        try:
            import psutil

            return round(psutil.Process().memory_info().peak_wset / 1e6, 1)
        except ImportError:
            return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--runtime", default="onnx", choices=["onnx", "onnx-fp32"])
    p.add_argument("--low-memory", action="store_true", help="ONNX Runtime without its memory arena (Pi)")
    args = p.parse_args()
    import cv2

    from lastscene_pipeline.runtime import OnnxYoloDetector

    model_dir = resolve_model_dir(args.model)
    name = "model-int8.onnx" if args.runtime == "onnx" else "model-fp32.onnx"
    if model_dir is None or not (model_dir / name).exists():
        raise PipelineError(EXIT_CONFIG, f"{args.model}: no {name} (run train/export on the laptop and copy the folder)")
    image = cv2.imread(str(args.image)) if args.image else np.full((720, 1280, 3), 127, np.uint8)
    if image is None:
        raise PipelineError(EXIT_CONFIG, f"{args.image}: not an image")
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    det = OnnxYoloDetector(model_dir / name, confidence=0.25, threads=args.threads, low_memory=args.low_memory)
    load_s = time.perf_counter() - t0
    det.detect(image)  # warm-up
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        found = det.detect(image)
        times.append(time.perf_counter() - t)
    times.sort()
    print(json.dumps({
        "model": f"{model_dir.name}/{name}", "bytes": (model_dir / name).stat().st_size, "imgsz": det.size,
        "threads": args.threads, "runs": args.runs, "load_s": round(load_s, 2),
        "median_s": round(statistics.median(times), 3), "p95_s": round(times[int(0.95 * (len(times) - 1))], 3),
        "max_fps": round(1 / statistics.median(times), 2), "peak_rss_mb": peak_rss_mb(),
        "rss_before_load_mb": rss_before, "detections_last_run": len(found),
        "machine": platform.machine(), "python": platform.python_version(), "system": platform.platform(),
    }))
    return EXIT_OK


if __name__ == "__main__":
    run_main(main)
