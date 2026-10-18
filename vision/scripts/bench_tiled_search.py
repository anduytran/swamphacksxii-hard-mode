"""Compare standard and tiled YOLO-World inference on the existing key clips.

Run from ``vision/`` with the GPU virtualenv:

    python scripts/bench_tiled_search.py --only m960d --out eval/reports/tiled_search/m960d.json
    python scripts/bench_tiled_search.py --only x960d --out eval/reports/tiled_search/x960d.json
    python scripts/bench_tiled_search.py --only x960d_tiled --out eval/reports/tiled_search/x960d_tiled.json

Each pass decodes the original clips sequentially at 2 Hz. ``x960d_tiled`` uses
the same XL detector for a full-frame pass and four overlapping crops. Frames
are never resized before tiling. The 04-return far window is explicitly
unscored because its key visibility is unconfirmed in the manifest.
"""

import argparse
from datetime import datetime, timezone
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import psutil
import torch

VISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION))

from memory_engine.detector import detect_tiled
from memory_engine.frames import FrameSource
from memory_engine.gpu_monitor import GpuMonitor
from memory_engine.memory import MemoryConfig, MemoryMaker
from memory_engine.presets import make_detector


CLIPS = VISION.parents[1] / "data" / "clips"
PROMPT = "bunch of keys"
CONFIDENCE = 0.05
WINDOWS = {
    "desk_a.mov": [("near", 0.0, 1.7, True), ("absent", 1.9, 3.5, False),
                   ("far", 3.8, 9.9, True)],
    "desk_b.mov": [("clutter", 0.0, 9.9, True)],
    "04_return_and_find.MOV": [("far_unscored", 0.0, 4.3, None)],
    "IMG_7688.MOV": [("no_keys", 0.0, 31.6, False)],
}


def model_path(name: str) -> Path:
    candidates = (VISION / name, VISION.parents[1] / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found; expected one of: {candidates}")


def window_for(clip: str, video_s: float):
    for group, start, end, target in WINDOWS[clip]:
        if start <= video_s <= end:
            return group, target
    return None


def detector_config(name: str):
    preset = "m960d" if name == "m960d" else "x960d"
    weights = "yolov8m-worldv2.pt" if preset == "m960d" else "yolov8x-worldv2.pt"
    return preset, model_path(weights)


def run_pass(name: str, gpu_log: str | None = None) -> dict:
    preset, weights = detector_config(name)
    detector = make_detector(preset, [PROMPT], conf=CONFIDENCE, model=str(weights))
    process = psutil.Process()
    process.cpu_percent(None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies: dict[str, list[float]] = {}
    scores: dict[str, list[float]] = {}
    targets: dict[str, bool | None] = {}
    frame_counts: dict[str, int] = {}
    stream_metadata: dict[str, dict] = {}
    peak_rss = process.memory_info().rss
    memory_maker = MemoryMaker(MemoryConfig(min_conf=CONFIDENCE, min_hits=2,
                                            min_seen_ms=500, max_gap_ms=1500))
    negative_memory_events = []
    total_frames = 0

    with GpuMonitor(interval_s=0.5, log_path=gpu_log) as gpu:
        started = time.perf_counter()
        for clip, windows in WINDOWS.items():
            probe = cv2.VideoCapture(str(CLIPS / clip))
            source_fps = probe.get(cv2.CAP_PROP_FPS)
            source_frames = probe.get(cv2.CAP_PROP_FRAME_COUNT)
            stream_metadata[clip] = {
                "source_fps": source_fps,
                "source_frames": int(source_frames),
                "source_duration_s": round(source_frames / source_fps, 3) if source_fps else None,
                "original_width": int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "original_height": int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            probe.release()
            source = FrameSource(str(CLIPS / clip), max_fps=2, realtime=False)
            for frame in source:
                video_s = frame.video_ms / 1000
                label = window_for(clip, video_s)
                if label is None:
                    continue
                group, target = label
                latencies.setdefault(group, [])
                scores.setdefault(group, [])
                targets[group] = target
                frame_counts[group] = frame_counts.get(group, 0) + 1
                torch.cuda.synchronize()
                tick = time.perf_counter()
                detections = (detect_tiled(detector, frame.image) if name == "x960d_tiled"
                              else detector.detect(frame.image))
                torch.cuda.synchronize()
                latencies[group].append((time.perf_counter() - tick) * 1000)
                key_confidence = max((d.conf for d in detections if d.label == PROMPT), default=0.0)
                scores[group].append(key_confidence)
                peak_rss = max(peak_rss, process.memory_info().rss)
                total_frames += 1

                if clip == "IMG_7688.MOV":
                    events = memory_maker.process(frame.frame_id, frame.video_ms, frame.wall_ms,
                                                  detections, image=frame.image,
                                                  observed_ms=frame.observed_ms)
                    negative_memory_events.extend(e for e in events if e["event"] == "created")
        elapsed = time.perf_counter() - started

    negative_memory_events.extend(e for e in memory_maker.finish() if e["event"] == "created")
    torch.cuda.synchronize()

    by_group = {}
    for group, values in latencies.items():
        count = len(values)
        hit_count = sum(score >= CONFIDENCE for score in scores[group])
        target = targets[group]
        by_group[group] = {
            "frames": count,
            "target_status": "positive" if target is True else "negative" if target is False else "unscored",
            "recall_at_0_05": f"{hit_count}/{count}" if target is True else None,
            "false_positive_frames_at_0_05": f"{hit_count}/{count}" if target is False else None,
            "median_key_score": round(statistics.median(scores[group]), 4),
            "max_key_score": round(max(scores[group]), 4),
            "median_ms": round(statistics.median(values), 1),
            "p95_ms": round(sorted(values)[min(count - 1, int(0.95 * (count - 1)))], 1),
        }

    return {
        "mode": name,
        "weights": str(weights),
        "preset": preset,
        "prompt": PROMPT,
        "distractors": ["laptop keyboard", "laptop"],
        "confidence_floor": CONFIDENCE,
        "tiling": {"grid": 2, "overlap": 0.25, "inferences_per_frame": 5 if name == "x960d_tiled" else 1},
        "sample_rate_hz": 2,
        "frames": total_frames,
        "elapsed_s": round(elapsed, 2),
        "effective_fps": round(total_frames / elapsed, 2),
        "peak_process_rss_mb": round(peak_rss / (1024 * 1024), 1),
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        "cpu_percent_mean": round(process.cpu_percent(None), 1),
        "gpu": gpu.summary(),
        "clips": stream_metadata,
        "groups": by_group,
        "no_keys_false_memories_created": len(negative_memory_events),
        "no_keys_false_memory_event_times_s": [round(e["video_ms"] / 1000, 2) for e in negative_memory_events],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VISION / "eval" / "reports" / "tiled_search_comparison.json")
    parser.add_argument("--only", nargs="+", choices=("m960d", "x960d", "x960d_tiled"))
    args = parser.parse_args()
    missing = [name for name in WINDOWS if not (CLIPS / name).is_file()]
    if missing:
        raise FileNotFoundError(f"recordings missing under {CLIPS}: {missing}")
    modes = args.only or ["m960d", "x960d", "x960d_tiled"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "Sequential original clips at 2 Hz; same windows and conf=0.05 for each mode; no resize before detection.",
        "limitations": [
            "desk_a and desk_b have only 360x640 source frames; tiling enlarges a crop but adds no captured detail.",
            "04_return_and_find far window is unscored and excluded from recall.",
            "IMG_7688.MOV is the labeled no-keys negative recording; one false memory matters even if frame-level false-positive rate is low.",
            "Phone-recording performance does not measure the live webcam stream or its JPEG quality and focus.",
        ],
        "runs": [],
    }
    for mode in modes:
        result = run_pass(mode, str(args.out.with_name(f"gpu_{mode}.csv")))
        report["runs"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written: {args.out}")


if __name__ == "__main__":
    main()
