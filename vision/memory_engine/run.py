r"""Run the memory engine on a video file, webcam or stream.

    python -m memory_engine.run --source data\clips\desk_b.mov
    python -m memory_engine.run --source 0 --show                       (webcam)
    python -m memory_engine.run --source http://<esp-ip>:81/stream      (ESP32 stream, once it exists)
    python -m memory_engine.run --source data\clips\desk_b.mov --preset xl --gpu-log data\gpu.csv

Detection presets are defined in memory_engine/presets.py (efficient = default; light, xl,
and the distractor/high-resolution candidates m480d, m640d, m960d, x960d).
Any explicit --model/--imgsz/--inference-fps/--min-conf/--distractors overrides the preset.

Inference is rate-limited (--inference-fps) whatever the camera's frame rate. Video files
play back in real time by default, so the loop waits between inferences like it would
on a live camera; --fast processes a file as quickly as possible (for offline evaluation).
"""

import argparse
import time
import uuid
from datetime import datetime

import cv2

from memory_engine.frames import FrameSource
from memory_engine.gpu_monitor import GpuMonitor
from memory_engine.memory import MemoryConfig, MemoryMaker
from memory_engine.presets import DEFAULT_PRESET, PRESETS, make_detector  # noqa: F401 (PRESETS re-exported)
from memory_engine.store import MemoryStore


def draw(image, detections, maker):
    shown = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(shown, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    for m in maker.memories.values():
        x1, y1, x2, y2 = m.bbox
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(shown, f"memories: {len(maker.memories)}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return shown


def memory_config(args, inference_fps) -> MemoryConfig:
    """Scale the time thresholds to the inference rate.

    At 2 inferences/s there are only 2 looks per second, so requiring 3 hits would take
    1.5 s, and a 1.5 s gap tolerance would end a track after a couple of misses. Explicit
    flags always win.
    """
    cfg = MemoryConfig(min_conf=args.min_conf)
    if args.min_seen_ms is not None:
        cfg.min_seen_ms = args.min_seen_ms
    interval_ms = 1000 / inference_fps if inference_fps else 0
    cfg.min_hits = args.min_hits if args.min_hits is not None else (2 if inference_fps and inference_fps <= 2 else 3)
    cfg.max_gap_ms = args.max_gap_ms if args.max_gap_ms is not None else max(MemoryConfig.max_gap_ms, int(3 * interval_ms))
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video file, webcam index, or stream URL")
    p.add_argument("--prompts", default="bunch of keys",
                   help='comma-separated; "bunch of keys" beat "keys" by a wide margin on our clips')
    p.add_argument("--out", default="data/memories_out")
    p.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--model")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--distractors", help='comma-separated distractor classes, or "none" (default: from the preset)')
    p.add_argument("--fp32", action="store_true", help="disable FP16")
    p.add_argument("--inference-fps", type=float, help="max detector runs per second (0 = every frame)")
    p.add_argument("--fast", action="store_true", help="don't pace video files to real time")
    p.add_argument("--stride", type=int, default=1, help="only consider every Nth captured frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--show", action="store_true", help="window with detections (green) and memories (red)")
    p.add_argument("--min-conf", type=float)
    p.add_argument("--min-hits", type=int)
    p.add_argument("--min-seen-ms", type=int)
    p.add_argument("--max-gap-ms", type=int)
    p.add_argument("--gpu-log", help="CSV path for GPU utilization/power/temperature (needs nvidia-smi)")
    p.add_argument("--scene-matching", choices=["on", "off"], default="on",
                   help="decide 'same place' from the surroundings (on) or image position only (off)")
    p.add_argument("--recorded-at", help='when a video file was recorded, local time, e.g. "2026-10-17 14:05:00". '
                                         "Without it, observed_ms stays null for files.")
    args = p.parse_args()

    preset = dict(PRESETS[args.preset])
    preset["model"] = args.model or preset["model"]
    preset["imgsz"] = args.imgsz or preset["imgsz"]
    preset["half"] = preset["half"] and not args.fp32
    if args.distractors is not None:
        preset["distractors"] = () if args.distractors == "none" else tuple(
            s.strip() for s in args.distractors.split(",") if s.strip())
    model, imgsz, half = preset["model"], preset["imgsz"], preset["half"]
    inference_fps = preset["inference_fps"] if args.inference_fps is None else args.inference_fps
    args.min_conf = preset["min_conf"] if args.min_conf is None else args.min_conf
    preset["min_conf"] = args.min_conf
    cfg = memory_config(args, inference_fps)

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    recorded_at_ms = int(datetime.fromisoformat(args.recorded_at).timestamp() * 1000) if args.recorded_at else None
    session_id = f"sess_{datetime.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:4]}"
    detector = make_detector(preset, prompts)
    from memory_engine.scene import SceneMatcher

    maker = MemoryMaker(cfg, scene_matcher=SceneMatcher() if args.scene_matching == "on" else None)
    frames = FrameSource(args.source, stride=args.stride, recorded_at_ms=recorded_at_ms,
                         max_fps=inference_fps or None, realtime=not args.fast)
    store = MemoryStore(args.out, args.source, session_id=session_id, source_kind=frames.kind)
    print(f"session {session_id} ({frames.kind}) | preset {args.preset}: {model} {imgsz}px {'FP16' if half else 'FP32'}"
          f"{' + distractors ' + str(list(preset['distractors'])) if preset['distractors'] else ''} | "
          f"{inference_fps or 'unlimited'} inferences/s | min_conf={cfg.min_conf} min_hits={cfg.min_hits} "
          f"max_gap_ms={cfg.max_gap_ms}")

    start, n, last_image, infer_ms = time.perf_counter(), 0, None, []
    with GpuMonitor(log_path=args.gpu_log) if args.gpu_log else _NullMonitor() as gpu:
        try:
            for frame in frames:
                t = time.perf_counter()
                detections = detector.detect(frame.image)
                infer_ms.append((time.perf_counter() - t) * 1000)
                for event in maker.process(frame.frame_id, frame.video_ms, frame.wall_ms, detections,
                                           image=frame.image, observed_ms=frame.observed_ms):
                    event = store.handle(event)
                    place = f" place={event['place_status']}" if event.get("place_status") else ""
                    print(f"[{frame.video_ms / 1000:7.2f}s] {event['event']:<15} {event['label']:<12} "
                          f"conf={event['confidence']:.2f} {event['memory_id']}{place}")
                last_image, n = frame.image, n + 1
                if args.show:
                    cv2.imshow("LastSeen memory engine (q quits)", draw(frame.image, detections, maker))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames and n >= args.max_frames:
                    break
        finally:
            for event in maker.finish():
                store.handle(event, last_image)
            store.close()
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    median_ms = sorted(infer_ms)[len(infer_ms) // 2] if infer_ms else 0
    print(f"\n{n} inferences in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.2f}/s, median {median_ms:.1f} ms each). "
          f"{len(maker.memories)} memories -> {args.out}\\memories.json")
    if args.gpu_log:
        print(f"GPU: {gpu.summary()}  (log: {args.gpu_log})")


class _NullMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
