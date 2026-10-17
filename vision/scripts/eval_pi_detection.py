"""Compare the Pi INT8 detector on the same labeled frames, including optional tiling.

Modes: native full frame, the same decoded frame downsampled to 640x480 (a
preview-size proxy, not a second camera capture), and native full frame plus
four overlapping tiles. The model remains at its exported fixed input size.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from lastscene_pipeline.bench import peak_rss_mb
from lastscene_pipeline.runtime import OnnxYoloDetector, decode, letterbox, sha256
from memory_engine.detector import Detection


def temperature_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def infer(detector, image, confidence):
    """One shared preprocessing/inference/decode pass, including raw class score below threshold."""
    tensor, transform = letterbox(image, detector.size)
    output = detector.session.run(None, {detector.input.name: tensor})[0]
    raw_max = float(np.max(output[0, 4:, :])) if output.shape[1] > 4 and output.shape[2] else 0.0
    detections = decode(output, transform, image.shape, detector.classes, confidence)
    return detections, raw_max


def infer_tiled(detector, image, confidence, grid, overlap):
    """Match detect_tiled's full-frame plus overlapping crop layout and preserve source pixels."""
    h, w = image.shape[:2]
    th, tw = int(h / grid * (1 + overlap)), int(w / grid * (1 + overlap))
    detections, full_score = infer(detector, image, confidence)
    raw_scores = [full_score]
    for row in range(grid):
        for col in range(grid):
            y0, x0 = min(int(row * h / grid), h - th), min(int(col * w / grid), w - tw)
            crop_detections, score = infer(detector, image[y0:y0 + th, x0:x0 + tw], confidence)
            raw_scores.append(score)
            for detection in crop_detections:
                x1, y1, x2, y2 = detection.bbox
                detections.append(Detection(detection.label, detection.conf,
                                           (x1 + x0, y1 + y0, x2 + x0, y2 + y0)))
    return detections, max(raw_scores, default=0.0)


def summarize(rows):
    by_scene = {}
    for scene in ("near", "mid", "far", "empty"):
        selected = [row for row in rows if row["scene"] == scene]
        if not selected:
            continue
        positive = scene != "empty"
        by_scene[scene] = {
            "frames": len(selected),
            "frames_with_detection_at_threshold": sum(row["detected"] for row in selected),
            "frame_rate": round(sum(row["detected"] for row in selected) / len(selected), 3),
            "false_positive_frames": None if positive else sum(row["detected"] for row in selected),
            "max_raw_class_score_per_frame": [row["max_raw_class_score"] for row in selected],
            "median_latency_ms": round(statistics.median(row["latency_ms"] for row in selected), 1),
            "p95_latency_ms": round(float(np.percentile([row["latency_ms"] for row in selected], 95)), 1),
            "effective_fps_at_median_latency": round(1000 / statistics.median(row["latency_ms"] for row in selected), 3),
        }
    all_rows = rows
    if all_rows:
        latencies = [row["latency_ms"] for row in all_rows]
        cpu_ms = sum(row["cpu_ms"] for row in all_rows)
        by_scene["overall"] = {
            "frames": len(all_rows),
            "median_latency_ms": round(statistics.median(latencies), 1),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
            "effective_fps_at_median_latency": round(1000 / statistics.median(latencies), 3),
            "aggregate_cpu_percent": round(cpu_ms / sum(latencies) * 100, 1) if sum(latencies) else None,
        }
    return by_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="version folder containing model-int8.onnx and model.json")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest.json from capture_pi_eval.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.35, help="keep the current preview acceptance floor")
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    model_path = (args.model / "model-int8.onnx").resolve()
    model_dir = model_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("samples", [])) == 0:
        parser.error("manifest contains no samples")
    if args.grid < 1 or not 0 <= args.overlap < 1:
        parser.error("grid must be >= 1 and overlap must be in [0, 1)")

    detector = OnnxYoloDetector(model_path, confidence=args.confidence, threads=args.threads, low_memory=True)
    if detector.classes != ["pin"]:
        parser.error(f"expected the current one-class pin model, found classes={detector.classes}")
    modes = ("full_native", "full_640x480_proxy", "native_tiled")
    rows = {mode: [] for mode in modes}
    per_image = []
    process_start = time.process_time()
    wall_start = time.perf_counter()
    temp_start = temperature_c()
    for sample in manifest["samples"]:
        image_path = (manifest_path.parent / sample["image"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode {image_path}")
        if (image.shape[1], image.shape[0]) != (sample["width"], sample["height"]):
            raise RuntimeError(f"dimensions differ from manifest for {image_path}")
        vga = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        inputs = {
            "full_native": image,
            "full_640x480_proxy": vga,
            "native_tiled": image,
        }
        image_row = {"id": sample["id"], "scene": sample["scene"], "target_visible": sample["target_visible"],
                     "width": image.shape[1], "height": image.shape[0], "image_sha256": sha256(image_path), "results": {}}
        # Rotate order across frames to reduce consistent thermal/order bias.
        order = list(modes)
        shift = len(per_image) % len(order)
        order = order[shift:] + order[:shift]
        for mode in order:
            frame = inputs[mode]
            started = time.perf_counter()
            cpu_started = time.process_time()
            if mode == "native_tiled":
                detections, raw_max = infer_tiled(detector, frame, args.confidence, args.grid, args.overlap)
            else:
                detections, raw_max = infer(detector, frame, args.confidence)
            elapsed_ms = (time.perf_counter() - started) * 1000
            cpu_ms = (time.process_time() - cpu_started) * 1000
            scores = [float(d.conf) for d in detections if d.label == "pin"]
            detected = any(score >= args.confidence for score in scores)
            result = {
                "detected": detected, "max_raw_class_score": round(raw_max, 6),
                "detections": [{"label": d.label, "confidence": d.conf, "bbox": list(d.bbox)} for d in detections],
                "latency_ms": round(elapsed_ms, 2),
            }
            image_row["results"][mode] = result
            rows[mode].append({"id": sample["id"], "scene": sample["scene"], "detected": detected,
                               "max_raw_class_score": result["max_raw_class_score"], "latency_ms": elapsed_ms,
                               "cpu_ms": cpu_ms})
        per_image.append(image_row)
    wall_s = time.perf_counter() - wall_start
    cpu_s = time.process_time() - process_start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model_version": detector.manifest["version"], "model": model_path.name,
        "model_sha256": sha256(model_path), "classes": detector.classes, "model_input": [1, 3, detector.size, detector.size],
        "confidence_threshold": args.confidence, "ort_threads": args.threads, "low_memory": True,
        "capture_manifest": str(manifest_path), "capture_dimensions": manifest.get("actual_capture"),
        "modes": {
            "full_native": "one inference on each original decoded camera frame",
            "full_640x480_proxy": "one inference after downscaling that same frame to 640x480; proxy only, not a second camera capture",
            "native_tiled": f"one full-frame plus {args.grid ** 2} overlapping tile inferences; tile boxes mapped to full-frame pixels",
        },
        "tiled": {"grid": args.grid, "overlap": args.overlap, "inferences_per_frame": args.grid ** 2 + 1},
        "summary": {mode: summarize(values) for mode, values in rows.items()},
        "resource": {"wall_seconds": round(wall_s, 2), "process_cpu_seconds": round(cpu_s, 2),
                     "aggregate_cpu_percent": round(cpu_s / wall_s * 100, 1) if wall_s else None,
                     "peak_rss_mb": peak_rss_mb(), "temperature_start_c": temp_start, "temperature_end_c": temperature_c()},
        "limitations": [
            "This is a same-item exploratory test across camera views, not identity testing against other pins.",
            "The detector accepts only the exported 320x320 tensor; a true 640-input model comparison requires another export.",
            "The 640x480 mode is a downsampled proxy from the native frame, not a separately captured camera mode.",
            "Frame hit rate is not memory-event recall and does not establish false-memory rate.",
            "Tiled inference uses five model calls per frame and is not enabled for automatic memories.",
        ],
        "frames": per_image,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for mode, scenes in report["summary"].items():
        print(f"\n{mode}")
        for scene, values in scenes.items():
            if scene == "overall":
                print(f"  overall: median {values['median_latency_ms']} ms, p95 {values['p95_latency_ms']} ms; "
                      f"{values['effective_fps_at_median_latency']} FPS at median, "
                      f"CPU {values['aggregate_cpu_percent']}% aggregate")
                continue
            fp = f", false-positive frames {values['false_positive_frames']}" if scene == "empty" else ""
            print(f"  {scene}: {values['frames_with_detection_at_threshold']}/{values['frames']} detected{fp}; "
                  f"median {values['median_latency_ms']} ms, p95 {values['p95_latency_ms']} ms; "
                  f"{values['effective_fps_at_median_latency']} FPS at median; "
                  f"raw max scores {values['max_raw_class_score_per_frame']}")
    print(f"\nwall {wall_s:.1f}s; CPU {cpu_s:.1f}s ({cpu_s / wall_s * 100:.1f}% aggregate); peak RSS {report['resource']['peak_rss_mb']} MiB")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
