"""Can a pretrained open-vocabulary detector see our target objects?

Runs YOLO-World on a webcam (or a video/image file), draws boxes, and at the end
prints how often each prompt was detected. Run it once per object/distance with a
--tag like "keys_1m_table" and the results append to data/detect_summary.csv, which
becomes the table we use at GO/NO-GO #1 to pick the demo objects.

Keys: q = quit, s = save a snapshot to data/snapshots/.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

DEFAULT_PROMPTS = "keys,wallet,sunglasses,cell phone,remote"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def summarize(frame_detections, total_frames, prompts):
    """Per-prompt detection rate and confidence.

    frame_detections: one list per processed frame, each holding (label, confidence)
    tuples. A frame counts as "detected" for a label if any box had that label.
    Every prompt gets a row, so a prompt that was never detected shows rate 0.
    """
    best_conf_per_frame = defaultdict(list)  # label -> best confidence in each frame it appeared
    for detections in frame_detections:
        best = {}
        for label, conf in detections:
            best[label] = max(conf, best.get(label, 0.0))
        for label, conf in best.items():
            best_conf_per_frame[label].append(conf)

    rows = []
    for label in prompts:
        confs = best_conf_per_frame.get(label, [])
        rows.append({
            "label": label,
            "frames_total": total_frames,
            "frames_detected": len(confs),
            "detect_rate": round(len(confs) / total_frames, 3) if total_frames else 0.0,
            "mean_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "max_conf": round(max(confs), 3) if confs else 0.0,
        })
    return rows


def open_source(source):
    """Return (kind, handle): an image array for image files, else a VideoCapture."""
    if Path(source).suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(source)
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        return "image", image
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return "video", cap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index, video file, image file, or MJPEG URL")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="comma-separated object names")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--conf", type=float, default=0.10, help="low on purpose so we see weak detections")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tag", default="untagged", help='what this run tests, e.g. "keys_1m_hand"')
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until q)")
    parser.add_argument("--no-show", action="store_true", help="don't open a window (for headless runs)")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works without loading torch

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    model = YOLO(args.model)
    model.set_classes(prompts)  # YOLO-World: turn text prompts into the class list

    out_dir = Path(args.out_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "detections.jsonl").open("a", encoding="utf-8")

    kind, handle = open_source(args.source)
    frame_detections = []
    frame_id = 0
    start = time.perf_counter()

    try:
        while True:
            if kind == "image":
                if frame_id >= max(1, args.max_frames):
                    break
                frame = handle
            else:
                ok, frame = handle.read()
                if not ok:
                    break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = []
            for box in result.boxes:
                label = prompts[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append({"label": label, "conf": round(conf, 3), "bbox": [x1, y1, x2, y2]})

            frame_detections.append([(d["label"], d["conf"]) for d in detections])
            log_file.write(json.dumps({
                "tag": args.tag, "frame_id": frame_id, "ts_ms": int(time.time() * 1000),
                "detections": detections,
            }) + "\n")
            frame_id += 1
            fps = frame_id / (time.perf_counter() - start)

            if not args.no_show:
                shown = frame.copy()
                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(shown, f'{d["label"]} {d["conf"]:.2f}', (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.putText(shown, f"{fps:.1f} fps  tag={args.tag}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("detect_test (q quits, s saves)", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    cv2.imwrite(str(out_dir / "snapshots" / f"{args.tag}_{frame_id}.jpg"), shown)

            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        log_file.close()
        if kind == "video":
            handle.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    rows = summarize(frame_detections, frame_id, prompts)
    print(f"\n{frame_id} frames in {elapsed:.1f}s ({frame_id / elapsed:.1f} fps), tag={args.tag}")
    print(f"{'label':<12} {'detected':>9} {'rate':>6} {'mean':>6} {'max':>6}")
    for r in rows:
        print(f"{r['label']:<12} {r['frames_detected']:>4}/{r['frames_total']:<4} "
              f"{r['detect_rate']:>6.2f} {r['mean_conf']:>6.2f} {r['max_conf']:>6.2f}")

    summary_path = out_dir / "detect_summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "fps", *rows[0].keys()])
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({"tag": args.tag, "fps": round(frame_id / elapsed, 1), **r})


if __name__ == "__main__":
    main()
