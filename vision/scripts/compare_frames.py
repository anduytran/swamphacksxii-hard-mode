r"""Compare two frames from real videos (or photos) with the scene matcher.

Single comparison: frame A vs frame B, saves a side-by-side picture with matched
features (thin lines), the object box in A (red) and its projection into B (green):

    python scripts\compare_frames.py --a data\clips\look_away.mp4 --a-time 1.0 --b-time 9.5 --box 400,300,470,340
    python scripts\compare_frames.py --a data\clips\desk.mp4 --a-time 1 --b data\clips\shelf.mp4 --b-time 2 --detect keys

Sweep: frame A vs every STEP seconds of video B, one line per comparison
(shows where the verdict flips while you walk around the desk):

    python scripts\compare_frames.py --a data\clips\walk_around.mp4 --a-time 1 --sweep 0.5 --detect keys

Box: give --box x1,y1,x2,y2 in frame A pixels, or --detect LABEL to take YOLO-World's
most confident box for that label in frame A. Without either, only the scene verdict is shown.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.scene import SceneMatcher, visible_fraction  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = {"same_scene": (0, 170, 0), "different_scene": (0, 0, 220), "unknown": (0, 160, 230)}


def grab(path, seconds):
    """One frame from a video at `seconds`, or the image itself."""
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(path)
    else:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, image = cap.read()
        cap.release()
        image = image if ok else None
    if image is None:
        raise SystemExit(f"Could not read a frame from {path} at {seconds}s")
    return image


def iter_frames(path, step):
    cap = cv2.VideoCapture(path)
    next_t = 0.0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t + 1e-6 >= next_t:
            yield t, image
            next_t += step
    cap.release()


def best_detection(detector, image):
    detections = detector.detect(image) if detector else []
    return max(detections, key=lambda d: d.conf) if detections else None


def visualize(a, b, result, box, projected, max_lines=80):
    height = 480
    sa, sb = height / a.shape[0], height / b.shape[0]
    va, vb = cv2.resize(a, None, fx=sa, fy=sa), cv2.resize(b, None, fx=sb, fy=sb)
    canvas = np.hstack([va, vb])
    offset = va.shape[1]
    if result.inlier_points_a is not None and result.verdict == "same_scene":
        step = max(1, len(result.inlier_points_a) // max_lines)
        for pa, pb in zip(result.inlier_points_a[::step], result.inlier_points_b[::step]):
            p1 = (int(pa[0] * sa), int(pa[1] * sa))
            p2 = (int(pb[0] * sb) + offset, int(pb[1] * sb))
            cv2.line(canvas, p1, p2, (255, 200, 0), 1)
            cv2.circle(canvas, p1, 2, (255, 200, 0), -1)
            cv2.circle(canvas, p2, 2, (255, 200, 0), -1)
    if box:
        x1, y1, x2, y2 = (int(v * sa) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if projected:
        x1, y1, x2, y2 = (int(v * sb) for v in projected)
        cv2.rectangle(canvas, (x1 + offset, y1), (x2 + offset, y2), (0, 220, 0), 2)
    text = (f"{result.verdict} ({result.reason})  inliers={result.inliers}  ratio={result.inlier_ratio}  "
            f"projection={'trusted' if result.projection_trusted else 'not trusted'}")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[result.verdict], 1, cv2.LINE_AA)
    return canvas


def describe(result, projected, b_shape, extract_ms):
    line = (f"{result.verdict:<15} reason={result.reason:<34} features={result.features_a}/{result.features_b} "
            f"good={result.good_matches} inliers={result.inliers} ratio={result.inlier_ratio} "
            f"spread={result.spread_a}/{result.spread_b} time={extract_ms + result.elapsed_ms:.1f}ms")
    if projected:
        line += f" box_in_B={projected} visible={visible_fraction(projected, b_shape):.0%}"
    return line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="video or image for the old memory")
    p.add_argument("--a-time", type=float, default=0.0, help="seconds into video A")
    p.add_argument("--b", help="video or image to compare against (default: same as --a)")
    p.add_argument("--b-time", type=float, default=0.0, help="seconds into video B")
    p.add_argument("--box", help="x1,y1,x2,y2 of the object in frame A")
    p.add_argument("--detect", metavar="LABEL", help="find the box in frame A with YOLO-World")
    p.add_argument("--sweep", type=float, metavar="STEP", help="compare A against all of video B every STEP seconds")
    p.add_argument("--out", default="data/compare", help="folder for visualization images")
    p.add_argument("--include-object", action="store_true",
                   help="also match features on the object itself (default: only its surroundings)")
    args = p.parse_args()

    b_path = args.b or args.a
    matcher = SceneMatcher()
    detector = None
    if args.detect:
        from memory_engine.detector import YoloWorldDetector

        detector = YoloWorldDetector([args.detect])
    frame_a = grab(args.a, args.a_time)
    if args.box:
        box = tuple(int(v) for v in args.box.split(","))
    elif detector:
        det = best_detection(detector, frame_a)
        if det is None:
            raise SystemExit(f'YOLO-World found no "{args.detect}" in frame A. Pass --box instead.')
        box = det.bbox
        print(f'frame A: "{args.detect}" at {box}, confidence {det.conf:.2f}')
    else:
        box = None
    features_a = matcher.features(frame_a, exclude_boxes=[box] if box and not args.include_object else ())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_name = f"{Path(args.a).stem}_{args.a_time:g}s__{Path(b_path).stem}" + ("_withobj" if args.include_object else "")
    rows = []

    import csv
    import time

    from memory_engine.memory import iou

    targets = iter_frames(b_path, args.sweep) if args.sweep else [(args.b_time, grab(b_path, args.b_time))]
    for t, frame_b in targets:
        start = time.perf_counter()
        features_b = matcher.features(frame_b)
        extract_ms = (time.perf_counter() - start) * 1000
        result = matcher.match(features_a, features_b)
        projected = matcher.project_box(result, box) if box else None
        line = f"B@{t:6.2f}s  " + describe(result, projected, frame_b.shape, extract_ms)
        # Independent check: where does the detector itself find the object in B?
        det_b = best_detection(detector, frame_b)
        agreement = round(iou(projected, det_b.bbox), 2) if projected and det_b else None
        if detector:
            line += f" | detector_in_B={'none' if det_b is None else f'{det_b.bbox}@{det_b.conf:.2f}'}"
            line += f" proj_vs_det_iou={agreement}" if agreement is not None else ""
        print(line)
        cv2.imwrite(str(out / f"{run_name}_{t:.2f}s.jpg"), visualize(frame_a, frame_b, result, box, projected))
        rows.append({"b_time_s": round(t, 2), "verdict": result.verdict, "reason": result.reason,
                     "inliers": result.inliers, "inlier_ratio": result.inlier_ratio,
                     "projection_trusted": result.projection_trusted, "projected_box": projected,
                     "detector_box": det_b.bbox if det_b else None,
                     "detector_conf": det_b.conf if det_b else None, "proj_vs_det_iou": agreement,
                     "ms": round(extract_ms + result.elapsed_ms, 1)})
    with (out / f"{run_name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nvisualizations + {run_name}.csv saved in {out}")


if __name__ == "__main__":
    main()
