"""Interactively save a small, labeled set of original Pi camera frames.

Run on the Pi from the vision checkout after stopping any process that owns the
camera. Files go under lastscene-data/ by default, which is ignored by git.
For each prompt, reposition the same target or remove it for the empty group,
then press Enter to save one PNG. Change viewpoint between shots.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2


GROUPS = (
    ("near", True, 5, "Put the pin close to the camera; vary angle and position between shots."),
    ("mid", True, 5, "Put the pin at a middle distance; vary angle and position between shots."),
    ("far", True, 5, "Put the pin at the far distance you want to test; vary its position between shots."),
    ("empty", False, 5, "Remove the pin. Capture similar backgrounds with no pin visible."),
)


def open_camera(source: str, width: int, height: int, fps: int, pixel_format: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera source {source!r} (is live_feed.py or another process using it?)")
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or stream URL (default: 0)")
    parser.add_argument("--out", type=Path, default=Path("lastscene-data/pi_eval"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pixel-format", default="MJPG", choices=("MJPG", "YUYV"))
    parser.add_argument("--show", action="store_true", help="briefly display each saved still; needs a local display")
    args = parser.parse_args()

    cap = open_camera(args.source, args.width, args.height, args.fps, args.pixel_format)
    samples = []
    try:
        # Let the camera settle and report the actual decoded array, not requested mode.
        frame = None
        for _ in range(8):
            ok, current = cap.read()
            if ok:
                frame = current
        if frame is None:
            raise RuntimeError("camera opened but returned no frames (is live_feed.py or another process using it?)")
        actual_h, actual_w = frame.shape[:2]
        print(f"Camera returned {actual_w}x{actual_h} pixels (requested {args.width}x{args.height}, {args.pixel_format}).")
        if (actual_w, actual_h) != (args.width, args.height):
            print("NOTE: use the actual dimensions in the report; the camera did not return the requested mode.")
        # Only now create the run folder: a busy or broken camera leaves no empty run behind.
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out / run_id
        out.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id, "source": args.source,
            "requested_capture": {"width": args.width, "height": args.height, "fps": args.fps,
                                  "pixel_format": args.pixel_format},
            "actual_capture": {"width": actual_w, "height": actual_h},
            "target_note": "Same physical pin across positive scenes; empty means no pin in frame.",
            "samples": samples,
        }

        for group, visible, count, instruction in GROUPS:
            print(f"\n{group.upper()} ({count} frames): {instruction}")
            for index in range(1, count + 1):
                input(f"Arrange view {index}/{count}, then press Enter to capture (Ctrl+C to stop)... ")
                # Drain queued frames so a snapshot is recent rather than stale.
                for _ in range(3):
                    cap.grab()
                ok, image = cap.read()
                if not ok or image is None:
                    print("Frame read failed; skipping this sample.")
                    continue
                stamp = datetime.now(timezone.utc)
                image_path = out / f"{group}_{index:02d}.png"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"failed to save {image_path}")
                row = {
                    "id": f"{group}_{index:02d}", "scene": group,
                    "target_visible": visible, "capture_utc": stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "image": image_path.name, "width": int(image.shape[1]), "height": int(image.shape[0]),
                }
                samples.append(row)
                # Rewritten after every frame, so frames saved before a Ctrl+C stay usable.
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(f"saved {image_path} ({image.shape[1]}x{image.shape[0]})")
                if args.show:
                    preview = cv2.resize(image, (960, 540)) if image.shape[1] > 960 else image
                    cv2.imshow("saved Pi frame (press any key)", preview)
                    cv2.waitKey(250)
                    cv2.destroyAllWindows()
        print(f"\nSaved {len(samples)} labeled frames. Manifest: {out / 'manifest.json'}")
        if len(samples) < 20:
            print("Some captures were skipped; capture additional views before evaluating if possible.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
