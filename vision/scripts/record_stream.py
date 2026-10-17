r"""Record a test clip from the ESP32 camera stream (or a webcam / file) for the eval pipeline.

    python scripts\record_stream.py --source http://<esp-ip>:81/stream --name 01_placement --seconds 30 --show
    python scripts\record_stream.py --source 0 --name webcam_test --seconds 10          (webcam)

Writes data/clips/esp32/<name>.mp4 plus <name>.json describing the stream:
resolution, measured fps, frame-interval jitter, and stalls (gaps > 0.5 s). MJPEG streams
don't have a fixed frame rate, so frames are buffered and the MP4 is written at the
*measured* rate: playback time then matches real time closely enough to label by eye.
If the stream drops, it reconnects (up to --retries times) and the gap is logged.
Keys while --show is on: q stops early.

Live recordings also get <name>.capture.json: the UTC wall-clock time of the FIRST frame
(this host's clock when it arrived), fps and frame count. The MP4 itself carries no creation
time, so this sidecar is what lets lastscene_bridge give the clip a real captured_at.
timing_error_ms says how far frame times in the MP4 (i / fps) drift from when frames arrived.
Replaying a video file gets no sidecar: "now" is not when that footage was captured.
"""

import argparse
import datetime
import json
import statistics
import time
from pathlib import Path

import cv2


def record(source: str, out_dir: str, name: str, seconds: float, show: bool = False, retries: int = 3,
           countdown: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    open_src = (lambda: cv2.VideoCapture(int(source))) if source.isdigit() else (lambda: cv2.VideoCapture(source))
    cap = open_src()
    if not cap.isOpened():
        raise RuntimeError(f"could not open {source}")
    is_file = Path(source).exists() or Path(source).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}  # incl. remote files
    file_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0  # a file is replayed at its own speed, like a live camera
    for i in range(countdown, 0, -1):
        print(f"recording in {i}...")
        time.sleep(1)
    frames, stamps, reconnects = [], [], 0
    first_frame_ms = None
    start = time.monotonic()
    print(f"recording {seconds:g}s from {source}")
    while time.monotonic() - start < seconds:
        if file_fps:
            due = start + len(frames) / file_fps
            if due > time.monotonic():
                time.sleep(due - time.monotonic())
        ok, img = cap.read()
        if not ok:
            if is_file:
                break  # a video file ended
            if reconnects >= retries:
                print("stream lost, giving up")
                break
            reconnects += 1
            print(f"stream dropped, reconnecting ({reconnects}/{retries})")
            cap.release()
            time.sleep(0.5)
            cap = open_src()
            continue
        stamps.append(time.monotonic() - start)
        if first_frame_ms is None:
            first_frame_ms = int(time.time() * 1000)
        frames.append(img)
        if show:
            vis = img.copy()
            cv2.circle(vis, (16, 16), 7, (0, 0, 255), -1)
            cv2.putText(vis, f"REC {stamps[-1]:4.1f}s  {len(frames)} frames", (30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("record_stream (q stops)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if not frames:
        raise RuntimeError("no frames received")

    duration = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
    fps = min(120.0, max(1.0, (len(frames) - 1) / duration if duration > 0 else 1.0))  # writer needs a sane rate
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img in frames:
        writer.write(img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h)))
    writer.release()
    info = {
        "source": source, "name": name, "file": f"{name}.mp4", "frames": len(frames),
        "resolution": [w, h], "duration_s": round(duration, 2), "measured_fps": round(fps, 2),
        "frame_interval_ms": {"median": round(1000 * statistics.median(gaps), 1) if gaps else None,
                              "max": round(1000 * max(gaps), 1) if gaps else None},
        "stalls_over_500ms": [round(s, 2) for s, g in zip(stamps[1:], gaps) if g > 0.5],
        "reconnects": reconnects,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),  # local time when writing finished, not the start
        "capture_sidecar": None if is_file else f"{name}.capture.json",
    }
    if not is_file:
        write_capture_sidecar(out / f"{name}.capture.json", source, first_frame_ms, fps, stamps)
    else:  # a sidecar left by an earlier live recording under this name would date this footage
        (out / f"{name}.capture.json").unlink(missing_ok=True)
    (out / f"{name}.json").write_text(json.dumps(info, indent=2))
    return info


def write_capture_sidecar(path, source: str, first_frame_ms: int, fps: float, stamps: list[float]) -> dict:
    """The format lastscene_bridge/capture_time.py reads (first_frame_unix_ms is the key it uses)."""
    drift = max(abs(i / fps - (t - stamps[0])) for i, t in enumerate(stamps)) if stamps else 0.0
    first_utc = datetime.datetime.fromtimestamp(first_frame_ms / 1000, tz=datetime.timezone.utc)
    sidecar = {
        "first_frame_unix_ms": first_frame_ms,
        "first_frame_utc": first_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "clock": "recording host wall clock when the first frame arrived (keep it NTP-synced)",
        "source": source, "fps": round(fps, 3), "frames": len(stamps),
        "timing_error_ms": round(1000 * drift, 1),
    }
    Path(path).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="stream URL, webcam index, or video file")
    p.add_argument("--name", required=True, help="clip name, e.g. 01_placement")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--out", default="data/clips/esp32")
    p.add_argument("--show", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--countdown", type=int, default=3)
    args = p.parse_args()
    info = record(args.source, args.out, args.name, args.seconds, args.show, args.retries, args.countdown)
    print(json.dumps(info, indent=2))
    print(f"\nnext: python scripts\\contact_sheet.py {Path(args.out) / info['file']}   (to set the label windows)")


if __name__ == "__main__":
    main()
