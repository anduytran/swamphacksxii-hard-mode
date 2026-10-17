r"""detect: watch frames, decide when something happened to an item, post one observation per event.

    python -m lastscene_pipeline.detect --source SRC [--model DIR] [--min-confidence F] [--dry-run]
                                        [--scene-label "grand ballroom table"] [--show]

--source   camera index ("0"), stream URL, a folder of JPEGs (testing), a video file, or "picam" (Raspberry Pi
           camera via picamera2)
--model    default $LASTSCENE_DATA/models/current. Runtime: best.pt with Ultralytics if installed (laptop),
           else model-int8.onnx (Pi). No trained model: YOLO-World set_classes(class names), laptop only.
--min-confidence  default 0.35 (the server drops detections under 0.35). Memories only form from detections
           at or above it, so nothing below it is ever posted.

Events (observation.event_for): placed = first confirmed sighting (>= 2 frames over >= 0.5 s), moved = ORB
places it elsewhere than its last memory, seen = resent at sighting end after 10 minutes. Never picked_up
(no hand tracking). Labels are the dashboard's class names. Scene is sent only with --scene-label: the
camera's location in the room's words, set by whoever mounts it (ORB has no room names).
Uploads: one per 2 s, 429 Retry-After, the same observation_id on retries, an on-disk queue for outages.
Reloads the model when models/current changes; runs sync every 5 minutes (--sync-every 0 = off).
While it runs: where / found / quit on the console, or w / f / q with --show.
"""

import argparse
import json
import threading
import time
from pathlib import Path

from lastscene_pipeline import EXIT_CONFIG, EXIT_OK
from lastscene_pipeline.common import PipelineError, data_dir, load_env_file, read_json, resolve_model_dir, run_main

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --- frame sources beyond memory_engine.frames (file / webcam / stream URL) -------------------------
def folder_frames(folder: Path, fps: float):
    """A folder of JPEGs, in name order. Capture time = each file's modification time (what a camera or
    record script writes when it saves the frame); copy with timestamps preserved or the times are wrong."""
    import cv2

    from memory_engine.frames import Frame

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise PipelineError(EXIT_CONFIG, f"{folder}: no JPEG/PNG frames")
    first = int(files[0].stat().st_mtime * 1000)
    for i, f in enumerate(files):
        image = cv2.imread(str(f))
        if image is None:
            continue
        observed = int(f.stat().st_mtime * 1000)
        yield Frame(i, observed - first, int(time.time() * 1000), image, observed)


def picam_frames(fps: float, size=(1280, 720)):
    """Raspberry Pi camera (libcamera) via picamera2: apt install python3-picamera2. Untested on hardware."""
    from picamera2 import Picamera2

    from memory_engine.frames import Frame

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": size, "format": "RGB888"}))  # BGR byte order
    cam.start()
    start, interval, i = time.monotonic(), 1 / fps, 0
    try:
        while True:
            image = cam.capture_array()
            now = int(time.time() * 1000)
            yield Frame(i, int((time.monotonic() - start) * 1000), now, image, now)
            i += 1
            time.sleep(max(0.0, start + i * interval - time.monotonic()))
    finally:
        cam.stop()


def open_source(source: str, fps: float, reconnect_seconds: float, recorded_at: str):
    from memory_engine.frames import FrameSource, is_file_source

    if source == "picam":
        return picam_frames(fps)
    if Path(source).is_dir():
        return folder_frames(Path(source), fps)
    recorded_at_ms = None
    if is_file_source(source):
        from lastscene_bridge.capture_time import capture_time_ms

        recorded_at_ms, basis = capture_time_ms(source) if recorded_at == "auto" else (None, "none")
        print(f"recording time: {recorded_at_ms or 'unknown -> nothing is posted'} ({basis})", flush=True)
    return FrameSource(source, recorded_at_ms=recorded_at_ms, max_fps=fps, realtime=False,
                       reconnect_seconds=reconnect_seconds)


# --- model + classes ---------------------------------------------------------------------------------
def load_classes(model_dir: Path | None, root: Path) -> list[str]:
    for f in ([model_dir / "classes.json"] if model_dir else []) + [root / "dataset" / "classes.json"]:
        if f.exists():
            return read_json(f)["classes"]
    return ["keys"]  # nothing synced yet: the keys demo


class ReloadingDetector:
    """Swaps in the new model when models/current points somewhere new (checked every `every` seconds).
    A model that fails to load is logged and the old one keeps running."""

    def __init__(self, pointer: Path | None, explicit: Path | None, build, every: float = 10, log=print):
        self.pointer, self.explicit, self.build, self.every, self.log = pointer, explicit, build, every, log
        self.model_dir = explicit or (resolve_model_dir(pointer) if pointer else None)
        self.inner = build(self.model_dir)
        self._checked = time.monotonic()

    @property
    def name(self):
        return self.inner.name

    def _maybe_reload(self):
        if self.explicit is not None or self.pointer is None or time.monotonic() - self._checked < self.every:
            return
        self._checked = time.monotonic()
        target = resolve_model_dir(self.pointer)
        if target is None or (self.model_dir and target.resolve() == self.model_dir.resolve()):
            return
        try:
            self.inner, self.model_dir = self.build(target), target
            self.log(f"model reloaded: {self.inner.name}")
        except Exception as exc:  # noqa: BLE001  keep detecting with the old model
            self.log(f"model {target} failed to load ({type(exc).__name__}: {exc}); keeping {self.inner.name}")

    def detect(self, image):
        self._maybe_reload()
        return self.inner.detect(image)


def background_sync(dataset: Path, every: float, log=print):
    """Run sync every `every` seconds on a daemon thread. New items need a new train; say so."""
    from lastscene_pipeline import EXIT_OK as SYNCED
    from lastscene_pipeline.sync import run_sync

    def loop():
        while True:
            time.sleep(every)
            try:
                if run_sync(dataset, log=lambda m: None) == SYNCED:
                    log(f"sync: dataset updated to {(dataset / 'version').read_text().strip()}; "
                        f"run train on the laptop to learn the new photos")
            except PipelineError as exc:
                log(f"sync skipped: {exc}")
            except Exception as exc:  # noqa: BLE001  never take detection down
                log(f"sync failed: {type(exc).__name__}: {exc}")

    threading.Thread(target=loop, name="lastscene-sync", daemon=True).start()


def pipeline_meta(detector) -> dict:
    from lastscene_bridge import BRIDGE_VERSION, PIPELINE_NAME

    return {"name": PIPELINE_NAME, "version": BRIDGE_VERSION, "detector": detector.name[:80], "matcher": "orb"}


def post_logger(summary, oid, status, http):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{stamp} {oid} {summary} {status} {http if http is not None else '-'}", flush=True)


# --- main --------------------------------------------------------------------------------------------
def build_parser():
    root = data_dir()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--model", type=Path, help="default $LASTSCENE_DATA/models/current")
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--dry-run", action="store_true", help="print each observation JSON instead of posting it")
    p.add_argument("--scene-label", help='where this camera is, in the room\'s words: "grand ballroom table"')
    p.add_argument("--scene-id", help="default: the label as a slug")
    p.add_argument("--runtime", default="auto", choices=["auto", "onnx", "onnx-fp32", "torch"])
    p.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads (Pi Zero 2 W: 4 cores)")
    p.add_argument("--fps", type=float, default=2.0, help="inference rate cap (memory thresholds scale with it)")
    p.add_argument("--out", type=Path, default=root / "memories", help="local memory folder")
    p.add_argument("--queue", type=Path, default=root / "upload_queue")
    p.add_argument("--recorded-at", default="auto", choices=["auto", "none"], help="video files: capture time source")
    p.add_argument("--reconnect-seconds", type=float, default=60)
    p.add_argument("--sync-every", type=float, default=300, help="seconds between background syncs (0 = off)")
    p.add_argument("--show", action="store_true", help="live window (laptop)")
    p.add_argument("--no-console", action="store_true")
    p.add_argument("--env-file", help="optional KEY=VALUE credentials file (never commit it)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)
    from lastscene_bridge.bridge import LastSceneBridge
    from lastscene_bridge.live_view import FrameDetections, LiveSession
    from lastscene_bridge.observation import scene_block
    from lastscene_pipeline.runtime import load_detector
    from memory_engine.api import LastSeen
    from memory_engine.memory import MemoryConfig

    root = data_dir()
    pointer = None if args.model else root / "models" / "current"
    explicit = resolve_model_dir(args.model) if args.model else None
    if args.model and explicit is None:
        raise PipelineError(EXIT_CONFIG, f"--model {args.model}: not a model folder")
    classes = load_classes(explicit or (resolve_model_dir(pointer) if pointer else None), root)
    started = time.monotonic()
    try:
        detector = ReloadingDetector(pointer, explicit, lambda d: load_detector(
            d, load_classes(d, root), args.min_confidence, args.runtime, args.threads))
    except (FileNotFoundError, ValueError) as exc:
        raise PipelineError(EXIT_CONFIG, str(exc)) from None
    print(f"detector: {detector.name} ({time.monotonic() - started:.1f}s); classes {classes}; "
          f"posting >= {args.min_confidence}", flush=True)
    if detector.model_dir is None and args.min_confidence >= 0.35:
        print("note: untrained YOLO-World scores real keys about 0.05-0.35, so few sightings reach "
              f"{args.min_confidence}; train a model from the Teach items photos", flush=True)

    uploader = None
    if not args.dry_run:
        from lastscene_bridge.uploader import uploader_from_env

        try:
            uploader = uploader_from_env(args.queue)
        except RuntimeError as exc:
            raise PipelineError(EXIT_CONFIG, f"device credentials: {exc}") from None
        uploader.on_result = post_logger
    if args.sync_every and (uploader is not None or _has_credentials()):
        background_sync(root / "dataset", args.sync_every)

    prompts = sorted(classes, key=lambda c: c != "keys")  # LiveSession's where/found act on prompts[0]
    cfg = MemoryConfig(min_conf=args.min_confidence)
    cfg.min_hits = 2 if args.fps <= 2 else 3
    cfg.max_gap_ms = max(cfg.max_gap_ms, int(3 * 1000 / args.fps))
    frame_detector = FrameDetections(detector)
    ls = LastSeen(out_dir=str(args.out), preset="m960d", prompts=prompts, source=args.source,
                  detector=frame_detector, search_detector=frame_detector, memory_config=cfg,
                  source_kind="live")
    bridge = LastSceneBridge(ls, uploader=uploader, min_confidence=args.min_confidence,
                             scene=scene_block(args.scene_label, args.scene_id), pipeline=pipeline_meta(detector),
                             outbox_dir=args.queue / "outbox" if args.dry_run else None)
    if args.dry_run:
        write = bridge._write_outbox

        def print_and_write(obs, jpeg):
            print(json.dumps(obs), flush=True)
            write(obs, jpeg)
        bridge._write_outbox = print_and_write

    session = LiveSession(ls, bridge, show=args.show, uploading=uploader is not None)
    if not args.no_console:
        session.start_console()
    frames = open_source(args.source, args.fps, args.reconnect_seconds, args.recorded_at)
    try:
        if uploader:
            uploader.start()
        session.run(frames)
        bridge.handle(ls.finish())
    finally:
        session.close()
        if uploader:
            uploader.stop()
        ls.close()
    print(json.dumps(bridge.stats), flush=True)
    return EXIT_OK


def _has_credentials() -> bool:
    import os

    return bool(os.environ.get("DEVICE_CLIENT_ID") or os.environ.get("AUTH0_CLIENT_ID"))


if __name__ == "__main__":
    run_main(main)
