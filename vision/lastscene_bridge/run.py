r"""Run LastSeen on a camera or clip and post selected memory events to LastScene.

    # dry run: writes contract payloads (+ preview images with the box drawn from the normalized bbox)
    python -m lastscene_bridge.run --source data\clips\01_normal_placement.MOV --dry-run

    # live demo: real upload (credentials from environment variables; see lastscene_bridge/auth.py)
    python -m lastscene_bridge.run --source http://<esp-ip>:81/stream --show

The default preset is m960d ("efficient" reads laptop keyboards as keys on 1080p frames).
While it runs, type where / search / found / quit in the console, or press w / s / f / q in
the --show window. "search" runs one manual XL tiled search on the current frame; its orange
candidate is not saved or uploaded. "where" prints the latest memory and starts guidance,
and "found" clears the keys' memories and calls the found endpoint.

Recorded clips need a real capture time: --recorded-at auto (default) reads it from the file's
metadata, else from the <clip>.capture.json sidecar that scripts/record_stream.py writes;
without either, memories stay local and are NOT uploaded (no invented timestamps).
Live sources use the host clock at frame decode (keep it NTP-synced); stream URLs and webcams
reconnect for up to --reconnect-seconds when frames stop.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lastscene_bridge.bridge import LastSceneBridge
from lastscene_bridge.capture_time import capture_time_ms
from lastscene_bridge.live_view import FrameDetections, LiveSession
from memory_engine.api import LastSeen
from memory_engine.frames import FrameSource, is_file_source
from memory_engine.presets import PRESETS, make_detector

DEFAULT_PRESET = "m960d"  # AGENTS.md: "efficient" uploads false "keys" on desks (laptop keyboards)


def load_env_file(path):
    """Optional KEY=VALUE file for local testing. Keep it out of git and out of any ZIP."""
    import os

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    p.add_argument("--out", default="data/memories_out", help="LastSeen memory folder")
    p.add_argument("--queue", default="data/upload_queue", help="offline upload queue")
    p.add_argument("--recorded-at", default="auto", help='"auto" (file metadata, else <clip>.capture.json), "none", or local time "2026-10-17 01:35:54"')
    p.add_argument("--dry-run", action="store_true", help="write payloads to <queue>/outbox instead of sending")
    p.add_argument("--fast", action="store_true", help="don't pace files to real time")
    p.add_argument("--enforce-contract-min-confidence", action="store_true",
                   help="drop objects below the contract's 0.35 (default: send and report them)")
    p.add_argument("--env-file", help="optional KEY=VALUE file with credentials (never commit it)")
    p.add_argument("--enrollment-config", help="opt-in local JSON: store, checkpoint, threshold, margin")
    p.add_argument("--prompt", action="append", help="category prompt; repeat for unvalidated extra categories")
    p.add_argument("--show", action="store_true", help="live window: detections, memory state, guidance")
    p.add_argument("--no-console", action="store_true", help="don't read where/found/quit from stdin")
    p.add_argument("--reconnect-seconds", type=float, default=60,
                   help="stream URL / webcam: keep reconnecting this long without frames (0 = stop at once)")
    return p


def main():
    args = build_parser().parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    recorded_at_ms = None
    if is_file_source(args.source):
        if not Path(args.source).exists():
            raise FileNotFoundError(args.source)
        basis = args.recorded_at
        if args.recorded_at == "auto":
            recorded_at_ms, basis = capture_time_ms(args.source)
        elif args.recorded_at != "none":
            recorded_at_ms = int(datetime.fromisoformat(args.recorded_at).timestamp() * 1000)
        print(f"recording time: {recorded_at_ms or 'unknown -> memories stay local, nothing is uploaded'} ({basis})")

    uploader = None
    if not args.dry_run:
        from lastscene_bridge.uploader import uploader_from_env

        uploader = uploader_from_env(args.queue)
    prompts = args.prompt or ["bunch of keys"]
    detector = None
    if args.enrollment_config:
        from memory_engine.enrollment import ClipEncoder, EnrollmentStore, EnrolledDetector
        config = json.loads(Path(args.enrollment_config).read_text(encoding="utf-8"))
        store = EnrollmentStore(config["store"], ClipEncoder(config["checkpoint"]))
        prompts = list(dict.fromkeys(i["category_prompt"] for i in store.items()))
        if not prompts:
            raise ValueError("enrollment store contains no items")
        detector = EnrolledDetector(make_detector(args.preset, prompts), store,
                                    config["threshold"], config.get("margin", 0.05),
                                    diagnostics=lambda row: print(json.dumps({"identity": row})))
    frame_detector = detector if detector is not None else make_detector(args.preset, prompts)
    started = time.monotonic()
    frame_detector.detect(np.zeros((720, 1280, 3), np.uint8))  # load + warm up before frames arrive
    print(f"detector ready ({args.preset}, {time.monotonic() - started:.1f}s)")
    ls = LastSeen(out_dir=args.out, preset=args.preset, source=args.source,
                  prompts=prompts, detector=FrameDetections(frame_detector), search_detector=detector,
                  source_kind="file" if Path(args.source).exists() else "live")
    bridge = LastSceneBridge(ls, uploader=uploader, preset_name=args.preset,
                             outbox_dir=Path(args.queue) / "outbox" if args.dry_run else None,
                             enforce_contract_min_confidence=args.enforce_contract_min_confidence)
    if args.enrollment_config:
        bridge.pipeline["matcher"] = "experimental CLIP ViT-B/32 reference matching + " + bridge.pipeline["matcher"]
    frames = FrameSource(args.source, recorded_at_ms=recorded_at_ms,
                         max_fps=PRESETS[args.preset]["inference_fps"], realtime=not args.fast,
                         reconnect_seconds=args.reconnect_seconds)
    session = LiveSession(ls, bridge, show=args.show, uploading=uploader is not None)
    if not args.no_console:
        session.start_console()
        print("commands: where / search / found / quit (or w / s / f / q in the --show window)", flush=True)
    try:
        if uploader:
            uploader.start()
        session.run(frames)
        bridge.handle(ls.finish())
    finally:
        session.close()
        if uploader:
            uploader.stop()
            for t, oid, status in uploader.log:
                print(f"  upload {oid}: {status}")
        ls.close()
    print(json.dumps(bridge.stats, indent=2))


if __name__ == "__main__":
    main()
