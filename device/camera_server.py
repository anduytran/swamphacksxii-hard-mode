"""The camera Pi (faker): one photo per request for the audio Pi (sreer), nothing in between.

The wearable is two Pi Zero 2 Ws. sreer has the mic, speaker, buttons and LEDs and runs
lastscene_device.py with CAMERA=remote; faker has the camera (and the item detector) and runs this.
For each memory sreer makes a single request; questions never involve faker:

    GET /photo                    a photo now                          -> image/jpeg
    GET /photo?for=<the words>    with VISION_DIR: retried until the item the words name is in it
    GET /photo?max=768            no side longer than 768 px (smaller to send, quicker for a model to read)
                                  (headers X-Item, X-Confidence, X-Attempts say what was found)
    GET /health                   {"camera": ..., "vision": ...}

Every request needs the header X-Camera-Token: <CAMERA_TOKEN> (the same value in both Pis' env).
Between requests faker can keep running the background item detection (VISION_EVERY), as before;
it pauses while a photo is being taken.

    CAMERA_TOKEN=... VISION_DIR=... python3 camera_server.py      # port 8765 (CAMERA_PORT)
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import lastscene_device as dev

log = logging.getLogger("lastscene.camera")
MAX_WORDS = 300


class CameraService:
    def __init__(self, cfg: dev.Config, camera=None, vision=None):
        self.cfg = cfg
        self.camera = camera if camera is not None else dev.make_camera(cfg)
        self.busy = threading.Lock()  # one photo at a time; the background loop waits too
        self.vision = vision
        if vision is None and cfg.vision_dir:
            try:
                self.vision = dev.Vision(cfg, self.camera, paused=self.busy.locked)
                log.info("Item detection: %s", self.vision.detector.name)
            except Exception:  # model or code missing: plain photos still work
                log.exception("Item detection unavailable (VISION_DIR=%s); plain photos only", cfg.vision_dir)

    def photo(self, words: str | None) -> tuple[bytes, dict[str, str]]:
        with self.busy:
            if self.vision is None:
                return self.camera.capture_jpeg(), {}
            frame, hit, attempts = self.vision.photo_for(words)
            headers = {"X-Attempts": str(attempts)}
            if hit is not None:
                headers.update({"X-Item": hit.label, "X-Confidence": f"{hit.conf:.2f}"})
            return dev.jpeg_bytes(frame), headers

    def health(self) -> dict:
        return {"camera": type(self.camera).__name__,
                "vision": self.vision.detector.name if self.vision is not None else None}


def shrink_jpeg(jpeg: bytes, max_side: int, quality: int = 85) -> bytes:
    """The same photo with no side longer than `max_side` (unchanged if it already fits, or can't be read)."""
    try:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # no OpenCV, or not a JPEG it can read: the full-size photo still works
        return jpeg
    if frame is None:
        return jpeg
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return jpeg
    scale = max_side / max(h, w)
    small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return dev.jpeg_bytes(small, quality)


def make_handler(service: CameraService, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "lastscene-camera"

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict):
            self._send(status, json.dumps(body).encode(), "application/json")

        def do_GET(self):  # noqa: N802 (http.server's naming)
            given = self.headers.get("X-Camera-Token", "")
            if not hmac.compare_digest(given.encode(), token.encode()):
                return self._json(401, {"detail": "wrong or missing X-Camera-Token"})
            url = urlparse(self.path)
            if url.path == "/health":
                return self._json(200, service.health())
            if url.path != "/photo":
                return self._json(404, {"detail": "GET /photo or /health"})
            query = parse_qs(url.query)
            words = (query.get("for") or [""])[0][:MAX_WORDS] or None
            max_side = int((query.get("max") or ["0"])[0]) if (query.get("max") or ["0"])[0].isdigit() else 0
            try:
                jpeg, headers = service.photo(words)
                if max_side >= 160:
                    jpeg = shrink_jpeg(jpeg, max_side)
            except Exception as exc:
                log.exception("Photo failed")
                return self._json(503, {"detail": f"camera error: {type(exc).__name__}"})
            log.info("Photo: %d bytes%s", len(jpeg), f", {headers['X-Item']} {headers['X-Confidence']}" if "X-Item" in headers else "")
            return self._send(200, jpeg, "image/jpeg", headers)

        def log_message(self, fmt, *args):  # requests are logged above, without the words
            pass

    return Handler


def serve(service: CameraService, token: str, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    if len(token) < 16:
        raise SystemExit("CAMERA_TOKEN must be set (16+ characters), the same on both Pis")
    return ThreadingHTTPServer((host, port), make_handler(service, token))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = dev.Config.local()  # Auth0 settings are only needed for the background uploads
    service = CameraService(cfg)
    server = serve(service, os.environ.get("CAMERA_TOKEN", ""), os.environ.get("CAMERA_BIND", "0.0.0.0"),
                   int(os.environ.get("CAMERA_PORT", 8765)))
    if service.vision is not None and cfg.vision_every > 0:
        service.vision.start_background()
        log.info("Watching for taught items every %.1f s when the scene changes", cfg.vision_every)
    log.info("Camera ready on port %d (%s)", server.server_address[1], service.health())
    server.serve_forever()


if __name__ == "__main__":
    main()
