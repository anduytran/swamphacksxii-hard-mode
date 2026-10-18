"""The two-Pi link: camera_server.py (faker, the camera) and RemoteCamera (sreer, everything else),
over real HTTP on localhost. No camera, model or network needed.

Run from the repo root:  backend/.venv/bin/python -m pytest device
"""

import threading
from types import SimpleNamespace

import httpx
import pytest

import camera_server
import lastscene_device as dev
from lastscene_device import Config, RemoteCamera

TOKEN = "t" * 24


class FakeCamera:
    def __init__(self):
        self.photos = 0

    def capture_jpeg(self):
        self.photos += 1
        return b"\xff\xd8plain"

    def frame(self):
        return "frame"


class FakeVision:
    """Finds the item when the words name "keys"."""

    def __init__(self):
        self.asked = []
        self.detector = SimpleNamespace(name="yolo11s-int8")

    def photo_for(self, words):
        self.asked.append(words)
        hit = SimpleNamespace(label="keys", conf=0.81) if words and "keys" in words else None
        return "frame", hit, 1 if hit else 4


@pytest.fixture
def faker(monkeypatch):
    """camera_server on a free localhost port; yields (base url, service)."""
    monkeypatch.setattr(dev, "jpeg_bytes", lambda frame: b"\xff\xd8checked")
    service = camera_server.CameraService(Config.local({}), camera=FakeCamera(), vision=None)
    server = camera_server.serve(service, TOKEN, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", service
    server.shutdown()


def test_a_photo_takes_one_request(faker):
    url, service = faker
    calls = []
    http = httpx.Client(event_hooks={"request": [calls.append]})
    assert RemoteCamera(url, TOKEN, http=http).capture_jpeg() == b"\xff\xd8plain"
    assert len(calls) == 1 and service.camera.photos == 1


def test_the_words_go_along_so_the_camera_pi_can_look_for_the_item(faker):
    url, service = faker
    service.vision = FakeVision()
    cam = RemoteCamera(url, TOKEN)
    assert cam.capture_jpeg("I'm putting my keys on the counter") == b"\xff\xd8checked"
    assert service.vision.asked == ["I'm putting my keys on the counter"]
    resp = httpx.get(f"{url}/photo", params={"for": "keys"}, headers={"X-Camera-Token": TOKEN})
    assert resp.headers["x-item"] == "keys" and resp.headers["x-confidence"] == "0.81"


def test_the_token_is_required(faker):
    url, _ = faker
    with pytest.raises(httpx.HTTPStatusError) as refused:
        RemoteCamera(url, "x" * 24).capture_jpeg()
    assert refused.value.response.status_code == 401
    assert httpx.get(f"{url}/photo").status_code == 401


def test_health_and_unknown_paths(faker):
    url, service = faker
    assert httpx.get(f"{url}/health", headers={"X-Camera-Token": TOKEN}).json() == {"camera": "FakeCamera", "vision": None,
                                                                                 "passive": None}
    assert httpx.get(f"{url}/video", headers={"X-Camera-Token": TOKEN}).status_code == 404


def test_a_camera_failure_is_a_503_not_a_crash(faker):
    url, service = faker

    def broken():
        raise RuntimeError("unplugged")

    service.camera.capture_jpeg = broken
    assert httpx.get(f"{url}/photo", headers={"X-Camera-Token": TOKEN}).status_code == 503
    service.camera.capture_jpeg = lambda: b"\xff\xd8back"
    assert httpx.get(f"{url}/photo", headers={"X-Camera-Token": TOKEN}).content == b"\xff\xd8back"


def test_a_short_token_refuses_to_start():
    with pytest.raises(SystemExit):
        camera_server.serve(camera_server.CameraService(Config.local({}), camera=FakeCamera()), "short", "127.0.0.1", 0)


def test_config_and_camera_choice():
    cfg = Config.local({"CAMERA": "remote", "CAMERA_URL": "http://faker:8765/", "CAMERA_TOKEN": TOKEN})
    cam = dev.make_camera(cfg)
    assert isinstance(cam, RemoteCamera) and cam.url == "http://faker:8765" and cam.checks_items
    with pytest.raises(ValueError):
        dev.make_camera(Config.local({"CAMERA": "remote"}))
    with pytest.raises(RuntimeError):
        cam.frame()  # no live frames over the link


def test_the_audio_pi_asks_with_the_words_and_beeps_if_the_camera_pi_is_down(monkeypatch):
    beeps = []
    monkeypatch.setattr(dev, "play_wav", lambda device, data: beeps.append(data))
    d = dev.Device.__new__(dev.Device)  # no GPIO: only the photo step
    d.cfg = Config.local({"CAMERA": "remote", "CAMERA_URL": "http://faker:8765", "CAMERA_TOKEN": TOKEN})
    d.vision = None
    asked = []
    d.camera = RemoteCamera("http://faker:8765", TOKEN, http=httpx.Client(transport=httpx.MockTransport(
        lambda req: asked.append(req.url.params.get("for")) or httpx.Response(200, content=b"\xff\xd8ok",
                                                                              headers={"content-type": "image/jpeg"}))))
    assert d._photo("my keys are on the hook") == b"\xff\xd8ok" and asked == ["my keys are on the hook"]

    def down(req):
        raise httpx.ConnectError("no route to faker")

    d.camera = RemoteCamera("http://faker:8765", TOKEN, http=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(httpx.ConnectError):
        d._photo("my keys are on the hook")
    assert len(beeps) == 1  # the low "didn't work" beep


def test_passive_detection_needs_device_credentials_and_says_so(caplog):
    """Without them the background loop would stop at once; it stays off, with the reason in /health."""
    assert dev.passive_missing({}) == ["DEVICE_CLIENT_ID", "DEVICE_CLIENT_SECRET", "AUTH0_DOMAIN", "AUTH0_AUDIENCE"]
    assert dev.passive_missing({"DEVICE_CLIENT_ID": "id", "DEVICE_CLIENT_SECRET": "s", "AUTH0_DOMAIN": "d",
                                "AUTH0_AUDIENCE": "a"}) == []
    assert dev.passive_missing({"AUTH0_CLIENT_ID": "id", "AUTH0_CLIENT_SECRET": "s", "AUTH0_DOMAIN": "d",
                                "AUTH0_AUDIENCE": "a"}) == []
    assert dev.passive_missing({"LASTSCENE_DEVICE_TOKEN": "t"}) == []

    vision = dev.Vision.__new__(dev.Vision)  # no model needed for the preflight
    vision.cfg, vision.camera, vision.paused = Config.local({}), FakeCamera(), lambda: False
    vision.background = {"state": "off", "error": None}
    import os
    saved = {k: os.environ.pop(k) for k in ("DEVICE_CLIENT_ID", "AUTH0_CLIENT_ID", "LASTSCENE_DEVICE_TOKEN")
             if k in os.environ}
    try:
        assert vision.start_background() is None
    finally:
        os.environ.update(saved)
    assert vision.background["state"] == "off" and "DEVICE_CLIENT_ID" in vision.background["error"]
    assert "Passive item detection OFF" in caplog.text

    vision.detector = SimpleNamespace(name="yolo11s-int8")
    service = camera_server.CameraService(Config.local({}), camera=FakeCamera(), vision=vision)
    assert service.health()["passive"]["state"] == "off"


def test_the_stream_sends_live_frames_with_the_latest_boxes():
    """GET /stream: MJPEG frames, with what the detector last saw drawn on them (needs OpenCV)."""
    pytest.importorskip("cv2")
    import time

    import numpy as np

    class FrameCamera(FakeCamera):
        def frame(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    remembering = dev.Remembering(SimpleNamespace(detect=lambda image: [], name="yolo11s-int8"))
    remembering.last = (time.monotonic(), [SimpleNamespace(label="pin", conf=0.87, bbox=(100, 100, 200, 220))],
                        (480, 640))
    vision = SimpleNamespace(detector=remembering, background={"state": "running", "error": None})
    service = camera_server.CameraService(Config.local({}), camera=FrameCamera(), vision=vision)
    jpeg = service.stream_jpeg()
    assert jpeg[:2] == b"\xff\xd8"
    import cv2
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (480, 640, 3) and image[100:220, 100].mean() > 40  # the box's left edge is drawn

    server = camera_server.serve(service, TOKEN, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert httpx.get(f"{url}/stream").status_code == 401
        with httpx.stream("GET", f"{url}/stream", headers={"X-Camera-Token": TOKEN}, timeout=10) as resp:
            assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
            got = b""
            for chunk in resp.iter_bytes():
                got += chunk
                if got.count(b"--frame") >= 2:
                    break
        assert b"Content-Type: image/jpeg" in got and b"\xff\xd8" in got
    finally:
        server.shutdown()


def test_each_sighting_carries_a_cached_wifi_scan():
    """CachedLocation: the scan voice commands send, as an observation's "location", rescanned once a minute."""
    now = [0.0]
    scans = []

    def scan():
        scans.append(now[0])
        return {"wifi": '[{"bssid": "aa:bb:cc:dd:ee:ff", "signal": -60}]'}

    where = dev.CachedLocation(Config.local({}), every=60, clock=lambda: now[0], scan=scan)
    assert where() == {"wifi": [{"bssid": "aa:bb:cc:dd:ee:ff", "signal": -60}]}
    now[0] = 30
    where()
    assert scans == [0.0]  # cached
    now[0] = 61
    where()
    assert scans == [0.0, 61]
    fixed = dev.CachedLocation(Config.local({}), scan=lambda: {"lat": "29.6", "lng": "-82.3", "accuracy_m": "100"})
    assert fixed() == {"lat": 29.6, "lng": -82.3, "accuracy_m": 100.0}
    assert dev.CachedLocation(Config.local({}), scan=lambda: {})() is None


def test_a_torn_frame_is_recognised():
    """A frame cut short on the USB bus decodes with one row repeated to the bottom; real frames have noise."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    real = rng.integers(60, 200, size=(720, 1280, 3), dtype=np.uint8)
    torn = real.copy()
    torn[600:] = torn[599]  # the missing part: the last good row, repeated
    assert camera_server.torn(torn) and not camera_server.torn(real)
    wall = np.clip(np.full((720, 1280, 3), 200) + rng.normal(0, 2, size=(720, 1280, 3)), 0, 255).astype(np.uint8)
    assert not camera_server.torn(wall)  # a plain wall still has sensor noise


def test_viewers_share_one_camera_feed_and_torn_frames_are_skipped():
    """Two viewers read the camera once per frame between them; a torn frame is never sent."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import time

    rng = np.random.default_rng(2)

    class CountingCamera(FakeCamera):
        reads = 0

        def frame(self):
            CountingCamera.reads += 1
            image = rng.integers(60, 200, size=(240, 320, 3), dtype=np.uint8)
            if CountingCamera.reads % 3 == 0:  # every third frame comes out torn
                image[200:] = image[199]
            return image

    service = camera_server.CameraService(Config.local({}), camera=CountingCamera(), vision=None)
    server = camera_server.serve(service, TOKEN, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/stream"
    got = {0: 0, 1: 0}

    def watch(i):
        with httpx.stream("GET", url, headers={"X-Camera-Token": TOKEN}, timeout=10) as resp:
            seen = b""
            for chunk in resp.iter_bytes():
                seen += chunk
                got[i] = seen.count(b"--frame")
                if got[i] >= 6:
                    return

    try:
        started = time.monotonic()
        watchers = [threading.Thread(target=watch, args=(i,)) for i in (0, 1)]
        [w.start() for w in watchers]
        [w.join(15) for w in watchers]
        elapsed = time.monotonic() - started
        assert got[0] >= 6 and got[1] >= 6
        # One feed: about STREAM_FPS reads a second for both viewers together (plus the torn ones re-read).
        assert CountingCamera.reads <= (camera_server.STREAM_FPS * elapsed) * 1.6 + 4
    finally:
        server.shutdown()
