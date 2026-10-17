"""LastScene wearable client for a Raspberry Pi Zero 2 W.

Two push-to-talk buttons:
  * REMEMBER (hold, speak, release): "I'm putting my keys on the counter"
      -> takes a photo, records while held, POSTs /api/v1/memory/log
  * ASK (hold, speak, release): "Where are my keys?"
      -> records while held, POSTs /api/v1/memory/query, plays the spoken answer

Authenticates with Auth0 Machine-to-Machine (client credentials) and caches the
token until shortly before it expires. Memories that fail to upload (no Wi-Fi)
are kept in a spool folder and retried in the background.

Config comes from environment variables (see device.env.example).
Run `python lastscene_device.py --help` for test modes that need no hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import httpx

log = logging.getLogger("lastscene")


# ---------------------------------------------------------------- config

@dataclass
class Config:
    api_url: str
    auth0_domain: str
    auth0_audience: str
    client_id: str
    client_secret: str
    remember_pin: int = 17   # physical pin 11
    ask_pin: int = 27        # physical pin 13
    led_pin: int = 22        # physical pin 15
    audio_in: str = "default"
    audio_out: str = "default"
    max_record_seconds: int = 15
    spool_dir: Path = Path("/var/lib/lastscene/spool")

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        return cls(
            api_url=env.get("API_URL", "http://localhost:8000").rstrip("/"),
            auth0_domain=env["AUTH0_DOMAIN"],
            auth0_audience=env["AUTH0_AUDIENCE"],
            client_id=env["DEVICE_CLIENT_ID"],
            client_secret=env["DEVICE_CLIENT_SECRET"],
            remember_pin=int(env.get("REMEMBER_PIN", 17)),
            ask_pin=int(env.get("ASK_PIN", 27)),
            led_pin=int(env.get("LED_PIN", 22)),
            audio_in=env.get("AUDIO_IN_DEVICE", "default"),
            audio_out=env.get("AUDIO_OUT_DEVICE", "default"),
            max_record_seconds=int(env.get("MAX_RECORD_SECONDS", 15)),
            spool_dir=Path(env.get("SPOOL_DIR", "/var/lib/lastscene/spool")),
        )


# ---------------------------------------------------------------- API client

class ApiClient:
    """Talks to the LastScene API with a cached Auth0 M2M token."""

    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            resp = self.http.post(
                f"https://{self.cfg.auth0_domain}/oauth/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": self.cfg.client_id,
                    "client_secret": self.cfg.client_secret,
                    "audience": self.cfg.auth0_audience,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            self._token = body["access_token"]
            self._expires_at = time.time() + int(body.get("expires_in", 3600))
            return self._token

    def _post(self, path: str, files: dict) -> httpx.Response:
        for attempt in (1, 2):
            resp = self.http.post(
                f"{self.cfg.api_url}{path}",
                headers={"Authorization": f"Bearer {self.token()}"},
                files=files,
            )
            if resp.status_code == 401 and attempt == 1:
                self._token = None  # revoked or rotated: fetch a fresh one once
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def log_memory(self, wav: bytes, jpeg: bytes) -> dict:
        return self._post(
            "/api/v1/memory/log",
            {"audio": ("memory.wav", wav, "audio/wav"), "image": ("scene.jpg", jpeg, "image/jpeg")},
        ).json()

    def ask(self, wav: bytes) -> tuple[bytes, str]:
        resp = self._post("/api/v1/memory/query", {"audio": ("question.wav", wav, "audio/wav")})
        return resp.content, unquote(resp.headers.get("x-answer", ""))


# ---------------------------------------------------------------- offline spool

class Spool:
    """Keeps memories that could not be uploaded and retries them later."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, wav: bytes, jpeg: bytes) -> Path:
        item = self.folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        tmp = item.with_suffix(".tmp")
        tmp.mkdir()
        (tmp / "audio.wav").write_bytes(wav)
        (tmp / "scene.jpg").write_bytes(jpeg)
        tmp.rename(item)  # atomic: a half-written item is never retried
        return item

    def pending(self) -> list[Path]:
        return sorted(p for p in self.folder.iterdir() if p.is_dir() and p.suffix != ".tmp")

    def flush(self, api: ApiClient) -> int:
        sent = 0
        for item in self.pending():
            try:
                api.log_memory((item / "audio.wav").read_bytes(), (item / "scene.jpg").read_bytes())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 422:  # e.g. no speech: retrying won't help
                    log.warning("Dropping spooled memory %s: %s", item.name, exc.response.text)
                    shutil.rmtree(item)
                    continue
                break
            except httpx.TransportError:
                break  # still offline; try again later
            shutil.rmtree(item)
            sent += 1
        return sent


# ---------------------------------------------------------------- hardware

class Recorder:
    """Push-to-talk recording via ALSA's arecord (16 kHz mono WAV, Whisper-friendly)."""

    def __init__(self, device: str, max_seconds: int):
        self.device = device
        self.max_seconds = max_seconds
        self._proc: subprocess.Popen | None = None
        self._path: Path | None = None

    def start(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._path = Path(path)
        self._proc = subprocess.Popen(
            ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", "16000", "-c", "1",
             "-d", str(self.max_seconds), "-t", "wav", path]
        )

    def stop(self) -> bytes:
        assert self._proc and self._path
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)  # arecord finalises the WAV header on SIGINT
        self._proc.wait(timeout=5)
        data = self._path.read_bytes()
        self._path.unlink(missing_ok=True)
        return data


def play_wav(device: str, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data)
        f.flush()
        subprocess.run(["aplay", "-q", "-D", device, f.name], check=False)


class Camera:
    def __init__(self):
        from picamera2 import Picamera2  # apt: python3-picamera2

        self.cam = Picamera2()
        # Small JPEGs upload fast and stay well under the API's 2 MB limit.
        self.cam.configure(self.cam.create_still_configuration(main={"size": (1280, 960)}))
        self.cam.start()
        time.sleep(1)  # let auto-exposure settle once at boot

    def capture_jpeg(self) -> bytes:
        import io

        buf = io.BytesIO()
        self.cam.capture_file(buf, format="jpeg")
        return buf.getvalue()


# ---------------------------------------------------------------- device loop

class Device:
    def __init__(self, cfg: Config):
        from gpiozero import LED, Button  # preinstalled on Raspberry Pi OS

        self.cfg = cfg
        self.api = ApiClient(cfg)
        self.spool = Spool(cfg.spool_dir)
        self.camera = Camera()
        self.led = LED(cfg.led_pin)
        self.remember = Button(cfg.remember_pin, pull_up=True, bounce_time=0.05)
        self.ask = Button(cfg.ask_pin, pull_up=True, bounce_time=0.05)
        self.busy = threading.Lock()  # one interaction at a time

        self.remember.when_pressed = lambda: self._begin("remember")
        self.remember.when_released = lambda: self._end("remember")
        self.ask.when_pressed = lambda: self._begin("ask")
        self.ask.when_released = lambda: self._end("ask")
        self._active: tuple[str, Recorder, bytes | None] | None = None

    def _begin(self, mode: str) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            photo = self.camera.capture_jpeg() if mode == "remember" else None
            rec = Recorder(self.cfg.audio_in, self.cfg.max_record_seconds)
            rec.start()
            self.led.on()
            self._active = (mode, rec, photo)
        except Exception:
            log.exception("Could not start %s", mode)
            self.busy.release()

    def _end(self, mode: str) -> None:
        if not self._active or self._active[0] != mode:
            return
        _, rec, photo = self._active
        self._active = None
        wav = rec.stop()
        self.led.blink(on_time=0.1, off_time=0.1)
        threading.Thread(target=self._send, args=(mode, wav, photo), daemon=True).start()

    def _send(self, mode: str, wav: bytes, photo: bytes | None) -> None:
        try:
            if mode == "remember":
                try:
                    result = self.api.log_memory(wav, photo)
                    log.info("Logged: %s", result.get("transcript"))
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                        log.warning("Server rejected memory: %s", exc.response.text)
                    else:
                        self.spool.add(wav, photo)
                        log.warning("Offline; memory spooled for later")
            else:
                audio, answer = self.api.ask(wav)
                log.info("Answer: %s", answer)
                self.led.on()
                play_wav(self.cfg.audio_out, audio)
        except Exception:
            log.exception("%s failed", mode)
        finally:
            self.led.off()
            self.busy.release()

    def run(self) -> None:
        log.info("LastScene ready: hold REMEMBER (GPIO%d) or ASK (GPIO%d)", self.cfg.remember_pin, self.cfg.ask_pin)
        while True:
            time.sleep(60)
            try:
                if sent := self.spool.flush(self.api):
                    log.info("Uploaded %d spooled memories", sent)
            except Exception:
                log.exception("Spool flush failed")


# ---------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-file", nargs=2, metavar=("WAV", "JPEG"), help="upload a memory from files, no hardware")
    parser.add_argument("--ask-file", metavar="WAV", help="ask a question from a file and save answer.wav, no hardware")
    parser.add_argument("--check", action="store_true", help="check the Auth0 token and API reachability")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.from_env()
    api = ApiClient(cfg)
    if args.check:
        api.token()
        print("Auth0 token OK;", "API", httpx.get(f"{cfg.api_url}/healthz", timeout=10).json())
    elif args.log_file:
        wav, jpeg = (Path(p).read_bytes() for p in args.log_file)
        print(json.dumps(api.log_memory(wav, jpeg), indent=2))
    elif args.ask_file:
        audio, answer = api.ask(Path(args.ask_file).read_bytes())
        Path("answer.wav").write_bytes(audio)
        print("Answer:", answer, "(saved answer.wav)")
    else:
        Device(cfg).run()


if __name__ == "__main__":
    main()
