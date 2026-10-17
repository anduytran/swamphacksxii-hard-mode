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
    pairing_code: str = ""  # printed on the device; a caregiver enters it to claim the device
    location_source: str = "wifi"  # wifi | fixed | none
    fixed_lat: float | None = None  # used for LOCATION_SOURCE=fixed (and as a Wi-Fi fallback)
    fixed_lng: float | None = None
    wifi_interface: str = "wlan0"

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
            pairing_code=env.get("PAIRING_CODE", ""),
            location_source=env.get("LOCATION_SOURCE", "wifi"),
            fixed_lat=float(env["HOME_LAT"]) if env.get("HOME_LAT") else None,
            fixed_lng=float(env["HOME_LNG"]) if env.get("HOME_LNG") else None,
            wifi_interface=env.get("WIFI_INTERFACE", "wlan0"),
        )


# ---------------------------------------------------------------- location (Wi-Fi positioning)

def _nmcli_scan() -> list[dict]:
    """NetworkManager (Raspberry Pi OS Bookworm): works without root."""
    out = subprocess.run(
        ["nmcli", "-t", "-f", "BSSID,SIGNAL", "device", "wifi", "list", "--rescan", "auto"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout
    aps = []
    for line in out.splitlines():
        # BSSID colons are escaped as "\:" in terse mode: AA\:BB\:..\:FF:72
        bssid, _, signal = line.replace("\\:", "-").rpartition(":")
        if bssid.count("-") == 5 and signal.isdigit():
            # nmcli reports 0-100 %; roughly linear to -100..-50 dBm
            aps.append({"bssid": bssid.replace("-", ":").lower(), "signal": int(signal) // 2 - 100})
    return aps


def _iw_scan(interface: str) -> list[dict]:
    """Fallback without NetworkManager; needs CAP_NET_ADMIN (see the systemd unit)."""
    out = subprocess.run(["iw", "dev", interface, "scan"], capture_output=True, text=True, timeout=20, check=True).stdout
    aps, current = [], None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("BSS "):
            current = {"bssid": line[4:21].lower()}
            aps.append(current)
        elif line.startswith("signal:") and current is not None:
            current["signal"] = int(float(line.split()[1]))
    return [a for a in aps if "signal" in a]


def scan_wifi(interface: str = "wlan0") -> list[dict]:
    """Nearby access points, strongest first. The server turns these into a position;
    nothing here leaves the device except BSSIDs and signal strengths."""
    for scan in (_nmcli_scan, lambda: _iw_scan(interface)):
        try:
            aps = scan()
        except (OSError, subprocess.SubprocessError):
            continue
        if aps:
            seen = {}
            for a in aps:  # one entry per BSSID (strongest)
                if a["bssid"] not in seen or a["signal"] > seen[a["bssid"]]["signal"]:
                    seen[a["bssid"]] = a
            return sorted(seen.values(), key=lambda a: a["signal"], reverse=True)[:15]
    return []


def whereabouts(cfg: Config) -> dict:
    """Form fields describing where the wearer is, per LOCATION_SOURCE."""
    fields: dict[str, str] = {}
    if cfg.location_source == "wifi":
        if aps := scan_wifi(cfg.wifi_interface):
            fields["wifi"] = json.dumps(aps)
    if cfg.location_source in ("wifi", "fixed") and cfg.fixed_lat is not None and cfg.fixed_lng is not None:
        # The server prefers the Wi-Fi fix and uses this only when it can't get one.
        fields |= {"lat": str(cfg.fixed_lat), "lng": str(cfg.fixed_lng), "accuracy_m": "100"}
    return fields


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

    def _post(self, path: str, files: dict, data: dict | None = None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.token()}"}
        if self.cfg.pairing_code:
            headers["X-Pairing-Code"] = self.cfg.pairing_code
        for attempt in (1, 2):
            resp = self.http.post(f"{self.cfg.api_url}{path}", headers=headers, files=files, data=data or {})
            if resp.status_code == 401 and attempt == 1:
                self._token = None  # revoked or rotated: fetch a fresh one once
                headers["Authorization"] = f"Bearer {self.token()}"
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def log_memory(self, wav: bytes, jpeg: bytes, where: dict | None = None) -> dict:
        return self._post(
            "/api/v1/memory/log",
            {"audio": ("memory.wav", wav, "audio/wav"), "image": ("scene.jpg", jpeg, "image/jpeg")},
            where,
        ).json()

    def ask(self, wav: bytes, where: dict | None = None) -> tuple[bytes, str]:
        resp = self._post("/api/v1/memory/query", {"audio": ("question.wav", wav, "audio/wav")}, where)
        return resp.content, unquote(resp.headers.get("x-answer", ""))


# ---------------------------------------------------------------- offline spool

class Spool:
    """Keeps memories that could not be uploaded and retries them later."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, wav: bytes, jpeg: bytes, where: dict | None = None) -> Path:
        item = self.folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        tmp = item.with_suffix(".tmp")
        tmp.mkdir()
        (tmp / "audio.wav").write_bytes(wav)
        (tmp / "scene.jpg").write_bytes(jpeg)
        (tmp / "where.json").write_text(json.dumps(where or {}))  # where it happened, not where it's sent from
        tmp.rename(item)  # atomic: a half-written item is never retried
        return item

    def pending(self) -> list[Path]:
        return sorted(p for p in self.folder.iterdir() if p.is_dir() and p.suffix != ".tmp")

    def flush(self, api: ApiClient) -> int:
        sent = 0
        for item in self.pending():
            try:
                where_file = item / "where.json"
                where = json.loads(where_file.read_text()) if where_file.exists() else None
                api.log_memory((item / "audio.wav").read_bytes(), (item / "scene.jpg").read_bytes(), where)
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
            where = whereabouts(self.cfg)  # a Wi-Fi scan takes a second or two
            if mode == "remember":
                try:
                    result = self.api.log_memory(wav, photo, where)
                    log.info("Logged: %s", result.get("transcript"))
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                        log.warning("Server rejected memory: %s", exc.response.text)
                    else:
                        self.spool.add(wav, photo, where)
                        log.warning("Offline; memory spooled for later")
            else:
                audio, answer = self.api.ask(wav, where)
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
    parser.add_argument("--where", action="store_true", help="print the location fields this device would send")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.from_env()
    api = ApiClient(cfg)
    if args.where:
        print(json.dumps(whereabouts(cfg), indent=2))
    elif args.check:
        api.token()
        print("Auth0 token OK;", "API", httpx.get(f"{cfg.api_url}/healthz", timeout=10).json())
    elif args.log_file:
        wav, jpeg = (Path(p).read_bytes() for p in args.log_file)
        print(json.dumps(api.log_memory(wav, jpeg, whereabouts(cfg)), indent=2))
    elif args.ask_file:
        audio, answer = api.ask(Path(args.ask_file).read_bytes(), whereabouts(cfg))
        Path("answer.wav").write_bytes(audio)
        print("Answer:", answer, "(saved answer.wav)")
    else:
        Device(cfg).run()


if __name__ == "__main__":
    main()
