"""LastScene wearable client for a Raspberry Pi Zero 2 W.

Optional wake phrase, "Last Scene" (WAKE_WORD = path to its openWakeWord model; hey_jarvis is a
ready-made stand-in for testing): say it, then speak.
  -> takes a photo, beeps, records until you stop talking, POSTs /api/v1/memory/voice;
     the server decides from the words: a statement is saved, a question is answered aloud.

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
import array
import json
import logging
import math
import os
import struct
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
    wake_word: str = ""  # path to the "Last Scene" .onnx (or a stock name like hey_jarvis); empty = buttons only
    wake_threshold: float = 0.6  # tune per model: with hey_jarvis, "hey Travis" scores ~0.45, the phrase ~1.0
    wake_max_seconds: int = 10

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
            wake_word=env.get("WAKE_WORD", ""),
            wake_threshold=float(env.get("WAKE_THRESHOLD", 0.6)),
            wake_max_seconds=int(env.get("WAKE_MAX_SECONDS", 10)),
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

    def voice(self, wav: bytes, jpeg: bytes | None, where: dict | None = None) -> tuple[str, bytes | dict]:
        """Wake-phrase clip: ("ask", spoken answer audio) or ("remember", the saved memory)."""
        files = {"audio": ("voice.wav", wav, "audio/wav")}
        if jpeg is not None:
            files["image"] = ("scene.jpg", jpeg, "image/jpeg")
        resp = self._post("/api/v1/memory/voice", files, where)
        if resp.headers.get("x-intent") == "ask" or resp.headers.get("content-type", "").startswith("audio/"):
            return "ask", resp.content
        return "remember", resp.json()

    def ask(self, wav: bytes, where: dict | None = None) -> tuple[bytes, str]:
        resp = self._post("/api/v1/memory/query", {"audio": ("question.wav", wav, "audio/wav")}, where)
        return resp.content, unquote(resp.headers.get("x-answer", ""))


# ---------------------------------------------------------------- offline spool

class Spool:
    """Keeps memories that could not be uploaded and retries them later."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, wav: bytes, jpeg: bytes, where: dict | None = None, kind: str = "log") -> Path:
        """kind "log": a REMEMBER press; "voice": a wake-phrase clip (the server decides what it was)."""
        item = self.folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        tmp = item.with_suffix(".tmp")
        tmp.mkdir()
        (tmp / "audio.wav").write_bytes(wav)
        (tmp / "scene.jpg").write_bytes(jpeg)
        (tmp / "kind").write_text(kind)
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
                kind = (item / "kind").read_text() if (item / "kind").exists() else "log"
                wav, jpeg = (item / "audio.wav").read_bytes(), (item / "scene.jpg").read_bytes()
                if kind == "voice":
                    api.voice(wav, jpeg, where)  # a question asked offline gets no late answer; a statement is saved
                else:
                    api.log_memory(wav, jpeg, where)
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


# ---------------------------------------------------------------- wake phrase

RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms: the frame size openWakeWord expects
FRAME_BYTES = FRAME_SAMPLES * 2


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    """16-bit mono PCM -> a WAV file."""
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


def tone(freq: float = 880, ms: int = 120, volume: float = 0.25) -> bytes:
    """A short beep (WAV) that says "I'm listening" / "saved"."""
    n = RATE * ms // 1000
    fade = max(1, n // 10)
    samples = array.array("h", (
        int(32767 * volume * math.sin(2 * math.pi * freq * i / RATE) * min(1, i / fade, (n - i) / fade)) for i in range(n)
    ))
    return wav_bytes(samples.tobytes())


def rms(frame: bytes) -> float:
    samples = array.array("h", frame)
    return math.sqrt(sum(x * x for x in samples) / len(samples)) if samples else 0.0


class Endpointer:
    """Decides when someone has finished speaking, from loudness alone (no extra libraries).

    The noise floor is learnt from the quiet before speech; speech is anything clearly louder; the
    clip ends after `silence_s` of quiet once speech started, or at `max_s`. `timeout_s` without
    any speech means nothing was said."""

    def __init__(self, max_s: float = 10, silence_s: float = 0.9, timeout_s: float = 4, floor: float = 300.0):
        self.frame_s = FRAME_SAMPLES / RATE
        self.max_frames = int(max_s / self.frame_s)
        self.silence_frames = int(silence_s / self.frame_s)
        self.timeout_frames = int(timeout_s / self.frame_s)
        self.floor = floor  # a starting guess; follows the room while nobody speaks
        self.frames = 0
        self.quiet = 0
        self.spoke = False

    def feed(self, frame: bytes) -> str:
        """'listening', 'speaking', 'done' or 'nothing'."""
        level = rms(frame)
        self.frames += 1
        loud = level > max(self.floor * 2.5, 500)
        if not self.spoke:
            if loud:
                self.spoke = True
            else:
                self.floor = 0.9 * self.floor + 0.1 * level
                return "nothing" if self.frames >= self.timeout_frames else "listening"
        self.quiet = 0 if loud else self.quiet + 1
        if self.quiet >= self.silence_frames or self.frames >= self.max_frames:
            return "done"
        return "speaking"


class WakeListener:
    """openWakeWord on 80 ms frames. `model` is anything with predict(np.int16 frame) -> {name: score}."""

    def __init__(self, name: str, threshold: float = 0.5, cooldown_s: float = 2.0, model=None, clock=time.monotonic):
        if model is None:
            from openwakeword.model import Model  # pip install openwakeword

            model = Model(wakeword_models=[name], inference_framework="onnx")
        self.model = model
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._last = -1e9

    def heard(self, frame: bytes) -> bool:
        import numpy as np

        scores = self.model.predict(np.frombuffer(frame, dtype=np.int16))
        if max(scores.values(), default=0) < self.threshold or self.clock() - self._last < self.cooldown_s:
            return False
        self._last = self.clock()
        self.reset()
        return True

    def reset(self) -> None:
        """Forget the audio so far (after a trigger, or after our own speaker played something)."""
        if hasattr(self.model, "reset"):
            self.model.reset()


class MicStream:
    """One arecord for everything: the wake listener and recordings both read its 80 ms frames.
    (On a Pi, two programs can't open the same microphone.)"""

    def __init__(self, device: str, on_frame):
        self.device = device
        self.on_frame = on_frame
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            frame = self._proc.stdout.read(FRAME_BYTES)
            if len(frame) < FRAME_BYTES:
                log.error("Microphone stream ended (arecord exited %s)", self._proc.poll())
                return
            try:
                self.on_frame(frame)
            except Exception:
                log.exception("Audio frame handler failed")


class StreamRecorder:
    """Push-to-talk recording from the shared MicStream (used when the wake phrase is on)."""

    def __init__(self, max_seconds: int):
        self.max_frames = int(max_seconds * RATE / FRAME_SAMPLES)
        self.frames: list[bytes] = []
        self.active = False

    def start(self) -> None:
        self.frames, self.active = [], True

    def add(self, frame: bytes) -> None:
        if self.active and len(self.frames) < self.max_frames:
            self.frames.append(frame)

    def stop(self) -> bytes:
        self.active = False
        return wav_bytes(b"".join(self.frames))


class VoiceCapture:
    """After the wake phrase: collects frames until the Endpointer says the speaker has stopped."""

    def __init__(self):
        self.listening = False
        self._endpointer: Endpointer | None = None
        self._frames: list[bytes] = []
        self._preroll: list[bytes] = []
        self._done = threading.Event()
        self._said = False

    def listen(self, endpointer: Endpointer, timeout_s: float) -> bytes | None:
        """Blocks until speech ends; the PCM spoken, or None when nothing was said."""
        self._endpointer, self._frames, self._preroll, self._said = endpointer, [], [], False
        self._done.clear()
        self.listening = True
        self._done.wait(timeout_s)
        self.listening = False
        return b"".join(self._frames) if self._said and self._frames else None

    def add(self, frame: bytes) -> None:
        if not self.listening or self._endpointer is None:
            return
        was_quiet = not self._endpointer.spoke
        state = self._endpointer.feed(frame)
        if self._endpointer.spoke:
            if was_quiet:  # speech just started: keep ~240 ms before it, so soft first sounds aren't cut
                self._frames = self._preroll[-3:]
            self._frames.append(frame)
        else:
            self._preroll = (self._preroll + [frame])[-3:]
        if state == "done":
            self._said = True
            self._done.set()
        elif state == "nothing":
            self._done.set()


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
        self._active: tuple[str, Recorder | StreamRecorder, bytes | None] | None = None

        # Wake phrase: one shared microphone stream for the listener and the buttons.
        self.mic: MicStream | None = None
        self.voice = VoiceCapture()
        if cfg.wake_word:
            try:
                self.wake = WakeListener(cfg.wake_word, cfg.wake_threshold)
            except Exception:  # openwakeword or its model missing: keep the buttons working
                log.exception("Wake phrase %r unavailable; buttons only (see README step 3)", cfg.wake_word)
            else:
                self.stream_rec = StreamRecorder(cfg.max_record_seconds)
                self.mic = MicStream(cfg.audio_in, self._on_frame)

    def _on_frame(self, frame: bytes) -> None:
        """Every 80 ms of microphone audio, when the wake phrase is on."""
        if self.stream_rec.active:  # a button is held
            self.stream_rec.add(frame)
        elif self.voice.listening:  # after the wake phrase
            self.voice.add(frame)
        elif not self.busy.locked() and self.wake.heard(frame):
            threading.Thread(target=self._wake, daemon=True).start()

    def _wake(self) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            self.led.on()
            photo = self.camera.capture_jpeg()
            play_wav(self.cfg.audio_out, tone(880))  # "I'm listening" (before recording, so it isn't heard)
            pcm = self.voice.listen(Endpointer(max_s=self.cfg.wake_max_seconds), self.cfg.wake_max_seconds + 3)
            if pcm is None:
                log.info("Wake phrase heard, but nothing said after it")
                play_wav(self.cfg.audio_out, tone(440))
                return
            self.led.blink(on_time=0.1, off_time=0.1)
            wav = wav_bytes(pcm)
            where = whereabouts(self.cfg)
            try:
                intent, result = self.api.voice(wav, photo, where)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    log.warning("Server rejected voice clip: %s", exc.response.text)
                    play_wav(self.cfg.audio_out, tone(440))
                else:
                    self.spool.add(wav, photo, where, kind="voice")
                    log.warning("Offline; voice clip spooled for later")
                return
            if intent == "ask":
                self.led.on()
                play_wav(self.cfg.audio_out, result)
            else:
                log.info("Logged by voice: %s", result.get("transcript"))
                play_wav(self.cfg.audio_out, tone(1320, 90))  # "saved"
        except Exception:
            log.exception("Wake phrase handling failed")
        finally:
            self.led.off()
            self.wake.reset()  # don't let our own beeps or answer trigger it again
            self.busy.release()

    def _begin(self, mode: str) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            photo = self.camera.capture_jpeg() if mode == "remember" else None
            # With the wake phrase on, the mic is already open: record from the shared stream.
            rec = self.stream_rec if self.mic else Recorder(self.cfg.audio_in, self.cfg.max_record_seconds)
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
        if self.mic:
            self.mic.start()
            log.info("Listening for the wake phrase (%s)", self.cfg.wake_word)
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
