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

Live conversation (LIVE=true, optional): the wake phrase, or holding ASK quietly for
LIVE_HOLD_SECONDS, starts a back-and-forth with Gemini Live using a one-use token from
/api/v1/live/token. Gemini looks memories up through /api/v1/memory/recall and saves new
ones (with a photo) through /api/v1/memory/log. It ends after LIVE_SILENCE_SECONDS of quiet,
LIVE_MAX_SECONDS, or any button press; if it can't start or fails, the one-shot flow above runs.

Speech-to-text on the device (STT_ENGINE=moonshine, or vosk): the wearable sends the words,
not the recording, so the audio never leaves it. If the local model is missing or hears nothing,
the clip goes to the server's speech-to-text instead (STT_FALLBACK=false turns that off).

Authenticates with Auth0 Machine-to-Machine (client credentials) and caches the
token until shortly before it expires. Memories that fail to upload (no Wi-Fi)
are kept in a spool folder and retried in the background.

Config comes from environment variables (see device.env.example).
Run `python lastscene_device.py --help` for test modes that need no hardware.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import io
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
import wave
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
    stt_engine: str = "server"  # moonshine | vosk | server (send the audio; the server transcribes)
    stt_model: str = ""  # model folder; empty = the default under /opt/lastscene/stt
    stt_fallback: bool = True  # local model missing or heard nothing: send the audio instead
    live: bool = False  # the wake phrase (or a quiet long press of ASK) starts a Gemini Live conversation
    live_max_seconds: int = 90
    live_silence_seconds: float = 8  # a conversation ends after this long with nobody talking
    live_hold_seconds: float = 2  # hold ASK this long without speaking to start one

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
            stt_engine=env.get("STT_ENGINE", "server").lower(),
            stt_model=env.get("STT_MODEL", ""),
            stt_fallback=env.get("STT_FALLBACK", "true").lower() not in ("0", "false", "no"),
            live=env.get("LIVE", "false").lower() in ("1", "true", "yes"),
            live_max_seconds=int(env.get("LIVE_MAX_SECONDS", 90)),
            live_silence_seconds=float(env.get("LIVE_SILENCE_SECONDS", 8)),
            live_hold_seconds=float(env.get("LIVE_HOLD_SECONDS", 2)),
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

    def _post(self, path: str, files: dict | None = None, data: dict | None = None, json: dict | None = None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.token()}"}
        if self.cfg.pairing_code:
            headers["X-Pairing-Code"] = self.cfg.pairing_code
        body = {"json": json} if json is not None else {"files": files or {}, "data": data or {}}
        for attempt in (1, 2):
            resp = self.http.post(f"{self.cfg.api_url}{path}", headers=headers, **body)
            if resp.status_code == 401 and attempt == 1:
                self._token = None  # revoked or rotated: fetch a fresh one once
                headers["Authorization"] = f"Bearer {self.token()}"
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    @staticmethod
    def _speech(name: str, wav: bytes | None, transcript: str | None, where: dict | None) -> tuple[dict, dict]:
        """The words if the device transcribed them (no audio sent), else the recording."""
        data = dict(where or {})
        if transcript:
            data["transcript"] = transcript
            return {}, data
        return {"audio": (name, wav, "audio/wav")}, data

    def log_memory(self, wav: bytes | None, jpeg: bytes, where: dict | None = None, transcript: str | None = None) -> dict:
        files, data = self._speech("memory.wav", wav, transcript, where)
        files["image"] = ("scene.jpg", jpeg, "image/jpeg")
        return self._post("/api/v1/memory/log", files, data).json()

    def voice(self, wav: bytes | None, jpeg: bytes | None, where: dict | None = None,
              transcript: str | None = None) -> tuple[str, bytes | dict]:
        """Wake-phrase clip: ("ask", spoken answer audio) or ("remember", the saved memory)."""
        files, data = self._speech("voice.wav", wav, transcript, where)
        if jpeg is not None:
            files["image"] = ("scene.jpg", jpeg, "image/jpeg")
        resp = self._post("/api/v1/memory/voice", files, data)
        if resp.headers.get("x-intent") == "ask" or resp.headers.get("content-type", "").startswith("audio/"):
            return "ask", resp.content
        return "remember", resp.json()

    def ask(self, wav: bytes | None, where: dict | None = None, transcript: str | None = None) -> tuple[bytes, str]:
        files, data = self._speech("question.wav", wav, transcript, where)
        resp = self._post("/api/v1/memory/query", files, data)
        return resp.content, unquote(resp.headers.get("x-answer", ""))

    def live_token(self) -> dict:
        """A one-use Gemini Live token: {"token": "auth_tokens/...", "model", "api_version", ...}.
        404 = live mode is off on the server; 429 = too many sessions this hour."""
        return self._post("/api/v1/live/token").json()

    def recall(self, question: str) -> str:
        """The saved memories about a question, as short text facts (no answer is generated)."""
        return self._post("/api/v1/memory/recall", json={"question": question}).json()["text"]


# ---------------------------------------------------------------- on-device speech-to-text

STT_DIR = Path("/opt/lastscene/stt")
STT_DEFAULT_MODELS = {"moonshine": STT_DIR / "moonshine-tiny", "vosk": STT_DIR / "vosk-model-small-en-us-0.15"}


def wav_pcm(wav: bytes) -> bytes:
    """16-bit mono PCM out of a WAV (what arecord and wav_bytes produce: 16 kHz mono)."""
    with wave.open(io.BytesIO(wav)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != RATE:
            raise ValueError(f"need 16-bit mono {RATE} Hz audio")
        return w.readframes(w.getnframes())


class LocalSTT:
    """Moonshine (default: tiny, quantized ONNX, ~28 MB) or Vosk (small-en, ~68 MB) on the Pi.

    Measured on a Mac, one thread: Moonshine tiny 40-90 ms per 1.5-2.6 s clip with punctuation kept
    ("Where did I leave my glasses?"), Vosk ~0.5 s and lower case. On the Pi: run --transcribe.
    """

    def __init__(self, engine: str, model_dir: str | Path = ""):
        self.engine = engine
        self.model_dir = Path(model_dir) if model_dir else STT_DEFAULT_MODELS[engine]
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"{engine} model folder {self.model_dir} not found (README step 3)")
        if engine == "moonshine":
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer  # pip install --no-deps useful-moonshine-onnx

            size = "base" if "base" in self.model_dir.name else "tiny"
            self._model = MoonshineOnnxModel(models_dir=str(self.model_dir), model_name=size)
            self._tokenizer = load_tokenizer()
        elif engine == "vosk":
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            self._model = Model(str(self.model_dir))
        else:
            raise ValueError(f"unknown STT_ENGINE {engine!r} (moonshine, vosk or server)")

    def transcribe(self, wav: bytes) -> str:
        pcm = wav_pcm(wav)
        seconds = len(pcm) / 2 / RATE
        if seconds < 0.15:
            return ""
        if self.engine == "moonshine":
            import numpy as np

            audio = (np.frombuffer(pcm[: 60 * RATE * 2], np.int16).astype(np.float32) / 32768.0)[None, :]
            return self._tokenizer.decode_batch(self._model.generate(audio))[0].strip()
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, RATE)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()


# ---------------------------------------------------------------- offline spool

class Spool:
    """Keeps memories that could not be uploaded and retries them later."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, wav: bytes | None, jpeg: bytes, where: dict | None = None, kind: str = "log",
            transcript: str | None = None) -> Path:
        """kind "log": a REMEMBER press; "voice": a wake-phrase clip (the server decides what it was).
        With a device transcript only the words are kept, not the recording."""
        item = self.folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        tmp = item.with_suffix(".tmp")
        tmp.mkdir()
        if transcript:
            (tmp / "transcript.txt").write_text(transcript)
        else:
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
                said = item / "transcript.txt"
                transcript = said.read_text() if said.exists() else None
                wav = None if transcript else (item / "audio.wav").read_bytes()
                jpeg = (item / "scene.jpg").read_bytes()
                if kind == "voice":
                    # a question asked offline gets no late answer; a statement is saved
                    api.voice(wav, jpeg, where, transcript)
                else:
                    api.log_memory(wav, jpeg, where, transcript)
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


# ---------------------------------------------------------------- live conversation (LIVE=true)

LIVE_RATE = 24000  # Gemini Live speaks 16-bit mono PCM at 24 kHz
LIVE_MIC_MIME = f"audio/pcm;rate={RATE}"


def gemini_live_connect(token: dict):
    """Gemini Live with the server's one-use token (pip install google-genai). The model, persona,
    tools and voice are locked into the token by the server; this only picks audio out."""
    from google import genai
    from google.genai import types

    # google-genai 2.31 (live.py): ephemeral tokens work with v1alpha only; the server says which.
    client = genai.Client(api_key=token["token"],
                          http_options=types.HttpOptions(api_version=token.get("api_version", "v1alpha")))
    return client.aio.live.connect(model=token["model"],
                                   config=types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO]))


class PcmPlayer:
    """Streams Gemini's 24 kHz PCM through one aplay. busy() is true while it is (by the bytes
    written) still playing, so the mic can stay muted: with no echo cancellation on the Pi,
    Gemini would otherwise hear itself and interrupt its own answer."""

    def __init__(self, device: str, rate: int = LIVE_RATE, tail_s: float = 0.3, clock=time.monotonic):
        self.device, self.rate, self.tail_s, self.clock = device, rate, tail_s, clock
        self._proc: subprocess.Popen | None = None
        self._until = 0.0

    def play(self, pcm: bytes) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                ["aplay", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE", "-r", str(self.rate), "-c", "1"],
                stdin=subprocess.PIPE,
            )
        self._until = max(self.clock(), self._until) + len(pcm) / 2 / self.rate
        self._proc.stdin.write(pcm)
        self._proc.stdin.flush()

    def busy(self) -> bool:
        return self.clock() < self._until + self.tail_s

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=max(0.0, self._until - self.clock()) + 2)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
        self._proc = None


class LiveSession:
    """One spoken back-and-forth with Gemini Live, after the wake phrase.

    `connect()` gives an async context manager for a session with send_realtime_input(audio=...),
    send_tool_response(function_responses=...) and receive() (google-genai's AsyncSession, or a
    fake in tests). `tools` maps tool names to blocking functions, run off the event loop. The mic
    thread calls feed() with 80 ms frames; nothing is sent until run() has connected, and nothing
    while Gemini's answer is playing. Ends on `silence_s` with nobody talking, at `max_s`, on
    stop() (a button) or when Gemini closes the session; errors propagate so the caller can fall back.
    """

    def __init__(self, connect, tools: dict, player, silence_s: float = 8, max_s: float = 90,
                 on_ready=None, clock=time.monotonic, tick_s: float = 0.1):
        self.connect, self.tools, self.player = connect, tools, player
        self.silence_s, self.max_s, self.on_ready = silence_s, max_s, on_ready
        self.clock, self.tick_s = clock, tick_s
        self.active = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frames: asyncio.Queue | None = None
        self._stop: asyncio.Event | None = None
        self._last = 0.0  # last time anyone spoke (or a tool ran)
        self._floor = 300.0  # the room's loudness, learnt while nobody speaks (as in Endpointer)

    def feed(self, frame: bytes) -> None:
        """From the mic thread."""
        if not self.active or self.player.busy():
            return
        level = rms(frame)
        if level > max(self._floor * 2.5, 500):
            self._last = self.clock()
        else:
            self._floor = 0.9 * self._floor + 0.1 * level
        try:
            self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)
        except RuntimeError:  # the loop just closed
            pass

    def stop(self) -> None:
        """From any thread: end the conversation (a button was pressed)."""
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass

    async def run(self) -> str:
        """Why it ended: "silence", "max", "button" or "closed"."""
        self._loop = asyncio.get_running_loop()
        self._frames, self._stop = asyncio.Queue(), asyncio.Event()
        async with self.connect() as session:
            if self.on_ready:
                await asyncio.to_thread(self.on_ready)  # e.g. the "go ahead" chime, before the mic opens
            started = self._last = self.clock()
            self.active = True
            try:
                tasks = [asyncio.create_task(c) for c in (self._send(session), self._receive(session), self._watch(started))]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for t in done:
                    if t.exception():
                        raise t.exception()
                return next(iter(done)).result()
            finally:
                self.active = False

    async def _send(self, session) -> None:
        while True:
            frame = await self._frames.get()
            await session.send_realtime_input(audio={"data": frame, "mime_type": LIVE_MIC_MIME})

    async def _receive(self, session) -> str:
        while True:
            got = False
            async for msg in session.receive():  # one model turn per loop
                got = True
                if audio := getattr(msg, "data", None):
                    self._last = self.clock()
                    await asyncio.to_thread(self.player.play, audio)
                call = getattr(msg, "tool_call", None)
                if call and call.function_calls:
                    responses = [await self._call(fc) for fc in call.function_calls]
                    await session.send_tool_response(function_responses=responses)
                content = getattr(msg, "server_content", None)
                for side in ("input_transcription", "output_transcription"):
                    if text := getattr(getattr(content, side, None), "text", None):
                        log.info("Live %s: %s", "heard" if side.startswith("input") else "said", text)
            if not got:
                return "closed"

    async def _call(self, fc) -> dict:
        self._last = self.clock()
        log.info("Live tool call %s(%s)", fc.name, fc.args)
        try:
            fn = self.tools[fc.name]
            response = {"result": await asyncio.to_thread(fn, **(fc.args or {}))}
        except Exception:
            log.exception("Live tool %s failed", fc.name)
            response = {"error": "That didn't work just now; say so kindly and don't guess."}
        self._last = self.clock()
        return {"id": fc.id, "name": fc.name, "response": response}

    async def _watch(self, started: float) -> str:
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), self.tick_s)
                return "button"
            except asyncio.TimeoutError:
                pass
            now = self.clock()
            if now - started >= self.max_s:
                return "max"
            if self.player.busy():
                self._last = now
            elif now - self._last >= self.silence_s:
                return "silence"


def held_quietly(wav: bytes, seconds: float) -> bool:
    """A press of ASK held at least `seconds` with no speech in it: start a live conversation."""
    try:
        pcm = wav_pcm(wav)
    except Exception:
        return False
    if len(pcm) / 2 / RATE < seconds:
        return False
    ep = Endpointer(max_s=1e9, timeout_s=1e9)
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        ep.feed(pcm[i:i + FRAME_BYTES])
    return not ep.spoke


# ---------------------------------------------------------------- device loop

class NothingHeard(Exception):
    """Local speech-to-text heard nothing and STT_FALLBACK=false: send nothing."""


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
        self.stt: LocalSTT | None = None
        if cfg.stt_engine != "server":
            try:
                self.stt = LocalSTT(cfg.stt_engine, cfg.stt_model)
                log.info("Speech-to-text on the device: %s (%s)", cfg.stt_engine, self.stt.model_dir)
            except Exception:  # package or model missing: the server transcribes instead
                log.exception("On-device speech-to-text %r unavailable; %s", cfg.stt_engine,
                              "sending audio to the server" if cfg.stt_fallback else "nothing can be sent")

        self.remember.when_pressed = lambda: self._begin("remember")
        self.remember.when_released = lambda: self._end("remember")
        self.ask.when_pressed = lambda: self._begin("ask")
        self.ask.when_released = lambda: self._end("ask")
        self._active: tuple[str, Recorder | StreamRecorder, bytes | None] | None = None

        # Wake phrase (and live mode): one shared microphone stream for the listener and the buttons.
        self.mic: MicStream | None = None
        self.wake: WakeListener | None = None
        self.live: LiveSession | None = None  # set while a live conversation runs
        self.voice = VoiceCapture()
        if cfg.wake_word:
            try:
                self.wake = WakeListener(cfg.wake_word, cfg.wake_threshold)
            except Exception:  # openwakeword or its model missing: keep the buttons working
                log.exception("Wake phrase %r unavailable; buttons only (see README step 3)", cfg.wake_word)
        if self.wake or cfg.live:
            self.stream_rec = StreamRecorder(cfg.max_record_seconds)
            self.mic = MicStream(cfg.audio_in, self._on_frame)

    def _words(self, wav: bytes) -> str | None:
        """The transcript to send instead of the audio, or None to send the audio. Raises
        NothingHeard when only local transcription is allowed and it heard nothing."""
        if self.stt is not None:
            try:
                t0 = time.monotonic()
                text = self.stt.transcribe(wav)
                log.info("Heard (%s, %.2f s): %r", self.cfg.stt_engine, time.monotonic() - t0, text)
                if text:
                    return text
            except Exception:
                log.exception("On-device speech-to-text failed")
        if self.cfg.stt_engine != "server" and not self.cfg.stt_fallback:
            raise NothingHeard()
        return None

    def _on_frame(self, frame: bytes) -> None:
        """Every 80 ms of microphone audio, when the wake phrase (or live mode) is on."""
        if (live := self.live) is not None:  # a live conversation: the audio goes to Gemini
            live.feed(frame)
        elif self.stream_rec.active:  # a button is held
            self.stream_rec.add(frame)
        elif self.voice.listening:  # after the wake phrase
            self.voice.add(frame)
        elif self.wake and not self.busy.locked() and self.wake.heard(frame):
            threading.Thread(target=self._wake, daemon=True).start()

    def _wake(self) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            self.led.on()
            if self.cfg.live and self._converse():
                return
            photo = self.camera.capture_jpeg()
            play_wav(self.cfg.audio_out, tone(880))  # "I'm listening" (before recording, so it isn't heard)
            pcm = self.voice.listen(Endpointer(max_s=self.cfg.wake_max_seconds), self.cfg.wake_max_seconds + 3)
            if pcm is None:
                log.info("Wake phrase heard, but nothing said after it")
                play_wav(self.cfg.audio_out, tone(440))
                return
            self.led.blink(on_time=0.1, off_time=0.1)
            wav = wav_bytes(pcm)
            try:
                words = self._words(wav)
            except NothingHeard:
                play_wav(self.cfg.audio_out, tone(440))
                return
            where = whereabouts(self.cfg)
            try:
                intent, result = self.api.voice(wav, photo, where, words)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    log.warning("Server rejected voice clip: %s", exc.response.text)
                    play_wav(self.cfg.audio_out, tone(440))
                else:
                    self.spool.add(wav, photo, where, kind="voice", transcript=words)
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

    def _converse(self) -> bool:
        """A live conversation with Gemini (LIVE=true). False if it failed at any point (no
        token, live mode off on the server, quota, network): the caller then uses /memory/voice."""
        player = PcmPlayer(self.cfg.audio_out)
        try:
            token = self.api.live_token()
            session = LiveSession(
                lambda: gemini_live_connect(token),
                {"recall_memories": self.api.recall, "save_memory": self._save_spoken},
                player, silence_s=self.cfg.live_silence_seconds, max_s=self.cfg.live_max_seconds,
                on_ready=lambda: play_wav(self.cfg.audio_out, tone(660)),  # "go ahead, I'm listening"
            )
            self.live = session
            reason = asyncio.run(session.run())
            log.info("Live conversation ended (%s)", reason)
            player.close()
            play_wav(self.cfg.audio_out, tone(440, 90))  # "bye"
            return True
        except Exception:
            log.exception("Live conversation failed; using the one-shot voice flow")
            return False
        finally:
            self.live = None
            player.close()

    def _save_spoken(self, text: str) -> str:
        """The save_memory tool: a photo now, saved with the words, like a REMEMBER press."""
        photo = self.camera.capture_jpeg()
        where = whereabouts(self.cfg)
        try:
            self.api.log_memory(None, photo, where, transcript=text)
            return "Saved."
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            self.spool.add(None, photo, where, transcript=text)
            return "Saved on the wearable; it will upload when it's back online."

    def _begin(self, mode: str) -> None:
        if (live := self.live) is not None:  # any button ends a live conversation
            live.stop()
            return
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
            if mode == "ask" and self.cfg.live and held_quietly(wav, self.cfg.live_hold_seconds):
                if not self._converse():  # a quiet long press of ASK: a conversation instead
                    play_wav(self.cfg.audio_out, tone(440))
                return
            try:
                words = self._words(wav)
            except NothingHeard:
                play_wav(self.cfg.audio_out, tone(440))
                return
            where = whereabouts(self.cfg)  # a Wi-Fi scan takes a second or two
            if mode == "remember":
                try:
                    result = self.api.log_memory(wav, photo, where, words)
                    log.info("Logged: %s", result.get("transcript"))
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                        log.warning("Server rejected memory: %s", exc.response.text)
                    else:
                        self.spool.add(wav, photo, where, transcript=words)
                        log.warning("Offline; memory spooled for later")
            else:
                audio, answer = self.api.ask(wav, where, words)
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
        if self.wake:
            log.info("Listening for the wake phrase (%s)%s", self.cfg.wake_word, "; live conversations on" if self.cfg.live else "")
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
    parser.add_argument("--live-check", action="store_true",
                        help="get a live token and open (then close) a Gemini Live session; sends no audio")
    parser.add_argument("--transcribe", metavar="WAV", nargs="+",
                        help="transcribe 16 kHz mono WAVs with STT_ENGINE on this device and time it; no network")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.transcribe:  # needs no Auth0 settings
        engine = os.environ.get("STT_ENGINE", "moonshine").lower()
        t0 = time.monotonic()
        stt = LocalSTT("moonshine" if engine == "server" else engine, os.environ.get("STT_MODEL", ""))
        print(f"{stt.engine} loaded from {stt.model_dir} in {time.monotonic() - t0:.1f} s")
        for path in args.transcribe:
            wav = Path(path).read_bytes()
            t0 = time.monotonic()
            text = stt.transcribe(wav)
            print(f"{path}: {time.monotonic() - t0:.2f} s for {len(wav_pcm(wav)) / 2 / RATE:.1f} s of audio: {text!r}")
        return

    cfg = Config.from_env()
    api = ApiClient(cfg)
    if args.where:
        print(json.dumps(whereabouts(cfg), indent=2))
    elif args.live_check:
        token = api.live_token()
        print("Live token OK:", token["model"], token["api_version"], "until", token["expires_at"])

        async def connect_once():
            async with gemini_live_connect(token):
                pass

        asyncio.run(connect_once())
        print("Gemini Live session opened and closed")
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
