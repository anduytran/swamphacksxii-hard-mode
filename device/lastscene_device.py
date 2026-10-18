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
not the recording, so the audio never leaves it.

Item detection on the device (VISION_DIR = the repo's vision/ folder, with a model from the vision
pipeline under LASTSCENE_DATA/models/current): the same process that listens also sees.
  * A statement ("I'm putting my keys on the counter") gets its photo after the words: if a taught
    item is mentioned and the photo doesn't show it, the camera tries 3 more times about 1 s apart
    (a short low beep says "point me at it"), and the clearest photo of the item is sent.
    Questions get no photo. Without on-device speech-to-text the photo is taken as before.
  * Between conversations, a background loop grabs a frame every VISION_EVERY seconds (default 2)
    and runs the detector only when the scene changed (lastscene_pipeline.gated), posting
    observations like `python -m lastscene_pipeline.detect`. It pauses while you talk to the device. If the local model is missing or hears nothing,
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
import re
import struct
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import httpx

try:  # speaker_eq.py sits next to this file (README step 4 copies it); without it sound plays flat
    import speaker_eq
except ImportError:
    speaker_eq = None

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
    wake_max_seconds: float = 5  # speech after the wake phrase is cut off here
    wake_drop_db: float = 15  # ...or ends sooner when loudness falls this far under the speech (0 = off)
    # Wake phrase as a command (WAKE_CONFIRM=true): the clip keeps the audio from just before the wake
    # (WAKE_PREROLL_SECONDS, so it holds "Last Scene" itself), a photo is taken at once and WAKE_PHOTOS - 1
    # more after speaking, WAKE_PHOTO_GAP_SECONDS apart; the server confirms the phrase with speech-to-text
    # (false wakes are dropped there) and a low-reasoning model reads the words and photos.
    wake_confirm: bool = True
    wake_preroll_seconds: float = 1.5
    wake_photos: int = 4
    wake_photo_gap_seconds: float = 0.75
    stt_engine: str = "server"  # moonshine | vosk | server (send the audio; the server transcribes)
    stt_model: str = ""  # model folder; empty = the default under /opt/lastscene/stt
    stt_fallback: bool = True  # local model missing or heard nothing: send the audio instead
    live: bool = False  # the wake phrase (or a quiet long press of ASK) starts a Gemini Live conversation
    live_max_seconds: int = 90
    live_silence_seconds: float = 8  # a conversation ends after this long with nobody talking
    live_hold_seconds: float = 2  # hold ASK this long without speaking to start one
    camera: str = "auto"  # auto (a Camera Module if one is connected, else USB) | picam | usb | remote
    camera_url: str = ""  # CAMERA=remote: the camera Pi's camera_server.py, e.g. http://faker:8765
    camera_token: str = ""  # CAMERA=remote: the camera Pi's CAMERA_TOKEN
    camera_index: int = 0  # USB: /dev/video<N>
    vision_dir: str = ""  # the repo's vision/ folder; empty = no on-device item detection
    min_confidence: float = 0.35  # the server drops detections under 0.35
    capture_retries: int = 3  # extra photos when the item you named isn't in the first one
    retry_gap_s: float = 1.0
    vision_every: float = 2.0  # background: seconds between frames; 0 = voice commands only
    motion_threshold: float = 4.0  # background: frame change (0-255) that counts as motion
    scene_label: str = ""  # background: where the wearer usually is, sent with observations

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env
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
            wake_max_seconds=float(env.get("WAKE_MAX_SECONDS", 5)),
            wake_drop_db=float(env.get("WAKE_DROP_DB", 15)),
            wake_confirm=env.get("WAKE_CONFIRM", "true").lower() not in ("0", "false", "no"),
            wake_preroll_seconds=float(env.get("WAKE_PREROLL_SECONDS", 1.5)),
            wake_photos=max(1, min(4, int(env.get("WAKE_PHOTOS", 4)))),
            wake_photo_gap_seconds=float(env.get("WAKE_PHOTO_GAP_SECONDS", 0.75)),
            stt_engine=env.get("STT_ENGINE", "server").lower(),
            stt_model=env.get("STT_MODEL", ""),
            stt_fallback=env.get("STT_FALLBACK", "true").lower() not in ("0", "false", "no"),
            live=env.get("LIVE", "false").lower() in ("1", "true", "yes"),
            live_max_seconds=int(env.get("LIVE_MAX_SECONDS", 90)),
            live_silence_seconds=float(env.get("LIVE_SILENCE_SECONDS", 8)),
            live_hold_seconds=float(env.get("LIVE_HOLD_SECONDS", 2)),
            camera=env.get("CAMERA", "auto").lower(),
            camera_index=int(env.get("CAMERA_INDEX", 0)),
            camera_url=env.get("CAMERA_URL", "").rstrip("/"),
            camera_token=env.get("CAMERA_TOKEN", ""),
            vision_dir=env.get("VISION_DIR", ""),
            min_confidence=float(env.get("MIN_CONFIDENCE", 0.35)),
            capture_retries=int(env.get("CAPTURE_RETRIES", 3)),
            retry_gap_s=float(env.get("RETRY_GAP_SECONDS", 1.0)),
            vision_every=float(env.get("VISION_EVERY", 2.0)),
            motion_threshold=float(env.get("MOTION_THRESHOLD", 4.0)),
            scene_label=env.get("SCENE_LABEL", ""),
        )

    @classmethod
    def local(cls, env=None) -> "Config":
        """For the offline test modes: everything from the environment, Auth0 settings optional."""
        env = dict(os.environ if env is None else env)
        for key in ("AUTH0_DOMAIN", "AUTH0_AUDIENCE", "DEVICE_CLIENT_ID", "DEVICE_CLIENT_SECRET"):
            env.setdefault(key, "")
        return cls.from_env(env)


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

def rejected(exc: Exception) -> bool:
    """The server refused this request for good (4xx), so retrying it won't help. Not 429:
    "too many requests" means later, so a memory waits in the spool like when offline."""
    return (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500
            and exc.response.status_code != 429)


class ApiClient:
    """Talks to the LastScene API with a cached Auth0 M2M token."""

    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.last: dict = {}  # what the last voice command's reply said (heard, answer, steps, followup)
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

    def voice(self, wav: bytes | None, jpeg: bytes | list[bytes] | None, where: dict | None = None,
              transcript: str | None = None, wake: bool = False, defer: bool = False,
              client: dict | None = None) -> tuple[str, bytes | dict]:
        """Wake-phrase clip, with one photo or several. Spoken audio back for "ask" and "found";
        otherwise the JSON's intent: "remember" (saved), "none" (not a request), "ignored" (wake=True and
        the clip didn't start with the wake phrase: a false wake)."""
        files, data = self._speech("voice.wav", wav, transcript, where)
        jpegs = [jpeg] if isinstance(jpeg, (bytes, bytearray)) else list(jpeg or [])
        parts = list(files.items()) + [("image", (f"scene{i}.jpg", j, "image/jpeg")) for i, j in enumerate(jpegs)]
        if wake:
            data["wake"] = "true"
        if defer:  # reply first; the slow work waits for followup()
            data["defer"] = "true"
        if client:
            data["client"] = json.dumps(client)
        resp = self._post("/api/v1/memory/voice", parts, data)
        h = resp.headers
        if h.get("content-type", "").startswith("audio/"):
            self.last = {"heard": unquote(h.get("x-heard", "")), "answer": unquote(h.get("x-answer", "")),
                         "answered_by": h.get("x-answered-by", ""), "steps": h.get("server-timing", ""),
                         "followup": h.get("x-followup")}
            return h.get("x-intent", "ask"), resp.content
        body = resp.json()
        steps = ", ".join(f"{k};dur={v}" for k, v in (body.get("steps") or {}).items())
        self.last = {"heard": body.get("heard", ""), "steps": steps, "followup": body.get("followup")}
        return body.get("intent", "remember"), body

    def followup(self, event_id, timeline: dict | None = None) -> dict:
        """Run the slow work a deferred voice command left on the server (indexing, photo pass, logging), and
        report the device's timeline of it (seconds since the wake) for the Live page."""
        return self._post(f"/api/v1/memory/followup/{event_id}", json={"timeline": timeline or {}}).json()

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
    if speaker_eq is not None:
        data = speaker_eq.apply_wav(data)  # AUDIO_GAIN_DB and SPEAKER_EQ
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
    any speech means nothing was said. With `drop_db`, a large fall in loudness also ends it: `drop_s`
    of frames `drop_db` under the speech's recent peak (after at least `min_speech_s` of speech).
    The first `ignore_s`, and any frame while `ignoring()` is true (the "I'm listening" beep is playing),
    can't start speech or set the peak. `reason` says why it ended: "drop", "silence", "max" or "nothing"."""

    def __init__(self, max_s: float = 10, silence_s: float = 0.9, timeout_s: float = 4, floor: float = 300.0,
                 drop_db: float | None = None, drop_s: float = 0.5, min_speech_s: float = 1.0, ignore_s: float = 0.0,
                 ignoring=None):
        self.frame_s = FRAME_SAMPLES / RATE
        self.max_frames = int(max_s / self.frame_s)
        self.silence_frames = int(silence_s / self.frame_s)
        self.timeout_frames = int(timeout_s / self.frame_s)
        self.drop_db, self.drop_frames = drop_db, max(1, round(drop_s / self.frame_s))
        self.min_speech_frames = round(min_speech_s / self.frame_s)
        self.ignore_frames = round(ignore_s / self.frame_s)
        self.ignoring = ignoring or (lambda: False)
        self.floor = floor  # a starting guess; follows the room while nobody speaks
        self.frames = 0
        self.quiet = 0
        self.spoke = False
        self.speech_frames = 0
        self.peak_db = 0.0  # the speech's recent loudest, falling 1 dB a frame
        self.declined = 0
        self.reason = ""

    def feed(self, frame: bytes) -> str:
        """'listening', 'speaking', 'done' or 'nothing'."""
        level = rms(frame)
        self.frames += 1
        if self.ignoring():  # our own beep: neither speech nor the room's quiet
            if self.frames >= self.timeout_frames + self.ignore_frames and not self.spoke:
                self.reason = "nothing"
                return "nothing"
            return "speaking" if self.spoke else "listening"
        loud = level > max(self.floor * 2.5, 500)
        if not self.spoke:
            if loud and self.frames > self.ignore_frames:
                self.spoke = True
            else:
                if not loud:
                    self.floor = 0.9 * self.floor + 0.1 * level
                if self.frames >= self.timeout_frames:
                    self.reason = "nothing"
                    return "nothing"
                return "listening"
        self.speech_frames += 1
        db = 20 * math.log10(level + 1)
        self.peak_db = max(db, self.peak_db - 1.0)
        dropped = self.drop_db is not None and db < self.peak_db - self.drop_db
        self.declined = self.declined + 1 if dropped else 0
        self.quiet = 0 if loud else self.quiet + 1
        if self.declined >= self.drop_frames and self.speech_frames >= self.min_speech_frames:
            self.reason = "drop"
        elif self.quiet >= self.silence_frames:
            self.reason = "silence"
        elif self.frames >= self.max_frames:
            self.reason = "max"
        else:
            return "speaking"
        return "done"


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
        self.score = max(scores.values(), default=0)
        if self.score < self.threshold or self.clock() - self._last < self.cooldown_s:
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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            frame = self._proc.stdout.read(FRAME_BYTES)
            if len(frame) < FRAME_BYTES:
                why = (self._proc.stderr.read() or b"").decode(errors="replace").strip() if self._proc.stderr else ""
                log.error("Microphone stream ended (arecord exited %s)%s", self._proc.wait(),
                          f": {why.splitlines()[-1]}" if why else "")
                if "capture slave is not defined" in why or "audio open error" in why:
                    log.error("AUDIO_IN_DEVICE=%s can't record. Pick the mic's card from `arecord -l` "
                              "(e.g. plughw:1,0 for the Voice Bonnet; README \"Voice Bonnet mic\")", self.device)
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

    def listen(self, endpointer: Endpointer, timeout_s: float, keep_all: bool = False) -> bytes | None:
        """Blocks until speech ends; the PCM spoken, or None when nothing was said. `keep_all`: every frame
        from the start (words said while the beep played aren't lost), not just from when speech began."""
        self._endpointer, self._frames, self._preroll, self._said = endpointer, [], [], False
        self._keep_all = keep_all
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
        if getattr(self, "_keep_all", False):
            self._frames.append(frame)
        elif self._endpointer.spoke:
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


def jpeg_bytes(frame, quality: int = 88) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the photo")
    return buf.tobytes()


class Camera:
    """A Raspberry Pi Camera Module (picamera2)."""

    def __init__(self):
        from picamera2 import Picamera2  # apt: python3-picamera2

        self.cam = Picamera2()
        # Small JPEGs upload fast and stay well under the API's 2 MB limit.
        self.cam.configure(self.cam.create_still_configuration(main={"size": (1280, 960), "format": "RGB888"}))
        self.cam.start()
        self._lock = threading.Lock()
        time.sleep(1)  # let auto-exposure settle once at boot

    def frame(self):
        """The current view as a BGR array (picamera2's RGB888 is BGR in memory)."""
        with self._lock:
            return self.cam.capture_array()

    def capture_jpeg(self) -> bytes:
        return jpeg_bytes(self.frame())


class UsbCamera:
    """A USB webcam (V4L2), kept open but read only when a frame is needed.

    Opening takes ~1.1 s and auto-exposure needs a few frames, so the camera stays open; nothing is
    decoded between photos. The frame waiting in the one-frame driver buffer is thrown away first:
    on the Zero 2 W it was captured while the detector had the CPU, and the USB controller drops data
    then (Oct 17, 2026, detector running between reads: 19 of 25 MJPEG 1280x720 frames corrupt, torn
    bands in uncompressed ones too). Reading the next frame instead: 0 of 25 corrupt, ~90 ms a photo."""

    def __init__(self, index: int = 0, size=(1280, 720)):
        import cv2

        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"no USB camera at /dev/video{index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        for _ in range(10):  # let auto-exposure settle once at start-up
            self.cap.read()

    def frame(self):
        with self._lock:
            self.cap.grab()  # the stale, possibly corrupt frame (see the class docstring)
            for _ in range(3):
                ok, image = self.cap.read()
                if ok:
                    return image
                time.sleep(0.2)
        raise RuntimeError("the USB camera returned no frame (unplugged?)")

    def capture_jpeg(self) -> bytes:
        return jpeg_bytes(self.frame())


class RemoteCamera:
    """The camera on the other Pi (camera_server.py on faker), for a two-Pi wearable.

    One GET per photo and nothing in between: no stream, no polling, and questions never ask for one.
    Given the words, the camera Pi retries until the item they name is in the photo (if it runs the
    item detector), so this waits for the words like on-device detection does."""

    checks_items = True  # the photo can wait for the words: the camera Pi looks for what they name

    def __init__(self, url: str, token: str, http: httpx.Client | None = None, timeout: float = 15.0):
        if not url or not token:
            raise ValueError("CAMERA=remote needs CAMERA_URL and CAMERA_TOKEN")
        self.url, self.token = url.rstrip("/"), token
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0))

    def capture_jpeg(self, words: str | None = None) -> bytes:
        params = {"for": words[:300]} if words else None
        for attempt in (1, 2):  # one quick retry for a dropped connection; anything else is final
            try:
                resp = self.http.get(f"{self.url}/photo", params=params, headers={"X-Camera-Token": self.token})
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
        resp.raise_for_status()
        if not resp.headers.get("content-type", "").startswith("image/jpeg"):
            raise RuntimeError("the camera Pi didn't send a JPEG")
        if item := resp.headers.get("x-item"):
            log.info("Camera Pi: photo shows %s (%s) after %s attempt(s)", item, resp.headers.get("x-confidence"),
                     resp.headers.get("x-attempts"))
        return resp.content

    def frame(self):
        raise RuntimeError("the camera is on the other Pi: it sends JPEGs, not live frames")


def make_camera(cfg: Config):
    if cfg.camera == "remote":
        return RemoteCamera(cfg.camera_url, cfg.camera_token)
    if cfg.camera in ("auto", "picam"):
        try:
            return Camera()
        except Exception:
            if cfg.camera == "picam":
                raise
            log.info("No Camera Module found; using the USB camera at /dev/video%d", cfg.camera_index)
    return UsbCamera(cfg.camera_index)


# ---------------------------------------------------------------- statement or question

# The same rules as the server's backend/app/intent.py (kept in step by test_device.py), so the device
# knows before it uploads whether the words need a photo: the server rejects a statement without one.
WAKE_PHRASES = (
    "hey last scene", "hey lastscene", "hey last seen", "okay last scene", "ok last scene",
    "last scene", "lastscene", "last seen", "hey jarvis",
)
_QUESTION_START = re.compile(
    r"^(where|what|when|which|who|whose|how|why|did|do|does|have|has|had|is|are|was|were|can|could|would|will|"
    r"should|tell me|remind me|help me find|find|show me|any idea|got any idea)\b"
)
_QUESTION_ANYWHERE = re.compile(
    r"\b(where('s| is| are| was| were| did| do| have| has)\b|where did i\b|have you seen|did i (leave|put|take|lock|turn)\b|"
    r"do you know where\b|can you find\b|i can'?t find\b|i lost\b|i'?ve lost\b)"
)


def strip_wake_phrase(text: str) -> str:
    t = text.strip()
    low = re.sub(r"[^\w\s']", "", t.lower())
    for phrase in WAKE_PHRASES:
        if low.startswith(phrase):
            words = len(phrase.split())
            return re.sub(r"^\W*" + r"\W+".join([r"\w+"] * words) + r"[\W]*", "", t, count=1).strip()
    return t


def intent_of(transcript: str) -> str:
    """'ask' or 'remember'."""
    low = strip_wake_phrase(transcript).lower().strip()
    if not low:
        return "remember"
    if low.endswith("?") or _QUESTION_START.match(low) or _QUESTION_ANYWHERE.search(low):
        return "ask"
    return "remember"


# ---------------------------------------------------------------- item detection on the device

class Vision:
    """The vision pipeline's detector, loaded once and shared by voice commands and the background loop.

    Imports lastscene_pipeline from VISION_DIR; the model is LASTSCENE_DATA/models/current (reloaded when
    `fetch` or `train` moves it). Class names and aliases come from the model's classes.json."""

    def __init__(self, cfg: Config, camera, paused=lambda: False):
        if cfg.vision_dir and cfg.vision_dir not in sys.path:
            sys.path.insert(0, cfg.vision_dir)
        from lastscene_pipeline.common import data_dir, read_json
        from lastscene_pipeline.detect import ReloadingDetector, load_classes
        from lastscene_pipeline.runtime import load_detector

        self.cfg, self.camera, self.paused = cfg, camera, paused
        self.root = data_dir()
        pointer = self.root / "models" / "current"

        def build(model_dir):
            return load_detector(model_dir, load_classes(model_dir, self.root), cfg.min_confidence, "onnx", 4)

        self.detector = ReloadingDetector(pointer, None, build, log=log.info)
        self._read_json = read_json
        self.stats = None

    @property
    def names(self) -> dict[str, list[str]]:
        """class name -> the words people use for it (the name itself, plural, and Teach aliases)."""
        items = {}
        model_dir = self.detector.model_dir
        vocab = self._read_json(model_dir / "classes.json") if model_dir and (model_dir / "classes.json").exists() else {}
        aliases = {i.get("name"): i.get("aliases") or [] for i in vocab.get("items", [])}
        for name in getattr(self.detector.inner, "classes", []):
            words = {name.lower(), *(a.lower() for a in aliases.get(name, []))}
            items[name] = sorted(words | {w + "s" for w in words if not w.endswith("s")})
        return items

    def targets(self, words: str | None) -> list[str]:
        """The taught items the words mention ("my pin", "the swamphacks pins"), in class-name form."""
        if not words:
            return []
        low = strip_wake_phrase(words).lower()
        return [name for name, said in self.names.items()
                if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in said)]

    def photo_for(self, words: str | None, on_retry=lambda attempt: None, sleep=time.sleep):
        """(frame, detection or None, attempts). With a taught item named, up to 1 + capture_retries
        photos until one shows it at >= min_confidence; otherwise a single photo."""
        targets = self.targets(words)
        attempts = 1 + (self.cfg.capture_retries if targets else 0)
        frame = None
        for attempt in range(attempts):
            if attempt:
                on_retry(attempt)
                sleep(self.cfg.retry_gap_s)
            frame = self.camera.frame()
            if not targets:
                return frame, None, 1
            hits = [d for d in self.detector.detect(frame) if d.label in targets and d.conf >= self.cfg.min_confidence]
            if hits:
                return frame, max(hits, key=lambda d: d.conf), attempt + 1
        return frame, None, attempts

    def start_background(self, dry_run: bool = False, stop=lambda: False) -> threading.Thread:
        """The detect loop on motion-gated frames from the shared camera, on a daemon thread."""
        from lastscene_pipeline.detect import build_parser, run
        from lastscene_pipeline.gated import GatedFrames

        gate = GatedFrames(self.camera.frame, every=self.cfg.vision_every, threshold=self.cfg.motion_threshold,
                           paused=self.paused, stop=stop)
        self.stats = gate.stats
        argv = ["--source", "wearable", "--fps", str(1 / self.cfg.vision_every), "--min-confidence",
                str(self.cfg.min_confidence), "--sync-every", "0", "--no-console"]
        if self.cfg.scene_label:
            argv += ["--scene-label", self.cfg.scene_label]
        if dry_run:
            argv.append("--dry-run")
        args = build_parser().parse_args(argv)

        def loop():
            try:
                run(args, frames=gate, detector=self.detector)
            except Exception:
                log.exception("Background item detection stopped")

        thread = threading.Thread(target=loop, name="lastscene-vision", daemon=True)
        thread.start()
        return thread


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
        self._eq = speaker_eq.from_env(rate) if speaker_eq is not None else None

    def play(self, pcm: bytes) -> None:
        if self._eq is not None:
            pcm = self._eq.process(pcm)
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

def optional_missing(msg: str, *args) -> None:
    """An optional part (speech-to-text model, wake phrase, item detector) isn't installed: one warning
    line, since the device works without it. The traceback only with LOG_LEVEL=DEBUG."""
    log.warning(msg, *args)
    log.debug("Details", exc_info=True)


class NothingHeard(Exception):
    """Local speech-to-text heard nothing and STT_FALLBACK=false: send nothing."""


class Device:
    def __init__(self, cfg: Config):
        from gpiozero import LED, Button  # preinstalled on Raspberry Pi OS

        self.cfg = cfg
        if speaker_eq is None:
            optional_missing("speaker_eq.py not next to lastscene_device.py; sound plays without AUDIO_GAIN_DB/SPEAKER_EQ")
        self.api = ApiClient(cfg)
        self.spool = Spool(cfg.spool_dir)
        self.camera = make_camera(cfg)
        self.led = LED(cfg.led_pin)
        self.remember = Button(cfg.remember_pin, pull_up=True, bounce_time=0.05)
        self.ask = Button(cfg.ask_pin, pull_up=True, bounce_time=0.05)
        self.busy = threading.Lock()  # one interaction at a time
        self.stt: LocalSTT | None = None
        if cfg.stt_engine != "server":
            try:
                self.stt = LocalSTT(cfg.stt_engine, cfg.stt_model)
                log.info("Speech-to-text on the device: %s (%s)", cfg.stt_engine, self.stt.model_dir)
            except Exception as exc:  # package or model missing: the server transcribes instead
                optional_missing("On-device speech-to-text %r unavailable (%s); %s", cfg.stt_engine, exc,
                                 "sending audio to the server" if cfg.stt_fallback else "nothing can be sent")

        # Item detection: one detector for voice commands and the background loop, paused while busy.
        self.vision: Vision | None = None
        if cfg.vision_dir and not isinstance(self.camera, RemoteCamera):  # remote: detection runs on the camera Pi
            try:
                self.vision = Vision(cfg, self.camera, paused=self.busy.locked)
                log.info("Item detection on the device: %s", self.vision.detector.name)
            except Exception as exc:  # vision code or model missing: photos are still taken, just not checked
                optional_missing("On-device item detection unavailable (%s; VISION_DIR=%s)", exc, cfg.vision_dir)

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
        self._recent: deque[bytes] = deque(maxlen=max(1, round(cfg.wake_preroll_seconds / 0.08)))  # 80 ms frames
        if cfg.wake_word:
            try:
                self.wake = WakeListener(cfg.wake_word, cfg.wake_threshold)
            except Exception as exc:  # openwakeword or its model missing: keep the buttons working
                optional_missing("Wake phrase %r unavailable (%s); buttons only (see README step 3)", cfg.wake_word, exc)
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

    def _photo(self, words: str | None) -> bytes:
        """The photo for a memory. With item detection: retried until the item the words name is in it."""
        if isinstance(self.camera, RemoteCamera):  # the camera Pi does the looking (one request)
            try:
                return self.camera.capture_jpeg(words)
            except Exception:
                log.exception("No photo from the camera Pi (%s)", self.cfg.camera_url)
                play_wav(self.cfg.audio_out, tone(440))  # the low beep: this memory can't be saved
                raise
        if self.vision is None:
            return self.camera.capture_jpeg()
        t0 = time.monotonic()
        frame, hit, attempts = self.vision.photo_for(
            words, on_retry=lambda n: n == 1 and play_wav(self.cfg.audio_out, tone(523, 80)))  # "point me at it"
        if hit is not None:
            log.info("Photo shows %s (%.2f) after %d attempt(s), %.1f s", hit.label, hit.conf, attempts, time.monotonic() - t0)
        elif attempts > 1:
            log.info("Item not seen in %d photos (%.1f s); sending the last one", attempts, time.monotonic() - t0)
        return jpeg_bytes(frame)

    def _on_frame(self, frame: bytes) -> None:
        """Every 80 ms of microphone audio, when the wake phrase (or live mode) is on."""
        if (live := self.live) is not None:  # a live conversation: the audio goes to Gemini
            live.feed(frame)
        elif self.stream_rec.active:  # a button is held
            self.stream_rec.add(frame)
        elif self.voice.listening:  # after the wake phrase
            self.voice.add(frame)
        else:
            self._recent.append(frame)  # the last WAKE_PREROLL_SECONDS: holds the wake phrase when it fires
            if self.wake and not self.busy.locked() and self.wake.heard(frame):
                preroll = b"".join(self._recent)
                self._recent.clear()
                threading.Thread(target=self._wake, args=(preroll,), daemon=True).start()

    def _wake(self, preroll: bytes = b"") -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            self.led.on()
            if self.cfg.live and self._converse():
                return
            if self.cfg.wake_confirm:
                return self._wake_command(preroll)
            # Without device speech-to-text the words come too late to choose: photograph now, as before.
            photo = self.camera.capture_jpeg() if self.stt is None else None
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
            if photo is None and (words is None or intent_of(words) == "remember"):
                photo = self._photo(words)  # a statement needs its photo; a question gets none
            where = whereabouts(self.cfg)
            try:
                intent, result = self.api.voice(wav, photo, where, words)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if rejected(exc):
                    log.warning("Server rejected voice clip: %s", exc.response.text)
                    play_wav(self.cfg.audio_out, tone(440))
                elif photo is not None:
                    self.spool.add(wav, photo, where, kind="voice", transcript=words)
                    log.warning("Offline; voice clip spooled for later")
                else:
                    log.warning("Offline; a question can't be answered later, so it isn't kept")
                    play_wav(self.cfg.audio_out, tone(440))
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
            if rejected(exc):
                raise
            self.spool.add(None, photo, where, transcript=text)
            return "Saved on the wearable; it will upload when it's back online."

    def _grab(self, photos: list, index: int) -> None:
        try:
            photos.append((index, self.camera.capture_jpeg()))
        except Exception as exc:  # one missing photo doesn't sink the command
            log.warning("Photo %d failed: %s", index + 1, exc)

    def _wake_command(self, preroll: bytes) -> None:
        """Photo now and more while they speak, listen (WAKE_MAX_SECONDS at most, ending sooner on silence or
        a sharp drop in loudness), then one request: the server confirms "Last Scene" was said, keeps the
        words after it, and replies as soon as it can; its slow work runs on a follow-up call afterwards."""
        t0 = time.monotonic()
        score = getattr(getattr(self, "wake", None), "score", 0) or 0
        log.info("Wake: \"Last Scene\" (score %.2f), listening up to %.0f s", score, self.cfg.wake_max_seconds)
        photos: list[tuple[int, bytes]] = []
        shot_at: list[float] = []
        stop_shooting = threading.Event()

        def shoot() -> None:  # photo 1 right away, then one every WAKE_PHOTO_GAP_SECONDS while they speak
            for index in range(self.cfg.wake_photos):
                if index and stop_shooting.wait(self.cfg.wake_photo_gap_seconds):
                    return
                self._grab(photos, index)
                shot_at.append(time.monotonic() - t0)

        shooter = threading.Thread(target=shoot, daemon=True)
        shooter.start()
        # The beep plays while the mic already listens, so words said over it aren't lost; until it has
        # finished (aplay can take a moment to start) it neither counts as speech nor sets the loudness peak.
        beeped = threading.Event()

        def beep() -> None:
            try:
                play_wav(self.cfg.audio_out, tone(880))
            finally:
                time.sleep(0.1)  # its echo
                beeped.set()

        threading.Thread(target=beep, daemon=True).start()
        ep = Endpointer(max_s=self.cfg.wake_max_seconds, timeout_s=3, ignore_s=0.2,
                        drop_db=self.cfg.wake_drop_db or None, ignoring=lambda: not beeped.is_set())
        pcm = self.voice.listen(ep, self.cfg.wake_max_seconds + 3, keep_all=True)
        recorded = time.monotonic() - t0
        timeline = {"speech_end": round(recorded, 2)}
        stop_shooting.set()
        if pcm is None:
            log.info("Nothing said within 3 s after the wake phrase")
            play_wav(self.cfg.audio_out, tone(440))
            return
        self.led.blink(on_time=0.1, off_time=0.1)
        shooter.join(timeout=1.5)  # a photo already on its way is worth a moment; no more than that
        shots = [jpeg for _, jpeg in sorted(photos)]
        timeline["photos_ready"] = round(time.monotonic() - t0, 2)
        ended = {"drop": "loudness dropped", "silence": "silence", "max": f"{self.cfg.wake_max_seconds:g} s limit"}
        log.info("Recorded %.1f s (%s), %d photo%s at %s s", recorded, ended.get(ep.reason, ep.reason), len(shots),
                 "" if len(shots) == 1 else "s", ", ".join(f"{t:.1f}" for t in shot_at) or "-")
        client = {"recorded_s": round(recorded, 2), "ended": ep.reason, "photos": len(shots),
                  "photo_at_s": [round(t, 2) for t in shot_at], "wake_score": round(float(score), 2),
                  "preroll_s": round(len(preroll) / 2 / RATE, 2)}
        wav = wav_bytes(preroll + pcm)  # with the wake phrase, for the server to confirm
        where = whereabouts(self.cfg)
        sent = time.monotonic()
        timeline["sent"] = round(sent - t0, 2)
        try:
            intent, result = self.api.voice(wav, shots, where, wake=True, defer=True, client=client)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if rejected(exc):
                log.warning("Server rejected the command: %s", exc.response.text)
                play_wav(self.cfg.audio_out, tone(440))
            elif shots:
                self.spool.add(wav, shots[0], where, kind="voice")
                log.warning("Offline; command spooled for later")
            else:
                play_wav(self.cfg.audio_out, tone(440))
            return
        timeline["reply"] = round(time.monotonic() - t0, 2)
        info = getattr(self.api, "last", {}) or {}
        server = " ".join(f"{p.split(';dur=')[0]} {p.split(';dur=')[1]}ms" for p in info.get("steps", "").split(", ") if ";dur=" in p)
        log.info("Reply in %.1f s (%.1f s since the wake)%s", time.monotonic() - sent, time.monotonic() - t0,
                 f" [server: {server}]" if server else "")
        heard = info.get("heard") or ""
        if intent in ("ask", "found"):
            log.info("Heard \"%s\" -> %s: \"%s\" (%s)", heard, intent, info.get("answer", ""), info.get("answered_by") or "-")
            self.led.on()
            play_wav(self.cfg.audio_out, result)
        elif intent == "remember":
            log.info("Heard \"%s\" -> saved \"%s\" (photo %s of %d)", heard, result.get("transcript"),
                     result.get("photo", 0) + 1, len(shots))
            play_wav(self.cfg.audio_out, tone(1320, 90))  # "saved"
        elif intent == "ignored":
            log.info("Heard \"%s\": no \"Last Scene\" in it, ignored", heard)
        else:
            log.info("Heard \"%s\": not a request", heard)
            play_wav(self.cfg.audio_out, tone(440))
        timeline["played"] = round(time.monotonic() - t0, 2)
        log.info("Timeline (s since wake): %s | listen %.1f, photos +%.1f, upload+server %.1f, play %.1f",
                 " ".join(f"{k}={v:.1f}" for k, v in timeline.items()), timeline["speech_end"],
                 timeline["photos_ready"] - timeline["speech_end"], timeline["reply"] - timeline["sent"],
                 timeline["played"] - timeline["reply"])
        if info.get("followup"):
            threading.Thread(target=self._followup, args=(info["followup"], timeline), daemon=True).start()

    def _followup(self, event_id, timeline: dict | None = None) -> None:
        """The server's slow work for that command (indexing, the photo pass, logging), after the wearer heard,
        and this command's timeline for the Live page."""
        try:
            done = self.api.followup(event_id, timeline)
            log.info("Follow-up %s: %s in %.1f s", event_id, done.get("followup", "nothing to do"),
                     (done.get("followup_ms") or 0) / 1000)
        except Exception as exc:  # the memory is saved either way; only its photo pass / indexing is missing
            log.warning("Follow-up %s failed: %s", event_id, exc)

    def _begin(self, mode: str) -> None:
        if (live := self.live) is not None:  # any button ends a live conversation
            live.stop()
            return
        if not self.busy.acquire(blocking=False):
            return
        try:
            # With item detection the photo waits for the words (to know what to look for).
            waits = self.vision is not None or getattr(self.camera, "checks_items", False)
            photo = self.camera.capture_jpeg() if mode == "remember" and not waits else None
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
            if mode == "remember" and photo is None:
                photo = self._photo(words)
            where = whereabouts(self.cfg)  # a Wi-Fi scan takes a second or two
            if mode == "remember":
                try:
                    result = self.api.log_memory(wav, photo, where, words)
                    log.info("Logged: %s", result.get("transcript"))
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if rejected(exc):
                        log.warning("Server rejected memory: %s", exc.response.text)
                    else:
                        self.spool.add(wav, photo, where, transcript=words)
                        log.warning("Offline; memory spooled for later")
            else:
                try:
                    audio, answer = self.api.ask(wav, where, words)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    log.warning("Question not answered: %s", exc)
                    play_wav(self.cfg.audio_out, tone(440))  # the low "didn't work" beep, not silence
                    return
                log.info("Answer: %s", answer)
                self.led.on()
                play_wav(self.cfg.audio_out, audio)
        except Exception:
            log.exception("%s failed", mode)
        finally:
            self.led.off()
            self.busy.release()

    def run(self) -> None:
        if self.vision is not None and self.cfg.vision_every > 0:
            self.vision.start_background()
            log.info("Watching for taught items every %.1f s when the scene changes", self.cfg.vision_every)
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
    parser.add_argument("--capture", metavar="WORDS",
                        help='take the photo a statement would get ("I put my keys here"), with retries, and save '
                             "capture.jpg; needs VISION_DIR; no network")
    parser.add_argument("--watch", metavar="SECONDS", type=float,
                        help="run background item detection for SECONDS, printing observations instead of posting "
                             "them, then report frames grabbed / passed / skipped; needs VISION_DIR; no network")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")

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

    if args.capture or args.watch:  # camera + model only
        cfg = Config.local()
        camera = make_camera(cfg)
        t0 = time.monotonic()
        vision = Vision(cfg, camera)
        print(f"detector {vision.detector.name} loaded in {time.monotonic() - t0:.1f} s; items {vision.names}")
        if args.capture:
            print(f"intent: {intent_of(args.capture)}; looking for: {vision.targets(args.capture) or 'nothing named'}")
            if intent_of(args.capture) == "ask":
                print("a question: the device takes no photo")
                return
            t0 = time.monotonic()
            frame, hit, attempts = vision.photo_for(args.capture, on_retry=lambda n: print(f"  retry {n}"))
            Path("capture.jpg").write_bytes(jpeg_bytes(frame))
            print(f"{'found ' + hit.label + f' ({hit.conf:.2f})' if hit else 'not found'} after {attempts} attempt(s) "
                  f"in {time.monotonic() - t0:.1f} s; saved capture.jpg")
        else:
            deadline = time.monotonic() + args.watch
            c0 = time.process_time()
            vision.start_background(dry_run=True, stop=lambda: time.monotonic() >= deadline).join(args.watch + 30)
            print(json.dumps({**vision.stats, "cpu_s": round(time.process_time() - c0, 1), "seconds": args.watch}))
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
        if (other := already_running()) is not None:
            sys.exit(f"LastScene is already running (pid {other}) and holds the mic and GPIO pins. Stop it first:\n"
                     "  sudo systemctl stop lastscene-device; sudo pkill -f '^/opt/lastscene/venv/bin/python /opt/lastscene/lastscene_device.py'")
        Device(cfg).run()


def already_running() -> int | None:
    """Another lastscene_device.py on this Pi that's running the device (not a --check and the like)."""
    try:
        out = subprocess.run(["pgrep", "-f", "lastscene_device.py"], capture_output=True, text=True).stdout
    except OSError:  # no pgrep: let the GPIO error speak for itself
        return None
    mine = {os.getpid(), os.getppid()}
    for pid in (int(p) for p in out.split() if p.isdigit()):
        if pid in mine:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(c.endswith(b"python") or b"python3" in c for c in cmd[:1]) and not any(c.startswith(b"--") for c in cmd):
            return pid
    return None


if __name__ == "__main__":
    main()
