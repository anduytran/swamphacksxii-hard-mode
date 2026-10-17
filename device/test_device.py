"""Network logic of the wearable client, against a fake Auth0 + API. No hardware needed.

Run from the repo root:  backend/.venv/bin/python -m pytest device
"""

import httpx
import pytest

from lastscene_device import ApiClient, Config, Spool


def make_client(handler, tmp_path=None):
    cfg = Config(
        api_url="https://api.test",
        auth0_domain="tenant.auth0.test",
        auth0_audience="https://api.lastscene",
        client_id="id",
        client_secret="secret",
        spool_dir=tmp_path or "/tmp",
    )
    return ApiClient(cfg, httpx.Client(transport=httpx.MockTransport(handler)))


class FakeServer:
    def __init__(self):
        self.tokens_issued = 0
        self.uploads = []
        self.reject_token = None  # a token the API should answer 401 to
        self.offline = False
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "tenant.auth0.test":
            self.tokens_issued += 1
            return httpx.Response(200, json={"access_token": f"tok{self.tokens_issued}", "expires_in": 86400})
        if self.offline:
            raise httpx.ConnectError("no wifi", request=request)
        if request.headers["authorization"] == f"Bearer {self.reject_token}":
            return httpx.Response(401, json={"detail": "revoked"})
        self.uploads.append(request.url.path)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "nope"})
        if request.url.path.endswith("/query"):
            return httpx.Response(200, content=b"RIFFwav", headers={"x-answer": "On%20the%20counter."})
        return httpx.Response(200, json={"id": "1", "transcript": "keys on the counter"})


def test_token_is_cached_between_calls():
    server = FakeServer()
    api = make_client(server)
    api.log_memory(b"wav", b"jpg")
    api.log_memory(b"wav", b"jpg")
    assert server.tokens_issued == 1
    assert server.uploads == ["/api/v1/memory/log"] * 2


def test_revoked_token_is_refreshed_once():
    server = FakeServer()
    api = make_client(server)
    api.token()
    server.reject_token = "tok1"
    assert api.log_memory(b"wav", b"jpg")["transcript"] == "keys on the counter"
    assert server.tokens_issued == 2


def test_ask_returns_audio_and_decoded_answer():
    api = make_client(FakeServer())
    audio, answer = api.ask(b"wav")
    assert audio == b"RIFFwav"
    assert answer == "On the counter."


def test_spool_keeps_memories_while_offline_and_uploads_later(tmp_path):
    server = FakeServer()
    api = make_client(server)
    spool = Spool(tmp_path / "spool")
    spool.add(b"wav1", b"jpg1")
    spool.add(b"wav2", b"jpg2")

    server.offline = True
    assert spool.flush(api) == 0
    assert len(spool.pending()) == 2

    server.offline = False
    assert spool.flush(api) == 2
    assert spool.pending() == []


def test_spool_drops_memories_the_server_rejects(tmp_path):
    server = FakeServer()
    server.status = 422  # e.g. no speech detected: retrying would never succeed
    spool = Spool(tmp_path / "spool")
    spool.add(b"silence", b"jpg")
    assert spool.flush(make_client(server)) == 0
    assert spool.pending() == []


def test_spool_retries_later_on_server_error(tmp_path):
    server = FakeServer()
    server.status = 503
    spool = Spool(tmp_path / "spool")
    spool.add(b"wav", b"jpg")
    assert spool.flush(make_client(server)) == 0
    assert len(spool.pending()) == 1


# ---------------------------------------------------------------- location + pairing

import json  # noqa: E402
import subprocess  # noqa: E402

import lastscene_device as dev  # noqa: E402

NMCLI_OUT = "AA\\:BB\\:CC\\:00\\:11\\:22:80\nAA\\:BB\\:CC\\:00\\:11\\:33:40\nAA\\:BB\\:CC\\:00\\:11\\:22:30\n"
IW_OUT = """BSS 12:34:56:78:9a:bc(on wlan0)
\tsignal: -48.00 dBm
\tSSID: Home
BSS de:ad:be:ef:00:01(on wlan0)
\tsignal: -71.00 dBm
"""


def fake_run(outputs):
    def run(cmd, **kwargs):
        out = outputs.get(cmd[0])
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out or "")
    return run


def test_nmcli_scan_parses_escaped_bssids_and_keeps_strongest(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": NMCLI_OUT}))
    aps = dev.scan_wifi()
    assert aps == [{"bssid": "aa:bb:cc:00:11:22", "signal": -60}, {"bssid": "aa:bb:cc:00:11:33", "signal": -80}]


def test_falls_back_to_iw_without_networkmanager(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": FileNotFoundError("nmcli"), "iw": IW_OUT}))
    assert dev.scan_wifi() == [{"bssid": "12:34:56:78:9a:bc", "signal": -48}, {"bssid": "de:ad:be:ef:00:01", "signal": -71}]


def test_whereabouts_by_source(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": NMCLI_OUT}))
    cfg = Config("https://api.test", "t", "a", "id", "s", location_source="wifi", fixed_lat=29.65, fixed_lng=-82.32)
    fields = dev.whereabouts(cfg)
    assert json.loads(fields["wifi"])[0]["bssid"] == "aa:bb:cc:00:11:22"
    assert fields["lat"] == "29.65"  # fallback for when Wi-Fi gives no fix
    cfg.location_source = "fixed"
    assert "wifi" not in dev.whereabouts(cfg)
    cfg.location_source = "none"
    assert dev.whereabouts(cfg) == {}


def test_pairing_code_and_location_are_sent():
    seen = {}

    def handler(request):
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        seen["code"] = request.headers.get("x-pairing-code")
        seen["body"] = request.content
        return httpx.Response(200, json={"id": "1"})

    api = make_client(handler)
    api.cfg.pairing_code = "ABCD-1234-EF56"
    api.log_memory(b"wav", b"jpg", {"wifi": '[{"bssid": "aa:bb:cc:00:11:22", "signal": -60}]'})
    assert seen["code"] == "ABCD-1234-EF56"
    assert b'name="wifi"' in seen["body"] and b"aa:bb:cc:00:11:22" in seen["body"]


def test_spooled_memory_keeps_where_it_happened(tmp_path):
    server = FakeServer()
    bodies = []
    original = server.__call__

    def capture(request):
        if request.url.host != "tenant.auth0.test":
            bodies.append(request.content)
        return original(request)

    spool = Spool(tmp_path / "spool")
    spool.add(b"wav", b"jpg", {"lat": "29.65", "lng": "-82.32"})
    assert spool.flush(make_client(capture)) == 1
    assert b'name="lat"' in bodies[0] and b"29.65" in bodies[0]


# ---------------------------------------------------------------- wake phrase

import array as _array  # noqa: E402
import io as _io  # noqa: E402
import math as _math  # noqa: E402
import threading as _threading  # noqa: E402
import wave as _wave  # noqa: E402

from lastscene_device import FRAME_SAMPLES, Endpointer, StreamRecorder, VoiceCapture, WakeListener, tone, wav_bytes  # noqa: E402


def frame(level=0, freq=220):
    """80 ms of a sine at the given amplitude (0 = silence)."""
    return _array.array("h", (int(level * _math.sin(2 * _math.pi * freq * i / 16000)) for i in range(FRAME_SAMPLES))).tobytes()


QUIET, SPEECH = frame(80), frame(6000)


def test_wav_bytes_and_tone_are_valid_wav():
    with _wave.open(_io.BytesIO(wav_bytes(QUIET * 3))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()) == (1, 16000, 2, FRAME_SAMPLES * 3)
    with _wave.open(_io.BytesIO(tone(880, 120))) as w:
        assert w.getnframes() == 1920


def test_endpointer_waits_for_speech_then_stops_after_silence():
    e = Endpointer(silence_s=0.4)
    assert [e.feed(QUIET) for _ in range(5)] == ["listening"] * 5
    assert e.feed(SPEECH) == "speaking" and e.spoke
    states = [e.feed(QUIET) for _ in range(6)]
    assert states[-1] == "done" and states.count("done") >= 1


def test_endpointer_gives_up_when_nothing_is_said_and_caps_long_speech():
    e = Endpointer(timeout_s=0.8)
    assert [e.feed(QUIET) for _ in range(10)][-1] == "nothing"
    e = Endpointer(max_s=1.0)
    assert [e.feed(SPEECH) for _ in range(13)][-1] == "done"


def test_endpointer_adapts_to_a_noisy_room():
    e = Endpointer()
    noisy = frame(900)  # a busy ballroom: louder than the default threshold of 500
    assert all(e.feed(noisy) == "listening" for _ in range(20))  # the floor rises; noise isn't speech
    assert e.feed(frame(9000)) == "speaking"


def test_voice_capture_returns_speech_with_a_little_preroll():
    v = VoiceCapture()
    result = {}
    t = _threading.Thread(target=lambda: result.setdefault("pcm", v.listen(Endpointer(silence_s=0.4), 5)))
    t.start()
    while not v.listening:
        pass
    for f in [QUIET] * 4 + [SPEECH] * 3 + [QUIET] * 6:
        v.add(f)
    t.join(2)
    pcm = result["pcm"]
    assert pcm.startswith(QUIET * 3 + SPEECH * 3)  # 240 ms before the first loud frame is kept


def test_voice_capture_none_when_nothing_said():
    v = VoiceCapture()
    result = {}
    t = _threading.Thread(target=lambda: result.setdefault("pcm", v.listen(Endpointer(timeout_s=0.5), 5)))
    t.start()
    while not v.listening:
        pass
    for _ in range(8):
        v.add(QUIET)
    t.join(2)
    assert result["pcm"] is None


def test_stream_recorder_caps_length():
    r = StreamRecorder(max_seconds=1)
    r.start()
    for _ in range(30):
        r.add(QUIET)
    with _wave.open(_io.BytesIO(r.stop())) as w:
        assert w.getnframes() == FRAME_SAMPLES * 12  # 1 s of 80 ms frames, no more
    r.add(QUIET)  # not recording any more
    assert len(r.frames) == 12


def test_wake_listener_threshold_cooldown_and_reset():
    pytest.importorskip("numpy")

    class Model:
        def __init__(self):
            self.score, self.resets = 0.0, 0

        def predict(self, x):
            assert len(x) == FRAME_SAMPLES
            return {"hey_jarvis": self.score}

        def reset(self):
            self.resets += 1

    now = [100.0]
    m = Model()
    w = WakeListener("hey_jarvis", threshold=0.5, cooldown_s=2.0, model=m, clock=lambda: now[0])
    assert not w.heard(QUIET)
    m.score = 0.9
    assert w.heard(QUIET) and m.resets == 1
    now[0] += 1.0
    assert not w.heard(QUIET)  # cooldown
    now[0] += 1.5
    assert w.heard(QUIET)


def test_voice_route_answer_or_saved():
    def handler(request):
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 86400})
        assert request.url.path == "/api/v1/memory/voice"
        body = request.read()
        if b"where" in body:
            return httpx.Response(200, content=b"RIFFanswer", headers={"content-type": "audio/wav", "x-intent": "ask"})
        return httpx.Response(200, json={"id": "m1", "transcript": "keys on the counter", "intent": "remember"})

    api = make_client(handler)
    assert api.voice(b"where are my keys", None) == ("ask", b"RIFFanswer")
    intent, saved = api.voice(b"keys on the counter", b"\xff\xd8")
    assert intent == "remember" and saved["transcript"] == "keys on the counter"


def test_spooled_voice_clip_goes_to_the_voice_route(tmp_path):
    server = FakeServer()
    api = make_client(server, tmp_path)
    spool = Spool(tmp_path)
    spool.add(b"wav", b"jpg", None, kind="voice")
    spool.add(b"wav", b"jpg", None)  # a REMEMBER press
    assert spool.flush(api) == 2
    assert sorted(server.uploads) == ["/api/v1/memory/log", "/api/v1/memory/voice"]


# ---------------------------------------------------------------- on-device speech-to-text

def test_a_device_transcript_is_sent_instead_of_audio():
    seen = []

    def handler(req):
        if req.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        seen.append(req.content)
        return httpx.Response(200, json={"transcript": "keys on the hook"})

    api = make_client(handler)
    api.log_memory(None, b"jpeg", {"lat": 1.0}, transcript="keys on the hook")
    api.log_memory(b"RIFFwav", b"jpeg", None)
    with_words, with_audio = seen
    assert b'name="transcript"' in with_words and b"keys on the hook" in with_words and b"audio" not in with_words
    assert b'name="audio"' in with_audio and b"transcript" not in with_audio


def test_spool_keeps_only_the_words_when_the_device_transcribed(tmp_path):
    item = Spool(tmp_path).add(None, b"jpeg", None, kind="voice", transcript="wallet in the car")
    assert (item / "transcript.txt").read_text() == "wallet in the car" and not (item / "audio.wav").exists()
    sent = []

    class Api:
        def voice(self, wav, jpeg, where, transcript=None):
            sent.append((wav, transcript))

    assert Spool(tmp_path).flush(Api()) == 1 and sent == [(None, "wallet in the car")]


class FakeSTT:
    def __init__(self, text="", fail=False):
        self.text, self.fail = text, fail

    def transcribe(self, wav):
        if self.fail:
            raise RuntimeError("model crashed")
        return self.text


def device_with(stt, engine="moonshine", fallback=True):
    d = dev.Device.__new__(dev.Device)  # no GPIO/camera: only the speech-to-text choice
    d.cfg = Config(api_url="", auth0_domain="", auth0_audience="", client_id="", client_secret="",
                   stt_engine=engine, stt_fallback=fallback)
    d.stt = stt
    return d


def test_words_local_then_fallback_to_audio():
    assert device_with(FakeSTT("where are my keys?"))._words(b"wav") == "where are my keys?"
    assert device_with(FakeSTT(""))._words(b"wav") is None  # heard nothing: let the server try
    assert device_with(FakeSTT(fail=True))._words(b"wav") is None
    assert device_with(None)._words(b"wav") is None  # model missing at start-up
    assert device_with(None, engine="server")._words(b"wav") is None
    with pytest.raises(dev.NothingHeard):
        device_with(FakeSTT(""), fallback=False)._words(b"wav")  # audio never leaves the device


def test_config_reads_stt_settings(monkeypatch):
    for k, v in {"AUTH0_DOMAIN": "d", "AUTH0_AUDIENCE": "a", "DEVICE_CLIENT_ID": "i", "DEVICE_CLIENT_SECRET": "s",
                 "STT_ENGINE": "Vosk", "STT_FALLBACK": "false"}.items():
        monkeypatch.setenv(k, v)
    cfg = Config.from_env()
    assert cfg.stt_engine == "vosk" and cfg.stt_fallback is False and cfg.stt_model == ""


def test_local_stt_needs_its_model_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        dev.LocalSTT("moonshine", tmp_path / "missing")
    with pytest.raises(ValueError):
        dev.wav_pcm(wav_bytes(b"\0\0" * 100, rate=8000))
