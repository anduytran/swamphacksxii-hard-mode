"""Live conversation mode, against a fake Gemini Live session. No hardware or network needed.

Run from the repo root:  backend/.venv/bin/python -m pytest device
"""

import array
import asyncio
import contextlib
import json
import threading
from types import SimpleNamespace

import httpx

import lastscene_device as ld
from lastscene_device import FRAME_SAMPLES, Config, Device, LiveSession, Spool, held_quietly, wav_bytes
from test_device import make_client

LOUD = array.array("h", [3000] * FRAME_SAMPLES).tobytes()
QUIET = bytes(FRAME_SAMPLES * 2)


def msg(data=None, tool_call=None, said=None):
    content = SimpleNamespace(input_transcription=None,
                              output_transcription=SimpleNamespace(text=said) if said else None)
    return SimpleNamespace(data=data, tool_call=tool_call, server_content=content)


def call(name, args, id="c1"):
    return SimpleNamespace(function_calls=[SimpleNamespace(id=id, name=name, args=args)])


class FakeLive:
    """Gemini Live's AsyncSession: receive() gives one model turn per call, as the SDK does."""

    def __init__(self, turns=(), stay_open=True):
        self.turns = [list(t) for t in turns]
        self.stay_open = stay_open
        self.audio, self.tool_responses = [], []
        self.connected = self.closed = False

    async def send_realtime_input(self, *, audio):
        self.audio.append(audio)

    async def send_tool_response(self, *, function_responses):
        self.tool_responses.append(function_responses)

    async def receive(self):
        if self.turns:
            for m in self.turns.pop(0):
                yield m
        elif self.stay_open:
            await asyncio.Event().wait()  # an open session with nothing to say

    @contextlib.asynccontextmanager
    async def connect(self):
        self.connected = True
        try:
            yield self
        finally:
            self.closed = True


class FakePlayer:
    def __init__(self):
        self.played, self.speaking = [], False

    def play(self, pcm):
        self.played.append(pcm)

    def busy(self):
        return self.speaking

    def close(self):
        pass


def session(live, tools=None, player=None, **kw):
    kw.setdefault("silence_s", 0.3)
    kw.setdefault("tick_s", 0.02)
    return LiveSession(live.connect, tools or {}, player or FakePlayer(), **kw)


class RecallServer:
    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 86400})
        self.requests.append((request.url.path, request.content))
        if request.url.path == "/api/v1/memory/recall":
            return httpx.Response(200, json={"question": "q", "facts": ["f"], "text": '(2 hours ago) said "keys on the hook"'})
        if request.url.path == "/api/v1/live/token":
            return httpx.Response(200, json={"token": "auth_tokens/x", "model": "gemini-3.8-live", "api_version": "v1alpha",
                                             "expires_at": "2026-10-17T20:00:00+00:00"})
        return httpx.Response(200, json={"id": "1", "transcript": "saved"})


def test_a_tool_call_is_answered_from_the_recall_endpoint():
    server = RecallServer()
    api = make_client(server)
    live = FakeLive([[msg(tool_call=call("recall_memories", {"question": "where are my keys?"}))],
                     [msg(data=b"pcm-answer", said="On the hook, two hours ago.")]])
    player = FakePlayer()
    reason = asyncio.run(session(live, {"recall_memories": api.recall}, player).run())

    assert reason == "silence" and live.closed
    path, body = server.requests[0]
    assert path == "/api/v1/memory/recall" and json.loads(body) == {"question": "where are my keys?"}
    [[response]] = live.tool_responses
    assert response == {"id": "c1", "name": "recall_memories",
                        "response": {"result": '(2 hours ago) said "keys on the hook"'}}
    assert player.played == [b"pcm-answer"]


def test_a_failing_or_unknown_tool_tells_gemini_instead_of_crashing():
    def broken(question):
        raise httpx.ConnectError("no wifi")

    live = FakeLive([[msg(tool_call=call("recall_memories", {"question": "keys?"}))],
                     [msg(tool_call=call("delete_everything", {}, id="c2"))]])
    asyncio.run(session(live, {"recall_memories": broken}).run())
    assert [r[0]["response"].keys() for r in live.tool_responses] == [{"error": 1}.keys()] * 2


def test_mic_audio_is_sent_only_while_connected_and_not_over_its_own_voice():
    live, player = FakeLive(), FakePlayer()
    s = session(live, player=player, silence_s=0.5)
    s.feed(LOUD)  # before the session (i.e. before the wake phrase): dropped

    async def talk():
        task = asyncio.create_task(s.run())
        while not s.active:
            await asyncio.sleep(0.01)
        threading.Thread(target=s.feed, args=(LOUD,)).start()  # the mic thread
        await asyncio.sleep(0.05)
        player.speaking = True
        s.feed(LOUD)  # Gemini is talking: the mic is muted
        await asyncio.sleep(0.05)
        player.speaking = False
        return await task

    assert asyncio.run(talk()) == "silence"
    assert live.audio == [{"data": LOUD, "mime_type": "audio/pcm;rate=16000"}]


def test_speech_keeps_it_going_and_silence_ends_it():
    live = FakeLive()
    s = session(live, silence_s=0.3)

    async def talk():
        task = asyncio.create_task(s.run())
        loop = asyncio.get_running_loop()
        start = loop.time()
        while not s.active:
            await asyncio.sleep(0.01)
        for _ in range(5):  # 0.5 s of talking, longer than silence_s
            s.feed(LOUD)
            await asyncio.sleep(0.1)
        s.feed(QUIET)  # quiet frames are still sent, but don't count as talking
        reason = await task
        return reason, loop.time() - start

    reason, took = asyncio.run(talk())
    # the last loud frame came at ~0.4 s, so it ends ~0.3 s later (not 0.3 s after starting)
    assert reason == "silence" and took >= 0.65
    assert len(live.audio) == 6


def test_button_max_time_and_a_closed_session_end_it():
    s = session(FakeLive(), silence_s=10)

    async def press():
        task = asyncio.create_task(s.run())
        while not s.active:
            await asyncio.sleep(0.01)
        threading.Thread(target=s.stop).start()
        return await task

    assert asyncio.run(press()) == "button"
    assert asyncio.run(session(FakeLive(), silence_s=10, max_s=0.2).run()) == "max"
    assert asyncio.run(session(FakeLive(stay_open=False), silence_s=10).run()) == "closed"


def test_connection_errors_propagate_for_the_fallback():
    @contextlib.asynccontextmanager
    async def refused():
        raise OSError("handshake failed")
        yield

    s = LiveSession(refused, {}, FakePlayer())
    try:
        asyncio.run(s.run())
    except OSError:
        pass
    else:
        raise AssertionError("expected the error")
    assert not s.active


def test_a_quiet_long_press_starts_a_conversation():
    assert held_quietly(wav_bytes(QUIET * 30), 2)  # 2.4 s, silent
    assert not held_quietly(wav_bytes(QUIET * 10), 2)  # too short
    assert not held_quietly(wav_bytes(QUIET * 10 + LOUD * 10 + QUIET * 10), 2)  # someone spoke


def test_config_reads_live_settings(monkeypatch):
    for k, v in {"AUTH0_DOMAIN": "d", "AUTH0_AUDIENCE": "a", "DEVICE_CLIENT_ID": "i", "DEVICE_CLIENT_SECRET": "s"}.items():
        monkeypatch.setenv(k, v)
    cfg = Config.from_env()
    assert not cfg.live and cfg.live_max_seconds == 90 and cfg.live_silence_seconds == 8
    monkeypatch.setenv("LIVE", "true")
    monkeypatch.setenv("LIVE_MAX_SECONDS", "45")
    cfg = Config.from_env()
    assert cfg.live and cfg.live_max_seconds == 45


# ---------------------------------------------------------------- the Device around it (no GPIO)

class FakeVoice:
    listening = False

    def __init__(self, pcm):
        self.pcm = pcm

    def listen(self, endpointer, timeout_s, keep_all=False):
        return self.pcm


def fake_device(monkeypatch, tmp_path, api, live=True):
    monkeypatch.setattr(ld, "play_wav", lambda device, data: None)
    monkeypatch.setattr(ld, "whereabouts", lambda cfg: {})
    dev = Device.__new__(Device)  # skips the GPIO / camera setup
    dev.cfg = api.cfg
    dev.cfg.live = live
    dev.api = api
    dev.spool = Spool(tmp_path / "spool")
    dev.camera = SimpleNamespace(capture_jpeg=lambda: b"\xff\xd8photo")
    dev.led = SimpleNamespace(on=lambda: None, off=lambda: None, blink=lambda **kw: None)
    dev.busy = threading.Lock()
    dev.stt = None
    dev.live = None
    dev.voice = FakeVoice(LOUD * 10)
    dev.wake = SimpleNamespace(reset=lambda: None)
    return dev


def test_wake_phrase_falls_back_to_the_voice_route_when_live_fails(monkeypatch, tmp_path):
    def server(request):
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 86400})
        server.paths.append(request.url.path)
        if request.url.path == "/api/v1/live/token":
            return httpx.Response(404, json={"detail": "Live mode is off"})
        return httpx.Response(200, content=b"RIFFanswer", headers={"x-intent": "ask", "content-type": "audio/wav"})

    server.paths = []
    dev = fake_device(monkeypatch, tmp_path, make_client(server))
    dev._wake()
    assert server.paths == ["/api/v1/live/token", "/api/v1/memory/voice"]
    assert not dev.busy.locked() and dev.live is None


def test_wake_phrase_starts_a_live_conversation(monkeypatch, tmp_path):
    server = RecallServer()
    dev = fake_device(monkeypatch, tmp_path, make_client(server))
    live = FakeLive([[msg(tool_call=call("save_memory", {"text": "keys on the hook"}))]])
    monkeypatch.setattr(ld, "gemini_live_connect", lambda token: live.connect())
    monkeypatch.setattr(ld, "PcmPlayer", lambda device: FakePlayer())
    dev.cfg.live_silence_seconds = 0.3
    dev._wake()
    assert [p for p, _ in server.requests] == ["/api/v1/live/token", "/api/v1/memory/log"]  # never /memory/voice
    log_body = server.requests[1][1]
    assert b"keys on the hook" in log_body and b"photo" in log_body  # the words and the photo
    assert live.tool_responses[0][0]["response"] == {"result": "Saved."}
    assert dev.live is None


def test_save_memory_is_spooled_when_offline(monkeypatch, tmp_path):
    def offline(request):
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 86400})
        raise httpx.ConnectError("no wifi", request=request)

    dev = fake_device(monkeypatch, tmp_path, make_client(offline))
    assert "upload" in dev._save_spoken("glasses on the nightstand")
    (item,) = dev.spool.pending()
    assert (item / "transcript.txt").read_text() == "glasses on the nightstand"


def test_any_button_ends_a_live_conversation(monkeypatch, tmp_path):
    dev = fake_device(monkeypatch, tmp_path, make_client(RecallServer()))
    stopped = []
    dev.live = SimpleNamespace(stop=lambda: stopped.append(True))
    dev._begin("remember")
    assert stopped == [True] and not dev.busy.locked()
