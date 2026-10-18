"""The wake phrase as a command (WAKE_CONFIRM=true, the default): the clip keeps the audio from before
the wake (it holds "Last Scene"), photo 1 is taken at once and the rest after speaking, and one request
goes to /memory/voice with wake=true. No hardware or network.

Run from the repo root:  backend/.venv/bin/python -m pytest device
"""

import threading
from types import SimpleNamespace

import httpx
import pytest

import lastscene_device as dev
from lastscene_device import ApiClient, Config


def config(**kw):
    return Config(api_url="https://api.test", auth0_domain="tenant.auth0.test", auth0_audience="a", client_id="id",
                  client_secret="s", **kw)


class Camera:
    def __init__(self, fail_at=()):
        self.shots, self.fail_at, self.lock = 0, set(fail_at), threading.Lock()

    def capture_jpeg(self):
        with self.lock:
            self.shots += 1
            n = self.shots
        if n in self.fail_at:
            raise RuntimeError("camera Pi busy")
        return b"\xff\xd8photo%d" % n


class Api:
    def __init__(self, answer=("remember", {"transcript": "keys on the counter", "photo": 2}), error=None):
        self.answer, self.error, self.sent = answer, error, []

    def voice(self, wav, jpegs, where, transcript=None, wake=False):
        self.sent.append((wav, list(jpegs), wake))
        if self.error:
            raise self.error
        return self.answer


def device(api, camera=None, said=b"\x01\x00" * 800, **cfg):
    d = dev.Device.__new__(dev.Device)
    d.cfg = config(**cfg)
    d.api, d.camera = api, camera or Camera()
    d.led = SimpleNamespace(on=lambda: None, off=lambda: None, blink=lambda **k: None)
    d.voice = SimpleNamespace(listen=lambda endpointer, timeout: said)
    d.spool = SimpleNamespace(added=[], add=lambda *a, **k: d.spool.added.append((a, k)))
    return d


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    sounds = []
    monkeypatch.setattr(dev, "play_wav", lambda device, data: sounds.append(data))
    monkeypatch.setattr(dev, "whereabouts", lambda cfg: {})
    monkeypatch.setattr(dev.time, "sleep", lambda s: None)
    return sounds


def test_one_request_with_the_wake_phrase_audio_and_four_photos_in_order(quiet):
    api = Api()
    preroll = b"\x07\x00" * 16000  # 2 s holding "Last Scene"
    device(api)._wake_command(preroll)
    ((wav, jpegs, wake),) = api.sent
    assert wake is True and jpegs == [b"\xff\xd8photo%d" % i for i in range(1, 5)]
    assert preroll in wav and wav.startswith(b"RIFF")  # the phrase is in the clip, for the server to confirm
    assert quiet[-1] == dev.tone(1320, 90)  # "saved"


def test_photo_count_and_a_failed_photo(quiet):
    api = Api()
    device(api, Camera(fail_at={2}), wake_photos=3)._wake_command(b"")
    assert api.sent[0][1] == [b"\xff\xd8photo1", b"\xff\xd8photo3"]  # photo 2 failed; the rest still go


def test_a_false_wake_is_silent(quiet):
    device(Api(answer=("ignored", {"intent": "ignored"})))._wake_command(b"")
    assert quiet == [dev.tone(880)]  # only the "listening" beep: nothing after a false wake


def test_answers_and_found_are_played(quiet):
    device(Api(answer=("ask", b"RIFFanswer")))._wake_command(b"")
    device(Api(answer=("found", b"RIFFglad")))._wake_command(b"")
    assert b"RIFFanswer" in quiet and b"RIFFglad" in quiet


def test_nothing_said_after_the_beep_sends_nothing(quiet):
    api = Api()
    device(api, said=None)._wake_command(b"")
    assert api.sent == [] and quiet[-1] == dev.tone(440)


def test_offline_keeps_the_command_with_its_first_photo(quiet):
    d = device(Api(error=httpx.ConnectError("offline")))
    d._wake_command(b"")
    ((args, kw),) = d.spool.added
    assert args[1] == b"\xff\xd8photo1" and kw == {"kind": "voice"}


def test_the_client_sends_every_photo_and_the_wake_flag():
    seen = []

    def handler(req):
        if req.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        seen.append(req.content)
        return httpx.Response(200, json={"intent": "ignored"})

    api = ApiClient(config(), httpx.Client(transport=httpx.MockTransport(handler)))
    assert api.voice(b"RIFFwav", [b"\xff\xd8a", b"\xff\xd8b", b"\xff\xd8c"], {}, wake=True) == ("ignored", {"intent": "ignored"})
    body = seen[0]
    assert body.count(b'name="image"') == 3 and b'name="wake"\r\n\r\ntrue' in body and b'name="audio"' in body
    api.voice(b"RIFFwav", b"\xff\xd8one", {})  # the original one-photo form
    assert seen[1].count(b'name="image"') == 1 and b'name="wake"' not in seen[1]


def test_the_frame_handler_hands_the_preroll_to_the_wake():
    d = dev.Device.__new__(dev.Device)
    d.cfg = config()
    d.live, d.busy = None, threading.Lock()
    d.stream_rec = SimpleNamespace(active=False)
    d.voice = SimpleNamespace(listening=False)
    d._recent = dev.deque(maxlen=25)
    fired = threading.Event()
    got = []
    d._wake = lambda preroll: (got.append(preroll), fired.set())
    frames = [bytes([i]) * 4 for i in range(30)]
    d.wake = SimpleNamespace(heard=lambda frame: frame == frames[-1])
    for f in frames:
        d._on_frame(f)
    assert fired.wait(2) and got == [b"".join(frames[-25:])]  # the last 2 s, wake frame included


def test_config():
    cfg = Config.local({"WAKE_PHOTOS": "9", "WAKE_PHOTO_GAP_SECONDS": "1.5", "WAKE_CONFIRM": "false"})
    assert cfg.wake_photos == 4 and cfg.wake_photo_gap_seconds == 1.5 and cfg.wake_confirm is False
    assert Config.local({}).wake_confirm is True and Config.local({}).wake_preroll_seconds == 2.0
