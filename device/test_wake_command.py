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

    def voice(self, wav, jpegs, where, transcript=None, wake=False, defer=False, client=None):
        self.sent.append((wav, list(jpegs), wake))
        self.defer, self.client = defer, client
        if self.error:
            raise self.error
        return self.answer


def device(api, camera=None, said=b"\x01\x00" * 800, **cfg):
    d = dev.Device.__new__(dev.Device)
    d.cfg = config(**{"wake_photo_gap_seconds": 0.01, **cfg})
    d.api, d.camera = api, camera or Camera()
    d.led = SimpleNamespace(on=lambda: None, off=lambda: None, blink=lambda **k: None)
    def listen(endpointer, timeout, keep_all=False):  # speaking long enough for every photo to be taken
        deadline = dev.time.monotonic() + 2
        while d.camera.shots < d.cfg.wake_photos and dev.time.monotonic() < deadline:
            threading.Event().wait(0.005)
        endpointer.reason = "silence"
        return said

    d.voice = SimpleNamespace(listen=listen)
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
    assert Config.local({}).wake_confirm is True and Config.local({}).wake_preroll_seconds == 1.5


def test_a_mic_device_that_cant_record_says_how_to_fix_it(caplog):
    mic = dev.MicStream("default", lambda frame: None)
    mic._proc = SimpleNamespace(
        stdout=SimpleNamespace(read=lambda n: b""),
        stderr=SimpleNamespace(read=lambda: b"ALSA lib pcm_asym.c:105:(_snd_pcm_asym_open) capture slave is not defined\n"
                                            b"arecord: main:850: audio open error: Invalid argument\n"),
        wait=lambda: 1,
    )
    with caplog.at_level("ERROR"):
        mic._read()
    text = caplog.text
    assert "audio open error" in text and "AUDIO_IN_DEVICE=default can't record" in text and "plughw:1,0" in text


# ---------------------------------------------------------------- how long it listens, and the follow-up

def frame(level: int) -> bytes:
    return (level.to_bytes(2, "little", signed=True) + (-level).to_bytes(2, "little", signed=True)) * 640


def feed(ep, levels):
    for i, lv in enumerate(levels):
        if ep.feed(frame(lv)) in ("done", "nothing"):
            return i + 1
    return None


def test_a_sharp_drop_in_loudness_ends_the_clip_quickly():
    ep = dev.Endpointer(max_s=5, drop_db=15)
    n = feed(ep, [100] * 3 + [8000] * 15 + [600] * 20)  # 1.2 s of speech, then a fall of ~22 dB (above the floor)
    assert ep.reason == "drop" and n == 3 + 15 + 6  # 0.5 s of the drop
    plain = dev.Endpointer(max_s=5)
    feed(plain, [100] * 3 + [8000] * 15 + [600] * 20)
    assert plain.reason != "drop"


def test_short_dips_between_words_and_the_beep_dont_end_it():
    ep = dev.Endpointer(max_s=5, drop_db=15, ignore_s=0.4)
    assert feed(ep, [9000] * 4) is None and not ep.spoke  # the beep, while the mic listens
    assert feed(ep, [8000] * 8 + [600] * 2 + [8000] * 8) is None and ep.spoke  # a 160 ms gap between words


def test_speech_is_cut_off_at_the_limit():
    ep = dev.Endpointer(max_s=5, drop_db=15)
    n = feed(ep, [8000] * 200)
    assert ep.reason == "max" and n == round(5 / 0.08)


def test_the_command_is_deferred_with_the_device_timings_then_followed_up(quiet):
    class Followed(Api):
        def __init__(self):
            super().__init__(answer=("ask", b"RIFFanswer"))
            self.last = {"heard": "Last Scene, where are my keys", "answer": "On the desk.", "steps": "transcribe;dur=400",
                         "followup": 7}
            self.followed = threading.Event()

        def followup(self, event_id, timeline=None):
            self.followed_id, self.timeline = event_id, timeline
            self.followed.set()
            return {"followup": "done", "followup_ms": 2100}

    api = Followed()
    device(api)._wake_command(b"\x07\x00" * 100)
    assert api.defer is True and api.client["photos"] == 4 and api.client["ended"] == "silence"
    assert set(api.client) >= {"recorded_s", "photo_at_s", "wake_score", "preroll_s"}
    assert api.followed.wait(2) and api.followed_id == 7 and b"RIFFanswer" in quiet
    t = api.timeline
    assert list(t) == ["speech_end", "photos_ready", "sent", "reply", "played"]
    assert t["speech_end"] <= t["photos_ready"] <= t["sent"] <= t["reply"] <= t["played"]


def test_a_late_beep_doesnt_set_the_peak_and_cut_the_speech_short():
    beeping = [True]
    ep = dev.Endpointer(max_s=5, drop_db=15, ignore_s=0.2, ignoring=lambda: beeping[0])
    assert feed(ep, [100] * 3 + [30000] * 4) is None and not ep.spoke  # the beep, 0.25 s late
    beeping[0] = False
    assert feed(ep, [3000] * 15) is None and ep.spoke  # ordinary speech, far quieter than the beep: still going
    assert ep.reason == ""


def test_a_second_client_is_spotted_but_not_a_check_or_its_shell(monkeypatch, tmp_path):
    procs = {101: [b"/opt/lastscene/venv/bin/python", b"/opt/lastscene/lastscene_device.py"],
             102: [b"bash", b"-c", b"set -a; /opt/lastscene/venv/bin/python /opt/lastscene/lastscene_device.py"],
             103: [b"/opt/lastscene/venv/bin/python", b"/opt/lastscene/lastscene_device.py", b"--check"]}
    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="\n".join(map(str, procs))))
    real = dev.Path

    def fake_path(p):
        if str(p).startswith("/proc/"):
            pid = int(str(p).split("/")[2])
            return SimpleNamespace(read_bytes=lambda: b"\0".join(procs[pid]))
        return real(p)

    monkeypatch.setattr(dev, "Path", fake_path)
    assert dev.already_running() == 101
    del procs[101]
    assert dev.already_running() is None
