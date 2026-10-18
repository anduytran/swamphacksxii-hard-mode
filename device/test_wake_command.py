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
    def __init__(self, answer=("remember", {"transcript": "keys on the counter", "photos_wanted": True}), error=None):
        self.answer, self.error, self.sent = answer, error, []
        self.followed = threading.Event()

    def voice(self, wav, jpegs, where, transcript=None, wake=False, defer=False, client=None):
        self.sent.append((wav, list(jpegs), wake))
        self.defer, self.client = defer, client
        self.last = {"followup": 7}
        if self.error:
            raise self.error
        return self.answer

    def followup(self, event_id, timeline=None, jpegs=None):
        self.followed_up = (event_id, timeline, list(jpegs or []))
        self.followed.set()
        return {"followup": "done", "saved_photo": 1}

    followed = None


def device(api, camera=None, said=b"\x01\x00" * 800, **cfg):
    d = dev.Device.__new__(dev.Device)
    d.cfg = config(**{"wake_photo_gap_seconds": 0.01, **cfg})
    d.api, d.camera = api, camera or Camera()
    d.led = SimpleNamespace(on=lambda: None, off=lambda: None, blink=lambda **k: None)
    def listen(endpointer, timeout, keep_all=False):  # long enough for photo 1, taken at the wake
        deadline = dev.time.monotonic() + 2
        while d.camera.shots < 1 and dev.time.monotonic() < deadline:
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


def test_saved_beeps_at_once_and_the_photos_follow(quiet):
    api = Api()
    preroll = b"\x07\x00" * 16000  # holding "Last Scene"
    device(api)._wake_command(preroll)
    ((wav, jpegs, wake),) = api.sent
    assert wake is True and api.defer is True and jpegs == [b"\xff\xd8photo1"]  # only photo 1 with the words
    assert preroll in wav and wav.startswith(b"RIFF")  # the phrase is in the clip, for the server to confirm
    assert quiet[-1] == dev.tone(1320, 90)  # "saved", before any later photo was waited for
    assert api.followed.wait(3)
    event_id, timeline, photos = api.followed_up
    assert event_id == 7 and photos == [b"\xff\xd8photo%d" % i for i in (1, 2, 3)]  # 1 at the wake, 2 and 3 after
    assert timeline["speech_end"] <= timeline["sent"] <= timeline["reply"] <= timeline["played"]


def test_a_failed_photo_still_sends_the_rest(quiet):
    api = Api()
    device(api, Camera(fail_at={2}))._wake_command(b"")
    assert api.followed.wait(3) and api.followed_up[2] == [b"\xff\xd8photo1", b"\xff\xd8photo3"]


def test_a_question_takes_no_photos_after_speaking(quiet):
    api, cam = Api(answer=("ask", b"RIFFanswer")), Camera()
    device(api, cam, wake_photo_gap_seconds=0.3)._wake_command(b"")
    assert api.followed.wait(3) and api.followed_up[2] == []
    threading.Event().wait(0.5)
    assert cam.shots == 2  # photo 1, and the one taken as speech ended; never the third
    assert b"RIFFanswer" in quiet


def test_a_false_wake_is_silent(quiet):
    device(Api(answer=("ignored", {"intent": "ignored"})))._wake_command(b"")
    assert quiet == [dev.LISTENING, dev.STOPPED]  # listening, stopped; nothing after a false wake


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
    d.voice = dev.VoiceCapture()
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


def test_the_command_sends_the_device_timings(quiet):
    api = Api(answer=("ask", b"RIFFanswer"))
    device(api)._wake_command(b"\x07\x00" * 100)
    assert api.client["photos"] == 1 and api.client["ended"] == "silence"
    assert set(api.client) >= {"recorded_s", "wake_score", "preroll_s"}
    assert api.followed.wait(2) and list(api.followed_up[1]) == ["speech_end", "sent", "stop_beep", "reply", "played"]


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


def test_never_ends_before_3_seconds_even_on_a_drop_or_silence():
    ep = dev.Endpointer(max_s=5, drop_db=15, min_s=3)
    n = feed(ep, [100] * 2 + [8000] * 14 + [600] * 60)  # a drop at 1.3 s, then quiet
    assert n == round(3 / 0.08) and ep.reason in ("drop", "silence")  # held until 3 s
    quiet = dev.Endpointer(max_s=5, min_s=3)
    n = feed(quiet, [8000] * 5 + [50] * 60)
    assert n == round(3 / 0.08) and quiet.reason == "silence"
    nothing = dev.Endpointer(max_s=5, min_s=3, timeout_s=3)
    feed(nothing, [50] * 60)
    assert nothing.reason == "nothing"


class SlowApi(Api):
    def __init__(self, delay, **kw):
        super().__init__(**kw)
        self.delay = delay

    def voice(self, *a, **k):
        threading.Event().wait(self.delay)
        return super().voice(*a, **k)


def test_a_filler_plays_only_while_a_slow_answer_is_on_its_way(quiet, tmp_path):
    fillers = dev.Fillers(tmp_path)
    fillers.clips = [b"RIFFlet-me-check"]
    slow = device(SlowApi(0.8, answer=("ask", b"RIFFanswer")))
    slow.fillers = fillers
    slow._wake_command(b"")
    assert quiet[-3:] == [dev.STOPPED, b"RIFFlet-me-check", b"RIFFanswer"]
    quiet.clear()
    fast = device(Api(answer=("ask", b"RIFFanswer")))
    fast.fillers = fillers
    fast._wake_command(b"")
    assert b"RIFFlet-me-check" not in quiet and quiet[-1] == b"RIFFanswer"


def test_fillers_are_fetched_once_and_kept(tmp_path):
    asked = []

    def fetch(i):
        asked.append(i)
        return (f"phrase {i}", b"RIFF%d" % i) if i < 2 else None

    f = dev.Fillers(tmp_path / "fillers", fetch)
    f.load()
    assert f.clips == [b"RIFF0", b"RIFF1"] and asked == [0, 1, 2]
    again = dev.Fillers(tmp_path / "fillers", fetch)
    again.load()
    assert again.clips == [b"RIFF0", b"RIFF1"] and asked == [0, 1, 2]  # from the folder, nothing fetched


def test_words_right_after_the_wake_are_kept_while_the_command_starts():
    v = dev.VoiceCapture()
    v.hold()
    for _ in range(5):
        v.add(frame(8000))  # "where are my keys", said before listen() began
    ep = dev.Endpointer(max_s=1.0)
    got = {}
    t = threading.Thread(target=lambda: got.update(pcm=v.listen(ep, 3, keep_all=True)))
    t.start()
    while t.is_alive():  # the mic keeps going, quiet now; max_s ends it
        v.add(frame(50))
        threading.Event().wait(0.002)
    assert got["pcm"] is not None and got["pcm"].startswith(frame(8000) * 5) and ep.spoke


def test_a_command_said_entirely_over_the_beep_still_counts():
    ep = dev.Endpointer(max_s=7, timeout_s=1, ignoring=lambda: ep.frames <= 8)
    n = feed(ep, [8000] * 8 + [50] * 40)  # talking while the beep plays, then quiet
    assert ep.reason == "said over the beep" and n is not None
    quiet = dev.Endpointer(max_s=7, timeout_s=1, ignoring=lambda: quiet.frames <= 2)
    feed(quiet, [8000] * 2 + [50] * 40)  # just the beep: nothing said
    assert quiet.reason == "nothing"


def test_the_listening_and_stopped_sounds_ship_with_the_device():
    import wave, io

    for clip in (dev.LISTENING, dev.STOPPED):
        with wave.open(io.BytesIO(clip)) as w:  # mono 16-bit, so the speaker EQ applies to them too
            assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() / w.getframerate() < 0.6
    assert dev.LISTENING != dev.tone(880) and dev.STOPPED != dev.chime(784, 523)
    assert dev.sound("missing.wav", b"RIFFbeep") == b"RIFFbeep"


def test_fillers_are_fetched_again_when_the_server_changes_them(tmp_path):
    phrases = ["Let me check.", "Hmm, let me see.", "Okay, let me think."]
    fetch = lambda i: (phrases[i], phrases[i].encode()) if i < len(phrases) else None
    folder = tmp_path / "fillers"
    dev.Fillers(folder, fetch, lambda: list(phrases)).load()
    assert len(dev.Fillers(folder).clips) == 3
    phrases.remove("Okay, let me think.")
    f = dev.Fillers(folder, fetch, lambda: list(phrases))
    f.load()
    assert f.clips == [b"Let me check.", b"Hmm, let me see."]
    offline = dev.Fillers(folder, fetch, lambda: (_ for _ in ()).throw(OSError("offline")))
    offline.load()
    assert len(offline.clips) == 2  # can't check: keeps what it has
