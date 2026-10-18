"""ElevenLabs speech (TTS_PROVIDER=elevenlabs): the request it sends, the plan and monthly
character limits, and falling back to Gemini speech when it can't speak."""

import asyncio
import io
import json
import wave

import httpx
import pytest

from app.ai import AIServices
from app.config import Settings
from app.elevenlabs import ElevenLabs
from app.limits import Limiter, MemorySlots
from app.main import missing_settings


def wav(seconds=0.1, rate=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))
    return buf.getvalue()


class Api:
    """ElevenLabs as far as we use it."""

    def __init__(self, left=10_000, status=200, retry_after=None):
        self.left, self.status, self.retry_after = left, status, retry_after
        self.spoken, self.subscription_checks = [], 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["xi-api-key"] == "el-key"
        if req.url.path == "/v1/user/subscription":
            self.subscription_checks += 1
            return httpx.Response(200, json={"character_count": 10_000 - self.left, "character_limit": 10_000})
        assert req.url.path == "/v1/text-to-speech/voice-1" and req.url.params["output_format"] == "wav_24000"
        if self.status != 200:
            return httpx.Response(self.status, headers={"retry-after": self.retry_after} if self.retry_after else {})
        body = json.loads(req.content)
        self.spoken.append(body)
        self.left -= len(body["text"])
        return httpx.Response(200, content=wav())


def settings(**kw):
    return Settings(_env_file=None, tts_provider="elevenlabs", elevenlabs_api_key="el-key",
                    elevenlabs_voice_id="voice-1", gemini_tts_fallback_models="", **kw)


class FakeGemini:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)
        return b"RIFFgemini"


def services(api, limiter=None, **kw):
    ai = AIServices(settings(**kw), limiter)
    ai.elevenlabs = ElevenLabs(ai.s, http=httpx.AsyncClient(transport=httpx.MockTransport(api)))
    ai.gemini = FakeGemini()
    return ai


def run(coro):
    return asyncio.run(coro)


def test_speaks_with_elevenlabs_as_wav():
    api = Api()
    ai = services(api)
    audio = run(ai.speak("Your keys are on the kitchen counter."))
    assert audio[:4] == b"RIFF" and ai.speech_media_type == "audio/wav"
    (body,) = api.spoken
    assert body["text"] == "Your keys are on the kitchen counter." and body["model_id"] == "eleven_multilingual_v2"
    assert body["voice_settings"]["speed"] == 0.9
    assert ai.gemini.spoken == []


def test_the_plan_balance_is_checked_once_in_a_while_not_per_answer():
    api = Api()
    ai = services(api)
    for _ in range(3):
        run(ai.speak("Your keys are on the hook."))
    assert api.subscription_checks == 1 and len(api.spoken) == 3


def test_never_spends_past_what_the_plan_has_left():
    api = Api(left=520)  # under the 500-character reserve once this answer is counted
    ai = services(api)
    run(ai.speak("Your keys are on the kitchen counter."))
    assert api.spoken == [] and ai.gemini.spoken == ["Your keys are on the kitchen counter."]


def test_monthly_character_cap_falls_back_to_gemini():
    api = Api()
    ai = services(api, Limiter(MemorySlots()), elevenlabs_monthly_chars=40)
    run(ai.speak("Your keys are on the hook."))  # 26 characters
    run(ai.speak("Your glasses are by the sink."))  # would make 55: over the cap
    assert len(api.spoken) == 1 and ai.gemini.spoken == ["Your glasses are by the sink."]


def test_an_elevenlabs_429_pauses_it_for_everyone_and_gemini_speaks():
    store = MemorySlots()
    api = Api(status=429, retry_after="120")
    first = services(api, Limiter(store))
    assert run(first.speak("Your keys are on the hook.")) == b"RIFFgemini"
    ok_api = Api()
    second = services(ok_api, Limiter(store))  # another server instance: still cooling down
    run(second.speak("Your keys are on the hook."))
    assert ok_api.spoken == [] and second.gemini.spoken == ["Your keys are on the hook."]


@pytest.mark.parametrize("status", [401, 500])
def test_any_failure_still_gets_an_answer_spoken(status):
    ai = services(Api(status=status))
    assert run(ai.speak("Your keys are on the hook.")) == b"RIFFgemini"


def test_without_a_key_it_uses_gemini_and_health_says_so():
    ai = AIServices(Settings(_env_file=None, tts_provider="elevenlabs", elevenlabs_api_key=""))
    ai.gemini = FakeGemini()
    assert run(ai.speak("Hello.")) == b"RIFFgemini"
    assert "ELEVENLABS_API_KEY" in missing_settings(Settings(_env_file=None, tts_provider="elevenlabs"))


# ---------------------------------------------------------------- speech-to-text (STT_PROVIDER=elevenlabs)

from app.ai import audio_seconds  # noqa: E402
from app.elevenlabs import keyterms  # noqa: E402


class Scribe:
    """The speech-to-text endpoint: records what was sent, answers `text`."""

    def __init__(self, text="I put my swamphacks pins on the desk", status=200):
        self.text, self.status, self.sent = text, status, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/speech-to-text" and req.headers["xi-api-key"] == "el-key"
        self.sent.append(req.content)
        if self.status != 200:
            return httpx.Response(self.status)
        return httpx.Response(200, json={"text": self.text, "language_code": "eng"})


class GeminiEars:
    def __init__(self):
        self.heard = []

    async def transcribe(self, audio, content_type, hints=None):
        self.heard.append(hints)
        return "from gemini"


def listening(api, limiter=None, **kw):
    ai = AIServices(Settings(_env_file=None, stt_provider="elevenlabs", elevenlabs_api_key="el-key",
                             gemini_fallback_models="", **kw), limiter)
    ai.elevenlabs = ElevenLabs(ai.s, http=httpx.AsyncClient(transport=httpx.MockTransport(api)))
    ai.gemini = GeminiEars()
    return ai


def test_scribe_transcribes_with_the_hints_as_keyterms():
    api = Scribe()
    ai = listening(api)
    text = run(ai.transcribe("c.wav", wav(2.0, 16000), "audio/wav", ["Last Scene", "swamphacks pins"]))
    assert text == "I put my swamphacks pins on the desk" and ai.gemini.heard == []
    (body,) = api.sent
    assert b'name="model_id"\r\n\r\nscribe_v2' in body and body.count(b'name="keyterms"') == 2
    assert b"swamphacks pins" in body


def test_scribe_failures_fall_back_to_gemini_with_the_same_hints():
    ai = listening(Scribe(status=500))
    assert run(ai.transcribe("c.wav", wav(), "audio/wav", ["Last Scene"])) == "from gemini"
    assert ai.gemini.heard == [["Last Scene"]]


def test_monthly_audio_cap_counts_seconds():
    api = Scribe()
    ai = listening(api, Limiter(MemorySlots()), elevenlabs_stt_monthly_seconds=5)
    run(ai.transcribe("a.wav", wav(3.0, 16000), "audio/wav"))  # 3 s
    run(ai.transcribe("b.wav", wav(3.0, 16000), "audio/wav"))  # would make 6 s: Gemini instead
    assert len(api.sent) == 1 and len(ai.gemini.heard) == 1


def test_audio_seconds_and_keyterm_rules():
    assert audio_seconds(wav(2.2, 16000)) == 3 and audio_seconds(b"\0" * 64000) == 2 and audio_seconds(b"") == 1
    assert keyterms(["Last Scene", "last scene", "a [b]", "one two three four five six", "x" * 60, ""]) == ["Last Scene", "a b"]


def test_what_served_each_step_is_noted_for_the_live_view():
    from app.ai import served

    seen = {}
    served.set(seen)
    ai = listening(Scribe())
    run(ai.transcribe("c.wav", wav(), "audio/wav"))
    assert seen == {"transcribe": "elevenlabs scribe_v2"}
    failing = listening(Scribe(status=500))
    run(failing.transcribe("c.wav", wav(), "audio/wav"))
    assert seen["transcribe"] == "gemini-3.1-flash-lite"  # the Gemini model that heard it
    served.set(None)
