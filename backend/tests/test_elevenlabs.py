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
    assert body["text"] == "Your keys are on the kitchen counter." and body["model_id"] == "eleven_flash_v2_5"
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
