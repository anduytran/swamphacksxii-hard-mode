"""ElevenLabs text-to-speech for the wearable's spoken answers (TTS_PROVIDER=elevenlabs).

POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}, `xi-api-key` header, WAV out
(output_format=wav_24000), which the Pi plays with aplay as is. eleven_flash_v2_5 by default:
~75 ms model latency, 32 languages, half the per-character price of the other models.

Plans are billed in characters, so AIServices caps characters per month (ELEVENLABS_MONTHLY_CHARS)
and this client also checks what the account really has left (GET /v1/user/subscription, cached
a few minutes), so a request that would go past the plan is never sent. Any failure falls back to
Gemini speech.
"""

import time

import httpx

from .config import Settings

API = "https://api.elevenlabs.io/v1"
SUBSCRIPTION_TTL_S = 300


class ElevenLabsError(Exception):
    """An API error, shaped like the SDK errors limits.provider_busy reads (status_code, response)."""

    def __init__(self, response: httpx.Response):
        super().__init__(f"ElevenLabs {response.status_code}")
        self.status_code, self.response = response.status_code, response


class ElevenLabs:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None, clock=time.monotonic):
        self.s = settings
        self.http = http
        self.clock = clock
        self._left: tuple[float, int | None] | None = None  # (checked at, characters left)

    def _client(self) -> httpx.AsyncClient:
        if self.http is None:
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
        return self.http

    @property
    def _headers(self) -> dict:
        return {"xi-api-key": self.s.elevenlabs_api_key}

    async def characters_left(self) -> int | None:
        """What the plan has left this billing period; None if ElevenLabs didn't say."""
        if self._left and self.clock() - self._left[0] < SUBSCRIPTION_TTL_S:
            return self._left[1]
        resp = await self._client().get(f"{API}/user/subscription", headers=self._headers)
        if resp.status_code >= 400:
            raise ElevenLabsError(resp)
        body = resp.json()
        left = None
        if isinstance(body.get("character_limit"), int) and isinstance(body.get("character_count"), int):
            left = body["character_limit"] - body["character_count"]
        self._left = (self.clock(), left)
        return left

    async def speak(self, text: str) -> bytes:
        left = await self.characters_left()
        if left is not None and left < len(text) + self.s.elevenlabs_reserve_chars:
            raise ElevenLabsError(httpx.Response(429, headers={"retry-after": "3600"}))  # plan nearly used up
        resp = await self._client().post(
            f"{API}/text-to-speech/{self.s.elevenlabs_voice_id}",
            params={"output_format": "wav_24000"},
            headers=self._headers,
            json={
                "text": text,
                "model_id": self.s.elevenlabs_model,
                # A little slower and steadier than default, for listeners with memory loss.
                "voice_settings": {"stability": 0.6, "similarity_boost": 0.75, "speed": self.s.elevenlabs_speed},
            },
        )
        if resp.status_code >= 400:
            raise ElevenLabsError(resp)
        audio = resp.content
        if audio[:4] != b"RIFF":
            raise ValueError("ElevenLabs didn't return WAV audio")
        if self._left and self._left[1] is not None:
            self._left = (self._left[0], self._left[1] - len(text))
        return audio
