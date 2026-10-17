"""Gemini implementations of every AI stage (google-genai SDK).

Kept separate from ai.py so each provider reads on its own; AIServices picks one per
stage from settings (STT_PROVIDER, EMBEDDING_PROVIDER, LLM_PROVIDER, TTS_PROVIDER,
VISION_PROVIDER).
"""

import io
import re
import wave

from google import genai
from google.genai import types
from pydantic import BaseModel

from .config import Settings

# Gemini TTS returns raw 16-bit mono PCM at 24 kHz; we wrap it as WAV for the wearable.
TTS_SAMPLE_RATE = 24000


def least_thinking(model: str) -> types.ThinkingConfig | None:
    """The least thinking `model` accepts, for calls where the wearer is waiting.

    None means leave the model's default: it doesn't think, or we don't know it.
    Gemini 3 takes thinking_level (Flash goes down to MINIMAL, Pro only to LOW); 2.5 takes
    thinking_budget in tokens (Flash can turn it off with 0, Pro needs at least 128,
    Flash-Lite doesn't think unless asked).
    """
    name = model.lower().removeprefix("models/")
    if m := re.match(r"gemini-(\d+)(?:\.\d+)?-", name):
        if int(m.group(1)) >= 3:
            return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW if "pro" in name else types.ThinkingLevel.MINIMAL)
    if name.startswith("gemini-2.5-flash-lite"):
        return None
    if name.startswith("gemini-2.5-flash"):
        return types.ThinkingConfig(thinking_budget=0)
    if name.startswith("gemini-2.5-pro"):
        return types.ThinkingConfig(thinking_budget=128)
    return None


def _embed_input(text: str, query: bool) -> str:
    # gemini-embedding-2 has no task_type; it uses these prefixes instead.
    return f"task: search result | query: {text}" if query else f"title: none | text: {text}"


class GeminiAI:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = genai.Client(api_key=settings.gemini_api_key or None)

    async def transcribe(self, audio: bytes, content_type: str | None) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_model,
            contents=[
                types.Part.from_bytes(data=audio, mime_type=content_type or "audio/wav"),
                "Transcribe this speech verbatim. Reply with only the words spoken, "
                "or nothing at all if there is no speech.",
            ],
        )
        return (response.text or "").strip()

    async def embed(self, text: str, *, query: bool) -> list[float]:
        [embedding] = await self.embed_many([text], query=query)
        return embedding

    async def embed_many(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """One batchEmbedContents request for all of them, one vector per text.

        Each text goes in its own Content: the SDK merges a plain list of strings into a
        single Content for gemini-embedding-2, which would give one vector for all of them.
        """
        result = await self.client.aio.models.embed_content(
            model=self.s.gemini_embedding_model,
            contents=[types.Content(role="user", parts=[types.Part(text=_embed_input(t, query))]) for t in texts],
            config=types.EmbedContentConfig(output_dimensionality=self.s.embedding_dimensions),
        )
        if len(result.embeddings) != len(texts):
            raise ValueError(f"Asked for {len(texts)} embeddings, got {len(result.embeddings)}")
        return [list(e.values) for e in result.embeddings]

    def _thinking(self, fast: bool) -> types.ThinkingConfig | None:
        return least_thinking(self.s.gemini_model) if fast else None

    async def generate(self, system: str, prompt: str, *, fast: bool = False) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, thinking_config=self._thinking(fast)),
        )
        return (response.text or "").strip()

    async def generate_json(
        self, system: str, contents: list, schema: type[BaseModel], *, fast: bool = False
    ) -> BaseModel | None:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=self._thinking(fast),
            ),
        )
        return response.parsed

    async def speak(self, text: str) -> bytes:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_tts_model,
            # The style prefix stays English (the TTS model reads it as direction, not words to say);
            # the text after it goes through unchanged and is spoken in whatever language it's in.
            contents=f"Say warmly and calmly: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.s.gemini_tts_voice)
                    )
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        return pcm_to_wav(pcm)


def pcm_to_wav(pcm: bytes, rate: int = TTS_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
