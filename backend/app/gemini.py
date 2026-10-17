"""Gemini implementations of every AI stage (google-genai SDK).

Kept separate from ai.py so each provider reads on its own; AIServices picks one per
stage from settings (STT_PROVIDER, EMBEDDING_PROVIDER, LLM_PROVIDER, TTS_PROVIDER,
VISION_PROVIDER).
"""

import io
import wave

from google import genai
from google.genai import types
from pydantic import BaseModel

from .config import Settings

# Gemini TTS returns raw 16-bit mono PCM at 24 kHz; we wrap it as WAV for the wearable.
TTS_SAMPLE_RATE = 24000


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
        # gemini-embedding-2 has no task_type; it uses these prefixes instead.
        content = f"task: search result | query: {text}" if query else f"title: none | text: {text}"
        result = await self.client.aio.models.embed_content(
            model=self.s.gemini_embedding_model,
            contents=content,
            config=types.EmbedContentConfig(output_dimensionality=self.s.embedding_dimensions),
        )
        return list(result.embeddings[0].values)

    async def generate(self, system: str, prompt: str) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return (response.text or "").strip()

    async def generate_json(self, system: str, contents: list, schema: type[BaseModel]) -> BaseModel | None:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return response.parsed

    async def speak(self, text: str) -> bytes:
        response = await self.client.aio.models.generate_content(
            model=self.s.gemini_tts_model,
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
