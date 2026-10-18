"""Gemini implementations of every AI stage (google-genai SDK).

Kept separate from ai.py so each provider reads on its own; AIServices picks one per
stage from settings (STT_PROVIDER, EMBEDDING_PROVIDER, LLM_PROVIDER, TTS_PROVIDER,
VISION_PROVIDER).
"""

import io
from contextlib import contextmanager
from contextvars import ContextVar
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


# The model this call should use, set by AIServices while it walks its fallback list
# (GEMINI_MODEL, then GEMINI_FALLBACK_MODELS). Unset: the configured primary model.
_text_model: ContextVar[str | None] = ContextVar("gemini_text_model", default=None)
_tts_model: ContextVar[str | None] = ContextVar("gemini_tts_model", default=None)


@contextmanager
def using(model: str | None, *, tts: bool = False):
    var = _tts_model if tts else _text_model
    token = var.set(model)
    try:
        yield
    finally:
        var.reset(token)


class GeminiAI:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key or None,
            # Retry real server errors once. "Too many requests" (429) and "overloaded" (503) are
            # left to limits.py, which makes every instance wait instead of each retrying for a minute.
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=2, http_status_codes=[500, 502, 504])
            ),
        )

    async def transcribe(self, audio: bytes, content_type: str | None, hints: list[str] | None = None) -> str:
        ask = "Transcribe this speech verbatim. Reply with only the words spoken, or nothing at all if there is no speech."
        if hints:
            ask += " Names and words that may come up, spelled like this: " + "; ".join(hints[:100]) + "."
        response = await self.client.aio.models.generate_content(
            model=self._model(),
            contents=[types.Part.from_bytes(data=audio, mime_type=content_type or "audio/wav"), ask],
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

    def _model(self) -> str:
        return _text_model.get() or self.s.gemini_model

    def _tts(self) -> str:
        return _tts_model.get() or self.s.gemini_tts_model

    def _thinking(self, fast: bool) -> types.ThinkingConfig | None:
        return least_thinking(self._model()) if fast else None

    async def generate(self, system: str, prompt: str, *, fast: bool = False) -> str:
        response = await self.client.aio.models.generate_content(
            model=self._model(),
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, thinking_config=self._thinking(fast)),
        )
        return (response.text or "").strip()

    async def generate_json(
        self, system: str, contents: list, schema: type[BaseModel], *, fast: bool = False
    ) -> BaseModel | None:
        response = await self.client.aio.models.generate_content(
            model=self._model(),
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
            model=self._tts(),
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


    async def name_place(self, lat: float, lng: float) -> dict | None:
        """Grounding with Google Maps (Interactions API): the place at lat/lng, as
        {"name": "Starbucks, Reitz Union", "sources": [{"title", "url"}]}, or None.

        English prompts only (a limit of the tool). store=False: nothing kept server-side."""
        interaction = await self.client.aio.interactions.create(
            model=self._model(),
            input=PLACE_NAME_PROMPT,
            tools=[{"type": "google_maps", "latitude": lat, "longitude": lng}],
            store=False,
        )
        return parse_place_name(interaction)


PLACE_NAME_PROMPT = (
    "What public place is at the user's location? Reply with only its name and, if it is "
    'inside a larger building or campus, that too, like "Starbucks, Reitz Union". '
    "If you can't tell, reply with nothing."
)


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _is_link(url) -> bool:
    # The dashboard links these; never pass on a javascript: or other non-web URL.
    return isinstance(url, str) and url.startswith("https://")


def parse_place_name(interaction) -> dict | None:
    """The answer text and its Google Maps sources (place citations, else the tool's places).

    Google's terms require showing these sources with the name, so no sources, no name."""
    text, cited, places = "", [], []
    for step in _get(interaction, "steps") or []:
        kind = _get(step, "type")
        if kind == "model_output":
            for content in _get(step, "content") or []:
                if _get(content, "type") != "text":
                    continue
                text += _get(content, "text") or ""
                for a in _get(content, "annotations") or []:
                    if _get(a, "type") == "place_citation" and _is_link(_get(a, "url")):
                        cited.append({"title": _get(a, "name") or "Google Maps", "url": _get(a, "url")})
        elif kind == "google_maps_result":
            for result in _get(step, "result") or []:
                for p in _get(result, "places") or []:
                    if _is_link(_get(p, "url")):
                        places.append({"title": _get(p, "name") or "Google Maps", "url": _get(p, "url")})
    sources = cited or places
    name = " ".join((text or _get(interaction, "output_text") or "").split()).strip(' ."')
    if not name or not sources or len(name) > 120:
        return None
    unique = list({s["url"]: s for s in sources}.values())[:3]
    return {"name": name, "sources": unique}


def pcm_to_wav(pcm: bytes, rate: int = TTS_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
