"""Every AI stage, each routed to a provider chosen in settings.

Defaults are Gemini end to end (see gemini.py). The original providers stay available
per stage: OpenAI for speech-to-text, embeddings and text-to-speech, Claude for the
language tasks.
"""

import functools
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache, cached_property
from typing import Literal

import anthropic
import openai
from pydantic import BaseModel, create_model

from .config import Settings
from .home_map import FloorPlan, floor_plan
from .rag import ITEM, TEXT, RetrievedMemory
from .elevenlabs import ElevenLabs
from .gemini import GeminiAI, using
from .limits import Limiter, RateLimited, Rule, per, provider_busy

log = logging.getLogger(__name__)
from . import languages

ANSWER_SYSTEM_PROMPT = """\
You are LastScene, a gentle voice assistant on a wearable that helps someone \
with memory loss find everyday items. Your reply is spoken aloud through a small speaker.

You are given the user's question and the memories retrieved for it. Each memory says \
how long ago it happened, what the user said, and what the wearable's camera saw \
(a description, the items and where they are, and any readable text). Answer using \
only those memories; when several are about the same item, the most recent one is \
where it is now.
- Reply in one or two short, warm, plain sentences. No lists, markdown or emoji.
- Say where the item is and roughly when it was left there, e.g. "about two hours ago".
- If none of the notes are about what they asked, say kindly that you don't have a \
note about it yet and suggest they tell you next time they put it down.
- Latency-sensitive; begin your visible answer immediately."""


EXTRACT_SYSTEM_PROMPT = """\
You read one sentence spoken to a memory-aid wearable. It is either a note about \
putting something down ("I'm putting my keys on the kitchen counter") or a question \
("Where are my glasses?").

List each physical item it is about:
- name: a short lowercase singular noun phrase a family member would recognise, with no \
possessive ("keys", "reading glasses", "wallet", "blue umbrella"). Use the same name \
every time for the same kind of item.
- location: where the item was put, as a short phrase ("kitchen counter", "coat pocket"). \
Use an empty string for questions or when no place is said.
Return an empty list if no physical item is mentioned."""


SCENE_SYSTEM_PROMPT = """\
This photo was taken by a chest-worn camera at the moment its wearer, who has memory \
loss, said: "{transcript}". Turn it into text a search engine can use to find the item later.
- description: one plain sentence (under 25 words) saying where things are, naming \
visible landmarks ("Keys on a wooden table by the front door, next to a blue bowl.").
- items: each physical item the wearer is putting down, from their words or, if they \
don't name it, from the photo.
  - name: a short lowercase singular noun phrase a family member would recognise, with no \
possessive ("keys", "reading glasses", "wallet", "blue umbrella"). Use the same name every \
time for the same kind of item.
  - location: where it was put, as a short phrase ("kitchen counter", "coat pocket"): the \
place they said, else the place the photo shows. Empty if neither tells.
- visible_text: any words readable in the photo (labels, notes, screens), or empty.
If the photo is too dark or blurry to tell, say so in the description and give only the \
items and places the wearer said."""

# Appended to SCENE_SYSTEM_PROMPT; the same room ids are the schema's enum (scene_schema).
SCENE_ROOM_PROMPT = """
- room: which room of the {plan} floor plan the photo was taken in, judged from what the \
camera sees (the wearer's words can help). One of these ids, or "unknown" if the photo \
doesn't show which:
{rooms}"""


SUMMARY_SYSTEM_PROMPT = """\
You write a short update for a family caregiver of someone with memory loss who wears \
the LastScene memory aid. You get counts and lists from the last few days.
Write 2-3 warm, plain sentences: what stood out, and one practical suggestion if the data \
supports one. No lists, markdown, headings or invented facts."""


class _ExtractedItem(BaseModel):
    name: str
    location: str


class ExtractedItems(BaseModel):
    items: list[_ExtractedItem]


class SceneDescription(BaseModel):
    description: str
    items: list[_ExtractedItem]
    visible_text: str
    room: str = "unknown"  # scene_schema limits it to a floor plan's room ids


@cache
def scene_schema(plan: FloorPlan) -> type[SceneDescription]:
    """SceneDescription with `room` an enum of the plan's room ids plus "unknown"."""
    ids = tuple(r.id for r in plan.rooms) + ("unknown",)
    return create_model(f"SceneDescription_{plan.id}", __base__=SceneDescription, room=(Literal[ids], ...))


def scene_room_prompt(plan: FloorPlan) -> str:
    rooms = "\n".join(f"  - {r.id}: {r.name}" + (f" ({', '.join(r.keywords)})" if r.keywords else "") for r in plan.rooms)
    return SCENE_ROOM_PROMPT.format(plan=plan.name, rooms=rooms)


@dataclass
class SceneText:
    """The image-to-text step's output: everything search needs from a photo."""

    description: str
    items: list["ItemMention"]
    visible_text: str
    room: str | None = None  # the floor plan room the photo shows, None if it can't tell


@dataclass
class ItemMention:
    name: str
    location: str


def _mentions(items: list[_ExtractedItem]) -> list[ItemMention]:
    return [
        ItemMention(name=i.name.strip().lower(), location=i.location.strip())
        for i in items
        if i.name.strip()
    ]


def _note(m: RetrievedMemory) -> str:
    parts = [f'said "{m.transcript}"']
    if m.scene:
        parts.append(f"camera saw: {m.scene}")
    if items := m.facts(ITEM):
        parts.append("items: " + "; ".join(items))
    if text := m.facts(TEXT):
        parts.append("readable text: " + " / ".join(text))
    return f"- ({humanize_ago(m.created_at)}) " + "; ".join(parts)


# Spoken replies that never change, so their audio is made once per server process.
DIDNT_CATCH = "Sorry, I didn't catch that. Could you ask again?"
REFUSED = "Sorry, I can't help with that one."
NO_ANSWER = "Sorry, I couldn't come up with an answer just now."
# With their translations (languages.py), so every language's audio is cached as well.
BUSY = "I'm getting a lot of questions right now. Please ask me again in a minute."
FIXED_PHRASES = frozenset().union(*(languages.all_translations(p) for p in (DIDNT_CATCH, REFUSED, NO_ANSWER, BUSY)))

_PREPOSITIONS = ("on", "in", "inside", "under", "underneath", "by", "next", "beside", "behind", "near",
                 "at", "between", "beneath", "above", "below", "against", "into", "onto")
# Places you put things *in* rather than *on*, by their last word.
_CONTAINERS = {"pocket", "drawer", "bag", "box", "cabinet", "cupboard", "fridge", "freezer", "car", "purse",
               "backpack", "basket", "bowl", "jar", "case", "closet", "wardrobe", "safe", "bin", "mug", "cup",
               "tray", "room", "kitchen", "bedroom", "bathroom", "garage", "office", "hallway", "hall",
               "jacket", "coat", "suitcase", "sink", "oven", "microwave", "trunk", "glovebox"}
_WORD = re.compile(r"[a-z]+")


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _head(name: str) -> str:
    """The noun an item name ends in: "reading glasses" -> "glasse", matched stem to stem."""
    words = _WORD.findall(name.lower())
    return _stem(words[-1]) if words else ""


def _items(memory: RetrievedMemory) -> list[tuple[str, str]]:
    return [(name.strip(), loc.strip()) for name, _, loc in (f.partition(":") for f in memory.facts(ITEM))]


def _place(location: str) -> str:
    """"kitchen counter" -> "on the kitchen counter"; "my coat pocket" -> "in your coat pocket"."""
    words = location.strip().rstrip(".").split()
    if words and words[0].lower() in _PREPOSITIONS:
        return " ".join("your" if w.lower() == "my" else w for w in words)
    if words and words[0].lower() in ("my", "the", "a", "an", "your"):
        det, words = ("your" if words[0].lower() in ("my", "your") else words[0].lower()), words[1:]
    else:
        det = "the"
    last = words[-1].lower() if words else ""
    prep = "in" if last in _CONTAINERS or _stem(last) in _CONTAINERS else "on"
    return " ".join([prep, det, *words])


def template_answer(question: str, memories: list[RetrievedMemory], now: datetime | None = None) -> str | None:
    """Answer "where is X?" from the top memory alone, with no language model, when that's safe.

    Safe means: it's a "where" question; exactly one item in the retrieved memories is the
    thing asked about (its name's last noun is in the question); the top memory has it with
    a place; and no other retrieved memory has it more recently. Otherwise None, and the
    language model answers. The wording is English, so callers skip it for other languages.
    """
    if not memories or not re.search(r"\bwhere\b", question, re.I):
        return None
    asked = {_stem(w) for w in _WORD.findall(question.lower())}
    heads = {_head(name) for m in memories for name, _ in _items(m) if _head(name) in asked}
    if len(heads) != 1:
        return None
    [head] = heads
    top = memories[0]
    found = [(name, loc) for name, loc in _items(top) if _head(name) == head and loc]
    if not found:
        return None
    if any(m.created_at > top.created_at and any(_head(n) == head for n, _ in _items(m)) for m in memories[1:]):
        return None
    name, location = found[0]
    plural = name.endswith("s") and not name.endswith("ss")
    return (f"Your {name} {'are' if plural else 'is'} {_place(location)}. "
            f"You put {'them' if plural else 'it'} there {humanize_ago(top.created_at, now)}.")


def audio_seconds(audio: bytes) -> int:
    """Whole seconds of audio, rounded up (at least 1): from the WAV header, else as 16 kHz 16-bit mono."""
    import io
    import math
    import wave

    try:
        with wave.open(io.BytesIO(audio)) as w:
            seconds = w.getnframes() / w.getframerate()
    except Exception:
        seconds = len(audio) / 32000
    return max(1, math.ceil(seconds))


def humanize_ago(then: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    minutes = int((now - then).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"about {days} day{'s' if days != 1 else ''} ago"


def limited(stage: str):
    """Guard a provider call with the shared rate limits for `stage` (see AIServices.rules), and
    fall back through Gemini models (AIServices.models): if a model is at its cap, cooling down,
    overloaded (429/503), down (5xx) or unknown (404), the next one is tried. A provider "too many
    requests" cools that model down on every server instance. If none can take the call, the
    caller gets RateLimited (or the last real error)."""

    def wrap(fn):
        @functools.wraps(fn)
        async def run(self, *args, **kwargs):
            chain = self.models(stage) or [None]
            last: Exception | None = None
            for model in chain:
                rules = self.rules(stage, model)
                try:
                    if self.limiter and rules:
                        await self.limiter.take(rules)
                    with using(model, tts=stage == "tts"):
                        return await fn(self, *args, **kwargs)
                except RateLimited as busy:
                    last = busy
                except Exception as exc:
                    if (wait := provider_busy(exc)) is not None:
                        if self.limiter and rules:
                            await self.limiter.cool_down(rules[0].bucket, wait)
                        last = RateLimited(rules[0].bucket if rules else stage, wait, "provider busy")
                    elif model is not None and _worth_another_model(exc):
                        last = exc
                    else:
                        raise
                if model is not None and model != chain[-1]:
                    log.warning("%s: %s unavailable (%s); trying the next model", stage, model, type(last).__name__)
            raise last

        return run

    return wrap


def _worth_another_model(exc: Exception) -> bool:
    """A Google-side failure another model might not have: 5xx, or a model that isn't there (404)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 404 or (isinstance(code, int) and 500 <= code < 600)


def model_list(primary: str, fallbacks: str) -> list[str]:
    """GEMINI_MODEL then GEMINI_FALLBACK_MODELS (comma-separated), without repeats."""
    seen: list[str] = []
    for m in [primary, *fallbacks.split(",")]:
        if (m := m.strip()) and m not in seen:
            seen.append(m)
    return seen


def model_limits(spec: str) -> dict[str, tuple[int, int]]:
    """"a=15/900,b=10/250" -> {"a": (15, 900), "b": (10, 250)}; malformed entries are skipped."""
    limits = {}
    for part in spec.split(","):
        name, _, caps = part.strip().partition("=")
        rpm, _, rpd = caps.partition("/")
        if name and rpm.strip().isdigit() and rpd.strip().isdigit():
            limits[name.strip()] = (int(rpm), int(rpd))
    return limits


class AIServices:
    def __init__(self, settings: Settings, limiter: Limiter | None = None):
        self.s = settings
        self.limiter = limiter  # None: no shared limits (tests, scripts)
        self._speech_cache: dict[tuple, bytes] = {}

    def _provider(self, stage: str) -> str:
        s = self.s
        # ElevenLabs is limited on its own (_speak_elevenlabs, _transcribe_elevenlabs); "tts" / "stt" are
        # its Gemini fallbacks.
        tts = "gemini" if s.tts_provider == "elevenlabs" else s.tts_provider
        stt = "gemini" if s.stt_provider == "elevenlabs" else s.stt_provider
        return {"stt": stt, "tts": tts, "embed": s.embedding_provider,
                "text": s.llm_provider, "vision": s.vision_provider, "maps": "gemini"}[stage]

    def models(self, stage: str) -> list[str]:
        """The Gemini models a `stage` call tries, in order; [] when it doesn't go to Gemini.
        Embeddings never fall back: another model's vectors wouldn't match the stored ones."""
        s = self.s
        if self._provider(stage) != "gemini":
            return []
        if stage == "tts":
            return model_list(s.gemini_tts_model, s.gemini_tts_fallback_models)
        if stage == "embed":
            return [s.gemini_embedding_model]
        return model_list(s.gemini_model, s.gemini_fallback_models)

    def rules(self, stage: str, model: str | None = None) -> list[Rule]:
        """The limits a call for `stage` counts against: the provider and model it goes to
        (by default the first model it tries). stt | tts | embed | text | vision | maps."""
        s = self.s
        provider = self._provider(stage)
        if provider == "gemini":
            model = model or self.models(stage)[0]
            if stage == "tts":
                return per(f"gemini:{model}", s.gemini_tts_rpm, s.gemini_tts_rpd)
            if stage == "embed":
                return per(f"gemini:{model}", s.gemini_embedding_rpm, s.gemini_embedding_rpd)
            rpm, rpd = model_limits(s.gemini_model_limits).get(model, (s.gemini_rpm, s.gemini_rpd))
            return per(f"gemini:{model}", rpm, rpd)
        if provider == "openai":
            return per("openai", s.openai_rpm, s.openai_rpd)
        if provider == "claude":
            return per("claude", s.claude_rpm, s.claude_rpd)
        return []  # e.g. VISION_PROVIDER=none: no call is made

    # Clients are created on first use so the API (and the dashboard) can start
    # before the AI keys are configured; the voice endpoints fail until they are.
    @cached_property
    def gemini(self) -> GeminiAI:
        return GeminiAI(self.s)

    @cached_property
    def openai(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(api_key=self.s.openai_api_key or None)

    @cached_property
    def claude(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(api_key=self.s.anthropic_api_key or None)

    @cached_property
    def elevenlabs(self) -> ElevenLabs:
        return ElevenLabs(self.s)

    @property
    def speech_media_type(self) -> str:
        if self.s.tts_provider in ("gemini", "elevenlabs"):
            return "audio/wav"
        return {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}.get(
            self.s.tts_format, "application/octet-stream"
        )

    # ---------------------------------------------------------------- speech in/out

    async def transcribe(self, filename: str, audio: bytes, content_type: str | None,
                         hints: list[str] | None = None) -> str:
        """Speech to text. ElevenLabs Scribe when chosen and configured, else (or when it fails) Gemini /
        OpenAI. `hints`: names and words likely to come up (see routers/memory.py)."""
        if self.s.stt_provider == "elevenlabs" and self.s.elevenlabs_api_key:
            try:
                return await self._transcribe_elevenlabs(filename, audio, content_type, hints)
            except Exception as exc:  # its limit, its plan, an outage: Gemini still hears it
                log.warning("ElevenLabs speech-to-text unavailable (%s); using Gemini", type(exc).__name__)
        return await self._transcribe_fallback(filename, audio, content_type, hints)

    async def _transcribe_elevenlabs(self, filename, audio, content_type, hints) -> str:
        s = self.s
        if self.limiter:
            await self.limiter.take(per("elevenlabs-stt", s.elevenlabs_stt_rpm))
            await self.limiter.take([Rule("elevenlabs-stt:seconds", s.elevenlabs_stt_monthly_seconds, "month")],
                                    cost=audio_seconds(audio))
        try:
            return await self.elevenlabs.transcribe(audio, filename, content_type, hints)
        except Exception as exc:
            if self.limiter and (wait := provider_busy(exc)) is not None:
                await self.limiter.cool_down("elevenlabs-stt", wait)
            raise

    @limited("stt")
    async def _transcribe_fallback(self, filename, audio, content_type, hints) -> str:
        if self.s.stt_provider in ("gemini", "elevenlabs"):
            return await self.gemini.transcribe(audio, content_type, hints)
        result = await self.openai.audio.transcriptions.create(
            model=self.s.stt_model,
            file=(filename or "audio.wav", audio, content_type or "audio/wav"),
            **({"prompt": ", ".join(hints)} if hints else {}),
        )
        return result.text.strip()

    async def speak(self, text: str) -> bytes:
        if text not in FIXED_PHRASES:
            return await self._speak(text)
        key = (self.s.tts_provider, self.s.gemini_tts_model, self.s.gemini_tts_voice, self.s.tts_model,
               self.s.tts_voice, self.s.tts_format, self.s.elevenlabs_voice_id, text)
        if key not in self._speech_cache:
            self._speech_cache[key] = await self._speak(text)
        return self._speech_cache[key]

    async def _speak(self, text: str) -> bytes:
        """ElevenLabs when chosen and configured, else (or when it fails) Gemini / OpenAI."""
        if self.s.tts_provider == "elevenlabs" and self.s.elevenlabs_api_key:
            try:
                return await self._speak_elevenlabs(text)
            except Exception as exc:  # its limit, its quota, an outage: the wearer still hears an answer
                log.warning("ElevenLabs speech unavailable (%s); using Gemini speech", type(exc).__name__)
        return await self._speak_fallback(text)

    async def _speak_elevenlabs(self, text: str) -> bytes:
        s = self.s
        if self.limiter:
            await self.limiter.take(per("elevenlabs", s.elevenlabs_rpm))
            await self.limiter.take([Rule("elevenlabs:characters", s.elevenlabs_monthly_chars, "month")], cost=len(text))
        try:
            return await self.elevenlabs.speak(text)
        except Exception as exc:
            if self.limiter and (wait := provider_busy(exc)) is not None:
                await self.limiter.cool_down("elevenlabs", wait)
            raise

    @limited("tts")
    async def _speak_fallback(self, text: str) -> bytes:
        if self.s.tts_provider in ("gemini", "elevenlabs"):
            return await self.gemini.speak(text)
        result = await self.openai.audio.speech.create(
            model=self.s.tts_model,
            voice=self.s.tts_voice,
            input=text,
            response_format=self.s.tts_format,
        )
        return result.content

    # ---------------------------------------------------------------- retrieval

    @limited("embed")
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed several memory chunks."""
        if not texts:
            return []
        if self.s.embedding_provider == "gemini":
            return await self.gemini.embed_many(texts, query=False)
        result = await self.openai.embeddings.create(
            model=self.s.embedding_model, input=texts, dimensions=self.s.embedding_dimensions
        )
        return [d.embedding for d in result.data]

    @limited("embed")
    async def embed(self, text: str, *, query: bool = False) -> list[float]:
        """Embed a memory (query=False) or a question (query=True) for vector search."""
        if self.s.embedding_provider == "gemini":
            return await self.gemini.embed(text, query=query)
        result = await self.openai.embeddings.create(
            model=self.s.embedding_model,
            input=text,
            dimensions=self.s.embedding_dimensions,
        )
        return result.data[0].embedding

    # ---------------------------------------------------------------- language

    # fast=True: the wearer is waiting (or the call is frequent), so Gemini thinks as
    # little as the model allows. Claude already runs at LLM_EFFORT.
    @limited("text")
    async def _generate(self, system: str, prompt: str, *, fast: bool = False) -> str | None:
        if self.s.llm_provider == "gemini":
            return await self.gemini.generate(system, prompt, fast=fast)
        response = await self.claude.beta.messages.create(
            model=self.s.llm_model,
            max_tokens=1024,
            system=system,
            output_config={"effort": self.s.llm_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return None
        return "".join(b.text for b in response.content if b.type == "text").strip()

    @limited("text")
    async def _generate_json(
        self, system: str, text: str, schema: type[BaseModel], *, fast: bool = False
    ) -> BaseModel | None:
        if self.s.llm_provider == "gemini":
            return await self.gemini.generate_json(system, [text], schema, fast=fast)
        response = await self.claude.beta.messages.parse(
            model=self.s.llm_model,
            max_tokens=1024,
            system=system,
            output_config={"effort": self.s.llm_effort},
            output_format=schema,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": text}],
        )
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output

    async def answer(self, question: str, memories: list[RetrievedMemory], language: str = languages.DEFAULT_LANGUAGE) -> str:
        notes = "\n".join(_note(m) for m in memories) or "(nothing retrieved)"
        system = f"{ANSWER_SYSTEM_PROMPT}\n- {languages.instruction(language)}"
        text = await self._generate(system, f"Memories:\n{notes}\n\nQuestion: {question}", fast=True)
        if text is None:
            return languages.phrase(REFUSED, language)
        return text or languages.phrase(NO_ANSWER, language)

    async def extract_items(self, text: str) -> list[ItemMention]:
        """Pull the items (and where they were put) out of a statement or question.

        Feeds the dashboard's Items and Insights pages. With vision on, a memory's items
        come from describe_scene instead; this is for questions and VISION_PROVIDER=none.
        """
        parsed = await self._generate_json(EXTRACT_SYSTEM_PROMPT, text, ExtractedItems, fast=True)
        return _mentions(parsed.items) if parsed else []

    @limited("vision")
    async def describe_scene(self, image: bytes, content_type: str, transcript: str) -> SceneText | None:
        """Image to text: a description, the items and where they are, and readable text.

        One call does the photo and the words: the items come from the transcript (with the
        camera filling in places it doesn't say), so extract_items isn't needed as well.
        The same call names the floor plan (FLOOR_PLAN) room the photo was taken in, for
        the dashboard map. None when vision is off."""
        if self.s.vision_provider != "gemini":
            return None
        from google.genai import types

        plan = floor_plan(self.s.floor_plan)
        parsed = await self.gemini.generate_json(
            SCENE_SYSTEM_PROMPT.format(transcript=transcript.replace('"', "'")) + scene_room_prompt(plan),
            [types.Part.from_bytes(data=image, mime_type=content_type), "Describe this scene."],
            scene_schema(plan),
        )
        if parsed is None:
            return None
        room = parsed.room if any(r.id == parsed.room for r in plan.rooms) else None
        return SceneText(parsed.description.strip(), _mentions(parsed.items), parsed.visible_text.strip(), room)

    async def caregiver_summary(self, facts: str, language: str = languages.DEFAULT_LANGUAGE) -> str | None:
        return await self._generate(f"{SUMMARY_SYSTEM_PROMPT}\n{languages.instruction(language)}", facts)

    @limited("vision")
    async def look(self, system: str, contents: list, schema: type[BaseModel]) -> BaseModel | None:
        """A photo question with a JSON answer (e.g. Teach's suggested boxes), within the limits."""
        return await self.gemini.generate_json(system, contents, schema, fast=True)

    @limited("maps")
    async def name_place(self, lat: float, lng: float) -> dict | None:
        """The public place at lat/lng from Grounding with Google Maps (Gemini only):
        {"name", "sources": [{"title", "url"}]} or None. Call it through
        maps_names.name_memory_place, which adds the opt-in, the daily cap and the cache.

        Dashboard only. Spoken answers keep the household's own place wording: Google
        requires the name to be shown with its linked sources and "Google Maps", which a
        speaker can't do."""
        return await self.gemini.name_place(lat, lng)
