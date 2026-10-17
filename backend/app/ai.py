"""Every AI stage, each routed to a provider chosen in settings.

Defaults are Gemini end to end (see gemini.py). The original providers stay available
per stage: OpenAI for speech-to-text, embeddings and text-to-speech, Claude for the
language tasks.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property

import anthropic
import openai
from pydantic import BaseModel

from .config import Settings
from .rag import ITEM, TEXT, RetrievedMemory
from .gemini import GeminiAI

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


@dataclass
class SceneText:
    """The image-to-text step's output: everything search needs from a photo."""

    description: str
    items: list["ItemMention"]
    visible_text: str


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
FIXED_PHRASES = frozenset({DIDNT_CATCH, REFUSED, NO_ANSWER})

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
    language model answers.
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


class AIServices:
    def __init__(self, settings: Settings):
        self.s = settings
        self._speech_cache: dict[tuple, bytes] = {}

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

    @property
    def speech_media_type(self) -> str:
        if self.s.tts_provider == "gemini":
            return "audio/wav"
        return {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}.get(
            self.s.tts_format, "application/octet-stream"
        )

    # ---------------------------------------------------------------- speech in/out

    async def transcribe(self, filename: str, audio: bytes, content_type: str | None) -> str:
        if self.s.stt_provider == "gemini":
            return await self.gemini.transcribe(audio, content_type)
        result = await self.openai.audio.transcriptions.create(
            model=self.s.stt_model,
            file=(filename or "audio.wav", audio, content_type or "audio/wav"),
        )
        return result.text.strip()

    async def speak(self, text: str) -> bytes:
        if text not in FIXED_PHRASES:
            return await self._speak(text)
        key = (self.s.tts_provider, self.s.gemini_tts_model, self.s.gemini_tts_voice, self.s.tts_model,
               self.s.tts_voice, self.s.tts_format, text)
        if key not in self._speech_cache:
            self._speech_cache[key] = await self._speak(text)
        return self._speech_cache[key]

    async def _speak(self, text: str) -> bytes:
        if self.s.tts_provider == "gemini":
            return await self.gemini.speak(text)
        result = await self.openai.audio.speech.create(
            model=self.s.tts_model,
            voice=self.s.tts_voice,
            input=text,
            response_format=self.s.tts_format,
        )
        return result.content

    # ---------------------------------------------------------------- retrieval

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

    async def answer(self, question: str, memories: list[RetrievedMemory]) -> str:
        notes = "\n".join(_note(m) for m in memories) or "(nothing retrieved)"
        text = await self._generate(ANSWER_SYSTEM_PROMPT, f"Memories:\n{notes}\n\nQuestion: {question}", fast=True)
        if text is None:
            return REFUSED
        return text or NO_ANSWER

    async def extract_items(self, text: str) -> list[ItemMention]:
        """Pull the items (and where they were put) out of a statement or question.

        Feeds the dashboard's Items and Insights pages. With vision on, a memory's items
        come from describe_scene instead; this is for questions and VISION_PROVIDER=none.
        """
        parsed = await self._generate_json(EXTRACT_SYSTEM_PROMPT, text, ExtractedItems, fast=True)
        return _mentions(parsed.items) if parsed else []

    async def describe_scene(self, image: bytes, content_type: str, transcript: str) -> SceneText | None:
        """Image to text: a description, the items and where they are, and readable text.

        One call does the photo and the words: the items come from the transcript (with the
        camera filling in places it doesn't say), so extract_items isn't needed as well.
        None when vision is off."""
        if self.s.vision_provider != "gemini":
            return None
        from google.genai import types

        parsed = await self.gemini.generate_json(
            SCENE_SYSTEM_PROMPT.format(transcript=transcript.replace('"', "'")),
            [types.Part.from_bytes(data=image, mime_type=content_type), "Describe this scene."],
            SceneDescription,
        )
        if parsed is None:
            return None
        return SceneText(parsed.description.strip(), _mentions(parsed.items), parsed.visible_text.strip())

    async def caregiver_summary(self, facts: str) -> str | None:
        return await self._generate(SUMMARY_SYSTEM_PROMPT, facts)
