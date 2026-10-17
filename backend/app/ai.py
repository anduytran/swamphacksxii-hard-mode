"""Speech-to-text, embeddings and text-to-speech (OpenAI) plus answer generation (Claude)."""

from datetime import datetime, timezone

import anthropic
import openai

from .config import Settings
from .db import MemoryRecord

ANSWER_SYSTEM_PROMPT = """\
You are Memory Anchor, a gentle voice assistant on a wearable that helps someone \
with memory loss find everyday items. Your reply is spoken aloud through a small speaker.

You are given the user's question and a few things they said out loud earlier while \
putting items down, each with how long ago it was said. Answer using only those notes.
- Reply in one or two short, warm, plain sentences. No lists, markdown or emoji.
- Say where the item is and roughly when it was left there, e.g. "about two hours ago".
- If none of the notes are about what they asked, say kindly that you don't have a \
note about it yet and suggest they tell you next time they put it down.
- Latency-sensitive; begin your visible answer immediately."""


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
        self.openai = openai.AsyncOpenAI(api_key=settings.openai_api_key or None)
        self.claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)

    async def transcribe(self, filename: str, audio: bytes, content_type: str | None) -> str:
        result = await self.openai.audio.transcriptions.create(
            model=self.s.stt_model,
            file=(filename or "audio.wav", audio, content_type or "audio/wav"),
        )
        return result.text.strip()

    async def embed(self, text: str) -> list[float]:
        result = await self.openai.embeddings.create(
            model=self.s.embedding_model,
            input=text,
            dimensions=self.s.embedding_dimensions,
        )
        return result.data[0].embedding

    async def answer(self, question: str, memories: list[MemoryRecord]) -> str:
        if memories:
            notes = "\n".join(
                f"- ({humanize_ago(m.created_at)}) \"{m.transcript}\"" for m in memories
            )
        else:
            notes = "(no notes recorded recently)"
        response = await self.claude.beta.messages.create(
            model=self.s.llm_model,
            max_tokens=1024,
            system=ANSWER_SYSTEM_PROMPT,
            output_config={"effort": self.s.llm_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    "content": f"Earlier notes:\n{notes}\n\nQuestion: {question}",
                }
            ],
        )
        if response.stop_reason == "refusal":
            return "Sorry, I can't help with that one."
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or "Sorry, I couldn't come up with an answer just now."

    async def speak(self, text: str) -> bytes:
        result = await self.openai.audio.speech.create(
            model=self.s.tts_model,
            voice=self.s.tts_voice,
            input=text,
            response_format=self.s.tts_format,
        )
        return result.content
