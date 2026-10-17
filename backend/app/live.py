"""Live conversation mode: ephemeral Gemini Live tokens for the wearable.

Vercel can't hold the long-lived websocket, so the Pi talks to Gemini Live itself, with a
one-use token minted here. The main GEMINI_API_KEY never leaves the server. Everything
that shapes the conversation is locked into the token: the model, the persona, the tools
and the voice. A device can't change any of it; it only streams audio and answers the
tool calls (via /api/v1/memory/recall and /api/v1/memory/log).

Field names are checked against google-genai 2.31.0 (google/genai/types.py, tokens.py,
live.py): ephemeral tokens are v1alpha only, both to create and to connect with.
"""

from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from .config import Settings

# google/genai/live.py warns unless an auth_tokens/... key connects with v1alpha.
LIVE_API_VERSION = "v1alpha"
TOKEN_TTL = timedelta(minutes=10)  # messages in the session are rejected after this
NEW_SESSION_TTL = timedelta(minutes=1)  # the device must connect within this

LIVE_SYSTEM_PROMPT = """\
You are LastScene, a gentle voice companion on a wearable that helps someone with memory \
loss find everyday items and remember what they did. You are talking with them out loud \
through a small speaker.
- Speak in one or two short, warm, plain sentences, then let them talk. No lists.
- Never guess where something is or what they did. Whenever they ask about an item, a place \
or something earlier, call recall_memories first and answer only from what it returns, \
saying roughly when, e.g. "about two hours ago".
- If recall_memories finds nothing about it, say kindly that you don't have a note yet and \
suggest they tell you next time they put it down.
- When they tell you they are putting something somewhere ("I'm leaving my keys on the \
counter"), call save_memory with their words, then confirm briefly.
- If they seem confused or upset, be calm and reassuring. Keep it simple."""

RECALL_TOOL = types.FunctionDeclaration(
    name="recall_memories",
    description="Look up the wearer's saved memories (what they said and what the camera saw) "
                "to answer a question about where an item is or what they did.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"question": types.Schema(type=types.Type.STRING, description='The question, e.g. "where are my keys?"')},
        required=["question"],
    ),
)
SAVE_TOOL = types.FunctionDeclaration(
    name="save_memory",
    description="Save a new memory when the wearer says they are putting an item somewhere. "
                "The wearable takes a photo at the same moment.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"text": types.Schema(type=types.Type.STRING, description="What they said, in their words.")},
        required=["text"],
    ),
)


def live_config(settings: Settings) -> types.LiveConnectConfig:
    """Everything locked into the token: persona, tools, voice, audio out."""
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=LIVE_SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=[RECALL_TOOL, SAVE_TOOL])],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=settings.gemini_tts_voice)
            )
        ),
        # Text of both sides, for the device's logs.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


def token_config(settings: Settings, now: datetime | None = None) -> types.CreateAuthTokenConfig:
    now = now or datetime.now(timezone.utc)
    return types.CreateAuthTokenConfig(
        uses=1,
        expire_time=now + TOKEN_TTL,
        new_session_expire_time=now + NEW_SESSION_TTL,
        # No lock_additional_fields: the whole constraint (model + config) is locked.
        live_connect_constraints=types.LiveConnectConstraints(
            model=settings.gemini_live_model, config=live_config(settings)
        ),
    )


class LiveTokens:
    """Mints tokens with the server's key; the client is made on first use (tests set .client)."""

    def __init__(self):
        self.client: genai.Client | None = None

    async def mint(self, settings: Settings) -> tuple[types.AuthToken, types.CreateAuthTokenConfig]:
        if self.client is None:
            self.client = genai.Client(
                api_key=settings.gemini_api_key or None,
                http_options=types.HttpOptions(api_version=LIVE_API_VERSION),
            )
        config = token_config(settings)
        return await self.client.aio.auth_tokens.create(config=config), config

