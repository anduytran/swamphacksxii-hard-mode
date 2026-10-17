"""Live conversation mode (see app/live.py): the token the wearable connects to Gemini Live
with, and the memory lookup its recall_memories tool calls."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..ai import AIServices, _note
from ..auth import Principal, require_device
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_ai, get_db
from ..live import LIVE_API_VERSION
from ..pipeline import retrieve

router = APIRouter(prefix="/api/v1", tags=["live"])
log = logging.getLogger(__name__)

NOTHING_FOUND = "No saved memories match that."


@router.post("/live/token")
async def live_token(
    request: Request,
    device: Principal = Depends(require_device),
    settings: Settings = Depends(get_settings),
):
    """A one-use Gemini Live token, valid to connect for a minute. Only the token's name
    (auth_tokens/...) is returned; the server's API key never is."""
    if not settings.live_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Live mode is off (LIVE_ENABLED)")
    if wait := request.app.state.live_limit.retry_after(device.sub, settings.live_tokens_per_hour):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"At most {settings.live_tokens_per_hour} live sessions per hour",
            headers={"Retry-After": str(max(1, round(wait)))},
        )
    try:
        token, config = await request.app.state.live_tokens.mint(settings)
    except Exception as exc:  # quota (429), bad key, network: the device falls back to /memory/voice
        log.warning("Minting a Live token failed: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Live mode unavailable right now") from exc
    if not token.name:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Live mode unavailable right now")
    return {
        "token": token.name,
        "model": settings.gemini_live_model,
        "api_version": LIVE_API_VERSION,
        "expires_at": config.expire_time.isoformat(),
        "new_session_expires_at": config.new_session_expire_time.isoformat(),
    }


class RecallRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@router.post("/memory/recall")
async def recall(
    body: RecallRequest,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """The memories retrieved for a question, as short text facts for Gemini Live to answer
    from. Search only (one embedding), no language model."""
    try:
        memories = await retrieve(db, ai, body.question.strip(), [device.sub], settings.query_lookback_hours)
    except Exception as exc:  # e.g. the embedding call hit Gemini's quota
        log.warning("Recall failed for %r: %s", body.question, exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Memory search unavailable right now") from exc
    facts = [_note(m).removeprefix("- ") for m in memories]
    return {"question": body.question, "facts": facts, "text": "\n".join(facts) or NOTHING_FOUND}
