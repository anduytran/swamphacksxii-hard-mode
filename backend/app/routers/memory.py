import logging
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ..ai import AIServices
from ..auth import Principal, require_device
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_ai, get_db
from ..pipeline import ingest_image, ingest_speech, retrieve

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 2 * 1024 * 1024  # photos live in Postgres; ESP32 JPEGs are ~50-200 KB
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read(upload: UploadFile, limit: int, label: str) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Empty {label} file")
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{label} exceeds {limit} bytes")
    return data


# Image-to-text indexing and question logging run after the response by default, so
# the wearable isn't kept waiting (ENRICH_IN_BACKGROUND=false runs them before it).
async def _record_query(
    db: MemoryDB, ai: AIServices, device_id: str, question: str, answer: str
) -> None:
    try:
        items = await ai.extract_items(question)
    except Exception:
        log.exception("Item extraction failed for question %r", question)
        items = []
    try:
        await db.log_query(device_id, question, answer, items[0].name if items else None)
    except Exception:
        log.exception("Failed to log question %r", question)


@router.post("/log")
async def log_memory(
    background: BackgroundTasks,
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    audio_bytes = await _read(audio, MAX_AUDIO_BYTES, "audio")
    image_bytes = await _read(image, MAX_IMAGE_BYTES, "image")

    transcript = await ai.transcribe(audio.filename, audio_bytes, audio.content_type)
    if not transcript:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")

    # The type is echoed back when the dashboard loads the photo, so never trust it blindly.
    image_type = image.content_type if image.content_type in ALLOWED_IMAGE_TYPES else "image/jpeg"
    record = await db.insert(device.sub, transcript, image_bytes, image_type)
    await ingest_speech(db, ai, record.id, transcript)
    image_args = (db, ai, record.id, transcript, image_bytes, image_type)
    if settings.enrich_in_background:
        background.add_task(ingest_image, *image_args)
    else:
        await ingest_image(*image_args)
    return {"id": record.id, "timestamp": record.created_at.isoformat(), "transcript": transcript}


@router.post("/query")
async def query_memory(
    background: BackgroundTasks,
    audio: UploadFile = File(...),
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    audio_bytes = await _read(audio, MAX_AUDIO_BYTES, "audio")
    question = await ai.transcribe(audio.filename, audio_bytes, audio.content_type)

    if question:
        memories = await retrieve(db, ai, question, device.sub, settings.query_lookback_hours)
        answer = await ai.answer(question, memories)
        if settings.enrich_in_background:
            background.add_task(_record_query, db, ai, device.sub, question, answer)
    else:
        answer = "Sorry, I didn't catch that. Could you ask again?"

    speech = await ai.speak(answer)
    if question and not settings.enrich_in_background:
        await _record_query(db, ai, device.sub, question, answer)
    return Response(
        content=speech,
        media_type=ai.speech_media_type,
        # Handy for debugging from curl / serial logs; the device can ignore them.
        headers={"X-Question": quote(question), "X-Answer": quote(answer)},
    )
