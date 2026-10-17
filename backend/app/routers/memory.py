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


# Enrichment: by default these run after the response is sent, so the wearable can go
# back to sleep (or play its answer) without waiting. A failure only costs the
# dashboard some detail; the memory itself is already saved and searchable.
async def _enrich_memory(
    db: MemoryDB, ai: AIServices, memory_id: str, transcript: str, image: bytes, image_type: str
) -> None:
    try:
        items = await ai.extract_items(transcript)
    except Exception:
        log.exception("Item extraction failed for memory %s", memory_id)
        items = []
    try:
        scene = await ai.describe_scene(image, image_type, transcript)
    except Exception:
        log.exception("Photo description failed for memory %s", memory_id)
        scene = None
    if scene:
        description, seen = scene
        # The camera fills in places the wearer didn't say ("I'm putting my keys down").
        seen_at = {i.name: i.location for i in seen if i.location}
        for item in items:
            if not item.location:
                item.location = seen_at.get(item.name, "")
        if not items:
            items = seen
        try:
            embedding = await ai.embed(f"{transcript}\nScene: {description}")
            await db.set_scene(memory_id, description, embedding)
        except Exception:
            log.exception("Saving scene failed for memory %s", memory_id)
    try:
        await db.add_items(memory_id, [(i.name, i.location) for i in items])
    except Exception:
        log.exception("Saving items failed for memory %s", memory_id)


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
    embedding = await ai.embed(transcript)
    record = await db.insert(device.sub, transcript, embedding, image_bytes, image_type)
    enrich_args = (db, ai, record.id, transcript, image_bytes, image_type)
    if settings.enrich_in_background:
        background.add_task(_enrich_memory, *enrich_args)
    else:
        await _enrich_memory(*enrich_args)
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
        embedding = await ai.embed(question, query=True)
        matches = await db.nearest(device.sub, embedding, settings.query_lookback_hours)
        answer = await ai.answer(question, matches)
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
