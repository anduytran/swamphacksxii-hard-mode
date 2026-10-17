from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from ..ai import AIServices
from ..auth import Principal, require_device
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_ai, get_db

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])

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


@router.post("/log")
async def log_memory(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
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
    return {"id": record.id, "timestamp": record.created_at.isoformat(), "transcript": transcript}


@router.post("/query")
async def query_memory(
    audio: UploadFile = File(...),
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    audio_bytes = await _read(audio, MAX_AUDIO_BYTES, "audio")
    question = await ai.transcribe(audio.filename, audio_bytes, audio.content_type)

    if question:
        embedding = await ai.embed(question)
        matches = await db.nearest(device.sub, embedding, settings.query_lookback_hours)
        answer = await ai.answer(question, matches)
    else:
        answer = "Sorry, I didn't catch that. Could you ask again?"

    speech = await ai.speak(answer)
    media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}.get(
        settings.tts_format, "application/octet-stream"
    )
    return Response(
        content=speech,
        media_type=media_type,
        # Handy for debugging from curl / serial logs; the device can ignore them.
        headers={"X-Question": quote(question), "X-Answer": quote(answer)},
    )
