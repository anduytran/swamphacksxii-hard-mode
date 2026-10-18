import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ..ai import BUSY, DIDNT_CATCH, GLAD_YOU_FOUND, IN_VIEW, AIServices, template_answer
from ..auth import Principal, require_device
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import device_limit, get_ai, get_db, get_geo
from ..geo import Geolocator, locate_and_record, record_public, resolve_location
from ..intent import classify, fallback_command, has_wake_phrase, strip_wake_phrase
from ..observations import scope_for
from ..languages import is_english, normalize, phrase
from ..limits import RateLimited
from ..maps_names import name_memory_place
from ..pipeline import ingest_image, ingest_speech, retrieve

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
log = logging.getLogger(__name__)
timing_log = logging.getLogger("lastscene.timing")  # shown by main.py's handler

MAX_IMAGE_BYTES = 2 * 1024 * 1024  # photos live in Postgres; ESP32 JPEGs are ~50-200 KB
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 2000  # a 15 s clip is ~40 words
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read(upload: UploadFile, limit: int, label: str) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Empty {label} file")
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{label} exceeds {limit} bytes")
    return data


TRANSCRIPT_FORM = Form(None, description="what was said, transcribed on the device (Moonshine/Vosk); "
                       "sent instead of audio, so the recording never leaves the wearable")


WAKE_HINT = "Last Scene"


async def stt_hints(db: MemoryDB, device_id: str) -> list[str]:
    """Words the wearer is likely to say that speech-to-text tends to get wrong: the wake phrase and the
    household's Teach items and their aliases ("swamphacks pins", not "Swampfax pins")."""
    hints = [WAKE_HINT]
    try:
        household = await db.household_of_device(device_id)
        if household:
            for item in await db.training_items(household[0]):
                hints += [item.name, *item.aliases]
    except Exception:  # hints are a nicety: transcribe without them
        log.warning("Speech-to-text hints unavailable for this device")
    return hints


async def saved_item_names(db: MemoryDB, device_id: str, hours: int) -> dict[str, str]:
    """Every word the household uses for its items -> the item's name: Teach items and their aliases,
    items the camera has a last sighting of, and items named in recent memories. Empty on any error."""
    names: dict[str, str] = {}
    try:
        household = await db.household_of_device(device_id)
        scope, devices = (household[0], await db.household_device_ids(household[0])) if household else (device_id, [device_id])
        for sighting in await db.item_sightings(hours, devices):
            names.setdefault(sighting.name.lower(), sighting.name)
        for seen in await db.last_seen(scope):
            names.setdefault(seen.item.lower(), seen.item)
        if household:
            for item in await db.training_items(household[0]):
                names[item.name.lower()] = item.name
                for alias in item.aliases:
                    names[alias.lower()] = item.name
    except Exception:
        log.warning("Saved item names unavailable for this device")
    return names


async def _heard(ai: AIServices, audio: UploadFile | None, transcript: str | None,
                 hints=None) -> str:
    """The words: the device's own transcript when it sent one, else speech-to-text on the audio.
    `hints`: a function giving the hint list, only called when audio is transcribed here."""
    if transcript is not None and transcript.strip():
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"transcript exceeds {MAX_TRANSCRIPT_CHARS} characters")
        return transcript.strip()
    if audio is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Send audio or a transcript")
    audio_bytes = await _read(audio, MAX_AUDIO_BYTES, "audio")
    return await ai.transcribe(audio.filename, audio_bytes, audio.content_type, await hints() if hints else None)


# Image-to-text indexing, positioning and question logging run after the response by
# default, so the wearable isn't kept waiting (ENRICH_IN_BACKGROUND=false runs them first).
@dataclass
class Whereabouts:
    """What the wearable sent about its position (see geo.resolve_location)."""

    wifi: str | None = None
    lat: float | None = None
    lng: float | None = None
    accuracy_m: float | None = None


def whereabouts(
    wifi: str | None = Form(None, description='JSON [{"bssid": "aa:bb:cc:dd:ee:ff", "signal": -60}, ...]'),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    accuracy_m: float | None = Form(None),
) -> Whereabouts:
    return Whereabouts(wifi, lat, lng, accuracy_m)


async def _locate(geo: Geolocator, db: MemoryDB, device_id: str, table: str, row_id: str, where: Whereabouts):
    try:
        loc = await resolve_location(geo, where.wifi, where.lat, where.lng, where.accuracy_m)
        place, public = await locate_and_record(db, device_id, table, row_id, loc)
        return loc, public
    except Exception:
        log.exception("Locating %s %s failed", table, row_id)
        return None, False


async def _enrich_memory(
    db: MemoryDB, ai: AIServices, geo: Geolocator, device_id: str, memory_id: str,
    transcript: str, image: bytes, image_type: str, where: Whereabouts, settings: Settings | None = None,
) -> None:
    items = await ingest_image(db, ai, memory_id, transcript, image, image_type)
    loc, public = await _locate(geo, db, device_id, "memory_logs", memory_id, where)
    if public:  # an item left at a public place, in a household that shares
        await record_public(db, device_id, loc, "left", [i.name for i in items])
    if loc is not None:  # names it only if the place is public and MAPS_GROUNDING is on
        await name_memory_place(db, ai, settings or get_settings(), memory_id)


async def _record_query(
    db: MemoryDB, ai: AIServices, geo: Geolocator, device_id: str, question: str, answer: str, where: Whereabouts
) -> None:
    try:
        items = await ai.extract_items(question)
    except Exception:
        log.exception("Item extraction failed for a question")  # not the question: it's personal
        items = []
    try:
        query_id = await db.log_query(device_id, question, answer, items[0].name if items else None)
    except Exception:
        log.exception("Failed to log a question")
        return
    loc, public = await _locate(geo, db, device_id, "query_logs", query_id, where)
    if public and items:  # someone realised something was missing while at a public place
        await record_public(db, device_id, loc, "asked", [items[0].name])


async def _note_pairing(db: MemoryDB, device_id: str, code: str | None) -> None:
    if code:
        await db.remember_pairing_code(device_id, code)


async def _save_memory(
    background: BackgroundTasks, db: MemoryDB, ai: AIServices, geo: Geolocator, settings: Settings,
    device_id: str, transcript: str, image_bytes: bytes, image_content_type: str | None, where: Whereabouts,
) -> dict:
    # The type is echoed back when the dashboard loads the photo, so never trust it blindly.
    image_type = image_content_type if image_content_type in ALLOWED_IMAGE_TYPES else "image/jpeg"
    record = await db.insert(device_id, transcript, image_bytes, image_type)
    try:
        await ingest_speech(db, ai, record.id, transcript)
    except RateLimited as busy:  # the memory is saved; only its search index waits for the photo pass
        log.warning("Indexing a memory's words skipped: %s", busy)
    enrich_args = (db, ai, geo, device_id, record.id, transcript, image_bytes, image_type, where, settings)
    if settings.enrich_in_background:
        background.add_task(_enrich_memory, *enrich_args)
    else:
        await _enrich_memory(*enrich_args)
    return {"id": record.id, "timestamp": record.created_at.isoformat(), "transcript": transcript}


async def _language(db: MemoryDB, device_id: str) -> str:
    """The language of the device's household ("en" if unclaimed, or if the lookup fails)."""
    try:
        return normalize(await db.household_language(device_id))
    except Exception:
        log.exception("Looking up the language for %s failed", device_id)
        return "en"


async def _spoken_answer(
    background: BackgroundTasks, db: MemoryDB, ai: AIServices, geo: Geolocator, settings: Settings,
    device_id: str, question: str, where: Whereabouts, extra_headers: dict | None = None,
) -> Response:
    timings: dict[str, float] = {}  # stage -> milliseconds, in the order they ran
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = (now - clock) * 1000
        clock = now

    record_args = None
    answered_by = "fixed"
    if question:
        language = None
        try:
            embedding, language = await asyncio.gather(ai.embed(question, query=True), _language(db, device_id))
            lap("embed")
            memories = await retrieve(db, ai, question, [device_id], settings.query_lookback_hours, embedding=embedding)
            lap("retrieve")
            # Template answers are English sentences; other languages always go to the model.
            use_template = settings.template_answers and is_english(language)
            answer = template_answer(question, memories) if use_template else None
            answered_by = "template"
            if answer is None:
                answer = await ai.answer(question, memories, language)
                answered_by = "llm"
            lap("answer")
            record_args = (db, ai, geo, device_id, question, answer, where)
            if settings.enrich_in_background:
                background.add_task(_record_query, *record_args)
        except RateLimited as busy:  # over a model's limit: say so (a cached phrase, no model call)
            log.warning("Answer rate limited: %s", busy)
            answer = phrase(BUSY, language or await _language(db, device_id))
            answered_by = "busy"
    else:
        answer = phrase(DIDNT_CATCH, await _language(db, device_id))

    speech = await ai.speak(answer)
    lap("speak")
    timing_log.info("Spoken answer (%s): %s, total %.0f ms", answered_by,
             ", ".join(f"{k} {v:.0f} ms" for k, v in timings.items()), sum(timings.values()))
    if record_args and not settings.enrich_in_background:
        await _record_query(*record_args)
    return Response(
        content=speech,
        media_type=ai.speech_media_type,
        # Handy for debugging from curl / serial logs; the device can ignore them.
        headers={
            "X-Question": quote(question),
            "X-Answer": quote(answer),
            "X-Answered-By": answered_by,
            "Server-Timing": ", ".join(f"{k};dur={v:.0f}" for k, v in timings.items()),
            **(extra_headers or {}),
        },
    )


@router.post("/log")
async def log_memory(
    background: BackgroundTasks,
    audio: UploadFile | None = File(None),
    image: UploadFile = File(...),
    transcript: str | None = TRANSCRIPT_FORM,
    where: Whereabouts = Depends(whereabouts),
    pairing_code: str | None = Header(None, alias="X-Pairing-Code"),
    device: Principal = Depends(device_limit()),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    geo: Geolocator = Depends(get_geo),
    settings: Settings = Depends(get_settings),
):
    image_bytes = await _read(image, MAX_IMAGE_BYTES, "image")
    transcript = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
    await _note_pairing(db, device.sub, pairing_code)
    if not transcript:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")
    return await _save_memory(background, db, ai, geo, settings, device.sub, transcript, image_bytes, image.content_type, where)


@router.post("/query")
async def query_memory(
    background: BackgroundTasks,
    audio: UploadFile | None = File(None),
    transcript: str | None = TRANSCRIPT_FORM,
    where: Whereabouts = Depends(whereabouts),
    pairing_code: str | None = Header(None, alias="X-Pairing-Code"),
    device: Principal = Depends(device_limit()),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    geo: Geolocator = Depends(get_geo),
    settings: Settings = Depends(get_settings),
):
    question = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
    await _note_pairing(db, device.sub, pairing_code)
    return await _spoken_answer(background, db, ai, geo, settings, device.sub, question, where)


MAX_WAKE_PHOTOS = 4


@router.post("/voice")
async def voice(
    background: BackgroundTasks,
    audio: UploadFile | None = File(None),
    image: list[UploadFile] | None = File(None, description="photos from the wake phrase on: one, or up to 4"),
    transcript: str | None = TRANSCRIPT_FORM,
    wake: bool = Form(False, description="the device's wake model fired: speech-to-text must confirm the phrase"),
    where: Whereabouts = Depends(whereabouts),
    pairing_code: str | None = Header(None, alias="X-Pairing-Code"),
    device: Principal = Depends(device_limit()),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    geo: Geolocator = Depends(get_geo),
    settings: Settings = Depends(get_settings),
):
    """Wake phrase, no button: the words decide. A statement is saved like /log (JSON back, with
    "intent": "remember"); a question is answered like /query (spoken audio back, X-Intent: ask).

    With wake=true (the device heard something like "Last Scene" on its own model) the clip must start
    with the wake phrase, or nothing happens ({"intent": "ignored"}): that filters the wake model's false
    alarms. Then a low-reasoning model reads the words alone to decide the command (remember, ask, found,
    none; the rules in intent.py decide if it can't). Only for remember and ask do the photos go to a
    model: to pick the one showing the item, and to say so when what's asked about is in view now."""
    photos = [(await _read(i, MAX_IMAGE_BYTES, "image"), i.content_type) for i in (image or [])[:MAX_WAKE_PHOTOS]]
    heard = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
    await _note_pairing(db, device.sub, pairing_code)
    if wake and audio is not None and not has_wake_phrase(heard or ""):
        log.info("Wake model fired, but the clip doesn't start with the wake phrase: ignored")
        return {"intent": "ignored"}
    if not heard:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")
    words = strip_wake_phrase(heard)

    # Step 1, words only: what is being asked. The photos stay out of it.
    command = None
    if wake:
        try:
            command = await ai.decipher(words)
        except Exception as exc:  # busy, quota, slow: the rules decide
            log.warning("Command model unavailable (%s); using the rules", type(exc).__name__)
    if command:
        intent, item = command.intent, command.item
    if not command or (not item and intent != "none"):
        # No model answer (slow, busy, down), or no item in it: match the household's own item names.
        names = await saved_item_names(db, device.sub, settings.query_lookback_hours)
        rule_intent, rule_item = fallback_command(heard, names)
        intent, item = (intent, item or rule_item) if command else (rule_intent, rule_item)

    # Step 2, only for remember and ask: the photos, to pick the one showing the item / see if it's in view.
    view = None
    if wake and photos and intent in ("remember", "ask"):
        try:
            typed = [(data, kind if kind in ALLOWED_IMAGE_TYPES else "image/jpeg") for data, kind in photos]
            view = await ai.look_for(item, words, typed)
        except Exception as exc:  # remember keeps photo 1; ask answers from the memories
            log.warning("Photo check unavailable (%s)", type(exc).__name__)

    if intent == "ask":
        if view and view.visible and item and view.where and is_english(await _language(db, device.sub)):
            answer = IN_VIEW.format(item=item, where=view.where.strip().rstrip("."))
            speech = await ai.speak(answer)
            return Response(content=speech, media_type=ai.speech_media_type,
                            headers={"X-Intent": "ask", "X-Answer": quote(answer), "X-Answered-By": "camera"})
        return await _spoken_answer(background, db, ai, geo, settings, device.sub, words, where, {"X-Intent": "ask"})
    if intent == "found" and item:
        scope, _ = await scope_for(db, device.sub)
        await db.mark_found(scope, " ".join(item.lower().split()), datetime.now(timezone.utc))
        language = await _language(db, device.sub)
        speech = await ai.speak(phrase(GLAD_YOU_FOUND, language).format(item=item))
        return Response(content=speech, media_type=ai.speech_media_type, headers={"X-Intent": "found"})
    if intent == "none":
        return {"intent": "none"}
    if not photos:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "A memory needs the photo; send image with the audio")
    best = view.best_photo if view and 0 <= view.best_photo < len(photos) else 0
    data, kind = photos[best]
    saved = await _save_memory(background, db, ai, geo, settings, device.sub, words, data, kind, where)
    return {**saved, "intent": "remember", "photo": best}
