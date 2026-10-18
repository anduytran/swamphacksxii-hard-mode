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

from ..ai import BUSY, DIDNT_CATCH, GLAD_YOU_FOUND, IN_VIEW, AIServices, served, template_answer
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


class Trace:
    """One voice command as the dashboard's Live view shows it (db_voice_events.py): what was heard, what
    was decided and by which model, the photo picked, what was said back, and how long each step took."""

    def __init__(self, kind: str):
        self.started = self.clock = time.perf_counter()
        self.served: dict[str, str] = {}  # step -> provider/model (ai.served)
        served.set(self.served)
        self.event: dict = {"kind": kind, "steps": {}, "outcome": "error"}

    def lap(self, step: str) -> None:
        now = time.perf_counter()
        self.event["steps"][step] = round((now - self.clock) * 1000)
        self.clock = now

    def add_steps(self, timings: dict[str, float]) -> None:
        self.event["steps"].update({k: round(v) for k, v in timings.items()})
        self.clock = time.perf_counter()

    def set(self, **fields) -> None:
        self.event.update({k: v for k, v in fields.items() if v is not None})

    def done(self) -> dict:
        return {**self.event, "served": dict(self.served), "total_ms": round((time.perf_counter() - self.started) * 1000)}


async def _record_trace(db: MemoryDB, device_id: str, event: dict) -> None:
    if not hasattr(db, "add_voice_event"):
        return
    try:
        await db.add_voice_event(device_id, event)
    except Exception:  # the Live view is for watching; never let it break a command
        log.exception("Recording a voice event failed")


async def _traced(background: BackgroundTasks, db: MemoryDB, device_id: str, trace: Trace, run):
    """Run a voice route's body; record its trace after the response (or before an error is raised)."""
    try:
        response = await run()
    except HTTPException as exc:
        trace.set(error=str(exc.detail))
        await _record_trace(db, device_id, trace.done())
        raise
    except Exception as exc:
        trace.set(error=type(exc).__name__)
        await _record_trace(db, device_id, trace.done())
        raise
    background.add_task(_record_trace, db, device_id, trace.done())
    return response


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
    device_id: str, question: str, where: Whereabouts, extra_headers: dict | None = None, trace: Trace | None = None,
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
    if trace:
        trace.add_steps(timings)
        trace.set(answer=answer, answered_by=answered_by, outcome="answered")
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
    trace = Trace("button: remember")

    async def run():
        image_bytes = await _read(image, MAX_IMAGE_BYTES, "image")
        heard = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
        trace.lap("transcribe")
        trace.set(heard=heard, intent="remember", photos=1)
        await _note_pairing(db, device.sub, pairing_code)
        if not heard:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")
        saved = await _save_memory(background, db, ai, geo, settings, device.sub, heard, image_bytes, image.content_type, where)
        trace.lap("save")
        trace.set(outcome="saved", memory_id=saved["id"])
        return saved

    return await _traced(background, db, device.sub, trace, run)


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
    trace = Trace("button: ask")

    async def run():
        question = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
        trace.lap("transcribe")
        trace.set(heard=question, intent="ask")
        await _note_pairing(db, device.sub, pairing_code)
        return await _spoken_answer(background, db, ai, geo, settings, device.sub, question, where, trace=trace)

    return await _traced(background, db, device.sub, trace, run)


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
    trace = Trace("wake" if wake else "voice")
    return await _traced(background, db, device.sub, trace, lambda: _voice(
        background, audio, image, transcript, wake, where, pairing_code, device, db, ai, geo, settings, trace))


async def _voice(background, audio, image, transcript, wake, where, pairing_code, device, db, ai, geo, settings,
                 trace: Trace):
    photos = [(await _read(i, MAX_IMAGE_BYTES, "image"), i.content_type) for i in (image or [])[:MAX_WAKE_PHOTOS]]
    heard = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
    trace.lap("transcribe")
    trace.set(heard=heard, photos=len(photos))
    await _note_pairing(db, device.sub, pairing_code)
    if wake and audio is not None and not has_wake_phrase(heard or ""):
        log.info("Wake model fired, but the clip doesn't start with the wake phrase: ignored")
        trace.set(outcome="ignored", intent="ignored")
        return {"intent": "ignored"}
    if not heard:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")
    words = strip_wake_phrase(heard)

    # Step 1, words only: what is being asked (the photos stay out of it). The item comes from matching the
    # household's own saved item names whenever they're said, so it's always the name they taught; the
    # model's item is used only when none matches, and the rules decide when the model is slow or down.
    intent, item, decided_by = classify(heard), "", "rules"
    if wake:
        command, names = await asyncio.gather(ai.decipher(words), saved_item_names(db, device.sub, settings.query_lookback_hours),
                                              return_exceptions=True)
        if isinstance(command, Exception):
            log.warning("Command model unavailable (%s); using the rules", type(command).__name__)
            trace.set(command_error=type(command).__name__)
            command = None
        rule_intent, rule_item = fallback_command(heard, names if isinstance(names, dict) else {})
        intent = command.intent if command else rule_intent
        item = rule_item or (command.item if command else "")
        decided_by = "model" if command else "rules"
        trace.set(item_from="saved name" if rule_item else "model" if item else None)
        trace.lap("decide")
    trace.set(intent=intent, item=item or None, decided_by=decided_by)

    # Step 2, only for remember and ask: the photos, to pick the one showing the item / see if it's in view.
    view = None
    if wake and photos and intent in ("remember", "ask"):
        try:
            typed = [(data, kind if kind in ALLOWED_IMAGE_TYPES else "image/jpeg") for data, kind in photos]
            view = await ai.look_for(item, words, typed)
        except Exception as exc:  # remember keeps photo 1; ask answers from the memories
            log.warning("Photo check unavailable (%s)", type(exc).__name__)
            trace.set(photo_error=type(exc).__name__)
        trace.lap("photos")
        if view:
            trace.set(best_photo=view.best_photo, in_view=view.visible, seen_where=view.where or None)

    if intent == "ask":
        if view and view.visible and item and view.where and is_english(await _language(db, device.sub)):
            answer = IN_VIEW.format(item=item, where=view.where.strip().rstrip("."))
            speech = await ai.speak(answer)
            trace.lap("speak")
            trace.set(answer=answer, answered_by="camera", outcome="answered")
            return Response(content=speech, media_type=ai.speech_media_type,
                            headers={"X-Intent": "ask", "X-Answer": quote(answer), "X-Answered-By": "camera"})
        return await _spoken_answer(background, db, ai, geo, settings, device.sub, words, where, {"X-Intent": "ask"},
                                    trace=trace)
    if intent == "found" and item:
        scope, _ = await scope_for(db, device.sub)
        await db.mark_found(scope, " ".join(item.lower().split()), datetime.now(timezone.utc))
        language = await _language(db, device.sub)
        answer = phrase(GLAD_YOU_FOUND, language).format(item=item)
        speech = await ai.speak(answer)
        trace.lap("speak")
        trace.set(answer=answer, outcome="found")
        return Response(content=speech, media_type=ai.speech_media_type, headers={"X-Intent": "found"})
    if intent == "none":
        trace.set(outcome="nothing to do")
        return {"intent": "none"}
    if not photos:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "A memory needs the photo; send image with the audio")
    best = view.best_photo if view and 0 <= view.best_photo < len(photos) else 0
    data, kind = photos[best]
    saved = await _save_memory(background, db, ai, geo, settings, device.sub, words, data, kind, where)
    trace.lap("save")
    trace.set(outcome="saved", memory_id=saved["id"], saved_photo=best)
    return {**saved, "intent": "remember", "photo": best}
