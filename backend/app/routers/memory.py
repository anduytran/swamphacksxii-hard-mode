import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
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
from ..intent import after_wake_phrase, classify, continue_command, fallback_command, strip_wake_phrase
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

    def __init__(self, kind: str, defer: bool = False):
        self.defer = defer  # slow follow-on work waits in event["pending"] for the device's follow-up request
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


async def _record_trace(db: MemoryDB, device_id: str, event: dict) -> int | None:
    if not hasattr(db, "add_voice_event"):
        return None
    try:
        return await db.add_voice_event(device_id, event)
    except Exception:  # the Live view is for watching; never let it break a command
        log.exception("Recording a voice event failed")
        return None


async def _traced(background: BackgroundTasks, db: MemoryDB, device_id: str, trace: Trace, run, deps=None):
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
    steps = trace.event["steps"]  # for the device's own log
    if isinstance(response, Response):
        response.headers["Server-Timing"] = ", ".join(f"{k};dur={v}" for k, v in steps.items())
    else:
        response = {**response, "steps": steps}
    if "pending" not in trace.event:
        background.add_task(_record_trace, db, device_id, trace.done())
        return response
    # Slow work waits in the event; the device asks for it with a follow-up request once the wearer has heard.
    event_id = await _record_trace(db, device_id, trace.done())
    if event_id is None:  # nowhere to keep it: do it after the response instead
        if deps:
            background.add_task(_run_pending, db, *deps, device_id, trace.event["pending"])
        return response
    if isinstance(response, Response):
        response.headers["X-Followup"] = str(event_id)
        return response
    return {**response, "followup": event_id}


async def _save_memory(
    background: BackgroundTasks, db: MemoryDB, ai: AIServices, geo: Geolocator, settings: Settings,
    device_id: str, transcript: str, image_bytes: bytes, image_content_type: str | None, where: Whereabouts,
    trace: "Trace | None" = None,
) -> dict:
    # The type is echoed back when the dashboard loads the photo, so never trust it blindly.
    image_type = image_content_type if image_content_type in ALLOWED_IMAGE_TYPES else "image/jpeg"
    record = await db.insert(device_id, transcript, image_bytes, image_type)
    if trace is not None and trace.defer:  # indexing and the photo pass wait for the wearable's follow-up
        trace.event["pending"] = {"type": "memory", "memory_id": record.id, "words": transcript,
                                  "image_type": image_type, "where": asdict(where)}
        return {"id": record.id, "timestamp": record.created_at.isoformat(), "transcript": transcript}
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


async def _answer_text(
    db: MemoryDB, ai: AIServices, settings: Settings, device_id: str, question: str,
    previous: tuple[str, str] | None = None,
) -> tuple[str, str, dict[str, float]]:
    """(answer, answered_by, stage -> ms) for a question, from the saved memories; nothing spoken yet."""
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = (now - clock) * 1000
        clock = now

    if not question and not previous:
        return phrase(DIDNT_CATCH, await _language(db, device_id)), "fixed", timings
    language = None
    try:
        embedding, language = await asyncio.gather(ai.embed(question or previous[0], query=True), _language(db, device_id))
        lap("embed")
        memories = await retrieve(db, ai, question or previous[0], [device_id], settings.query_lookback_hours,
                                  embedding=embedding)
        lap("retrieve")
        # Template answers are English sentences; other languages and follow-ups always go to the model.
        use_template = settings.template_answers and is_english(language) and not previous
        answer = template_answer(question, memories) if use_template else None
        answered_by = "template"
        if answer is None:
            answer = await (ai.answer(question or "Tell me more.", memories, language, previous) if previous
                            else ai.answer(question, memories, language))
            answered_by = "llm"
        lap("answer")
    except RateLimited as busy:  # over a model's limit: say so (a cached phrase, no model call)
        log.warning("Answer rate limited: %s", busy)
        answer, answered_by = phrase(BUSY, language or await _language(db, device_id)), "busy"
    return answer, answered_by, timings


async def _spoken_answer(
    background: BackgroundTasks, db: MemoryDB, ai: AIServices, geo: Geolocator, settings: Settings,
    device_id: str, question: str, where: Whereabouts, extra_headers: dict | None = None, trace: Trace | None = None,
    previous: tuple[str, str] | None = None, answer: tuple[str, str, dict[str, float]] | None = None,
) -> Response:
    """Answer out loud. `answer`: an (answer, answered_by, timings) already worked out (_answer_text)."""
    answer_text, answered_by, timings = answer or await _answer_text(db, ai, settings, device_id, question, previous)
    clock = time.perf_counter()
    speech = await ai.speak(answer_text)
    timings["speak"] = (time.perf_counter() - clock) * 1000
    timing_log.info("Spoken answer (%s): %s, total %.0f ms", answered_by,
                    ", ".join(f"{k} {v:.0f} ms" for k, v in timings.items()), sum(timings.values()))
    if trace:
        trace.add_steps(timings)
        trace.set(question=question or None, answer=answer_text, answered_by=answered_by, outcome="answered")
    if answered_by in ("template", "llm"):  # a real answer: log the question (for Insights)
        record_args = (db, ai, geo, device_id, question or (previous[0] if previous else ""), answer_text, where)
        if trace is not None and trace.defer:
            trace.event["pending"] = {"type": "query", "question": record_args[4], "answer": answer_text,
                                      "where": asdict(where)}
        elif settings.enrich_in_background:
            background.add_task(_record_query, *record_args)
        else:
            await _record_query(*record_args)
    return Response(
        content=speech,
        media_type=ai.speech_media_type,
        # Handy for debugging from curl / serial logs; the device can ignore them.
        headers={
            "X-Question": quote(question),
            "X-Answer": quote(answer_text),
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
    defer: bool = Form(False, description="reply as soon as the wearer can hear it; the slow follow-on work "
                       "(indexing, the photo pass, logging the question) runs when the device calls /followup/{id}"),
    client: str | None = Form(None, description="the device's own timings as JSON, shown on the Live page"),
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

    With wake=true (the device heard something like "Last Scene" on its own model) only the words after
    "Last Scene" count, and if speech-to-text doesn't hear the phrase at all nothing happens
    ({"intent": "ignored"}): that filters the wake model's false alarms. Then a low-reasoning model reads
    the words alone to decide the command (remember, ask, found, none; the rules in intent.py decide if it
    can't). Only for remember and ask do the photos go to a model: to pick the one showing the item, and to
    say so when what's asked about is in view now. "Last Scene, continue ..." is a follow-up question that
    keeps the last answer as context."""
    trace = Trace("wake" if wake else "voice", defer=defer)
    if client:
        try:
            trace.set(device=json.loads(client[:2000]))
        except ValueError:
            pass
    return await _traced(background, db, device.sub, trace, lambda: _voice(
        background, audio, image, transcript, wake, where, pairing_code, device, db, ai, geo, settings, trace),
        deps=(ai, geo, settings))


CONTINUE_WITHIN_S = 600  # "continue" follows on from an answer this recent


async def _previous_exchange(db: MemoryDB, device_id: str) -> tuple[str, str] | None:
    """The last (question, answer) this device heard, for "continue"."""
    if not hasattr(db, "voice_events"):
        return None
    now = datetime.now(timezone.utc)
    for e in await db.voice_events([device_id], 0, 10):
        when = datetime.fromisoformat(e["created_at"])
        if (now - when).total_seconds() > CONTINUE_WITHIN_S:
            return None
        if e.get("answer") and (e.get("question") or e.get("heard")):
            return e.get("question") or e["heard"], e["answer"]
    return None


async def _voice(background, audio, image, transcript, wake, where, pairing_code, device, db, ai, geo, settings,
                 trace: Trace):
    photos = [(await _read(i, MAX_IMAGE_BYTES, "image"), i.content_type) for i in (image or [])[:MAX_WAKE_PHOTOS]]
    heard = await _heard(ai, audio, transcript, settings.stt_hints and (lambda: stt_hints(db, device.sub)))
    trace.lap("transcribe")
    trace.set(heard=heard, photos=len(photos))
    await _note_pairing(db, device.sub, pairing_code)
    if wake and audio is not None:
        words = after_wake_phrase(heard or "")
        if words is None:
            log.info("Wake model fired, but speech-to-text didn't hear the wake phrase: ignored")
            trace.set(outcome="ignored", intent="ignored")
            return {"intent": "ignored", "heard": heard}
    else:
        words = strip_wake_phrase(heard or "")
    trace.set(words=words)
    if not words:
        if wake:
            trace.set(outcome="nothing to do", intent="none")
            return {"intent": "none", "heard": heard}
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech detected")

    # "continue": a follow-up to the last answer, with it as context.
    follow_up = continue_command(words)
    if follow_up is not None:
        previous = await _previous_exchange(db, device.sub)
        trace.set(intent="ask", continued=previous[0] if previous else None, decided_by="rules")
        trace.lap("decide")
        if previous is None and not follow_up:
            trace.set(outcome="nothing to do")
            return {"intent": "none", "heard": heard}
        return await _spoken_answer(background, db, ai, geo, settings, device.sub, follow_up, where,
                                    {"X-Intent": "ask", "X-Heard": quote(heard)}, trace=trace, previous=previous)

    # Step 1, words only: what is being asked (the photos stay out of it). The item comes from matching the
    # household's own saved item names whenever they're said, so it's always the name they taught; the
    # model's item is used only when none matches, and the rules decide when the model is slow or down.
    intent, item, decided_by = classify(words), "", "rules"
    if wake:
        command, names = await asyncio.gather(ai.decipher(words), saved_item_names(db, device.sub, settings.query_lookback_hours),
                                              return_exceptions=True)
        if isinstance(command, Exception):
            log.warning("Command model unavailable (%s); using the rules", type(command).__name__)
            trace.set(command_error=type(command).__name__)
            command = None
        rule_intent, rule_item = fallback_command(words, names if isinstance(names, dict) else {})
        intent = command.intent if command else rule_intent
        item = rule_item or (command.item if command else "")
        decided_by = "model" if command else "rules"
        trace.set(item_from="saved name" if rule_item else "model" if item else None)
        trace.lap("decide")
    trace.set(intent=intent, item=item or None, decided_by=decided_by)
    heard_header = {"X-Heard": quote(heard)}

    async def check_photos():
        """Only for remember and ask: which photo shows the item, and whether it's in view now."""
        try:
            typed = [(data, kind if kind in ALLOWED_IMAGE_TYPES else "image/jpeg") for data, kind in photos]
            return await ai.look_for(item, words, typed)
        except Exception as exc:  # remember keeps photo 1; ask answers from the memories
            log.warning("Photo check unavailable (%s)", type(exc).__name__)
            trace.set(photo_error=type(exc).__name__)
            return None

    def note_view(view) -> None:
        if view:
            trace.set(best_photo=view.best_photo, in_view=view.visible, seen_where=view.where or None)

    if intent == "ask":
        # The photo check and the memory search run side by side: whichever answer is used, nobody waits twice.
        look = wake and photos
        started = time.perf_counter()
        view, answered = await asyncio.gather(
            check_photos() if look else asyncio.sleep(0, None),
            _answer_text(db, ai, settings, device.sub, words),
        )
        if look:
            trace.event["steps"]["photos"] = round((time.perf_counter() - started) * 1000)
            note_view(view)
        trace.clock = time.perf_counter()
        if view and view.visible and item and view.where and is_english(await _language(db, device.sub)):
            answer = IN_VIEW.format(item=item, where=view.where.strip().rstrip("."))
            speech = await ai.speak(answer)
            trace.lap("speak")
            trace.set(answer=answer, answered_by="camera", outcome="answered")
            return Response(content=speech, media_type=ai.speech_media_type,
                            headers={"X-Intent": "ask", "X-Answer": quote(answer), "X-Answered-By": "camera", **heard_header})
        return await _spoken_answer(background, db, ai, geo, settings, device.sub, words, where,
                                    {"X-Intent": "ask", **heard_header}, trace=trace, answer=answered)
    if intent == "found" and item:
        scope, _ = await scope_for(db, device.sub)
        await db.mark_found(scope, " ".join(item.lower().split()), datetime.now(timezone.utc))
        language = await _language(db, device.sub)
        answer = phrase(GLAD_YOU_FOUND, language).format(item=item)
        speech = await ai.speak(answer)
        trace.lap("speak")
        trace.set(answer=answer, outcome="found")
        return Response(content=speech, media_type=ai.speech_media_type,
                        headers={"X-Intent": "found", "X-Answer": quote(answer), **heard_header})
    if intent == "none":
        trace.set(outcome="nothing to do")
        return {"intent": "none", "heard": heard}
    if not photos:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "A memory needs the photo; send image with the audio")
    view = await check_photos() if wake else None
    if wake:
        trace.lap("photos")
        note_view(view)
    best = view.best_photo if view and 0 <= view.best_photo < len(photos) else 0
    data, kind = photos[best]
    saved = await _save_memory(background, db, ai, geo, settings, device.sub, words, data, kind, where, trace=trace)
    trace.lap("save")
    trace.set(outcome="saved", memory_id=saved["id"], saved_photo=best)
    return {**saved, "intent": "remember", "photo": best, "heard": heard}


async def _run_pending(db: MemoryDB, ai: AIServices, geo: Geolocator, settings: Settings | None, device_id: str,
                       pending: dict) -> None:
    """A wake command's deferred work: index and describe a saved memory, or log an answered question."""
    where = Whereabouts(**pending.get("where", {}))
    if pending["type"] == "memory":
        memory_id, words = pending["memory_id"], pending["words"]
        try:
            await ingest_speech(db, ai, memory_id, words)
        except RateLimited as busy:
            log.warning("Indexing a memory's words skipped: %s", busy)
        stored = await db.get_image(memory_id)
        if stored:
            await _enrich_memory(db, ai, geo, device_id, memory_id, words, stored[0], stored[1], where, settings)
    elif pending["type"] == "query":
        await _record_query(db, ai, geo, device_id, pending["question"], pending["answer"], where)


@router.post("/followup/{event_id}")
async def followup(
    event_id: int,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    geo: Geolocator = Depends(get_geo),
    settings: Settings = Depends(get_settings),
):
    """The slow work a wake command left for later (its "followup" id). The device calls this right after
    the wearer heard the reply and doesn't wait on it. Runs once; a repeat is a no-op."""
    event = await db.voice_event(device.sub, event_id) if hasattr(db, "voice_event") else None
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such command")
    pending = event.get("pending")
    if not pending:
        return {"done": False}
    await db.finish_voice_event(event_id, {"followup": "running"})  # claim it, so a retry doesn't run it twice
    started = time.perf_counter()
    try:
        await _run_pending(db, ai, geo, settings, device.sub, pending)
        result = {"followup": "done"}
    except Exception as exc:
        log.exception("Follow-up %s failed", event_id)
        result = {"followup": f"failed: {type(exc).__name__}"}
    result["followup_ms"] = round((time.perf_counter() - started) * 1000)
    await db.finish_voice_event(event_id, result)
    return {"done": True, **result}
