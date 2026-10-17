import logging
import time
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, Depends

from .. import signing
from ..ai import AIServices
from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import ItemSighting, MemoryDB, QueryRecord
from ..deps import get_ai, get_db

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])
log = logging.getLogger(__name__)


def _image_url(settings: Settings, memory_id: str) -> str:
    return signing.image_url(settings.image_url_secret, memory_id, settings.image_url_ttl_seconds)


@router.get("/history")
async def history(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    hours = settings.dashboard_history_hours
    records = await db.recent(hours)
    items_by_memory: dict[str, list[dict]] = {}
    for s in await db.item_sightings(hours):
        items_by_memory.setdefault(s.memory_id, []).append({"name": s.name, "location": s.location})
    return [
        {
            "id": r.id,
            "device_id": r.device_id,
            "timestamp": r.created_at.isoformat(),
            "transcript": r.transcript,
            "scene": r.scene,
            "image_url": _image_url(settings, r.id) if r.has_image else None,
            "items": items_by_memory.get(r.id, []),
        }
        for r in records
    ]


@router.get("/items")
async def items(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """One entry per item: where it was last put, plus where it was before that."""
    window = settings.retention_hours
    sightings = await db.item_sightings(window)
    asked = Counter(q.item for q in await db.recent_queries(window) if q.item)

    by_name: dict[str, list[ItemSighting]] = {}
    for s in sightings:  # newest first, so each list starts with the latest sighting
        by_name.setdefault(s.name, []).append(s)

    result = []
    for name, seen in by_name.items():
        latest = seen[0]
        photo = next((s for s in seen if s.has_image), None)
        result.append(
            {
                "name": name,
                "location": latest.location,
                "last_seen": latest.created_at.isoformat(),
                "transcript": latest.transcript,
                "scene": latest.scene,
                "image_url": _image_url(settings, photo.memory_id) if photo else None,
                "mentions": len(seen),
                "times_asked": asked.get(name, 0),
                "previous": [
                    {"location": s.location, "timestamp": s.created_at.isoformat()}
                    for s in seen[1:5]
                ],
            }
        )
    return {"window_hours": window, "items": result}


def _was_answerable(q: QueryRecord, first_seen: dict[str, datetime]) -> bool:
    """True if the item asked about had been logged before the question."""
    return q.item is not None and q.item in first_seen and first_seen[q.item] <= q.created_at


async def _insights(db: MemoryDB, settings: Settings) -> dict:
    window = settings.retention_hours
    memories = await db.recent(window)
    sightings = await db.item_sightings(window)
    queries = await db.recent_queries(window)

    first_seen: dict[str, datetime] = {}
    for s in reversed(sightings):  # oldest first
        first_seen.setdefault(s.name, s.created_at)

    unanswered = [q for q in queries if not _was_answerable(q, first_seen)]
    return {
        "window_hours": window,
        "memories_logged": len(memories),
        "questions_asked": len(queries),
        "items_tracked": len(first_seen),
        "unanswered_count": len(unanswered),
        "most_asked": [
            {"name": n, "count": c}
            for n, c in Counter(q.item for q in queries if q.item).most_common(5)
        ],
        "most_logged": [
            {"name": n, "count": c} for n, c in Counter(s.name for s in sightings).most_common(5)
        ],
        "unanswered": [
            {"timestamp": q.created_at.isoformat(), "question": q.question, "item": q.item}
            for q in unanswered[:10]
        ],
        "recent_questions": [
            {
                "timestamp": q.created_at.isoformat(),
                "question": q.question,
                "answer": q.answer,
                "item": q.item,
                "answerable": _was_answerable(q, first_seen),
            }
            for q in queries[:20]
        ],
        # Raw timestamps so the browser can bucket them by the viewer's local hour.
        "activity": (
            [{"timestamp": m.created_at.isoformat(), "kind": "memory"} for m in memories]
            + [{"timestamp": q.created_at.isoformat(), "kind": "question"} for q in queries]
        ),
    }


@router.get("/insights")
async def insights(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return await _insights(db, settings)


# One summary per process for a few minutes: it's an LLM call, and the data moves slowly.
_SUMMARY_TTL_SECONDS = 600
_summary_cache: dict[str, tuple[float, str | None]] = {}


@router.get("/summary")
async def summary(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """A few plain sentences for the caregiver, written by the LLM (Gemini by default)."""
    data = await _insights(db, settings)
    if not data["memories_logged"] and not data["questions_asked"]:
        return {"summary": None}
    def counts(rows: list[dict]) -> str:
        return ", ".join(f"{r['name']} ({r['count']})" for r in rows) or "none"

    never_logged = sorted({q["item"] for q in data["unanswered"] if q["item"]})
    facts = (
        f"Window: last {data['window_hours']} hours. "
        f"Things put down: {data['memories_logged']}. Questions asked: {data['questions_asked']}. "
        f"Most asked about: {counts(data['most_asked'])}. "
        f"Most often put down: {counts(data['most_logged'])}. "
        f"Asked about but never logged: {', '.join(never_logged) or 'none'}."
    )
    cached = _summary_cache.get(facts)
    if cached and time.monotonic() - cached[0] < _SUMMARY_TTL_SECONDS:
        return {"summary": cached[1]}
    try:
        text = await ai.caregiver_summary(facts)
    except Exception:
        log.exception("Caregiver summary failed")
        return {"summary": None}
    _summary_cache[facts] = (time.monotonic(), text)
    return {"summary": text}
