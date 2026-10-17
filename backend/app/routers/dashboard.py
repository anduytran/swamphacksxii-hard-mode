from collections import Counter
from datetime import datetime

from fastapi import APIRouter, Depends

from .. import signing
from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import ItemSighting, MemoryDB, QueryRecord
from ..deps import get_db

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


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


@router.get("/insights")
async def insights(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
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
