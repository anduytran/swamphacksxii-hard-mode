import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from .. import signing
from ..ai import AIServices
from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import ACTIVITY_COUNTS_DAYS, ItemSighting, MemoryDB, QueryRecord
from ..deps import get_ai, get_db
from ..home_map import floor_plan, room_for
from ..languages import normalize
from ..geo import home_place_id, place_label
from ..pipeline import retrieve

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


async def household_devices(
    caregiver: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
) -> list[str]:
    """Devices in the caller's households: the only data a caregiver may see."""
    return await db.device_ids_for(caregiver.sub)

log = logging.getLogger(__name__)


def _image_url(settings: Settings, memory_id: str) -> str:
    return signing.image_url(settings.image_url_secret, memory_id, settings.image_url_ttl_seconds)


@router.get("/history")
async def history(
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    hours = settings.dashboard_history_hours
    records = await db.recent(hours, devices)
    items_by_memory: dict[str, list[dict]] = {}
    for s in await db.item_sightings(hours, devices):
        items_by_memory.setdefault(s.memory_id, []).append({"name": s.name, "location": s.location})
    detections = await _detections(db, [r.id for r in records if r.source == "auto"], settings)
    return [
        {
            "id": r.id,
            "device_id": r.device_id,
            "timestamp": r.created_at.isoformat(),
            "source": r.source,
            "transcript": r.transcript,
            "scene": r.scene,
            # Automatic observations have no speech; this sentence stands in for it.
            "summary": r.scene.removeprefix(AUTO_PREFIX) if r.source == "auto" and r.scene else None,
            "detections": detections.get(r.id, []),
            "image_url": _image_url(settings, r.id) if r.has_image else None,
            "items": items_by_memory.get(r.id, []),
        }
        for r in records
    ]


AUTO_PREFIX = "Seen automatically: "


async def _detections(db: MemoryDB, memory_ids: list[str], settings: Settings) -> dict[str, list[dict]]:
    """Bounding boxes per memory, flagged "maybe" below the confidence threshold."""
    if not memory_ids or not hasattr(db, "detections_for"):
        return {}
    found = await db.detections_for(memory_ids)
    for boxes in found.values():
        for d in boxes:
            d["maybe"] = d["confidence"] < settings.maybe_confidence
    return found


@router.get("/items")
async def items(
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """One entry per item: where it was last put, plus where it was before that."""
    window = settings.retention_hours
    sightings = await db.item_sightings(window, devices)
    asked = Counter(q.item for q in await db.recent_queries(window, devices) if q.item)

    by_name: dict[str, list[ItemSighting]] = {}
    for s in sightings:  # newest first, so each list starts with the latest sighting
        by_name.setdefault(s.name, []).append(s)

    photos = {name: next((s for s in seen if s.has_image), None) for name, seen in by_name.items()}
    boxes = await _detections(db, [p.memory_id for p in photos.values() if p], settings)

    result = []
    for name, seen in by_name.items():
        latest = seen[0]
        photo = photos[name]
        box = next((d for d in boxes.get(photo.memory_id, []) if d["label"] == name), None) if photo else None
        result.append(
            {
                "name": name,
                "location": latest.location,
                "last_seen": latest.created_at.isoformat(),
                "transcript": latest.transcript,
                "scene": latest.scene,
                "image_url": _image_url(settings, photo.memory_id) if photo else None,
                "detection": box,  # the item's box in that photo, when the vision pipeline saw it
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


async def _insights(db: MemoryDB, settings: Settings, devices: list[str], hours: int | None = None) -> dict:
    # Any window up to what is kept: nothing older than retention_hours exists.
    window = min(hours or settings.retention_hours, settings.retention_hours)
    memories = await db.recent(window, devices)
    sightings = await db.item_sightings(window, devices)
    queries = await db.recent_queries(window, devices)

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
    hours: int | None = Query(None, ge=1, description="Window in hours; capped at the retention period"),
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return await _insights(db, settings, devices, hours)


@router.get("/missing-times")
async def missing_times(
    days: int = Query(28, ge=7, le=ACTIVITY_COUNTS_DAYS, description="How far back to look"),
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
):
    """When things go missing: questions asked (and things put down) per hour, for the
    hour-of-day x day-of-week grid. Hours are UTC instants; the dashboard places them in the
    viewer's own time zone. Older than 72 h, only anonymous hourly counts are left."""
    hours = await db.missing_grid(devices, days) if devices else []
    return {
        "days": days,
        "since": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
        "hours": [{"at": at.isoformat(), "asked": asked, "put_down": put_down} for at, asked, put_down in hours],
    }


# One summary per process for a few minutes: it's an LLM call, and the data moves slowly.
_SUMMARY_TTL_SECONDS = 600
_summary_cache: dict[str, tuple[float, str | None]] = {}


@router.get("/summary")
async def summary(
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """A few plain sentences for the caregiver, written by the LLM (Gemini by default)."""
    data = await _insights(db, settings, devices)
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
    # In the household language (a caregiver in several households gets the first one's).
    language = normalize(await db.household_language(devices[0])) if devices else "en"
    cache_key = f"{sorted(devices)}|{language}|{facts}"
    cached = _summary_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _SUMMARY_TTL_SECONDS:
        return {"summary": cached[1]}
    try:
        text = await ai.caregiver_summary(facts, language)
    except Exception:
        log.exception("Caregiver summary failed")
        return {"summary": None}
    _summary_cache[cache_key] = (time.monotonic(), text)
    return {"summary": text}


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=300),
    limit: int = Query(5, ge=1, le=20),
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """The wearable's retrieval step, shown in full: which memories a question finds and
    which chunks matched, by meaning (vector) and by words (keyword). Embeds the query,
    so each search makes one embedding call: only for households with a wearable (an
    account with nothing to search can't spend embedding calls)."""
    if not devices:
        return {"query": q, "results": []}
    memories = await retrieve(db, ai, q, devices, settings.retention_hours, limit)
    results = []
    for m in memories:
        matched = {h.chunk_id: h for h in m.matched}
        results.append(
            {
                "memory_id": m.memory_id,
                "timestamp": m.created_at.isoformat(),
                "score": round(m.score, 5),
                "image_url": _image_url(settings, m.memory_id),
                "chunks": [
                    {
                        "kind": c.kind,
                        "content": c.content,
                        "matched": c.chunk_id in matched,
                        "vector_rank": matched[c.chunk_id].vector_rank if c.chunk_id in matched else None,
                        "keyword_rank": matched[c.chunk_id].keyword_rank if c.chunk_id in matched else None,
                        "distance": (
                            round(matched[c.chunk_id].distance, 4)
                            if c.chunk_id in matched and matched[c.chunk_id].distance is not None
                            else None
                        ),
                    }
                    for c in m.chunks
                ],
            }
        )
    return {"query": q, "results": results}


def _curve(a: list[float], b: list[float], steps: int = 12) -> list[list[float]]:
    """A gentle arc from a to b (quadratic Bezier), so trails read as movement."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    cx, cy = mx - dy * 0.25, my + dx * 0.25
    return [
        [
            round((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * cx + t**2 * b[0], 2),
            round((1 - t) ** 2 * a[1] + 2 * (1 - t) * t * cy + t**2 * b[1], 2),
        ]
        for t in (i / steps for i in range(steps + 1))
    ]


@router.get("/map")
async def home_map(
    devices: list[str] = Depends(household_devices),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Where each item is, which rooms things get lost in, and how items moved."""
    window = settings.retention_hours
    sightings = await db.item_sightings(window, devices)  # newest first
    asked = Counter(q.item for q in await db.recent_queries(window, devices) if q.item)

    history: dict[str, list] = {}
    for s in reversed(sightings):  # oldest first
        history.setdefault(s.name, []).append(s)

    plan = floor_plan(settings.floor_plan)
    rooms = {r.id: {"id": r.id, "name": r.name, "kind": r.kind, "level": r.level, "polygon": r.polygon, "putdowns": 0, "lost": 0, "items": []} for r in (*plan.rooms, plan.unknown)}
    places = {name: [room_for(s.location, s.scene, plan=plan) for s in seen] for name, seen in history.items()}
    for name, rooms_visited in places.items():
        here = rooms_visited[-1]
        for room in rooms_visited:
            rooms[room.id]["putdowns"] += 1
        rooms[here.id]["lost"] += asked.get(name, 0)  # questions count against where it was
        rooms[here.id]["items"].append(name)
    # Items sharing a room get evenly spaced spots; past rooms use a stable per-item spot.
    position = {
        name: plan.room(room_id).slot(i, len(r["items"]))
        for room_id, r in rooms.items()
        for i, name in enumerate(r["items"])
    }

    items, trips = [], []
    start = min((s.created_at for s in sightings), default=None)
    for name, seen in history.items():
        visited = places[name]
        here, latest = visited[-1], seen[-1]
        items.append(
            {
                "name": name,
                "room": here.id,
                "room_name": here.name,
                "location": latest.location,
                "last_seen": latest.created_at.isoformat(),
                "mentions": len(seen),
                "times_asked": asked.get(name, 0),
                "position": position[name],
            }
        )
        # One animated trip per item: its spot in each room it was put down in, over time.
        path, stamps = [], []
        for i, (prev, cur, s) in enumerate(zip([None, *visited], visited, seen)):
            hours = round((s.created_at - start).total_seconds() / 3600, 3)
            point = position[name] if i == len(visited) - 1 else cur.spot(name)
            if prev is None:
                path.append(point)
                stamps.append(hours)
            elif prev.id != cur.id:
                arc = _curve(path[-1], point)[1:]
                last = stamps[-1]
                path += arc
                stamps += [round(last + (hours - last) * (i + 1) / len(arc), 3) for i in range(len(arc))]
        if len(path) > 1:
            trips.append({"name": name, "path": path, "timestamps": stamps, "rooms": [p.name for p in visited]})

    used = [r for r in rooms.values() if r["id"] != plan.unknown.id or r["putdowns"]]
    return {
        "window_hours": window,
        "plan": {"id": plan.id, "name": plan.name},
        "start": start.isoformat() if start else None,  # trip timestamps are hours after this
        "span_hours": round((max(s.created_at for s in sightings) - start).total_seconds() / 3600, 3) if sightings else 0,
        "rooms": used,
        "items": items,
        "trips": trips,
    }


@router.get("/places")
async def our_places(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """The household's own map: its places (private ones included) and where each
    memory happened. Only members see this."""
    households = []
    for h in await db.households_for(caregiver.sub):
        places = await db.places_for(h.id)
        home = home_place_id(places)
        households.append(
            {
                "id": h.id,
                "name": h.name,
                "places": [
                    {"id": p.id, "label": place_label(p, home, i), "kind": p.kind, "is_home": p.id == home,
                     "lat": p.lat, "lng": p.lng, "radius_m": p.radius_m, "visits": p.visits}
                    for i, p in enumerate(places, 1)
                ],
            }
        )
    memories = await db.located_memories(await db.device_ids_for(caregiver.sub), settings.retention_hours)
    return {
        "households": households,
        "memories": [
            {"lat": m["lat"], "lng": m["lng"], "accuracy_m": m["accuracy_m"], "place_id": m["place_id"],
             "items": m["items"], "transcript": m["transcript"], "timestamp": m["created_at"].isoformat()}
            for m in memories
        ],
    }
