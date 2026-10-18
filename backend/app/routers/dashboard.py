import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .. import signing
from ..ai import AIServices
from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import ACTIVITY_COUNTS_DAYS, ItemSighting, MemoryDB, QueryRecord
from ..deps import caregiver_ai_limit, get_ai, get_db, get_geo
from ..home_map import FloorPlan, floor_plan, room_for
from ..languages import normalize
from ..rooms import Room
from ..geo import home_place_id, place_label
from ..lastscene import relative_to_item
from ..wifi_places import best_place
from ..pipeline import retrieve

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


async def household_devices(
    caregiver: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
) -> list[str]:
    """Devices in the caller's households: the only data a user may see."""
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
    place_names = await _place_names(db, [r.id for r in records])
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
            # Public place name from Google Maps: {"name", "sources": [{"title", "url"}]}.
            # The dashboard must show the sources and "Google Maps" right after the name.
            "place_name": place_names.get(r.id),
        }
        for r in records
    ]


async def _place_names(db: MemoryDB, memory_ids: list[str]) -> dict[str, dict]:
    if not memory_ids or not hasattr(db, "place_names_for"):
        return {}
    try:
        return await db.place_names_for(memory_ids)
    except Exception:
        log.exception("Could not read place names")
        return {}


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
    """One entry per item: its LastScene (where it was last put), plus where it was before that."""
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
    _limit: Principal = Depends(caregiver_ai_limit),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """A few plain sentences for the dashboard, written by the LLM (Gemini by default)."""
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
    # In the household language (someone in several households gets the first one's).
    language = normalize(await db.household_language(devices[0])) if devices else "en"
    cache_key = f"{sorted(devices)}|{language}|{facts}"
    cached = _summary_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _SUMMARY_TTL_SECONDS:
        return {"summary": cached[1]}
    try:
        text = await ai.caregiver_summary(facts, language)
    except Exception:
        log.exception("Dashboard summary failed")
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
    _limit: Principal = Depends(caregiver_ai_limit),
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


def room_and_source(s: ItemSighting, plan: FloorPlan) -> tuple[Room, str]:
    """The room its Wi-Fi scan matched ("wifi", a room surveyed by saying "Last Scene, this is the Grand
    Ballroom"), else the room the photo showed ("photo"), else the one the words name ("words"). A room id
    from another floor plan (FLOOR_PLAN changed since) is ignored."""
    wifi = next((r for r in plan.rooms if r.id == s.wifi_room), None) if getattr(s, "wifi_room", None) else None
    if wifi:
        return wifi, "wifi"
    photo = next((r for r in plan.rooms if r.id == s.room_id), None) if s.room_id else None
    return (photo, "photo") if photo else (room_for(s.location, s.scene, plan=plan), "words")


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
    surveyed = await db.surveyed_rooms(devices) if hasattr(db, "surveyed_rooms") else {}
    rooms = {r.id: {"id": r.id, "name": r.name, "kind": r.kind, "level": r.level, "polygon": r.polygon, "putdowns": 0, "lost": 0,
                    "items": [], "wifi_scans": surveyed.get(r.id, 0)} for r in (*plan.rooms, plan.unknown)}
    places = {name: [room_and_source(s, plan)[0] for s in seen] for name, seen in history.items()}
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
                "room_source": room_and_source(latest, plan)[1],
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

    used = [r for r in rooms.values() if r["id"] != plan.unknown.id or r["putdowns"]]  # every room, surveyed or not
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
    place_names = await _place_names(db, [m["id"] for m in memories])
    return {
        "households": households,
        "memories": [
            {"lat": m["lat"], "lng": m["lng"], "accuracy_m": m["accuracy_m"], "place_id": m["place_id"],
             "items": m["items"], "transcript": m["transcript"], "timestamp": m["created_at"].isoformat(),
             "place_name": place_names.get(m["id"])}
            for m in memories
        ],
    }


@router.get("/live")
async def live(
    after: int = Query(0, ge=0, description="only events newer than this id (the newest id you have)"),
    devices: list[str] = Depends(household_devices),
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Voice commands as they happen (db_voice_events.py), newest first, for the Live view: what was heard,
    what was decided and by which model, the photo picked, what was said back, and each step's time. With them,
    the camera's passive sightings (db_history.py), newest first: the frame and box, what raia read in it, and
    where (the surveyed room its Wi-Fi matches, or the map place)."""
    if not devices or not hasattr(db, "voice_events"):
        return {"events": [], "sightings": []}
    events = await db.voice_events(devices, after)
    for e in events:
        e.pop("pending", None)  # the follow-up's inputs (Wi-Fi scan included) stay server-side
        if e.get("memory_id"):
            e["image_url"] = _image_url(settings, e["memory_id"])
    return {"events": events, "sightings": await _live_sightings(db, caregiver, settings)}


LIVE_SIGHTINGS = 30


async def _live_sightings(db: MemoryDB, caregiver: Principal, settings: Settings) -> list[dict]:
    if not hasattr(db, "recent_sightings"):
        return []
    try:
        _, scopes, places = await _scopes_and_places(db, caregiver)
        found = await db.recent_sightings(scopes, LIVE_SIGHTINGS)
        plan = floor_plan(settings.floor_plan)
        named = {scope: [p for p in await db.wifi_places(scope) if p.get("room_id")] if hasattr(db, "wifi_places") else []
                 for scope in {s.scope for s, _ in found}}
        current = {pair: await db.last_seen_keys([pair[0]], pair[1]) for pair in {(s.scope, s.item) for s, _ in found}}
        out = []
        for s, wifi in found:
            match = best_place(named[s.scope], wifi or {}) if wifi else None
            room = plan.room(match[0]["room_id"]) if match else None
            out.append({**_sighting_json(s, settings, places, current[(s.scope, s.item)]), "item": s.item,
                        "read": bool(s.context), "wifi_room": room.name if room else None})
        return out
    except Exception:  # the Live view is for watching; sightings never break the voice events
        log.exception("Live sightings failed")
        return []


# ---------------------------------------------------------------- an item's last few sightings (db_history)

def _label(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


async def _scopes_and_places(db: MemoryDB, caregiver: Principal) -> tuple[list[str], list[str], dict[str, str]]:
    """(the user's household ids, every scope they may see: those and their devices, place id -> label)."""
    households = await db.households_for(caregiver.sub)
    labels: dict[str, str] = {}
    for h in households:
        places = await db.places_for(h.id)
        home = home_place_id(places)
        labels |= {p.id: place_label(p, home, i) for i, p in enumerate(places, 1)}
    ids = [h.id for h in households]
    return ids, ids + await db.device_ids_for(caregiver.sub), labels


@router.get("/items/{label}/sightings")
async def item_sightings(
    label: str,
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """The item's last few passive sightings, newest first, with raia's context and the place, so a wrong
    one can be trashed. "last_seen" marks the one the item's last-seen comes from."""
    item = _label(label)
    _, scopes, places = await _scopes_and_places(db, caregiver)
    sightings = await db.sightings(scopes, item)
    current = await db.last_seen_keys(scopes, item)
    return {"item": item, "sightings": [_sighting_json(s, settings, places, current) for s in sightings]}


def _sighting_json(s, settings: Settings, places: dict[str, str], current: set[str]) -> dict:
    return {
        "id": s.id,
        "captured_at": s.captured_at.isoformat(),
        "source": s.source,
        "confidence": s.confidence,
        "maybe": s.confidence is not None and s.confidence < settings.maybe_confidence,
        "bbox": s.bbox,
        "image_size": {"width": s.image_width, "height": s.image_height} if s.image_width else None,
        "image_url": signing.image_url(settings.image_url_secret, f"ih-{s.id}", settings.image_url_ttl_seconds)
        if s.has_image else None,
        "where": (s.context or {}).get("where") or None,
        "summary": (s.context or {}).get("summary") or None,
        "nearby": (s.context or {}).get("nearby") or [],
        "room": (s.context or {}).get("room") or None,
        "scene": s.scene_label,
        "location": {"lat": s.lat, "lng": s.lng, "accuracy_m": s.accuracy_m} if s.lat is not None else None,
        "place": {"id": s.place_id, "label": places.get(s.place_id)} if s.place_id else None,
        "last_seen": (s.observation_id or s.captured_at.isoformat()) in current,
    }


@router.get("/items/{label}/direction")
async def item_direction(
    label: str,
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    geo=Depends(get_geo),
    settings: Settings = Depends(get_settings),
):
    """Where the item's LastScene is from the wearer right now: distance and compass direction (Wi-Fi positions),
    floors and rooms (surveyed fingerprints), and the sentence the wearer would hear (lastscene.py)."""
    item = _label(label)
    households, _, _ = await _scopes_and_places(db, caregiver)
    for household in households:
        devices = await db.household_device_ids(household)
        rel = await relative_to_item(db, geo, settings, household, devices, item, minutes=settings.wearer_fix_minutes)
        if rel is None:
            continue
        fix_at = await db.latest_fix_at(devices, settings.wearer_fix_minutes) if hasattr(db, "latest_fix_at") else None
        return {
            "item": item,
            "known": True,
            "here": rel.here,
            "meters": round(rel.meters) if rel.meters is not None else None,
            "bearing": round(rel.bearing) if rel.bearing is not None else None,
            "compass": rel.compass,
            "floors": rel.floors,
            "item_room": rel.item_room,
            "wearer_room": rel.wearer_room,
            "item_place": rel.item_place,
            "wearer_place": rel.wearer_place,
            "accuracy_m": round(rel.accuracy_m) if rel.accuracy_m else None,
            "lastscene_at": rel.item_at.isoformat(),
            "wearer_at": fix_at.isoformat() if fix_at else None,
            "sentence": rel.sentence(),
        }
    return {"item": item, "known": False}


@router.delete("/items/{label}/sightings/{sighting_id}")
async def trash_item_sighting(
    label: str,
    sighting_id: str,
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
):
    """Trash a wrong sighting: it and the memory it alone stands for go, and the item's last-seen points at the
    newest sighting left."""
    item = _label(label)
    households, scopes, _ = await _scopes_and_places(db, caregiver)
    gone = await db.trash_sighting(scopes, item, sighting_id)
    if gone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such sighting of that item")
    devices = await db.household_device_ids(gone.scope) if gone.scope in households else [gone.scope]
    await db.restore_last_seen(gone.scope, item, devices)
    return {"trashed": sighting_id, "item": item, "remaining": len(await db.sightings(scopes, item))}
