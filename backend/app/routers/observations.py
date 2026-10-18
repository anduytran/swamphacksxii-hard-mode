"""Vision pipeline API: ingest observations, and the navigation endpoints built on them.

    POST   /api/v1/observations                     one frame + what was detected
    DELETE /api/v1/observations/{observation_id}    undo a wrong detection
    GET    /api/v1/items/last-seen                  every item's last-seen frame + box
    GET    /api/v1/items/{label}/last-seen          one item's (latest_memory / locate)
    POST   /api/v1/items/{label}/found              the wearer found it (found)

All use the device's Auth0 machine-to-machine token; each device sees its household.
"""

import json
import logging
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, ValidationError

from .. import signing
from ..ai import AIServices
from ..auth import Principal, require_device
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..db_observations import LastSeen
from ..deps import device_limit, get_ai, get_db, get_geo, get_raia
from ..geo import Geolocator, locate_and_record, record_public
from ..maps_names import name_memory_place
from ..observations import Observation, observation_location, scope_for, store_observation, too_far_ahead
from ..raia import Raia, candidates, extract_contexts
from ..wifi_places import from_wifi_json, room_of_scan, usable
from .memory import ALLOWED_IMAGE_TYPES, MAX_AUDIO_BYTES, MAX_IMAGE_BYTES, _read

router = APIRouter(prefix="/api/v1", tags=["observations"])
log = logging.getLogger(__name__)


def _label(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


def _absolute(request: Request, path: str) -> str:
    """Links for the pipeline host, which fetches them directly (behind Vercel's proxy too)."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}{path}"


def _last_seen_json(request: Request, settings: Settings, ls: LastSeen) -> dict:
    return {
        "label": ls.item,
        "memory_id": ls.memory_id,
        "observation_id": ls.observation_id,
        "source": ls.source,
        "captured_at": ls.captured_at.isoformat(),
        "scene": {"id": ls.scene_id, "label": ls.scene_label} if ls.scene_id or ls.scene_label else None,
        "bbox": ls.bbox,
        "confidence": ls.confidence,
        "maybe": ls.confidence is not None and ls.confidence < settings.maybe_confidence,
        "image_size": {"width": ls.image_width, "height": ls.image_height} if ls.image_width else None,
        "image_url": (
            _absolute(request, signing.image_url(settings.image_url_secret, f"ls-{ls.id}", settings.image_url_ttl_seconds))
            if ls.has_image else None
        ),
        "found_at": ls.found_at.isoformat() if ls.found_at else None,
    }


@router.post("/observations", status_code=status.HTTP_201_CREATED)
async def ingest_observation(
    response: Response,
    background: BackgroundTasks,
    image: UploadFile = File(...),
    observation: str = Form(..., description="JSON, see the integration contract"),
    audio: UploadFile | None = File(None),
    pairing_code: str | None = Header(None, alias="X-Pairing-Code"),
    device: Principal = Depends(device_limit("observations")),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
    raia: Raia | None = Depends(get_raia),
    geo: Geolocator = Depends(get_geo),
):
    try:
        obs = Observation.model_validate(json.loads(observation))
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"observation is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"observation: {problems}") from exc
    if too_far_ahead(obs.captured_at):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "captured_at is more than 5 minutes in the future; check the host clock")

    # Retries with the same observation_id are safe: answer with what was stored.
    if existing := await db.find_observation(str(obs.observation_id)):
        memory_id, owner = existing
        if owner != device.sub:
            raise HTTPException(status.HTTP_409_CONFLICT, "observation_id already used by another device")
        response.status_code = status.HTTP_200_OK
        labels = sorted({d["label"] for d in (await db.detections_for([memory_id])).get(memory_id, [])})
        return {"memory_id": memory_id, "items": labels, "duplicate": True}

    if not any(o.confidence >= settings.min_detection_confidence for o in obs.objects):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"no object at or above confidence {settings.min_detection_confidence}",
        )
    if await db.recent_auto_uploads(device.sub, settings.observation_min_interval_seconds):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "send events, not frames: one observation per device every "
            f"{settings.observation_min_interval_seconds:g} s",
            headers={"Retry-After": str(max(1, round(settings.observation_min_interval_seconds)))},
        )

    image_bytes = await _read(image, MAX_IMAGE_BYTES, "image")
    image_type = image.content_type if image.content_type in ALLOWED_IMAGE_TYPES else "image/jpeg"
    transcript = ""
    if audio is not None:
        audio_bytes = await _read(audio, MAX_AUDIO_BYTES, "audio")
        transcript = await ai.transcribe(audio.filename, audio_bytes, audio.content_type)
    if pairing_code:
        await db.remember_pairing_code(device.sub, pairing_code)

    try:
        stored = await store_observation(db, ai, settings, device.sub, obs, image_bytes, image_type, transcript)
    except asyncpg.UniqueViolationError:
        # Two copies of the same observation raced; the other one won.
        existing = await db.find_observation(str(obs.observation_id))
        response.status_code = status.HTTP_200_OK
        return {"memory_id": existing[0] if existing else None, "items": [], "duplicate": True}
    # After the response, like the memory enrichment: where it was seen, then context from the frame (raia.py).
    # Only a new memory counts on the public map: a merged refresh is the same item seen again.
    new_items = stored.items if stored.created else []
    args = (db, ai, geo, raia, settings, device.sub, stored.memory_id, obs, image_bytes, image_type, new_items)
    if settings.enrich_in_background:
        background.add_task(_after_observation, *args)
    else:
        await _after_observation(*args)
    if not stored.created:  # refreshed the item's recent passive memory: one timeline entry per item
        response.status_code = status.HTTP_200_OK
    return {"memory_id": stored.memory_id, "items": stored.items, "duplicate": False, "merged": not stored.created}


async def _after_observation(db, ai, geo, raia: Raia | None, settings: Settings, device_id: str, memory_id: str,
                             obs: Observation, image: bytes, image_type: str, new_items: list[str] = ()) -> None:
    """Locate the sighting (its own Wi-Fi scan, else the wearable's last position) and its place; a new memory
    at a public place, in a household that shares, adds its items to the public map; then, for the user's items,
    raia's read of the frame, saved with the same location. Best effort, never raises."""
    loc, scan = None, {}
    try:
        scan = from_wifi_json([a.model_dump() for a in obs.location.wifi]) if obs.location else {}
        if not usable(scan) and hasattr(db, "recent_scan"):  # faker often hears one access point; sreer many
            _, device_ids = await scope_for(db, device_id)
            scan = await db.recent_scan(device_ids, settings.location_reuse_minutes) or {}
        if usable(scan) and hasattr(db, "set_scan"):
            await db.set_scan("memory_logs", memory_id, scan)
            await db.set_sighting_scan(str(obs.observation_id), scan)
            scope, _ = await scope_for(db, device_id)
            if room := await room_of_scan(db, scope, scan):
                await db.set_wifi_room(memory_id, room)
    except Exception:
        log.exception("Keeping the Wi-Fi scan of observation %s failed", memory_id)
    try:
        loc = await observation_location(geo, db, device_id, obs, settings)
        if loc is not None:
            place, public = await locate_and_record(db, device_id, "memory_logs", memory_id, loc)
            if hasattr(db, "locate_sightings"):
                await db.locate_sightings(str(obs.observation_id), loc.lat, loc.lng, loc.accuracy_m,
                                          place.id if place else None)
            if public and new_items:  # an item left at a public place: like a saved memory (memory._enrich_memory)
                await record_public(db, device_id, loc, "left", list(new_items))
                await name_memory_place(db, ai, settings, memory_id)  # the cell's Google Maps name, if MAPS_GROUNDING
    except Exception:
        log.exception("Locating observation %s failed", memory_id)

    async def locate(raia_memory_id: str) -> None:
        if usable(scan):
            await db.set_scan("memory_logs", raia_memory_id, scan)
            scope, _ = await scope_for(db, device_id)
            if room := await room_of_scan(db, scope, scan):
                await db.set_wifi_room(raia_memory_id, room)
        if loc is not None:
            await locate_and_record(db, device_id, "memory_logs", raia_memory_id, loc)

    if raia is not None and raia.configured and candidates(obs, settings.raia_min_confidence):
        await extract_contexts(db, ai, raia, settings, device_id, memory_id, obs, image, image_type, locate)


@router.delete("/observations/{observation_id}")
async def delete_observation(
    observation_id: str,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
):
    labels = await db.delete_observation(device.sub, observation_id)
    if labels is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No observation with that id from this device")
    scope, device_ids = await scope_for(db, device.sub)
    for label in labels:  # last-seen falls back to the previous sighting, if any
        await db.rebuild_last_seen(scope, label, device_ids)
    return {"deleted": observation_id, "items": labels}


@router.get("/items/last-seen")
async def all_last_seen(
    request: Request,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    scope, _ = await scope_for(db, device.sub)
    return {"items": [_last_seen_json(request, settings, ls) for ls in await db.last_seen(scope)]}


@router.get("/items/{label}/last-seen")
async def one_last_seen(
    label: str,
    request: Request,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    scope, _ = await scope_for(db, device.sub)
    found = await db.last_seen(scope, _label(label))
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sighting of {_label(label)!r}")
    return _last_seen_json(request, settings, found[0])


class Found(BaseModel):
    found_at: datetime | None = None
    observation_id: str | None = None


@router.post("/items/{label}/found")
async def item_found(
    label: str,
    body: Found | None = None,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
):
    scope, _ = await scope_for(db, device.sub)
    when = (body.found_at if body and body.found_at else datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not await db.mark_found(scope, _label(label), when):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sighting of {_label(label)!r}")
    return {"label": _label(label), "found_at": when.isoformat()}
