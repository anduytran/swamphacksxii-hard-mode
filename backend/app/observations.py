"""Automatic memories from the vision pipeline: the request format and how one is stored.

See the integration contract: the pipeline (YOLO-World + ORB on a host machine) decides
when something happened and posts one observation, a frame plus what it detected. The
backend stores the real frame and boxes, makes the observation searchable from its
structured labels (no Gemini caption), and keeps each item's last-seen frame for
navigation.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .ai import AIServices
from .config import Settings
from .db import MemoryDB
from .db_observations import LOCATING_EVENTS
from .rag import ITEM, SCENE, SPEECH, item_chunk

MAX_FUTURE_SKEW = timedelta(minutes=5)


class Pipeline(BaseModel):
    name: str = Field(..., max_length=80)
    version: str = Field(..., max_length=40)
    detector: str | None = Field(None, max_length=80)
    matcher: str | None = Field(None, max_length=80)


class ImageSize(BaseModel):
    width: int = Field(..., gt=0, le=10000)
    height: int = Field(..., gt=0, le=10000)


class Scene(BaseModel):
    id: str = Field(..., max_length=80)
    label: str | None = Field(None, max_length=120)
    match_score: float | None = Field(None, ge=0, le=1)


class DetectedObject(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    confidence: float = Field(..., ge=0, le=1)
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h], normalised 0-1")
    track_id: str | None = Field(None, max_length=80)
    event: Literal["placed", "seen", "moved", "picked_up"]

    @field_validator("label")
    @classmethod
    def normalise_label(cls, v: str) -> str:
        return " ".join(v.strip().lower().split())

    @field_validator("bbox")
    @classmethod
    def bbox_inside_frame(cls, v: list[float]) -> list[float]:
        x, y, w, h = v
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001):
            raise ValueError("bbox must be [x, y, w, h] normalised to 0-1 and inside the frame")
        return [round(c, 5) for c in v]


class Observation(BaseModel):
    observation_id: uuid.UUID
    captured_at: datetime
    source: Literal["auto"]
    pipeline: Pipeline
    image_size: ImageSize
    scene: Scene | None = None
    objects: list[DetectedObject] = Field(..., min_length=1, max_length=50)

    @field_validator("captured_at")
    @classmethod
    def aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("captured_at needs a timezone (use UTC, e.g. 2026-10-17T14:03:12Z)")
        return v.astimezone(timezone.utc)


@dataclass
class Stored:
    memory_id: str
    items: list[str]
    created: bool  # False: a recent passive memory of the same items was refreshed instead (merged)


def summary(obs: Observation, kept: list[DetectedObject]) -> str:
    """A plain sentence for the dashboard and retrieval: 'keys on the kitchen counter'."""
    where = f" on the {obs.scene.label}" if obs.scene and obs.scene.label else ""
    names = sorted({o.label for o in kept})
    return f"{', '.join(names)}{where}"


def text_chunks(obs: Observation, kept: list[DetectedObject], transcript: str, settings: Settings) -> list[tuple[str, str]]:
    """Searchable text built from the structured detection, one fact per chunk."""
    chunks: list[tuple[str, str]] = []
    if transcript:
        chunks.append((SPEECH, transcript))
    if obs.scene and obs.scene.label:
        chunks.append((SCENE, f"Camera saw the {obs.scene.label}"))
    seen = set()
    for o in kept:
        if o.label in seen:
            continue
        seen.add(o.label)
        place = obs.scene.label if obs.scene and obs.scene.label else ""
        text = item_chunk(o.label, place)
        if o.event == "picked_up":
            text = f"{o.label}: picked up" + (f" from the {place}" if place else "")
        if o.confidence < settings.maybe_confidence:
            text += " (maybe)"
        chunks.append((ITEM, text))
    return chunks


async def scope_for(db: MemoryDB, device_id: str) -> tuple[str, list[str]]:
    """Last-seen records are shared across a household; an unclaimed device keeps its own."""
    household = await db.household_of_device(device_id)
    if household:
        return household[0], await db.household_device_ids(household[0])
    return device_id, [device_id]


async def store_observation(
    db: MemoryDB, ai: AIServices, settings: Settings, device_id: str, obs: Observation,
    image: bytes, image_type: str, transcript: str,
) -> Stored:
    kept = [o for o in obs.objects if o.confidence >= settings.min_detection_confidence]
    scope, device_ids = await scope_for(db, device_id)
    merge_into = (await _recent_memory(db, device_ids, {o.label for o in kept}, obs.captured_at, settings)
                  if not transcript else None)
    if merge_into:
        memory_id = merge_into
        await db.refresh_observation(memory_id, str(obs.observation_id), obs.captured_at, image, image_type)
    else:
        memory_id = await db.insert_observation(
            device_id, str(obs.observation_id), obs.captured_at, transcript, image, image_type
        )
    await db.add_detections(
        memory_id,
        [o.model_dump() for o in kept],
        obs.scene.model_dump() if obs.scene else None,
        obs.pipeline.model_dump(),
        (obs.image_size.width, obs.image_size.height),
    )
    chunks = text_chunks(obs, kept, transcript, settings)
    if merge_into:  # index only what the memory doesn't say yet (a new place), so a refresh costs no embeddings
        have = {(c.kind, c.content) for c in (await db.chunks_for([memory_id])).get(memory_id, [])}
        chunks = [c for c in chunks if c not in have]
    if chunks:
        embeddings = await ai.embed_many([text for _, text in chunks])
        await db.add_chunks(memory_id, [(kind, text, emb) for (kind, text), emb in zip(chunks, embeddings)])
    await db.set_scene(memory_id, f"Seen automatically: {summary(obs, kept)}")

    place = obs.scene.label if obs.scene and obs.scene.label else ""
    locating = [o for o in kept if o.event in LOCATING_EVENTS]
    await db.add_items(memory_id, list({o.label: place for o in locating}.items()))

    best: dict[str, DetectedObject] = {}
    for o in locating:  # one record per item: the most confident box in this frame
        if o.label not in best or o.confidence > best[o.label].confidence:
            best[o.label] = o
    for label, o in best.items():
        await db.upsert_last_seen(
            scope, label, memory_id, str(obs.observation_id), "auto", obs.captured_at,
            obs.scene.id if obs.scene else None, obs.scene.label if obs.scene else None,
            o.bbox, o.confidence, image, image_type, obs.image_size.width, obs.image_size.height,
        )
    return Stored(memory_id, sorted({o.label for o in kept}), not merge_into)


async def _recent_memory(db: MemoryDB, device_ids: list[str], labels: set[str], captured_at: datetime,
                         settings: Settings) -> str | None:
    """The passive memory to refresh: one from the last PASSIVE_MERGE_MINUTES holding every label, else None."""
    if settings.passive_merge_minutes <= 0 or not labels or not hasattr(db, "recent_passive_memory"):
        return None
    found = {await db.recent_passive_memory(device_ids, label, settings.passive_merge_minutes, captured_at)
             for label in labels}
    return found.pop() if len(found) == 1 and None not in found else None


def too_far_ahead(captured_at: datetime) -> bool:
    return captured_at > datetime.now(timezone.utc) + MAX_FUTURE_SKEW
