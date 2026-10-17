"""Teaching the vision pipeline a household's own items.

Dashboard (household members, caregiver token):
    GET    /api/v1/households/{household_id}/training/items
    POST   /api/v1/households/{household_id}/training/items                    {name, aliases}
    PATCH  /api/v1/households/{household_id}/training/items/{item_id}          {name?, aliases?}
    DELETE /api/v1/households/{household_id}/training/items/{item_id}
    POST   /api/v1/households/{household_id}/training/items/{item_id}/samples  multipart image + bbox + width + height
    DELETE /api/v1/households/{household_id}/training/samples/{sample_id}
    POST   /api/v1/households/{household_id}/training/items/{item_id}/suggest-box  image or memory_id -> Gemini's box
    GET    /api/v1/households/{household_id}/training/review    "Needs a look": photos the detector is unsure of
    POST   /api/v1/households/{household_id}/training/review    {item_id, memory_id, decision, bbox?, width?, height?}

Vision pipeline (device token; the device's household):
    GET    /api/v1/training/vocabulary   names + other words, for YOLO-World's set_classes
    GET    /api/v1/training/dataset      every photo as a signed link + its YOLO label line
                                         (scripts/fetch_training_set.py turns it into a YOLO folder)

Trained models (device token), uploaded and downloaded in parts of up to 3 MB, because Vercel
caps a request or response at about 4.5 MB and the Pi's INT8 model is about 10 MB:
    POST   /api/v1/training/models                                  {version, files: [{name, size, sha256}], metrics}
    PUT    /api/v1/training/models/{model_id}/files/{name}/parts/{n} raw bytes
    POST   /api/v1/training/models/{model_id}/complete               checks sizes + sha256, makes it current
    GET    /api/v1/training/models/current                          the newest complete model's manifest
    GET    /api/v1/training/models/{model_id}/files/{name}/parts/{n} raw bytes

Dashboard: GET /api/v1/households/{household_id}/training/model   is the model up to date with the photos?
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from .. import signing
from ..auth import Principal, require_caregiver, require_device
from ..box_suggest import suggest_box
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..db_training import (
    MAX_MODEL_BYTES,
    MAX_SAMPLES_PER_ITEM,
    MAX_TRAINING_BYTES,
    MAX_TRAINING_ITEMS,
    MODEL_FILES,
    MODEL_PART_BYTES,
    TrainedModel,
    TrainingItem,
)
from ..deps import get_ai, get_db
from ..review import Candidate, candidates
from .households import _require_member
from .memory import ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES, _read
from .observations import _absolute

router = APIRouter(prefix="/api/v1", tags=["training"])
log = logging.getLogger("lastscene.training")

MAX_ALIASES = 8


def _clean(text: str) -> str:
    return " ".join(text.strip().lower().split())


class ItemBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    aliases: list[str] = Field(default_factory=list, max_length=MAX_ALIASES)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not (v := _clean(v)):
            raise ValueError("Name can't be blank")
        return v

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for a in v:
            if (a := _clean(a)[:40]) and a not in out:
                out.append(a)
        return out


class ItemPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=40)
    aliases: list[str] | None = Field(None, max_length=MAX_ALIASES)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return _clean(v) if v is not None else None

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, v: list[str] | None) -> list[str] | None:
        return ItemBody._aliases(v) if v is not None else None


def _sample_url(request: Request, settings: Settings, sample_id: str, absolute: bool = False) -> str:
    path = signing.image_url(settings.image_url_secret, f"tr-{sample_id}", settings.image_url_ttl_seconds)
    return _absolute(request, path) if absolute else path


def _item_json(request: Request, settings: Settings, item: TrainingItem) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "aliases": item.aliases,
        "created_at": item.created_at.isoformat(),
        "samples": [
            {
                "id": s.id,
                "bbox": s.bbox,
                "width": s.width,
                "height": s.height,
                "image_url": _sample_url(request, settings, s.id),
                "created_at": s.created_at.isoformat(),
            }
            for s in item.samples
        ],
    }


# ---------------------------------------------------------------- dashboard


async def _require_wearable(db: MemoryDB, household_id: str) -> None:
    """Teaching needs a paired wearable: photos are for the pipeline that serves it, and a
    stranger who signs up can't pair one, so they can't fill the database with photos."""
    if not await db.household_device_ids(household_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Add the wearable to this household first")


@router.get("/households/{household_id}/training/items")
async def list_items(
    household_id: UUID,
    request: Request,
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    await _require_member(db, str(household_id), user)
    items = await db.training_items(str(household_id))
    return {
        "items": [_item_json(request, settings, i) for i in items],
        "limits": {"items": MAX_TRAINING_ITEMS, "samples_per_item": MAX_SAMPLES_PER_ITEM},
    }


@router.post("/households/{household_id}/training/items", status_code=status.HTTP_201_CREATED)
async def create_item(
    household_id: UUID,
    body: ItemBody,
    request: Request,
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    hid = str(household_id)
    await _require_member(db, hid, user)
    await _require_wearable(db, hid)
    if await db.count_training_items(hid) >= MAX_TRAINING_ITEMS:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A household can teach up to {MAX_TRAINING_ITEMS} items")
    try:
        item = await db.create_training_item(hid, body.name, [a for a in body.aliases if a != body.name])
    except asyncpg.UniqueViolationError:
        raise HTTPException(status.HTTP_409_CONFLICT, f"“{body.name}” is already on the list") from None
    return _item_json(request, settings, item)


@router.patch("/households/{household_id}/training/items/{item_id}")
async def update_item(
    household_id: UUID,
    item_id: UUID,
    body: ItemPatch,
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
):
    await _require_member(db, str(household_id), user)
    if body.name == "":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Name can't be blank")
    try:
        ok = await db.update_training_item(str(household_id), str(item_id), body.name, body.aliases)
    except asyncpg.UniqueViolationError:
        raise HTTPException(status.HTTP_409_CONFLICT, f"“{body.name}” is already on the list") from None
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    return {"updated": True}


@router.delete("/households/{household_id}/training/items/{item_id}")
async def delete_item(
    household_id: UUID, item_id: UUID, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, str(household_id), user)
    if not await db.delete_training_item(str(household_id), str(item_id)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    return {"deleted": True}


def _parse_bbox(raw: str) -> list[float]:
    """[x, y, w, h] in 0..1 with the box inside the photo and not a sliver."""
    try:
        box = [float(v) for v in json.loads(raw)]
    except (ValueError, TypeError):
        box = []
    if len(box) != 4:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "bbox must be [x, y, w, h]")
    x, y, w, h = box
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0.01 <= w <= 1 and 0.01 <= h <= 1 and x + w <= 1.001 and y + h <= 1.001):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "bbox must lie inside the photo, values 0 to 1")
    return [round(v, 5) for v in (x, y, min(w, 1 - x), min(h, 1 - y))]


@router.post("/households/{household_id}/training/items/{item_id}/samples", status_code=status.HTTP_201_CREATED)
async def add_sample(
    household_id: UUID,
    item_id: UUID,
    request: Request,
    image: UploadFile = File(...),
    bbox: str = Form(..., description="[x, y, w, h], 0..1, top-left origin"),
    width: int = Form(..., ge=32, le=8192),
    height: int = Form(..., ge=32, le=8192),
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    hid = str(household_id)
    await _require_member(db, hid, user)
    await _require_wearable(db, hid)
    box = _parse_bbox(bbox)
    await _require_room(db, hid, str(item_id))
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Photos must be JPEG, PNG or WebP")
    data = await _read(image, MAX_IMAGE_BYTES, "image")
    sample = await _store_sample(db, hid, str(item_id), data, image.content_type, width, height, box)
    return _sample_json(request, settings, sample)


async def _require_room(db: MemoryDB, household_id: str, item_id: str) -> None:
    count = await db.training_item_sample_count(household_id, item_id)
    if count is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    if count >= MAX_SAMPLES_PER_ITEM:
        raise HTTPException(status.HTTP_409_CONFLICT, f"An item can have up to {MAX_SAMPLES_PER_ITEM} photos")


async def _store_sample(db: MemoryDB, household_id: str, item_id: str, data: bytes, content_type: str,
                        width: int, height: int, box: list[float]):
    """Every way a photo becomes a training sample (Add photos, a review answer) ends here."""
    if await db.training_bytes(household_id) + len(data) > MAX_TRAINING_BYTES:
        raise HTTPException(status.HTTP_409_CONFLICT, "This household's photos are full (100 MB); delete some first")
    return await db.add_training_sample(item_id, data, content_type, width, height, box)


def _sample_json(request: Request, settings: Settings, sample) -> dict:
    return {
        "id": sample.id,
        "bbox": sample.bbox,
        "width": sample.width,
        "height": sample.height,
        "image_url": _sample_url(request, settings, sample.id),
        "created_at": sample.created_at.isoformat(),
    }


@router.delete("/households/{household_id}/training/samples/{sample_id}")
async def delete_sample(
    household_id: UUID, sample_id: UUID, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, str(household_id), user)
    if not await db.delete_training_sample(str(household_id), str(sample_id)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    return {"deleted": True}


# ---------------------------------------------------------------- suggested boxes and review


async def _item(db: MemoryDB, household_id: str, item_id: str) -> TrainingItem:
    item = next((i for i in await db.training_items(household_id) if i.id == item_id), None)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    return item


async def _memory_photo(db: MemoryDB, household_id: str, memory_id: str) -> tuple[bytes, str]:
    """A memory photo taken by one of the household's wearables (anyone else's is "not found")."""
    found = await db.household_memory_image(memory_id, await db.household_device_ids(household_id))
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    return found


async def _suggest(db: MemoryDB, ai, settings: Settings, household_id: str, item: TrainingItem, image: bytes, content_type: str) -> dict:
    """Gemini's box, or why there isn't one. Never raises: the caregiver can always draw by hand.
    status: ok (bbox set) | none (item not found in the photo) | off (VISION_PROVIDER isn't gemini)
            | busy (this household's hourly limit) | failed (Gemini errored, e.g. quota)"""
    if settings.vision_provider != "gemini":
        return {"bbox": None, "note": "", "status": "off"}
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    if await db.suggest_calls_since(household_id, since) >= settings.suggest_box_per_hour:
        return {"bbox": None, "note": "", "status": "busy"}
    await db.log_suggest_call(household_id)
    try:
        bbox, note = await suggest_box(ai, image, content_type, item.name, item.aliases)
    except Exception as exc:  # quota (429), network, a malformed answer: fall back to drawing
        log.warning("box suggestion failed: %s", type(exc).__name__)
        return {"bbox": None, "note": "", "status": "failed"}
    return {"bbox": bbox, "note": note, "status": "ok" if bbox else "none"}


@router.post("/households/{household_id}/training/items/{item_id}/suggest-box")
async def suggest_item_box(
    household_id: UUID,
    item_id: UUID,
    image: UploadFile | None = File(None, description="A JPEG (or PNG/WebP) about to be added"),
    memory_id: UUID | None = Form(None, description="Or a memory photo from the household's wearable"),
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    ai=Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    """Where Gemini thinks the item is: {bbox: [x, y, w, h] | null, note, status}, for the box editor to pre-draw."""
    hid = str(household_id)
    await _require_member(db, hid, user)
    item = await _item(db, hid, str(item_id))
    if memory_id is not None:
        data, content_type = await _memory_photo(db, hid, str(memory_id))
    elif image is not None:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Photos must be JPEG, PNG or WebP")
        data, content_type = await _read(image, MAX_IMAGE_BYTES, "image"), image.content_type
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Send an image or a memory_id")
    result = await _suggest(db, ai, settings, hid, item, data, content_type)
    if memory_id is not None and result["status"] in ("ok", "none"):  # the review card won't ask again
        await db.cache_suggestion(item.id, str(memory_id), result["bbox"], result["note"])
    return result


def _candidate_json(settings: Settings, c: Candidate) -> dict:
    return {
        "item_id": c.item.id,
        "item_name": c.item.name,
        "memory_id": c.memory_id,
        "kind": c.kind,
        "image_url": signing.image_url(settings.image_url_secret, c.memory_id, settings.image_url_ttl_seconds),
        "captured_at": c.captured_at.isoformat(),
        "bbox": c.bbox,
        "confidence": c.confidence,
        "width": c.width,
        "height": c.height,
        "suggested": c.suggested,
        "note": c.note,
    }


@router.get("/households/{household_id}/training/review")
async def review_queue(
    household_id: UUID,
    limit: int = Query(10, ge=0, le=50),
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """"Needs a look": photos the detector is unsure of, newest first. `count` and `per_item` cover
    all of them; `candidates` only the first `limit`. Items that already have the most photos are left out."""
    hid = str(household_id)
    await _require_member(db, hid, user)
    items = [i for i in await db.training_items(hid) if len(i.samples) < MAX_SAMPLES_PER_ITEM]
    devices = await db.household_device_ids(hid)
    found: list[Candidate] = []
    if items and devices:
        hours = settings.retention_hours
        found = candidates(
            items,
            await db.review_detections(devices, settings.review_confidence_min, settings.review_confidence_max, hours),
            await db.review_memory_photos(devices, hours),
            await db.reviews(hid),
        )
    per_item: dict[str, int] = {}
    for c in found:
        per_item[c.item.id] = per_item.get(c.item.id, 0) + 1
    return {"count": len(found), "per_item": per_item, "candidates": [_candidate_json(settings, c) for c in found[:limit]]}


class ReviewAnswer(BaseModel):
    item_id: UUID
    memory_id: UUID
    decision: Literal["yes", "fix", "not"]
    bbox: list[float] | None = Field(None, description="[x, y, w, h], 0..1: the box confirmed (yes) or drawn (fix)")
    width: int | None = Field(None, ge=32, le=8192)
    height: int | None = Field(None, ge=32, le=8192)


@router.post("/households/{household_id}/training/review")
async def answer_review(
    household_id: UUID,
    body: ReviewAnswer,
    request: Request,
    user: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Yes / Fix: the photo and its box become a training sample (same limits as Add photos).
    Not: remembered, so the photo isn't asked about again for that item."""
    hid, item_id, memory_id = str(household_id), str(body.item_id), str(body.memory_id)
    await _require_member(db, hid, user)
    data, content_type = await _memory_photo(db, hid, memory_id)
    if (done := (await db.reviews(hid)).get((item_id, memory_id))) and done.decision:
        raise HTTPException(status.HTTP_409_CONFLICT, "This photo was already answered")
    if body.decision == "not":
        if await db.training_item_sample_count(hid, item_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
        await db.decide_review(item_id, memory_id, "not", None, None)
        return {"decision": "not", "sample": None}
    if body.bbox is None or body.width is None or body.height is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Yes and Fix need the box and the photo's width and height")
    box = _parse_bbox(json.dumps(body.bbox))
    await _require_wearable(db, hid)
    await _require_room(db, hid, item_id)
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "This photo's format can't be used for teaching")
    sample = await _store_sample(db, hid, item_id, data, content_type, body.width, body.height, box)
    decision = "yes" if body.decision == "yes" else "fixed"
    await db.decide_review(item_id, memory_id, decision, box, sample.id)
    return {"decision": decision, "sample": _sample_json(request, settings, sample)}


# ---------------------------------------------------------------- vision pipeline


async def _device_items(db: MemoryDB, device: Principal) -> tuple[str, list[TrainingItem]]:
    household = await db.household_of_device(device.sub)
    if household is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This device isn't in a household yet")
    return household[0], await db.training_items(household[0])


def _version(items: list[TrainingItem]) -> str:
    """Changes whenever a name, other word or photo does, so the pipeline knows to re-fetch."""
    key = json.dumps([[i.id, i.name, i.aliases, [s.id for s in i.samples]] for i in items])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@router.get("/training/vocabulary")
async def vocabulary(device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)):
    """The household's items as YOLO-World classes: `classes[i]` is class id i everywhere."""
    household_id, items = await _device_items(db, device)
    return {
        "household_id": household_id,
        "version": _version(items),
        "classes": [i.name for i in items],
        "items": [{"class_id": n, "name": i.name, "aliases": i.aliases, "samples": len(i.samples)} for n, i in enumerate(items)],
    }


@router.get("/training/dataset")
async def dataset(
    request: Request,
    device: Principal = Depends(require_device),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Every photo with a signed link and its YOLO label line ("class cx cy w h", 0..1).
    Links expire after IMAGE_URL_TTL_SECONDS; fetch the photos right away."""
    household_id, items = await _device_items(db, device)
    samples = []
    for class_id, item in enumerate(items):
        for s in item.samples:
            x, y, w, h = s.bbox
            samples.append(
                {
                    "id": s.id,
                    "class_id": class_id,
                    "name": item.name,
                    "image_url": _sample_url(request, settings, s.id, absolute=True),
                    "width": s.width,
                    "height": s.height,
                    "bbox": s.bbox,
                    "bbox_xyxy_px": [round(x * s.width), round(y * s.height), round((x + w) * s.width), round((y + h) * s.height)],
                    "yolo": f"{class_id} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}",
                }
            )
    return {
        "household_id": household_id,
        "version": _version(items),
        "names": {str(n): i.name for n, i in enumerate(items)},
        "samples": samples,
    }


# ---------------------------------------------------------------- trained models


class ModelFile(BaseModel):
    name: str
    size: int = Field(..., gt=0)
    sha256: str = Field(..., pattern="^[0-9a-f]{64}$")


class NewModel(BaseModel):
    version: str = Field(..., min_length=1, max_length=64, pattern="^[A-Za-z0-9._-]+$")
    files: list[ModelFile] = Field(..., min_length=1, max_length=len(MODEL_FILES))
    metrics: dict = Field(default_factory=dict)

    @field_validator("files")
    @classmethod
    def _files(cls, v: list[ModelFile]) -> list[ModelFile]:
        names = [f.name for f in v]
        if len(set(names)) != len(names) or not set(names) <= set(MODEL_FILES):
            raise ValueError(f"files must be distinct, from {', '.join(MODEL_FILES)}")
        if "model-int8.onnx" not in names:
            raise ValueError("model-int8.onnx is required")
        if sum(f.size for f in v) > MAX_MODEL_BYTES:
            raise ValueError(f"a model can be at most {MAX_MODEL_BYTES // (1024 * 1024)} MB")
        return v


def _parts(size: int) -> int:
    return math.ceil(size / MODEL_PART_BYTES)


def _manifest(model: TrainedModel) -> dict:
    return {
        "id": model.id,
        "version": model.version,
        "files": [{**f, "parts": _parts(f["size"])} for f in model.files],
        "metrics": model.metrics,
        "part_bytes": MODEL_PART_BYTES,
        "completed_at": model.completed_at.isoformat() if model.completed_at else None,
    }


async def _device_household(db: MemoryDB, device: Principal) -> str:
    household = await db.household_of_device(device.sub)
    if household is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This device isn't in a household yet")
    return household[0]


async def _uploading(db: MemoryDB, household_id: str, model_id: UUID, name: str) -> tuple[TrainedModel, dict]:
    model = await db.get_model(household_id, str(model_id))
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    entry = next((f for f in model.files if f["name"] == name), None)
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{name} isn't part of this model")
    return model, entry


@router.post("/training/models", status_code=status.HTTP_201_CREATED)
async def create_model(body: NewModel, device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)):
    household_id = await _device_household(db, device)
    model_id = await db.create_model(household_id, body.version, [f.model_dump() for f in body.files], body.metrics)
    return {"id": model_id, "part_bytes": MODEL_PART_BYTES, "parts": {f.name: _parts(f.size) for f in body.files}}


@router.put("/training/models/{model_id}/files/{name}/parts/{part}", status_code=status.HTTP_204_NO_CONTENT)
async def put_model_part(
    model_id: UUID, name: str, part: int, request: Request, device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)
):
    household_id = await _device_household(db, device)
    model, entry = await _uploading(db, household_id, model_id, name)
    if model.complete:
        raise HTTPException(status.HTTP_409_CONFLICT, "This model is already complete")
    if not 0 <= part < _parts(entry["size"]):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"part must be 0 to {_parts(entry['size']) - 1}")
    data = await request.body()
    if not data or len(data) > MODEL_PART_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"a part is 1 byte to {MODEL_PART_BYTES} bytes")
    await db.put_model_part(model.id, name, part, data)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/training/models/{model_id}/complete")
async def complete_model(model_id: UUID, device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)):
    household_id = await _device_household(db, device)
    model = await db.get_model(household_id, str(model_id))
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    for f in model.files:
        data = await db.model_file(model.id, f["name"])
        if len(data) != f["size"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{f['name']} is incomplete or doesn't match its sha256")
    await db.complete_model(household_id, model.id)
    return {"id": model.id, "version": model.version, "current": True}


@router.get("/training/models/current")
async def current_model(device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)):
    model = await db.current_model(await _device_household(db, device))
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No trained model yet")
    return _manifest(model)


@router.get("/training/models/{model_id}/files/{name}/parts/{part}")
async def get_model_part(model_id: UUID, name: str, part: int, device: Principal = Depends(require_device), db: MemoryDB = Depends(get_db)):
    household_id = await _device_household(db, device)
    model, _ = await _uploading(db, household_id, model_id, name)
    if not model.complete:
        raise HTTPException(status.HTTP_409_CONFLICT, "This model is still uploading")
    data = await db.model_part(model.id, name, part)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such part")
    return Response(content=data, media_type="application/octet-stream")


@router.get("/households/{household_id}/training/model")
async def model_status(household_id: UUID, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    """For the Teach items page: the household's current model, and whether photos changed since."""
    hid = str(household_id)
    await _require_member(db, hid, user)
    items = await db.training_items(hid)
    model = await db.current_model(hid)
    version = _version(items)
    return {
        "dataset_version": version,
        "photos": sum(len(i.samples) for i in items),
        "model": None
        if model is None
        else {
            "version": model.version,
            "trained_at": model.completed_at.isoformat() if model.completed_at else None,
            "map50": model.metrics.get("map50"),
            "base_model": model.metrics.get("base_model"),
        },
        "up_to_date": bool(model and model.version == version),
    }
