from fastapi import APIRouter, Depends

from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_db, get_store
from ..storage import ObjectStore

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/history")
async def history(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    store: ObjectStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    records = await db.recent(settings.dashboard_history_hours)
    return [
        {
            "id": r.id,
            "device_id": r.device_id,
            "timestamp": r.created_at.isoformat(),
            "transcript": r.transcript,
            # Presigning is a local HMAC computation, no network round trip.
            "image_url": store.presigned_url(r.image_key) if r.image_key else None,
        }
        for r in records
    ]
