from fastapi import APIRouter, Depends

from .. import signing
from ..auth import Principal, require_caregiver
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_db

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/history")
async def history(
    caregiver: Principal = Depends(require_caregiver),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    records = await db.recent(settings.dashboard_history_hours)
    return [
        {
            "id": r.id,
            "device_id": r.device_id,
            "timestamp": r.created_at.isoformat(),
            "transcript": r.transcript,
            "image_url": (
                signing.image_url(settings.image_url_secret, r.id, settings.image_url_ttl_seconds)
                if r.has_image
                else None
            ),
        }
        for r in records
    ]
