from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from .. import signing
from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_db

router = APIRouter(prefix="/api/v1/images", tags=["images"])


@router.get("/{record_id}")
async def get_image(
    record_id: str,
    exp: int = Query(...),
    sig: str = Query(...),
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    if not signing.verify(settings.image_url_secret, record_id, exp, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Link expired or invalid")
    # "ls-<id>" = an item's last-seen frame (kept apart from memories; see db_observations).
    if record_id.startswith("ls-"):
        found = await db.last_seen_image(record_id.removeprefix("ls-"))
    else:
        found = await db.get_image(record_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    data, content_type = found
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": f"private, max-age={settings.image_url_ttl_seconds}",
            "X-Content-Type-Options": "nosniff",
        },
    )
