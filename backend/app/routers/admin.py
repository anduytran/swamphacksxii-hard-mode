import hmac

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ..config import Settings, get_settings
from ..db import MemoryDB
from ..deps import get_db

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def require_cron_secret(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = f"Bearer {settings.cron_secret}"
    if not settings.cron_secret or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid cron secret")


@router.get("/purge", dependencies=[Depends(require_cron_secret)])
async def purge(
    db: MemoryDB = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """The 72h privacy purge, for schedulers that call a URL (Vercel Cron).

    Vercel sends "Authorization: Bearer <CRON_SECRET>". Disabled when no secret is set.
    """
    deleted = await db.delete_older_than(settings.retention_hours)
    last_seen = await db.purge_last_seen(settings.last_seen_retention_hours)
    geo_cache = await db.purge_geolocation_cache(settings.geolocation_cache_days)
    raia = await db.purge_raia_contexts(settings.retention_hours) if hasattr(db, "purge_raia_contexts") else 0
    if hasattr(db, "prune_limits"):  # rate-limit counters older than any window still in use
        await db.prune_limits(datetime.now(timezone.utc) - timedelta(days=2))
    return {
        "deleted": deleted,
        "retention_hours": settings.retention_hours,
        "last_seen_deleted": last_seen,
        "geolocation_cache_deleted": geo_cache,
        "raia_contexts_deleted": raia,
    }
