"""Daily privacy purge: delete memory logs (and their photos) older than RETENTION_HOURS.

Run from cron:  python -m app.retention
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from .config import get_settings
from .db import MemoryDB

log = logging.getLogger("retention")


async def purge() -> int:
    settings = get_settings()
    db = MemoryDB(settings)
    await db.connect()
    try:
        deleted = await db.delete_older_than(settings.retention_hours)
        await db.purge_last_seen(settings.last_seen_retention_hours)
        await db.purge_raia_contexts(settings.retention_hours)
        await db.prune_limits(datetime.now(timezone.utc) - timedelta(days=2))  # old rate-limit counters
    finally:
        await db.close()
    log.info("Purged %d records older than %dh", deleted, settings.retention_hours)
    return deleted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(purge())
