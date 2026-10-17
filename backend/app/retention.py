"""Daily privacy purge: delete memory logs and images older than RETENTION_HOURS.

Run from cron:  python -m app.retention
"""

import asyncio
import logging

from .config import get_settings
from .db import MemoryDB
from .storage import ObjectStore

log = logging.getLogger("retention")


async def purge() -> int:
    settings = get_settings()
    db = MemoryDB(settings)
    store = ObjectStore(settings)
    await db.connect()
    try:
        keys = await db.delete_older_than(settings.retention_hours)
    finally:
        await db.close()
    if keys:
        store.delete_keys(keys)
    log.info("Purged %d records older than %dh", len(keys), settings.retention_hours)
    return len(keys)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(purge())
