"""Names for public places, from Gemini's Grounding with Google Maps, under a spend guard.

  * Off unless MAPS_GROUNDING=true.
  * Only memories at a place its household marked public are sent; home and other private
    places never leave the server (db.public_memory_position checks it in SQL).
  * Answers are cached in Postgres by position (~11 m), so a familiar spot costs nothing.
  * At most MAPS_DAILY_LIMIT calls per UTC day across the app, counted before each call.
  * Fail-soft: any error leaves the memory without a name.

Runs in the background enrichment after a memory is located, never while the wearer waits.
"""

import logging
from datetime import datetime, timezone

from .config import Settings

log = logging.getLogger(__name__)


def usage_day():
    return datetime.now(timezone.utc).date()


async def name_memory_place(db, ai, settings: Settings, memory_id: str) -> dict | None:
    """Name where this memory happened, if it was a public place. {"name", "sources"} or None."""
    if not settings.maps_grounding:
        return None
    try:
        position = await db.public_memory_position(memory_id)
        if position is None:
            return None
        lat, lng = position
        found, named = await db.cached_place_name(lat, lng, settings.maps_cache_days)
        if not found:
            if not await db.reserve_maps_call(usage_day(), settings.maps_daily_limit):
                log.info("Maps grounding daily limit (%d) reached; not naming %s", settings.maps_daily_limit, memory_id)
                return None
            named = await ai.name_place(lat, lng)
            await db.cache_place_name(lat, lng, named)
        if named:
            await db.link_memory_place_name(memory_id, lat, lng)
        return named
    except Exception:
        log.exception("Naming the place of memory %s failed", memory_id)
        return None
