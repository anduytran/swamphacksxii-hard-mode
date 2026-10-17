"""Public place names from Grounding with Google Maps (mixed into MemoryDB; see maps_names.py).

  * maps_place_names caches one answer per position rounded to 4 decimals (~11 m), so a
    familiar spot is asked about once. A miss (no name) is cached too.
  * maps_grounding_usage counts calls per UTC day, for the MAPS_DAILY_LIMIT cap.
  * memory_maps_places links a memory to the cached name of where it happened. Reads also
    require the memory's place to still be public, so a place made private loses its name.
"""

import json

from .db_places import cell_of


def maps_key(lat: float, lng: float) -> tuple[float, float]:
    """~11 m: close enough to be the same shop, coarse enough to reuse answers."""
    return round(lat, 4), round(lng, 4)


def _named(row) -> dict | None:
    if row is None or not row["name"]:
        return None
    return {"name": row["name"], "sources": json.loads(row["sources"]) if row["sources"] else []}


class MapsNamesMixin:
    pool: object  # asyncpg.Pool, from MemoryDB

    @staticmethod
    async def init_maps_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS maps_place_names (
                lat4       double precision NOT NULL,
                lng4       double precision NOT NULL,
                cell_lat   double precision NOT NULL,  -- the public map's ~100 m cell (db_places.cell_of)
                cell_lng   double precision NOT NULL,
                name       text,                       -- NULL: Google Maps had nothing to say
                sources    text,                       -- JSON [{"title", "url"}]
                cached_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (lat4, lng4)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS maps_grounding_usage (
                day    date PRIMARY KEY,  -- UTC
                calls  integer NOT NULL DEFAULT 0
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_maps_places (
                memory_id  uuid PRIMARY KEY REFERENCES memory_logs (id) ON DELETE CASCADE,
                lat4       double precision NOT NULL,
                lng4       double precision NOT NULL
            )
            """
        )

    async def public_memory_position(self, memory_id: str) -> tuple[float, float] | None:
        """The memory's position, only when it happened at a place its household marked public."""
        row = await self.pool.fetchrow(
            "SELECT m.lat, m.lng FROM memory_logs m JOIN places p ON p.id = m.place_id "
            "WHERE m.id = $1::uuid AND p.kind = 'public' AND m.lat IS NOT NULL",
            memory_id,
        )
        return (row["lat"], row["lng"]) if row else None

    async def cached_place_name(self, lat: float, lng: float, max_age_days: int) -> tuple[bool, dict | None]:
        """(found, name): found=False means never asked (or expired); name may be None for a cached miss."""
        lat4, lng4 = maps_key(lat, lng)
        row = await self.pool.fetchrow(
            "SELECT name, sources FROM maps_place_names "
            "WHERE lat4 = $1 AND lng4 = $2 AND cached_at > now() - make_interval(days => $3)",
            lat4, lng4, max_age_days,
        )
        return (row is not None), _named(row)

    async def cache_place_name(self, lat: float, lng: float, named: dict | None) -> None:
        lat4, lng4 = maps_key(lat, lng)
        cell_lat, cell_lng = cell_of(lat, lng)
        await self.pool.execute(
            "INSERT INTO maps_place_names (lat4, lng4, cell_lat, cell_lng, name, sources) VALUES ($1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (lat4, lng4) DO UPDATE SET name = EXCLUDED.name, sources = EXCLUDED.sources, cached_at = now()",
            lat4, lng4, cell_lat, cell_lng,
            named["name"] if named else None,
            json.dumps(named["sources"]) if named else None,
        )

    async def link_memory_place_name(self, memory_id: str, lat: float, lng: float) -> None:
        lat4, lng4 = maps_key(lat, lng)
        await self.pool.execute(
            "INSERT INTO memory_maps_places (memory_id, lat4, lng4) VALUES ($1::uuid, $2, $3) "
            "ON CONFLICT (memory_id) DO UPDATE SET lat4 = EXCLUDED.lat4, lng4 = EXCLUDED.lng4",
            memory_id, lat4, lng4,
        )

    async def reserve_maps_call(self, day, limit: int) -> bool:
        """Count one call against the day's cap; False (nothing counted) once it is reached.
        Atomic, like reserve_geolocation_call, so serverless instances can't overshoot."""
        if limit <= 0:
            return False
        calls = await self.pool.fetchval(
            "INSERT INTO maps_grounding_usage (day, calls) VALUES ($1, 1) "
            "ON CONFLICT (day) DO UPDATE SET calls = maps_grounding_usage.calls + 1 "
            "WHERE maps_grounding_usage.calls < $2 RETURNING calls",
            day, limit,
        )
        return calls is not None

    async def place_names_for(self, memory_ids: list[str]) -> dict[str, dict]:
        """memory id -> {"name", "sources"}, for memories still at a public place."""
        if not memory_ids:
            return {}
        rows = await self.pool.fetch(
            """
            SELECT l.memory_id, c.name, c.sources
            FROM memory_maps_places l
            JOIN maps_place_names c USING (lat4, lng4)
            JOIN memory_logs m ON m.id = l.memory_id
            JOIN places p ON p.id = m.place_id AND p.kind = 'public'
            WHERE l.memory_id = ANY($1::uuid[]) AND c.name IS NOT NULL
            """,
            memory_ids,
        )
        return {str(r["memory_id"]): _named(r) for r in rows}

    async def place_names_by_cell(self) -> dict[tuple[float, float], dict]:
        """Public-map cell -> the most recently cached name in it. Only public places are ever
        named (maps_names.name_memory_place), so these are shops and parks, never homes."""
        rows = await self.pool.fetch(
            """
            SELECT DISTINCT ON (n.cell_lat, n.cell_lng) n.cell_lat, n.cell_lng, n.name, n.sources
            FROM maps_place_names n
            WHERE n.name IS NOT NULL
              -- only while a memory there is still at a public place (a place made private hides its name)
              AND EXISTS (
                SELECT 1 FROM memory_maps_places l
                JOIN memory_logs m ON m.id = l.memory_id
                JOIN places p ON p.id = m.place_id AND p.kind = 'public'
                WHERE l.lat4 = n.lat4 AND l.lng4 = n.lng4
              )
            ORDER BY n.cell_lat, n.cell_lng, n.cached_at DESC
            """
        )
        return {(r["cell_lat"], r["cell_lng"]): _named(r) for r in rows}
