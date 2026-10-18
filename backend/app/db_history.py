"""Each item's last few passive sightings (mixed into MemoryDB), kept so a wrong one can be trashed.

item_history keeps the newest KEEP sightings of every item per scope (household, or an unclaimed device): the
camera's frame and box, its location and place, and raia's JSON context when raia read that frame. Passive
memories merge (one timeline entry per item), so this is where the earlier sightings survive. Trashing one
(the dashboard's item drawer) removes it, removes the memory it alone stands for, and points the item's
last-seen back at the newest sighting left. Rows go with the last-seen retention purge.
"""

import json
from dataclasses import dataclass
from datetime import datetime

KEEP = 5  # sightings kept per item

_COLUMNS = """id::text, scope, item, captured_at, source, memory_id::text, observation_id::text, image IS NOT NULL
    AS has_image, image_width, image_height, bbox, confidence, scene_label, context, lat, lng, accuracy_m,
    place_id::text"""


@dataclass
class Sighting:
    id: str
    scope: str
    item: str
    captured_at: datetime
    source: str  # auto: a camera sighting | raia: a wake command's photo that raia read
    memory_id: str | None
    observation_id: str | None
    has_image: bool
    image_width: int | None
    image_height: int | None
    bbox: list[float] | None
    confidence: float | None
    scene_label: str | None
    context: dict | None
    lat: float | None
    lng: float | None
    accuracy_m: float | None
    place_id: str | None


def _sighting(r) -> Sighting:
    context = r["context"]
    if isinstance(context, str):
        context = json.loads(context)
    return Sighting(
        r["id"], r["scope"], r["item"], r["captured_at"], r["source"], r["memory_id"], r["observation_id"],
        r["has_image"], r["image_width"], r["image_height"],
        [round(float(v), 5) for v in r["bbox"]] if r["bbox"] else None,
        round(float(r["confidence"]), 4) if r["confidence"] is not None else None, r["scene_label"], context,
        r["lat"], r["lng"], r["accuracy_m"], r["place_id"],
    )


class HistoryMixin:
    @staticmethod
    async def init_history_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS item_history (
                id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                scope           text NOT NULL,
                item            text NOT NULL,
                captured_at     timestamptz NOT NULL,
                source          text NOT NULL,
                memory_id       uuid REFERENCES memory_logs (id) ON DELETE SET NULL,
                observation_id  uuid,
                image           bytea,
                image_type      text,
                image_width     integer,
                image_height    integer,
                bbox            real[],
                confidence      real,
                scene_label     text,
                context         jsonb,
                lat             double precision,
                lng             double precision,
                accuracy_m      real,
                place_id        uuid REFERENCES places (id) ON DELETE SET NULL,
                created_at      timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS item_history_item_idx ON item_history (scope, item, captured_at DESC)"
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS item_history_observation_idx ON item_history (observation_id)")

    async def add_sighting(
        self, scope: str, item: str, captured_at: datetime, source: str, memory_id: str | None,
        observation_id: str | None, image: bytes | None, image_type: str | None, size: tuple[int, int] | None,
        bbox: list[float] | None, confidence: float | None, scene_label: str | None = None,
        context: dict | None = None,
    ) -> str:
        """Record a sighting and keep only the item's newest KEEP."""
        async with self.pool.acquire() as conn, conn.transaction():
            sighting_id = await conn.fetchval(
                """
                INSERT INTO item_history (scope, item, captured_at, source, memory_id, observation_id, image,
                    image_type, image_width, image_height, bbox, confidence, scene_label, context)
                VALUES ($1, $2, $3, $4, $5::uuid, $6::uuid, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
                RETURNING id::text
                """,
                scope, item, captured_at, source, memory_id, observation_id, image, image_type,
                size[0] if size else None, size[1] if size else None, bbox, confidence, scene_label,
                json.dumps(context) if context else None,
            )
            await conn.execute(
                """
                DELETE FROM item_history WHERE scope = $1 AND item = $2 AND id NOT IN (
                    SELECT id FROM item_history WHERE scope = $1 AND item = $2 ORDER BY captured_at DESC LIMIT $3)
                """,
                scope, item, KEEP,
            )
        return sighting_id

    async def locate_sightings(self, observation_id: str, lat: float, lng: float, accuracy_m: float | None,
                               place_id: str | None) -> None:
        await self.pool.execute(
            "UPDATE item_history SET lat = $2, lng = $3, accuracy_m = $4, place_id = $5::uuid "
            "WHERE observation_id = $1::uuid",
            observation_id, lat, lng, accuracy_m, place_id,
        )

    async def locate_sighting_like_memory(self, sighting_id: str) -> None:
        """A raia sighting takes the position its memory was given."""
        await self.pool.execute(
            "UPDATE item_history h SET lat = m.lat, lng = m.lng, accuracy_m = m.accuracy_m, place_id = m.place_id "
            "FROM memory_logs m WHERE h.id = $1::uuid AND m.id = h.memory_id",
            sighting_id,
        )

    async def add_sighting_context(self, scope: str, item: str, observation_id: str, context: dict) -> None:
        await self.pool.execute(
            "UPDATE item_history SET context = $4::jsonb WHERE scope = $1 AND item = $2 AND observation_id = $3::uuid",
            scope, item, observation_id, json.dumps(context),
        )

    async def sightings(self, scopes: list[str], item: str) -> list[Sighting]:
        rows = await self.pool.fetch(
            f"SELECT {_COLUMNS} FROM item_history WHERE scope = ANY($1::text[]) AND item = $2 "
            "ORDER BY captured_at DESC LIMIT $3",
            scopes, item, KEEP,
        )
        return [_sighting(r) for r in rows]

    async def lastscene_position(self, scope: str, device_ids: list[str], item: str):
        """The item's LastScene position, (lat, lng, accuracy_m, place_id, when): the newest of its located
        sightings, the memory its last-seen frame came from, and located memories that name it. None if unknown."""
        row = await self.pool.fetchrow(
            """
            SELECT lat, lng, accuracy_m, place_id, at FROM (
                SELECT lat, lng, accuracy_m, place_id::text, captured_at AS at FROM item_history
                WHERE scope = $1 AND item = $2 AND lat IS NOT NULL
                UNION ALL
                SELECT m.lat, m.lng, m.accuracy_m, m.place_id::text, ls.captured_at FROM item_last_seen ls
                JOIN memory_logs m ON m.id = ls.memory_id WHERE ls.scope = $1 AND ls.item = $2 AND m.lat IS NOT NULL
                UNION ALL
                SELECT m.lat, m.lng, m.accuracy_m, m.place_id::text, m.created_at FROM memory_items i
                JOIN memory_logs m ON m.id = i.memory_id
                WHERE m.device_id = ANY($3::text[]) AND lower(i.name) = $2 AND m.lat IS NOT NULL
            ) located ORDER BY at DESC LIMIT 1
            """,
            scope, item, device_ids,
        )
        return (row["lat"], row["lng"], row["accuracy_m"], row["place_id"], row["at"]) if row else None

    async def last_seen_keys(self, scopes: list[str], item: str) -> set[str]:
        """What the item's last-seen comes from: its observation id, else its capture time (isoformat)."""
        rows = await self.pool.fetch(
            "SELECT observation_id::text, captured_at FROM item_last_seen WHERE scope = ANY($1::text[]) AND item = $2",
            scopes, item,
        )
        return {r["observation_id"] or r["captured_at"].isoformat() for r in rows}

    async def sighting_image(self, sighting_id: str) -> tuple[bytes, str] | None:
        row = await self.pool.fetchrow(
            "SELECT image, image_type FROM item_history WHERE id = $1::uuid AND image IS NOT NULL", sighting_id
        )
        return (bytes(row["image"]), row["image_type"] or "image/jpeg") if row else None

    async def trash_sighting(self, scopes: list[str], item: str, sighting_id: str) -> Sighting | None:
        """Delete a sighting the caller may see, and the memory it alone stands for (a passive memory now showing
        this sighting, or raia's memory of it). The caller then repoints last-seen (restore_last_seen)."""
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"DELETE FROM item_history WHERE id = $1::uuid AND scope = ANY($2::text[]) AND item = $3 "
                f"RETURNING {_COLUMNS}",
                sighting_id, scopes, item,
            )
            if row is None:
                return None
            gone = _sighting(row)
            if gone.memory_id:
                await conn.execute(
                    "DELETE FROM memory_logs WHERE id = $1::uuid AND (($2 = 'raia') OR "
                    "(source = 'auto' AND observation_id IS NOT DISTINCT FROM $3::uuid))",
                    gone.memory_id, gone.source, gone.observation_id,
                )
        return gone

    async def restore_last_seen(self, scope: str, item: str, device_ids: list[str]) -> None:
        """Point the item's last-seen at its newest remaining sighting (keeping found_at if it was found after);
        with none left, rebuild it from the memories (db_observations.rebuild_last_seen)."""
        row = await self.pool.fetchrow(
            "SELECT id, memory_id::text, observation_id::text, source, captured_at, scene_label, bbox, confidence, "
            "image, image_type, image_width, image_height FROM item_history WHERE scope = $1 AND item = $2 "
            "ORDER BY captured_at DESC LIMIT 1",
            scope, item,
        )
        if row is None:
            await self.rebuild_last_seen(scope, item, device_ids)
            return
        async with self.pool.acquire() as conn, conn.transaction():
            found_at = await conn.fetchval(
                "SELECT found_at FROM item_last_seen WHERE scope = $1 AND item = $2 FOR UPDATE", scope, item
            )
            await conn.execute("DELETE FROM item_last_seen WHERE scope = $1 AND item = $2", scope, item)
            await conn.execute(
                """
                INSERT INTO item_last_seen (scope, item, memory_id, observation_id, source, captured_at, scene_label,
                    bbox, confidence, image, image_type, image_width, image_height, found_at)
                VALUES ($1, $2, $3::uuid, $4::uuid, 'auto', $5, $6, $7, $8, $9, $10, $11, $12,
                        CASE WHEN $13::timestamptz > $5 THEN $13::timestamptz END)
                """,
                scope, item, row["memory_id"], row["observation_id"], row["captured_at"], row["scene_label"],
                row["bbox"], row["confidence"], row["image"], row["image_type"], row["image_width"],
                row["image_height"], found_at,
            )

    async def purge_history(self, hours: int) -> int:
        result = await self.pool.execute(
            "DELETE FROM item_history WHERE captured_at < now() - make_interval(hours => $1)", hours
        )
        return int(result.split()[-1])
