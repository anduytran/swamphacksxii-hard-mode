"""Automatic observations from the vision pipeline (mixed into MemoryDB).

An observation is a memory with source='auto': the frame lives in memory_logs like any
photo, each detected object in memory_detections, and every item's most recent sighting
in item_last_seen. item_last_seen keeps its own copy of the frame so "where did I last
see it" can outlive the 72-hour purge if LAST_SEEN_RETENTION_HOURS says so.

Last-seen records are scoped to a household (or, for a device not yet in one, to the
device itself), so every device in a household shares one view of where things are.
"""

import json
from dataclasses import dataclass
from datetime import datetime

# Events that say where an item is now; picked_up only records the moment.
LOCATING_EVENTS = ("placed", "seen", "moved")


@dataclass
class LastSeen:
    id: str
    scope: str
    item: str
    memory_id: str | None
    observation_id: str | None
    source: str
    captured_at: datetime
    scene_id: str | None
    scene_label: str | None
    bbox: list[float] | None
    confidence: float | None
    image_width: int | None
    image_height: int | None
    has_image: bool
    found_at: datetime | None


_LAST_SEEN_COLUMNS = """id, scope, item, memory_id, observation_id, source, captured_at, scene_id,
    scene_label, bbox, confidence, image_width, image_height, image IS NOT NULL AS has_image, found_at"""


class ObservationsMixin:
    pool: object  # asyncpg.Pool, from MemoryDB

    @staticmethod
    async def init_observations_schema(conn) -> None:
        await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'manual'")
        await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS observation_id uuid")
        await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS received_at timestamptz NOT NULL DEFAULT now()")
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS memory_logs_observation_idx "
            "ON memory_logs (observation_id) WHERE observation_id IS NOT NULL"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_detections (
                id           bigserial PRIMARY KEY,
                memory_id    uuid NOT NULL REFERENCES memory_logs (id) ON DELETE CASCADE,
                label        text NOT NULL,
                confidence   real NOT NULL,
                bbox         real[] NOT NULL,      -- [x, y, w, h], 0..1, top-left origin
                track_id     text,
                event        text NOT NULL,        -- placed | seen | moved | picked_up
                scene_id     text,
                scene_label  text,
                scene_score  real,
                image_width  integer,
                image_height integer,
                pipeline     jsonb
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_detections_memory_idx ON memory_detections (memory_id)"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS item_last_seen (
                id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                scope           text NOT NULL,     -- household id, or device id if unclaimed
                item            text NOT NULL,
                memory_id       uuid REFERENCES memory_logs (id) ON DELETE SET NULL,
                observation_id  uuid,
                source          text NOT NULL,
                captured_at     timestamptz NOT NULL,
                scene_id        text,
                scene_label     text,
                bbox            real[],
                confidence      real,
                image           bytea,
                image_type      text,
                image_width     integer,
                image_height    integer,
                found_at        timestamptz,
                UNIQUE (scope, item)
            )
            """
        )

    # ---------------------------------------------------------------- observations

    async def find_observation(self, observation_id: str) -> tuple[str, str] | None:
        row = await self.pool.fetchrow(
            "SELECT id, device_id FROM memory_logs WHERE observation_id = $1::uuid", observation_id
        )
        return (str(row["id"]), row["device_id"]) if row else None

    async def recent_auto_uploads(self, device_id: str, seconds: float) -> int:
        return await self.pool.fetchval(
            "SELECT count(*) FROM memory_logs WHERE device_id = $1 AND source = 'auto' "
            "AND received_at > now() - make_interval(secs => $2)",
            device_id,
            seconds,
        )

    async def insert_observation(
        self, device_id: str, observation_id: str, captured_at: datetime,
        transcript: str, image: bytes, image_type: str,
    ) -> str:
        return await self.pool.fetchval(
            """
            INSERT INTO memory_logs (device_id, transcript, image, image_type, source, observation_id, created_at)
            VALUES ($1, $2, $3, $4, 'auto', $5::uuid, $6)
            RETURNING id::text
            """,
            device_id,
            transcript,
            image,
            image_type,
            observation_id,
            captured_at,
        )

    async def recent_position(self, device_ids: list[str], minutes: float) -> tuple[float, float, float | None] | None:
        """The newest located memory or question of these devices from the last `minutes`: (lat, lng, accuracy)."""
        row = await self.pool.fetchrow(
            """
            SELECT lat, lng, accuracy_m FROM (
                SELECT lat, lng, accuracy_m, created_at FROM memory_logs
                WHERE device_id = ANY($1::text[]) AND lat IS NOT NULL AND created_at > now() - make_interval(secs => $2)
                UNION ALL
                SELECT lat, lng, accuracy_m, created_at FROM query_logs
                WHERE device_id = ANY($1::text[]) AND lat IS NOT NULL AND created_at > now() - make_interval(secs => $2)
            ) located ORDER BY created_at DESC LIMIT 1
            """,
            device_ids,
            minutes * 60,
        )
        return (row["lat"], row["lng"], row["accuracy_m"]) if row else None

    async def recent_passive_memory(self, device_ids: list[str], label: str, minutes: float,
                                    captured_at: datetime) -> str | None:
        """The household's newest passive memory with `label`, if it's from the last `minutes` and not newer
        than `captured_at` (a late upload of an older frame never replaces a newer one)."""
        return await self.pool.fetchval(
            """
            SELECT m.id::text FROM memory_logs m JOIN memory_detections d ON d.memory_id = m.id
            WHERE m.source = 'auto' AND m.device_id = ANY($1::text[]) AND d.label = $2
              AND m.created_at > now() - make_interval(secs => $3) AND m.created_at <= $4
            ORDER BY m.created_at DESC LIMIT 1
            """,
            device_ids,
            label,
            minutes * 60,
            captured_at,
        )

    async def refresh_observation(
        self, memory_id: str, observation_id: str, captured_at: datetime, image: bytes, image_type: str,
    ) -> None:
        """A newer sighting of the same thing: its frame, time and id replace the memory's; detections go."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE memory_logs SET image = $2, image_type = $3, observation_id = $4::uuid, "
                "created_at = GREATEST(created_at, $5), received_at = now() WHERE id = $1::uuid",
                memory_id,
                image,
                image_type,
                observation_id,
                captured_at,
            )
            await conn.execute("DELETE FROM memory_detections WHERE memory_id = $1::uuid", memory_id)
            await conn.execute("DELETE FROM memory_items WHERE memory_id = $1::uuid", memory_id)

    async def add_detections(
        self, memory_id: str, detections: list[dict], scene: dict | None, pipeline: dict, size: tuple[int, int]
    ) -> None:
        await self.pool.executemany(
            """
            INSERT INTO memory_detections (memory_id, label, confidence, bbox, track_id, event, scene_id,
                scene_label, scene_score, image_width, image_height, pipeline)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
            """,
            [
                (
                    memory_id, d["label"], d["confidence"], d["bbox"], d.get("track_id"), d["event"],
                    (scene or {}).get("id"), (scene or {}).get("label"), (scene or {}).get("match_score"),
                    size[0], size[1], json.dumps(pipeline),
                )
                for d in detections
            ],
        )

    async def detections_for(self, memory_ids: list[str]) -> dict[str, list[dict]]:
        rows = await self.pool.fetch(
            "SELECT memory_id, label, confidence, bbox, event, scene_label FROM memory_detections "
            "WHERE memory_id = ANY($1::uuid[]) ORDER BY id",
            memory_ids,
        )
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(str(r["memory_id"]), []).append(
                {"label": r["label"], "confidence": _round(r["confidence"], 4), "bbox": _bbox(r["bbox"]),
                 "event": r["event"], "scene_label": r["scene_label"]}
            )
        return out

    async def delete_observation(self, device_id: str, observation_id: str) -> list[str] | None:
        """Delete a device's own observation; returns the labels it held (None if not found)."""
        async with self.pool.acquire() as conn, conn.transaction():
            memory_id = await conn.fetchval(
                "SELECT id FROM memory_logs WHERE observation_id = $1::uuid AND device_id = $2",
                observation_id,
                device_id,
            )
            if memory_id is None:
                return None
            labels = [r["label"] for r in await conn.fetch(
                "SELECT DISTINCT label FROM memory_detections WHERE memory_id = $1", memory_id
            )]
            # The caller's rebuild_last_seen replaces the row in the device's current scope, keeping
            # found_at; a row left in an older scope (sent before the device joined a household) goes now.
            await conn.execute(
                "DELETE FROM item_last_seen WHERE memory_id = $1 AND scope <> COALESCE("
                "(SELECT household_id::text FROM household_devices WHERE device_id = $2), $2)",
                memory_id,
                device_id,
            )
            await conn.execute("DELETE FROM memory_logs WHERE id = $1", memory_id)
        return labels

    # ---------------------------------------------------------------- last seen

    async def upsert_last_seen(
        self, scope: str, item: str, memory_id: str, observation_id: str, source: str, captured_at: datetime,
        scene_id: str | None, scene_label: str | None, bbox: list[float], confidence: float,
        image: bytes, image_type: str, width: int, height: int, conn=None,
    ) -> None:
        """Record a sighting unless a newer one is already stored (late or replayed uploads).

        Only a sighting captured after found_at reopens a found item: one taken before "found it"
        but uploaded after it (an offline queue) moves the record but leaves it found.
        """
        await (conn or self.pool).execute(
            """
            INSERT INTO item_last_seen (scope, item, memory_id, observation_id, source, captured_at, scene_id,
                scene_label, bbox, confidence, image, image_type, image_width, image_height)
            VALUES ($1, $2, $3::uuid, $4::uuid, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (scope, item) DO UPDATE SET
                memory_id = EXCLUDED.memory_id, observation_id = EXCLUDED.observation_id, source = EXCLUDED.source,
                captured_at = EXCLUDED.captured_at, scene_id = EXCLUDED.scene_id, scene_label = EXCLUDED.scene_label,
                bbox = EXCLUDED.bbox, confidence = EXCLUDED.confidence, image = EXCLUDED.image,
                image_type = EXCLUDED.image_type, image_width = EXCLUDED.image_width,
                image_height = EXCLUDED.image_height,
                found_at = CASE WHEN EXCLUDED.captured_at > item_last_seen.found_at THEN NULL
                                ELSE item_last_seen.found_at END
            WHERE item_last_seen.captured_at <= EXCLUDED.captured_at
            """,
            scope, item, memory_id, observation_id, source, captured_at, scene_id, scene_label,
            bbox, confidence, image, image_type, width, height,
        )

    async def rebuild_last_seen(self, scope: str, item: str, device_ids: list[str]) -> None:
        """After a deletion: the latest remaining sighting of the item becomes last-seen.

        It stays found if it was found after that sighting (same rule as upsert_last_seen). One
        transaction, so a failure part-way never loses found_at or leaves the item missing.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            found_at = await conn.fetchval(
                "SELECT found_at FROM item_last_seen WHERE scope = $1 AND item = $2 FOR UPDATE", scope, item
            )
            row = await conn.fetchrow(
                """
                SELECT m.id, m.observation_id, m.created_at, m.image, m.image_type,
                       d.scene_id, d.scene_label, d.bbox, d.confidence, d.image_width, d.image_height
                FROM memory_detections d JOIN memory_logs m ON m.id = d.memory_id
                WHERE d.label = $1 AND d.event = ANY($2::text[]) AND m.device_id = ANY($3::text[])
                ORDER BY m.created_at DESC, d.id DESC LIMIT 1
                """,
                item,
                list(LOCATING_EVENTS),
                device_ids,
            )
            await conn.execute("DELETE FROM item_last_seen WHERE scope = $1 AND item = $2", scope, item)
            if row is None:
                return
            await self.upsert_last_seen(
                scope, item, str(row["id"]), str(row["observation_id"]) if row["observation_id"] else None, "auto",
                row["created_at"], row["scene_id"], row["scene_label"], list(row["bbox"]), row["confidence"],
                row["image"], row["image_type"], row["image_width"], row["image_height"], conn=conn,
            )
            if found_at is not None:
                await conn.execute(
                    "UPDATE item_last_seen SET found_at = $3 WHERE scope = $1 AND item = $2 AND captured_at <= $3",
                    scope, item, found_at,
                )

    async def last_seen(self, scope: str, item: str | None = None) -> list[LastSeen]:
        rows = await self.pool.fetch(
            f"SELECT {_LAST_SEEN_COLUMNS} FROM item_last_seen WHERE scope = $1 AND ($2::text IS NULL OR item = $2) "
            "ORDER BY captured_at DESC",
            scope,
            item,
        )
        return [_last_seen(r) for r in rows]

    async def last_seen_image(self, last_seen_id: str) -> tuple[bytes, str] | None:
        row = await self.pool.fetchrow(
            "SELECT image, image_type FROM item_last_seen WHERE id = $1::uuid AND image IS NOT NULL", last_seen_id
        )
        return (row["image"], row["image_type"] or "image/jpeg") if row else None

    async def mark_found(self, scope: str, item: str, found_at: datetime) -> bool:
        result = await self.pool.execute(
            "UPDATE item_last_seen SET found_at = $3 WHERE scope = $1 AND item = $2", scope, item, found_at
        )
        return result.endswith("1")

    async def household_device_ids(self, household_id: str) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT device_id FROM household_devices WHERE household_id = $1::uuid", household_id
        )
        return [r["device_id"] for r in rows]

    async def purge_last_seen(self, hours: int) -> int:
        result = await self.pool.execute(
            "DELETE FROM item_last_seen WHERE captured_at < now() - make_interval(hours => $1)", hours
        )
        return int(result.split()[-1])


def _round(v: float | None, digits: int) -> float | None:
    # Stored as 32-bit `real`: 0.1 comes back as 0.10000000149; trim to what was sent.
    return round(v, digits) if v is not None else None


def _bbox(v) -> list[float] | None:
    return [round(c, 5) for c in v] if v is not None else None


def _last_seen(r) -> LastSeen:
    return LastSeen(
        id=str(r["id"]),
        scope=r["scope"],
        item=r["item"],
        memory_id=str(r["memory_id"]) if r["memory_id"] else None,
        observation_id=str(r["observation_id"]) if r["observation_id"] else None,
        source=r["source"],
        captured_at=r["captured_at"],
        scene_id=r["scene_id"],
        scene_label=r["scene_label"],
        bbox=_bbox(r["bbox"]),
        confidence=_round(r["confidence"], 4),
        image_width=r["image_width"],
        image_height=r["image_height"],
        has_image=r["has_image"],
        found_at=r["found_at"],
    )
