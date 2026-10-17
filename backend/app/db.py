"""TigerData (Timescale Postgres) with the pgvector extension.

Photos are stored in the same row as their transcript, so the retention purge
removes both in one statement and no separate object store is needed.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from .config import Settings


def _vec(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


@dataclass
class MemoryRecord:
    id: str
    device_id: str
    created_at: datetime
    transcript: str
    has_image: bool
    scene: str | None = None  # what the camera saw (Gemini vision), filled in after upload
    distance: float | None = None


@dataclass
class ItemSighting:
    name: str
    location: str
    memory_id: str
    created_at: datetime
    transcript: str
    has_image: bool
    scene: str | None = None


@dataclass
class QueryRecord:
    created_at: datetime
    question: str
    answer: str
    item: str | None


# Every query that lists records selects this instead of the image bytes.
_COLUMNS = "id, device_id, created_at, transcript, scene, image IS NOT NULL AS has_image"


class MemoryDB:
    def __init__(self, settings: Settings):
        self.dsn = settings.database_url
        self.dims = settings.embedding_dimensions
        self.pool: asyncpg.Pool | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=0, max_size=5)
        self._loop = asyncio.get_running_loop()

    async def ensure_ready(self) -> None:
        """Connect and create the schema on first use.

        Serverless hosts (Vercel) may skip the startup hook or run requests on a new
        event loop; an asyncpg pool can't cross loops, so reconnect when that happens.
        """
        loop = asyncio.get_running_loop()
        if self.pool is not None and self._loop is loop:
            return
        async with self._locks.setdefault(loop, asyncio.Lock()):
            if self.pool is not None and self._loop is loop:
                return
            await self.connect()
            await self.init_schema()

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_logs (
                    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    device_id   text        NOT NULL,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    transcript  text        NOT NULL,
                    embedding   vector({self.dims}) NOT NULL,
                    image       bytea,
                    image_type  text
                )
                """
            )
            # Tables created by the earlier object-storage version lack these columns.
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS image bytea")
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS image_type text")
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS scene text")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_created_at_idx ON memory_logs (created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_embedding_idx "
                "ON memory_logs USING hnsw (embedding vector_cosine_ops)"
            )
            # Items named in each memory; removed with the memory by the retention purge.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id         bigserial PRIMARY KEY,
                    memory_id  uuid NOT NULL REFERENCES memory_logs (id) ON DELETE CASCADE,
                    name       text NOT NULL,
                    location   text NOT NULL DEFAULT ''
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_items_memory_id_idx ON memory_items (memory_id)"
            )
            # Questions the wearer asked, for the caregiver's Insights page.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    device_id   text        NOT NULL,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    question    text        NOT NULL,
                    answer      text        NOT NULL,
                    item        text
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS query_logs_created_at_idx ON query_logs (created_at DESC)"
            )

    async def insert(
        self,
        device_id: str,
        transcript: str,
        embedding: list[float],
        image: bytes | None,
        image_type: str | None,
    ) -> MemoryRecord:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO memory_logs (device_id, transcript, embedding, image, image_type)
            VALUES ($1, $2, $3::vector, $4, $5)
            RETURNING {_COLUMNS}
            """,
            device_id,
            transcript,
            _vec(embedding),
            image,
            image_type,
        )
        return _record(row)

    async def nearest(
        self, device_id: str, embedding: list[float], lookback_hours: int, limit: int = 3
    ) -> list[MemoryRecord]:
        rows = await self.pool.fetch(
            f"""
            SELECT {_COLUMNS}, embedding <=> $2::vector AS distance
            FROM memory_logs
            WHERE device_id = $1
              AND created_at > now() - make_interval(hours => $3)
            ORDER BY embedding <=> $2::vector
            LIMIT $4
            """,
            device_id,
            _vec(embedding),
            lookback_hours,
            limit,
        )
        return [_record(r) for r in rows]

    async def recent(self, hours: int) -> list[MemoryRecord]:
        rows = await self.pool.fetch(
            f"""
            SELECT {_COLUMNS}
            FROM memory_logs
            WHERE created_at > now() - make_interval(hours => $1)
            ORDER BY created_at DESC
            """,
            hours,
        )
        return [_record(r) for r in rows]

    async def get_image(self, record_id: str) -> tuple[bytes, str] | None:
        row = await self.pool.fetchrow(
            "SELECT image, image_type FROM memory_logs WHERE id = $1::uuid AND image IS NOT NULL",
            record_id,
        )
        return (row["image"], row["image_type"] or "image/jpeg") if row else None

    async def set_scene(self, memory_id: str, scene: str, embedding: list[float]) -> None:
        """Store what the camera saw, and re-embed so search matches it too."""
        await self.pool.execute(
            "UPDATE memory_logs SET scene = $2, embedding = $3::vector WHERE id = $1::uuid",
            memory_id,
            scene,
            _vec(embedding),
        )

    async def add_items(self, memory_id: str, items: list[tuple[str, str]]) -> None:
        if items:
            await self.pool.executemany(
                "INSERT INTO memory_items (memory_id, name, location) VALUES ($1::uuid, $2, $3)",
                [(memory_id, name, location) for name, location in items],
            )

    async def item_sightings(self, hours: int) -> list[ItemSighting]:
        """Every item mention in the window, newest first."""
        rows = await self.pool.fetch(
            """
            SELECT i.name, i.location, m.id AS memory_id, m.created_at, m.transcript, m.scene,
                   m.image IS NOT NULL AS has_image
            FROM memory_items i
            JOIN memory_logs m ON m.id = i.memory_id
            WHERE m.created_at > now() - make_interval(hours => $1)
            ORDER BY m.created_at DESC
            """,
            hours,
        )
        return [
            ItemSighting(
                name=r["name"],
                location=r["location"],
                memory_id=str(r["memory_id"]),
                created_at=r["created_at"],
                transcript=r["transcript"],
                has_image=r["has_image"],
                scene=r["scene"],
            )
            for r in rows
        ]

    async def log_query(self, device_id: str, question: str, answer: str, item: str | None) -> None:
        await self.pool.execute(
            "INSERT INTO query_logs (device_id, question, answer, item) VALUES ($1, $2, $3, $4)",
            device_id,
            question,
            answer,
            item,
        )

    async def recent_queries(self, hours: int) -> list[QueryRecord]:
        rows = await self.pool.fetch(
            """
            SELECT created_at, question, answer, item
            FROM query_logs
            WHERE created_at > now() - make_interval(hours => $1)
            ORDER BY created_at DESC
            """,
            hours,
        )
        return [QueryRecord(**dict(r)) for r in rows]

    async def delete_older_than(self, hours: int) -> int:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM query_logs WHERE created_at < now() - make_interval(hours => $1)",
                hours,
            )
            # memory_items rows go with their memory via ON DELETE CASCADE.
            result = await conn.execute(
                "DELETE FROM memory_logs WHERE created_at < now() - make_interval(hours => $1)",
                hours,
            )
        return int(result.split()[-1])  # asyncpg returns e.g. "DELETE 12"


def _record(row) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["id"]),
        device_id=row["device_id"],
        created_at=row["created_at"],
        transcript=row["transcript"],
        has_image=row["has_image"],
        scene=row["scene"],
        distance=row["distance"] if "distance" in row.keys() else None,
    )
