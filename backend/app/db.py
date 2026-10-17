"""TigerData (Timescale Postgres) with the pgvector extension."""

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
    image_key: str | None
    distance: float | None = None


class MemoryDB:
    def __init__(self, settings: Settings):
        self.dsn = settings.database_url
        self.dims = settings.embedding_dimensions
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

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
                    image_key   text
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_created_at_idx ON memory_logs (created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_embedding_idx "
                "ON memory_logs USING hnsw (embedding vector_cosine_ops)"
            )

    async def insert(
        self, device_id: str, transcript: str, embedding: list[float], image_key: str | None
    ) -> MemoryRecord:
        row = await self.pool.fetchrow(
            """
            INSERT INTO memory_logs (device_id, transcript, embedding, image_key)
            VALUES ($1, $2, $3::vector, $4)
            RETURNING id, device_id, created_at, transcript, image_key
            """,
            device_id,
            transcript,
            _vec(embedding),
            image_key,
        )
        return _record(row)

    async def nearest(
        self, device_id: str, embedding: list[float], lookback_hours: int, limit: int = 3
    ) -> list[MemoryRecord]:
        rows = await self.pool.fetch(
            """
            SELECT id, device_id, created_at, transcript, image_key,
                   embedding <=> $2::vector AS distance
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
            """
            SELECT id, device_id, created_at, transcript, image_key
            FROM memory_logs
            WHERE created_at > now() - make_interval(hours => $1)
            ORDER BY created_at DESC
            """,
            hours,
        )
        return [_record(r) for r in rows]

    async def delete_older_than(self, hours: int) -> list[str]:
        """Delete expired rows and return their image keys so the caller can purge storage."""
        rows = await self.pool.fetch(
            """
            DELETE FROM memory_logs
            WHERE created_at < now() - make_interval(hours => $1)
            RETURNING image_key
            """,
            hours,
        )
        return [r["image_key"] for r in rows if r["image_key"]]


def _record(row) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["id"]),
        device_id=row["device_id"],
        created_at=row["created_at"],
        transcript=row["transcript"],
        image_key=row["image_key"],
        distance=row.get("distance") if "distance" in row.keys() else None,
    )
