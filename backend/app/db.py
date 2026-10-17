"""TigerData (Timescale Postgres) with the pgvector extension.

Photos are stored in the same row as their transcript, so the retention purge
removes both in one statement and no separate object store is needed.
"""

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
    distance: float | None = None


# Every query that lists records selects this instead of the image bytes.
_COLUMNS = "id, device_id, created_at, transcript, image IS NOT NULL AS has_image"


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
                    image       bytea,
                    image_type  text
                )
                """
            )
            # Tables created by the earlier object-storage version lack these columns.
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS image bytea")
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS image_type text")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_created_at_idx ON memory_logs (created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_embedding_idx "
                "ON memory_logs USING hnsw (embedding vector_cosine_ops)"
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

    async def delete_older_than(self, hours: int) -> int:
        result = await self.pool.execute(
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
        distance=row["distance"] if "distance" in row.keys() else None,
    )
