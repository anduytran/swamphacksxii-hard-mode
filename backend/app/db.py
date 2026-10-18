"""TigerData (Timescale Postgres) with the pgvector extension.

memory_logs holds one row per memory (transcript, photo, timestamps). Everything used
for retrieval lives in memory_chunks: text chunks derived from speech and from the
photo, each with an embedding and a full-text index (see rag.py). Chunks, items and the
photo all go with their memory in the retention purge.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from .config import Settings
from .db_maps import MapsNamesMixin
from .db_limits import LimitsMixin
from .db_mcp import McpMixin
from .db_observations import ObservationsMixin
from .db_places import PlacesMixin
from .db_review import ReviewMixin
from .db_training import TrainingMixin, TrainingModelsMixin
from .db_voice_events import VoiceEventsMixin
from .rag import ChunkHit


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
    source: str = "manual"  # manual (Remember button) | auto (vision pipeline observation)


@dataclass
class ItemSighting:
    name: str
    location: str
    memory_id: str
    created_at: datetime
    transcript: str
    has_image: bool
    scene: str | None = None
    room_id: str | None = None  # the floor plan room the photo showed (describe_scene), if any


@dataclass
class QueryRecord:
    created_at: datetime
    question: str
    answer: str
    item: str | None


# How long the anonymous hourly counts are kept (the memories themselves: RETENTION_HOURS).
ACTIVITY_COUNTS_DAYS = 90

# Every query that lists records selects this instead of the image bytes.
_COLUMNS = "id, device_id, created_at, transcript, scene, source, image IS NOT NULL AS has_image"


class MemoryDB(PlacesMixin, ObservationsMixin, TrainingMixin, TrainingModelsMixin, MapsNamesMixin, ReviewMixin, McpMixin, LimitsMixin,
               VoiceEventsMixin):
    def __init__(self, settings: Settings):
        self.dsn = settings.database_url
        self.dims = settings.embedding_dimensions
        self.pool: asyncpg.Pool | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._used_at = 0.0

    async def connect(self) -> None:
        # Small and short-lived, because every serverless instance has its own pool and Postgres
        # has few connection slots: a burst of requests (a page of photos) starts many instances,
        # and idle connections they keep fill the slots, so later requests fail with
        # "remaining connection slots are reserved". The pool drops connections idle for 10 s,
        # and Postgres itself closes ones idle for 60 s, which also covers instances that
        # Vercel froze between requests (their timers don't run).
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=0,
            max_size=3,
            max_inactive_connection_lifetime=10,
            server_settings={"idle_session_timeout": "60000"},
        )
        self._loop = asyncio.get_running_loop()

    async def ensure_ready(self) -> None:
        """Connect and create the schema on first use.

        Serverless hosts (Vercel) may skip the startup hook or run requests on a new
        event loop; an asyncpg pool can't cross loops, so reconnect when that happens.
        """
        loop = asyncio.get_running_loop()
        if self.pool is not None and self._loop is loop:
            # Idle longer than the pool's own 10 s (e.g. frozen by Vercel): Postgres may have closed
            # its connections, so open fresh ones rather than hand out a dead one.
            if time.time() - self._used_at > 10:
                await self.pool.expire_connections()
            self._used_at = time.time()
            return
        async with self._locks.setdefault(loop, asyncio.Lock()):
            if self.pool is not None and self._loop is loop:
                return
            if self.pool is not None:  # made on another event loop: can't be closed from here, so drop it
                self.pool.terminate()
            await self.connect()
            await self.init_schema()
            self._used_at = time.time()

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
            await conn.execute("ALTER TABLE memory_logs ADD COLUMN IF NOT EXISTS room_id text")
            # Retrieval moved to memory_chunks; the per-memory vector is no longer written.
            await conn.execute("ALTER TABLE memory_logs ALTER COLUMN embedding DROP NOT NULL")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_created_at_idx ON memory_logs (created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_logs_embedding_idx "
                "ON memory_logs USING hnsw (embedding vector_cosine_ops)"
            )
            # The vector store: text chunks per memory, searchable by meaning and by words.
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id         bigserial PRIMARY KEY,
                    memory_id  uuid NOT NULL REFERENCES memory_logs (id) ON DELETE CASCADE,
                    kind       text NOT NULL,
                    content    text NOT NULL,
                    embedding  vector({self.dims}) NOT NULL,
                    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_memory_id_idx ON memory_chunks (memory_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx "
                "ON memory_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_tsv_idx ON memory_chunks USING gin (tsv)"
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
            await self.init_places_schema(conn)
            await self.init_observations_schema(conn)
            await self.init_training_schema(conn)
            await self.init_training_models_schema(conn)
            await self.init_maps_schema(conn)
            await self.init_review_schema(conn)
            await self.init_mcp_schema(conn)
            await self.init_limits_schema(conn)
            await self.init_voice_events_schema(conn)
            # Anonymous hourly counts that outlive the 72-hour purge (see delete_older_than):
            # no words, photos or item names, just how many questions / put-downs per hour.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_hourly (
                    device_id  text        NOT NULL,
                    hour       timestamptz NOT NULL,
                    kind       text        NOT NULL,   -- asked | put_down
                    count      integer     NOT NULL,
                    PRIMARY KEY (device_id, hour, kind)
                )
                """
            )

    async def insert(
        self, device_id: str, transcript: str, image: bytes | None, image_type: str | None
    ) -> MemoryRecord:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO memory_logs (device_id, transcript, image, image_type)
            VALUES ($1, $2, $3, $4)
            RETURNING {_COLUMNS}
            """,
            device_id,
            transcript,
            image,
            image_type,
        )
        return _record(row)

    # ---------------------------------------------------------------- vector store

    async def add_chunks(self, memory_id: str, chunks: list[tuple[str, str, list[float]]]) -> None:
        """Store (kind, content, embedding) chunks for a memory."""
        if chunks:
            await self.pool.executemany(
                "INSERT INTO memory_chunks (memory_id, kind, content, embedding) "
                "VALUES ($1::uuid, $2, $3, $4::vector)",
                [(memory_id, kind, content, _vec(emb)) for kind, content, emb in chunks],
            )

    async def vector_hits(
        self, device_ids: list[str] | None, embedding: list[float], hours: int, limit: int
    ) -> list[ChunkHit]:
        """Chunks closest in meaning to the question (cosine distance)."""
        rows = await self.pool.fetch(
            """
            SELECT c.id, c.memory_id, c.kind, c.content, m.created_at,
                   c.embedding <=> $1::vector AS distance
            FROM memory_chunks c
            JOIN memory_logs m ON m.id = c.memory_id
            WHERE m.created_at > now() - make_interval(hours => $2)
              AND ($3::text[] IS NULL OR m.device_id = ANY($3::text[]))
            ORDER BY c.embedding <=> $1::vector
            LIMIT $4
            """,
            _vec(embedding),
            hours,
            device_ids,
            limit,
        )
        return [_hit(r, distance=r["distance"]) for r in rows]

    async def keyword_hits(self, device_ids: list[str] | None, text: str, hours: int, limit: int) -> list[ChunkHit]:
        """Chunks sharing words with the question (Postgres full-text, any word, stemmed)."""
        # Keyword matching only finds memories in the question's language (and stems as English):
        # a Spanish question won't keyword-match an English note. Vector search covers that
        # case, since gemini-embedding-2 is multilingual.
        rows = await self.pool.fetch(
            """
            WITH q AS (
                SELECT to_tsquery('english', coalesce(string_agg(quote_literal(w), ' | '), '')) AS query
                FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS w
            )
            SELECT c.id, c.memory_id, c.kind, c.content, m.created_at
            FROM memory_chunks c
            JOIN memory_logs m ON m.id = c.memory_id, q
            WHERE c.tsv @@ q.query
              AND m.created_at > now() - make_interval(hours => $2)
              AND ($3::text[] IS NULL OR m.device_id = ANY($3::text[]))
            ORDER BY ts_rank_cd(c.tsv, q.query) DESC, m.created_at DESC
            LIMIT $4
            """,
            text,
            hours,
            device_ids,
            limit,
        )
        return [_hit(r) for r in rows]

    async def chunks_for(self, memory_ids: list[str]) -> dict[str, list[ChunkHit]]:
        rows = await self.pool.fetch(
            """
            SELECT c.id, c.memory_id, c.kind, c.content, m.created_at
            FROM memory_chunks c JOIN memory_logs m ON m.id = c.memory_id
            WHERE c.memory_id = ANY($1::uuid[])
            ORDER BY c.id
            """,
            memory_ids,
        )
        out: dict[str, list[ChunkHit]] = {}
        for r in rows:
            out.setdefault(str(r["memory_id"]), []).append(_hit(r))
        return out

    async def recent(self, hours: int, device_ids: list[str] | None = None) -> list[MemoryRecord]:
        """device_ids=None means every device (internal use); dashboards pass the household's."""
        rows = await self.pool.fetch(
            f"""
            SELECT {_COLUMNS}
            FROM memory_logs
            WHERE created_at > now() - make_interval(hours => $1)
              AND ($2::text[] IS NULL OR device_id = ANY($2::text[]))
            ORDER BY created_at DESC
            """,
            hours,
            device_ids,
        )
        return [_record(r) for r in rows]

    async def get_image(self, record_id: str) -> tuple[bytes, str] | None:
        row = await self.pool.fetchrow(
            "SELECT image, image_type FROM memory_logs WHERE id = $1::uuid AND image IS NOT NULL",
            record_id,
        )
        return (row["image"], row["image_type"] or "image/jpeg") if row else None

    async def set_scene(self, memory_id: str, scene: str, room_id: str | None = None) -> None:
        """What the camera saw, for the dashboard (its chunk is what search uses), and
        the floor plan room it showed, for the map."""
        await self.pool.execute(
            "UPDATE memory_logs SET scene = $2, room_id = $3 WHERE id = $1::uuid", memory_id, scene, room_id
        )

    async def add_items(self, memory_id: str, items: list[tuple[str, str]]) -> None:
        if items:
            await self.pool.executemany(
                "INSERT INTO memory_items (memory_id, name, location) VALUES ($1::uuid, $2, $3)",
                [(memory_id, name, location) for name, location in items],
            )

    async def item_sightings(self, hours: int, device_ids: list[str] | None = None) -> list[ItemSighting]:
        """Every item mention in the window, newest first."""
        rows = await self.pool.fetch(
            """
            SELECT i.name, i.location, m.id AS memory_id, m.created_at, m.transcript, m.scene, m.room_id,
                   m.image IS NOT NULL AS has_image
            FROM memory_items i
            JOIN memory_logs m ON m.id = i.memory_id
            WHERE m.created_at > now() - make_interval(hours => $1)
              AND ($2::text[] IS NULL OR m.device_id = ANY($2::text[]))
            ORDER BY m.created_at DESC
            """,
            hours,
            device_ids,
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
                room_id=r["room_id"],
            )
            for r in rows
        ]

    async def log_query(self, device_id: str, question: str, answer: str, item: str | None) -> str:
        return await self.pool.fetchval(
            "INSERT INTO query_logs (device_id, question, answer, item) VALUES ($1, $2, $3, $4) RETURNING id::text",
            device_id,
            question,
            answer,
            item,
        )

    async def recent_queries(self, hours: int, device_ids: list[str] | None = None) -> list[QueryRecord]:
        rows = await self.pool.fetch(
            """
            SELECT created_at, question, answer, item
            FROM query_logs
            WHERE created_at > now() - make_interval(hours => $1)
              AND ($2::text[] IS NULL OR device_id = ANY($2::text[]))
            ORDER BY created_at DESC
            """,
            hours,
            device_ids,
        )
        return [QueryRecord(**dict(r)) for r in rows]

    async def delete_older_than(self, hours: int, keep_counts_days: int = ACTIVITY_COUNTS_DAYS) -> int:
        """The privacy purge. In the same transaction, what's about to go is first counted
        into activity_hourly (per device and hour, no content), so the "when things go
        missing" grid can look back further than the memories themselves."""
        async with self.pool.acquire() as conn, conn.transaction():
            # now() is fixed for the transaction, so the counted rows are exactly the deleted ones.
            for kind, source in (
                ("asked", "SELECT device_id, created_at FROM query_logs"),
                ("put_down", "SELECT m.device_id, m.created_at FROM memory_items i JOIN memory_logs m ON m.id = i.memory_id"),
            ):
                await conn.execute(
                    f"""
                    INSERT INTO activity_hourly (device_id, hour, kind, count)
                    SELECT device_id, date_trunc('hour', created_at), '{kind}', count(*)
                    FROM ({source}) AS rows
                    WHERE created_at < now() - make_interval(hours => $1)
                    GROUP BY 1, 2
                    ON CONFLICT (device_id, hour, kind) DO UPDATE SET count = activity_hourly.count + EXCLUDED.count
                    """,
                    hours,
                )
            await conn.execute("DELETE FROM activity_hourly WHERE hour < now() - make_interval(days => $1)", keep_counts_days)
            await conn.execute(
                "DELETE FROM query_logs WHERE created_at < now() - make_interval(hours => $1)",
                hours,
            )
            # memory_items and memory_chunks rows go with their memory via ON DELETE CASCADE.
            result = await conn.execute(
                "DELETE FROM memory_logs WHERE created_at < now() - make_interval(hours => $1)",
                hours,
            )
        return int(result.split()[-1])  # asyncpg returns e.g. "DELETE 12"


    async def missing_grid(self, device_ids: list[str], days: int) -> list[tuple[datetime, int, int]]:
        """(hour, questions asked, things put down) for every hour with any activity in the
        last `days` days: recent rows from the live tables, older ones from activity_hourly."""
        rows = await self.pool.fetch(
            """
            WITH counts AS (
                SELECT date_trunc('hour', created_at) AS hour, 'asked' AS kind, count(*) AS n
                FROM query_logs
                WHERE device_id = ANY($1::text[]) AND created_at > now() - make_interval(days => $2)
                GROUP BY 1
                UNION ALL
                SELECT date_trunc('hour', m.created_at), 'put_down', count(*)
                FROM memory_items i JOIN memory_logs m ON m.id = i.memory_id
                WHERE m.device_id = ANY($1::text[]) AND m.created_at > now() - make_interval(days => $2)
                GROUP BY 1
                UNION ALL
                SELECT hour, kind, sum(count)
                FROM activity_hourly
                WHERE device_id = ANY($1::text[]) AND hour > now() - make_interval(days => $2)
                GROUP BY 1, 2
            )
            SELECT hour,
                   coalesce(sum(n) FILTER (WHERE kind = 'asked'), 0)::int AS asked,
                   coalesce(sum(n) FILTER (WHERE kind = 'put_down'), 0)::int AS put_down
            FROM counts GROUP BY hour ORDER BY hour
            """,
            device_ids,
            days,
        )
        return [(r["hour"], r["asked"], r["put_down"]) for r in rows]


def _record(row) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["id"]),
        device_id=row["device_id"],
        created_at=row["created_at"],
        transcript=row["transcript"],
        has_image=row["has_image"],
        scene=row["scene"],
        source=row["source"] if "source" in row.keys() else "manual",
    )


def _hit(row, distance: float | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=row["id"],
        memory_id=str(row["memory_id"]),
        kind=row["kind"],
        content=row["content"],
        created_at=row["created_at"],
        distance=distance,
    )
