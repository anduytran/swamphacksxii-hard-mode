"""Shared rate-limit counters and provider cooldowns (see limits.py).

rate_slots: one row per (bucket:window, window start), `used` counted up atomically and never
past the limit. rate_cooldowns: a provider bucket nobody should call until `until`.
"""

from datetime import datetime


class LimitsMixin:
    @staticmethod
    async def init_limits_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_slots (
                bucket       text NOT NULL,
                window_start timestamptz NOT NULL,
                used         integer NOT NULL,
                PRIMARY KEY (bucket, window_start)
            )
            """
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_cooldowns (bucket text PRIMARY KEY, until timestamptz NOT NULL)"
        )

    async def take_slots(
        self, slots: list[tuple[str, datetime, int, int]], cooldown_buckets: list[str], now: datetime
    ) -> tuple[str, str, datetime | None] | None:
        """Take `cost` from each (key, window start, limit, cost), all or none, in one transaction.
        None when taken; ("cooldown", bucket, until) or ("full", key, None) when not."""
        try:
            return await self._take_slots(slots, cooldown_buckets, now)
        except _Full as full:  # raised inside the transaction, so the slots it took are rolled back
            return "full", full.key, None

    async def _take_slots(self, slots, cooldown_buckets, now):
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT bucket, until FROM rate_cooldowns WHERE bucket = ANY($1::text[]) AND until > $2 "
                "ORDER BY until DESC LIMIT 1",
                cooldown_buckets,
                now,
            )
            if row:
                return "cooldown", row["bucket"], row["until"]
            for key, start, limit, cost in slots:
                used = await conn.fetchval(
                    """
                    INSERT INTO rate_slots (bucket, window_start, used) VALUES ($1, $2, $4)
                    ON CONFLICT (bucket, window_start) DO UPDATE SET used = rate_slots.used + $4
                    WHERE rate_slots.used + $4 <= $3
                    RETURNING used
                    """,
                    key,
                    start,
                    limit,
                    cost,
                )
                if used is None or used > limit:
                    raise _Full(key)
            return None

    async def set_cooldown(self, bucket: str, until: datetime) -> None:
        await self.pool.execute(
            """
            INSERT INTO rate_cooldowns (bucket, until) VALUES ($1, $2)
            ON CONFLICT (bucket) DO UPDATE SET until = greatest(rate_cooldowns.until, EXCLUDED.until)
            """,
            bucket,
            until,
        )

    async def slot_usage(self, keys: list[str], since: datetime) -> dict[str, int]:
        """Current counts, for /healthz: key -> used in the newest window since `since`."""
        rows = await self.pool.fetch(
            "SELECT DISTINCT ON (bucket) bucket, used FROM rate_slots WHERE bucket = ANY($1::text[]) "
            "AND window_start >= $2 ORDER BY bucket, window_start DESC",
            keys,
            since,
        )
        return {r["bucket"]: r["used"] for r in rows}

    async def prune_limits(self, before: datetime) -> None:
        await self.pool.execute("DELETE FROM rate_slots WHERE window_start < $1", before)
        await self.pool.execute("DELETE FROM rate_cooldowns WHERE until < $1", before)


class _Full(Exception):
    def __init__(self, key: str):
        self.key = key

