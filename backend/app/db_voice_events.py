"""Voice commands as they happen, for the dashboard's Live view: what was heard, what was decided
and by which model, the photo picked, what was said back, and how long each step took.

One row per request to /memory/voice, /log and /query. Kept KEEP_DAYS (it's for watching tests and the
demo, not history; the memories themselves are in memory_logs). No audio or photos are stored here.
"""

import json
from datetime import datetime

KEEP_DAYS = 3


class VoiceEventsMixin:
    @staticmethod
    async def init_voice_events_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_events (
                id         bigserial PRIMARY KEY,
                device_id  text        NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                event      jsonb       NOT NULL
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS voice_events_device_idx ON voice_events (device_id, id DESC)"
        )

    async def add_voice_event(self, device_id: str, event: dict) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO voice_events (device_id, event) VALUES ($1, $2::jsonb)", device_id, json.dumps(event)
            )
            await conn.execute(
                f"DELETE FROM voice_events WHERE created_at < now() - interval '{KEEP_DAYS} days'"
            )

    async def voice_events(self, device_ids: list[str], after: int = 0, limit: int = 50) -> list[dict]:
        """Newest first: {"id", "device_id", "created_at", **event}, only ids above `after`."""
        rows = await self.pool.fetch(
            "SELECT id, device_id, created_at, event FROM voice_events "
            "WHERE device_id = ANY($1::text[]) AND id > $2 ORDER BY id DESC LIMIT $3",
            device_ids,
            after,
            limit,
        )
        return [_row(r) for r in rows]


def _row(r) -> dict:
    event = r["event"] if isinstance(r["event"], dict) else json.loads(r["event"])
    created: datetime = r["created_at"]
    return {**event, "id": r["id"], "device_id": r["device_id"], "created_at": created.isoformat()}
