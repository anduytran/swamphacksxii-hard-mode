"""Wi-Fi fingerprints (mixed into MemoryDB): each save's scan, and the household's named places (wifi_places.py).

A save's scan lives on its row (memory_logs.wifi, query_logs.wifi, item_history.wifi) and goes with it in the
retention purge. Named places (wifi_places) stay until the household deletes them.
"""

import json


def _aps(value) -> dict[str, int]:
    if isinstance(value, str):
        value = json.loads(value)
    return {k: int(v) for k, v in (value or {}).items()}


class WifiMixin:
    @staticmethod
    async def init_wifi_schema(conn) -> None:
        for table in ("memory_logs", "query_logs", "item_history"):
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS wifi jsonb")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wifi_places (
                id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                scope       text NOT NULL,
                name        text NOT NULL,
                aps         jsonb NOT NULL,
                samples     integer NOT NULL DEFAULT 1,
                created_at  timestamptz NOT NULL DEFAULT now(),
                updated_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS wifi_places_name_idx ON wifi_places (scope, lower(name))")

    async def set_scan(self, table: str, row_id: str, aps: dict[str, int]) -> None:
        assert table in ("memory_logs", "query_logs")
        await self.pool.execute(f"UPDATE {table} SET wifi = $2::jsonb WHERE id = $1::uuid", row_id, json.dumps(aps))

    async def set_sighting_scan(self, observation_id: str, aps: dict[str, int]) -> None:
        await self.pool.execute(
            "UPDATE item_history SET wifi = $2::jsonb WHERE observation_id = $1::uuid", observation_id, json.dumps(aps)
        )

    async def recent_scan(self, device_ids: list[str], minutes: float) -> dict[str, int] | None:
        """The newest scan these devices sent with a memory or question in the last `minutes`."""
        value = await self.pool.fetchval(
            """
            SELECT wifi FROM (
                SELECT wifi, created_at FROM memory_logs WHERE device_id = ANY($1::text[]) AND wifi IS NOT NULL
                    AND created_at > now() - make_interval(secs => $2)
                UNION ALL
                SELECT wifi, created_at FROM query_logs WHERE device_id = ANY($1::text[]) AND wifi IS NOT NULL
                    AND created_at > now() - make_interval(secs => $2)
            ) scans ORDER BY created_at DESC LIMIT 1
            """,
            device_ids,
            minutes * 60,
        )
        return _aps(value) if value else None

    async def wifi_places(self, scope: str) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT id::text, name, aps, samples, updated_at FROM wifi_places WHERE scope = $1 ORDER BY name", scope
        )
        return [{"id": r["id"], "name": r["name"], "aps": _aps(r["aps"]), "samples": r["samples"],
                 "updated_at": r["updated_at"]} for r in rows]

    async def save_wifi_place(self, scope: str, name: str, aps: dict[str, int], place_id: str | None = None) -> dict:
        if place_id:
            row = await self.pool.fetchrow(
                "UPDATE wifi_places SET aps = $3::jsonb, samples = samples + 1, updated_at = now() "
                "WHERE id = $1::uuid AND scope = $2 RETURNING id::text, name, samples",
                place_id, scope, json.dumps(aps),
            )
        else:
            row = await self.pool.fetchrow(
                "INSERT INTO wifi_places (scope, name, aps) VALUES ($1, $2, $3::jsonb) "
                "ON CONFLICT (scope, lower(name)) DO UPDATE SET aps = EXCLUDED.aps, samples = wifi_places.samples + 1, "
                "updated_at = now() RETURNING id::text, name, samples",
                scope, name, json.dumps(aps),
            )
        return dict(row)

    async def delete_wifi_place(self, scopes: list[str], place_id: str) -> bool:
        result = await self.pool.execute(
            "DELETE FROM wifi_places WHERE id = $1::uuid AND scope = ANY($2::text[])", place_id, scopes
        )
        return result.endswith(" 1")
