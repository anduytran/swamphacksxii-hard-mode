"""Read-only MCP access (mixed into MemoryDB): household MCP keys and the request counter.

A key is shown once when it is made; only its SHA-256 is stored. Keys are 256-bit random,
so a plain hash is enough (no salt or slow hash needed, unlike a password).
"""

import hashlib
import secrets

KEY_PREFIX = "lsk_"


def mcp_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class McpMixin:
    pool: object  # asyncpg.Pool, from MemoryDB

    @staticmethod
    async def init_mcp_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_keys (
                id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                created_by    text NOT NULL,
                label         text NOT NULL,
                key_hash      text NOT NULL UNIQUE,
                hint          text NOT NULL,   -- first characters, so people can tell keys apart
                created_at    timestamptz NOT NULL DEFAULT now(),
                last_used_at  timestamptz,
                revoked_at    timestamptz
            )
            """
        )
        # One row per caller: a fixed one-minute window shared by every serverless instance.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_rate (
                subject       text PRIMARY KEY,
                window_start  timestamptz NOT NULL,
                count         integer NOT NULL
            )
            """
        )

    async def create_mcp_key(self, household_id: str, user_sub: str, label: str) -> tuple[str, str]:
        """Returns (key id, the key itself). The key is not stored and can't be shown again."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        key_id = await self.pool.fetchval(
            "INSERT INTO mcp_keys (household_id, created_by, label, key_hash, hint) "
            "VALUES ($1::uuid, $2, $3, $4, $5) RETURNING id::text",
            household_id, user_sub, label, mcp_key_hash(key), key[: len(KEY_PREFIX) + 4],
        )
        return key_id, key

    async def mcp_keys(self, household_id: str) -> list[dict]:
        """Active keys, never their hashes."""
        rows = await self.pool.fetch(
            "SELECT id::text AS id, label, hint, created_at, last_used_at FROM mcp_keys "
            "WHERE household_id = $1::uuid AND revoked_at IS NULL ORDER BY created_at",
            household_id,
        )
        return [dict(r) for r in rows]

    async def revoke_mcp_key(self, household_id: str, key_id: str) -> bool:
        result = await self.pool.execute(
            "UPDATE mcp_keys SET revoked_at = now() WHERE household_id = $1::uuid AND id = $2::uuid AND revoked_at IS NULL",
            household_id, key_id,
        )
        return result.endswith(" 1")

    async def mcp_key_caller(self, key_hash: str) -> tuple[str, str, str] | None:
        """(key id, household id, created_by) for an active key whose creator is still a
        member, recording the use."""
        row = await self.pool.fetchrow(
            """
            UPDATE mcp_keys k SET last_used_at = now()
            WHERE k.key_hash = $1 AND k.revoked_at IS NULL
              AND EXISTS (SELECT 1 FROM household_members m
                          WHERE m.household_id = k.household_id AND m.user_sub = k.created_by)
            RETURNING k.id::text AS id, k.household_id::text AS household_id, k.created_by
            """,
            key_hash,
        )
        return (row["id"], row["household_id"], row["created_by"]) if row else None

    async def mcp_rate_hit(self, subject: str, window_seconds: int) -> int:
        """Count one request for `subject`; returns how many it has made in the current window."""
        return await self.pool.fetchval(
            """
            INSERT INTO mcp_rate (subject, window_start, count) VALUES ($1, now(), 1)
            ON CONFLICT (subject) DO UPDATE SET
                window_start = CASE WHEN mcp_rate.window_start < now() - make_interval(secs => $2)
                                    THEN now() ELSE mcp_rate.window_start END,
                count = CASE WHEN mcp_rate.window_start < now() - make_interval(secs => $2)
                             THEN 1 ELSE mcp_rate.count + 1 END
            RETURNING count
            """,
            subject, window_seconds,
        )

