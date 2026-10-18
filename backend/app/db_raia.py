"""raia frame context (mixed into MemoryDB): which items are the user's, and when each was last read.

raia_contexts holds one row per (scope, item): when its frame was last sent to raia, what came back (the JSON
context) and the "LastScene: " memory it was saved as. A new read claims the row only once RAIA_COOLDOWN_MINUTES
have passed, in one statement, so two server instances never both send the same item. Rows go in the retention
purge.
"""


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


class RaiaMixin:
    @staticmethod
    async def init_raia_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raia_contexts (
                scope        text        NOT NULL,
                item         text        NOT NULL,
                memory_id    uuid        REFERENCES memory_logs (id) ON DELETE SET NULL,
                claimed_at   timestamptz NOT NULL DEFAULT now(),
                finished_at  timestamptz,
                context      text,
                error        text,
                PRIMARY KEY (scope, item)
            )
            """
        )
        # The raia conversation that read the frame: follow-up questions about the item continue in it.
        await conn.execute("ALTER TABLE raia_contexts ADD COLUMN IF NOT EXISTS conversation_id text")

    async def logged_items(self, scope: str, device_ids: list[str]) -> set[str]:
        """The user's items: the household's Teach items and their aliases, and items named in spoken memories."""
        rows = await self.pool.fetch(
            """
            SELECT name, aliases FROM training_items WHERE household_id::text = $1
            UNION ALL
            SELECT DISTINCT i.name, '{}'::text[] FROM memory_items i JOIN memory_logs m ON m.id = i.memory_id
            WHERE m.device_id = ANY($2::text[]) AND m.source = 'manual'
            """,
            scope,
            device_ids,
        )
        return {_norm(n) for r in rows for n in [r["name"], *r["aliases"]] if n and n.strip()}

    async def claim_raia_context(self, scope: str, item: str, memory_id: str | None, cooldown_minutes: float) -> bool:
        """True if this item may be read now (never read, or last read over cooldown_minutes ago)."""
        claimed = await self.pool.fetchval(
            """
            INSERT INTO raia_contexts (scope, item, memory_id) VALUES ($1, $2, $3::uuid)
            ON CONFLICT (scope, item) DO UPDATE SET
                memory_id = EXCLUDED.memory_id, claimed_at = now(), finished_at = NULL, context = NULL, error = NULL
            WHERE raia_contexts.claimed_at < now() - make_interval(secs => $4)
            RETURNING 1
            """,
            scope,
            item,
            memory_id,
            cooldown_minutes * 60,
        )
        return claimed is not None

    async def finish_raia_context(self, scope: str, item: str, context: str | None, error: str | None,
                                  memory_id: str | None = None, conversation_id: str | None = None) -> None:
        """context: the JSON the agent returned (as text); memory_id: the "LastScene: " memory it was saved as;
        conversation_id: the raia conversation that read it."""
        await self.pool.execute(
            "UPDATE raia_contexts SET finished_at = now(), context = $3, error = $4, "
            "memory_id = COALESCE($5::uuid, memory_id), conversation_id = $6 WHERE scope = $1 AND item = $2",
            scope,
            item,
            context,
            error,
            memory_id,
            conversation_id,
        )

    async def retry_raia_context(self, scope: str, item: str, cooldown_minutes: float, retry_minutes: float) -> None:
        """After a failed read: let the item be claimed again in retry_minutes, not the whole cooldown."""
        await self.pool.execute(
            "UPDATE raia_contexts SET claimed_at = now() - make_interval(secs => $3) WHERE scope = $1 AND item = $2",
            scope,
            item,
            max(0.0, cooldown_minutes - retry_minutes) * 60,
        )

    async def raia_context(self, scope: str, item: str) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT memory_id, claimed_at, finished_at, context, error, conversation_id FROM raia_contexts "
            "WHERE scope = $1 AND item = $2",
            scope,
            item,
        )
        return dict(row) if row else None

    async def purge_raia_contexts(self, hours: int) -> int:
        result = await self.pool.execute(
            "DELETE FROM raia_contexts WHERE claimed_at < now() - make_interval(hours => $1)", hours
        )
        return int(result.split()[-1])
