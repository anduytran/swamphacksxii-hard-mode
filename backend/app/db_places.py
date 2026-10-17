"""Households, devices, places and public statistics (mixed into MemoryDB).

Privacy model
  * A household is the wearer plus the caregivers they share with. Members see their
    household's devices' memories; nobody else does.
  * Places are GPS (Wi-Fi-located) clusters per household. They start private; a
    member can mark one public (a shop, a park). Only public places, in households
    that opted in, feed public_lost_stats.
  * public_lost_stats holds no transcripts, photos, device ids or exact positions:
    item names and counts per ~100 m cell, per day, per one-way household hash. The
    public map only shows cells with at least PUBLIC_MIN_HOUSEHOLDS households.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime

PUBLIC_MIN_HOUSEHOLDS = 3
PUBLIC_CELL_DEGREES = 0.001  # ~110 m north-south
PUBLIC_WINDOW_DAYS = 90


@dataclass
class Household:
    id: str
    name: str
    share_public: bool
    role: str  # the requesting user's role: owner | member


@dataclass
class Place:
    id: str
    household_id: str
    name: str | None
    kind: str  # private | public
    lat: float
    lng: float
    radius_m: float
    visits: int = 0


def household_hash(household_id: str) -> str:
    # One-way and salted per deployment would be stronger; this keeps households
    # distinct for counting without storing their ids in the public table.
    return hashlib.sha256(f"lastscene-public:{household_id}".encode()).hexdigest()[:16]


def pairing_hash(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode()).hexdigest()


def cell_of(lat: float, lng: float) -> tuple[float, float]:
    """Snap to the public grid (cell centre), so exact positions are never published."""
    def snap(v: float) -> float:
        return round((v // PUBLIC_CELL_DEGREES) * PUBLIC_CELL_DEGREES + PUBLIC_CELL_DEGREES / 2, 6)

    return snap(lat), snap(lng)


class PlacesMixin:
    pool: object  # asyncpg.Pool, from MemoryDB

    @staticmethod
    async def init_places_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS households (
                id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name          text NOT NULL,
                share_public  boolean NOT NULL DEFAULT false,
                created_by    text NOT NULL,
                created_at    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS household_members (
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                user_sub      text NOT NULL,
                role          text NOT NULL DEFAULT 'member',
                display_name  text,
                joined_at     timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (household_id, user_sub)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS household_devices (
                device_id     text PRIMARY KEY,
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                label         text,
                claimed_at    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # Trust on first use: the first pairing code a device sends is the one to claim it with.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS device_pairing (
                device_id   text PRIMARY KEY,
                code_hash   text NOT NULL,
                first_seen  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS household_invites (
                code          text PRIMARY KEY,
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                created_by    text NOT NULL,
                expires_at    timestamptz NOT NULL,
                used_by       text
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS places (
                id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                name          text,
                kind          text NOT NULL DEFAULT 'private',
                lat           double precision NOT NULL,
                lng           double precision NOT NULL,
                radius_m      real NOT NULL DEFAULT 75,
                created_at    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for table in ("memory_logs", "query_logs"):
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS lat double precision")
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS lng double precision")
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS accuracy_m real")
            await conn.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS place_id uuid REFERENCES places (id) ON DELETE SET NULL"
            )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_lost_stats (
                cell_lat         double precision NOT NULL,
                cell_lng         double precision NOT NULL,
                item             text NOT NULL,
                kind             text NOT NULL,  -- left: put down here | asked: asked about here
                household_hash   text NOT NULL,
                day              date NOT NULL,
                count            integer NOT NULL DEFAULT 0,
                PRIMARY KEY (cell_lat, cell_lng, item, kind, household_hash, day)
            )
            """
        )

        # Google Geolocation spend guard: answers cached by access-point set, calls counted per month.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS geolocation_cache (
                key         text PRIMARY KEY,  -- sha256 of the strongest BSSIDs, never the BSSIDs
                lat         double precision NOT NULL,
                lng         double precision NOT NULL,
                accuracy_m  real,
                cached_at   timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS geolocation_usage (
                month  text PRIMARY KEY,  -- YYYY-MM, UTC
                calls  integer NOT NULL DEFAULT 0
            )
            """
        )

    # ---------------------------------------------------------------- geolocation spend guard

    async def cached_position(self, key: str, max_age_days: int) -> tuple[float, float, float | None] | None:
        row = await self.pool.fetchrow(
            "SELECT lat, lng, accuracy_m FROM geolocation_cache WHERE key = $1 AND cached_at > now() - make_interval(days => $2)",
            key,
            max_age_days,
        )
        return (row["lat"], row["lng"], row["accuracy_m"]) if row else None

    async def cache_position(self, key: str, lat: float, lng: float, accuracy_m: float | None) -> None:
        await self.pool.execute(
            "INSERT INTO geolocation_cache (key, lat, lng, accuracy_m) VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (key) DO UPDATE SET lat = EXCLUDED.lat, lng = EXCLUDED.lng, accuracy_m = EXCLUDED.accuracy_m, cached_at = now()",
            key, lat, lng, accuracy_m,
        )

    async def reserve_geolocation_call(self, month: str, limit: int) -> bool:
        """Count one call against the month's budget; False (nothing counted) once it is spent.
        Atomic, so concurrent requests on different serverless instances can't overshoot."""
        if limit <= 0:
            return False
        calls = await self.pool.fetchval(
            "INSERT INTO geolocation_usage (month, calls) VALUES ($1, 1) "
            "ON CONFLICT (month) DO UPDATE SET calls = geolocation_usage.calls + 1 "
            "WHERE geolocation_usage.calls < $2 RETURNING calls",
            month,
            limit,
        )
        return calls is not None

    async def geolocation_calls(self, month: str) -> int:
        return await self.pool.fetchval("SELECT calls FROM geolocation_usage WHERE month = $1", month) or 0

    async def purge_geolocation_cache(self, max_age_days: int) -> int:
        result = await self.pool.execute(
            "DELETE FROM geolocation_cache WHERE cached_at <= now() - make_interval(days => $1)", max_age_days
        )
        return int(result.split()[-1])

    # ---------------------------------------------------------------- households

    async def households_for(self, user_sub: str) -> list[Household]:
        rows = await self.pool.fetch(
            """
            SELECT h.id, h.name, h.share_public, m.role
            FROM households h JOIN household_members m ON m.household_id = h.id
            WHERE m.user_sub = $1 ORDER BY h.created_at
            """,
            user_sub,
        )
        return [Household(str(r["id"]), r["name"], r["share_public"], r["role"]) for r in rows]

    async def device_ids_for(self, user_sub: str) -> list[str]:
        rows = await self.pool.fetch(
            """
            SELECT d.device_id FROM household_devices d
            JOIN household_members m ON m.household_id = d.household_id
            WHERE m.user_sub = $1
            """,
            user_sub,
        )
        return [r["device_id"] for r in rows]

    async def create_household(self, user_sub: str, name: str, display_name: str | None) -> str:
        async with self.pool.acquire() as conn, conn.transaction():
            household_id = await conn.fetchval(
                "INSERT INTO households (name, created_by) VALUES ($1, $2) RETURNING id", name, user_sub
            )
            await conn.execute(
                "INSERT INTO household_members (household_id, user_sub, role, display_name) VALUES ($1, $2, 'owner', $3)",
                household_id,
                user_sub,
                display_name,
            )
        return str(household_id)

    async def member_role(self, household_id: str, user_sub: str) -> str | None:
        return await self.pool.fetchval(
            "SELECT role FROM household_members WHERE household_id = $1::uuid AND user_sub = $2",
            household_id,
            user_sub,
        )

    async def household_detail(self, household_id: str) -> dict:
        members = await self.pool.fetch(
            "SELECT user_sub, role, display_name, joined_at FROM household_members WHERE household_id = $1::uuid ORDER BY joined_at",
            household_id,
        )
        devices = await self.pool.fetch(
            """
            SELECT d.device_id, d.label, d.claimed_at,
                   (SELECT max(created_at) FROM memory_logs m WHERE m.device_id = d.device_id) AS last_seen
            FROM household_devices d WHERE d.household_id = $1::uuid ORDER BY d.claimed_at
            """,
            household_id,
        )
        return {
            "members": [dict(r) for r in members],
            "devices": [dict(r) for r in devices],
        }

    async def set_share_public(self, household_id: str, share: bool) -> None:
        await self.pool.execute("UPDATE households SET share_public = $2 WHERE id = $1::uuid", household_id, share)

    async def create_invite(self, household_id: str, user_sub: str, hours: int = 72) -> str:
        code = "-".join(secrets.token_hex(2).upper() for _ in range(3))
        await self.pool.execute(
            "INSERT INTO household_invites (code, household_id, created_by, expires_at) "
            "VALUES ($1, $2::uuid, $3, now() + make_interval(hours => $4))",
            code,
            household_id,
            user_sub,
            hours,
        )
        return code

    async def accept_invite(self, code: str, user_sub: str, display_name: str | None) -> str | None:
        async with self.pool.acquire() as conn, conn.transaction():
            household_id = await conn.fetchval(
                "UPDATE household_invites SET used_by = $2 "
                "WHERE code = $1 AND used_by IS NULL AND expires_at > now() RETURNING household_id",
                code.strip().upper(),
                user_sub,
            )
            if household_id is None:
                return None
            await conn.execute(
                "INSERT INTO household_members (household_id, user_sub, role, display_name) "
                "VALUES ($1, $2, 'member', $3) ON CONFLICT DO NOTHING",
                household_id,
                user_sub,
                display_name,
            )
        return str(household_id)

    # ---------------------------------------------------------------- devices

    async def remember_pairing_code(self, device_id: str, code: str) -> None:
        await self.pool.execute(
            "INSERT INTO device_pairing (device_id, code_hash) VALUES ($1, $2) ON CONFLICT (device_id) DO NOTHING",
            device_id,
            pairing_hash(code),
        )

    async def claim_device(self, household_id: str, device_id: str, code: str, label: str | None) -> str:
        """Returns 'ok', 'bad_code' (unknown device or wrong code) or 'taken'."""
        stored = await self.pool.fetchval("SELECT code_hash FROM device_pairing WHERE device_id = $1", device_id)
        if stored is None or not secrets.compare_digest(stored, pairing_hash(code)):
            return "bad_code"
        owner = await self.pool.fetchval("SELECT household_id FROM household_devices WHERE device_id = $1", device_id)
        if owner is not None and str(owner) != household_id:
            return "taken"
        await self.pool.execute(
            "INSERT INTO household_devices (device_id, household_id, label) VALUES ($1, $2::uuid, $3) "
            "ON CONFLICT (device_id) DO UPDATE SET label = EXCLUDED.label",
            device_id,
            household_id,
            label,
        )
        return "ok"

    async def household_of_device(self, device_id: str) -> tuple[str, bool] | None:
        row = await self.pool.fetchrow(
            "SELECT h.id, h.share_public FROM household_devices d JOIN households h ON h.id = d.household_id "
            "WHERE d.device_id = $1",
            device_id,
        )
        return (str(row["id"]), row["share_public"]) if row else None

    # ---------------------------------------------------------------- places

    async def places_for(self, household_id: str) -> list[Place]:
        rows = await self.pool.fetch(
            """
            SELECT p.*, (SELECT count(*) FROM memory_logs m WHERE m.place_id = p.id) AS visits
            FROM places p WHERE p.household_id = $1::uuid ORDER BY p.created_at
            """,
            household_id,
        )
        return [_place(r) for r in rows]

    async def create_place(self, household_id: str, lat: float, lng: float, radius_m: float) -> Place:
        row = await self.pool.fetchrow(
            "INSERT INTO places (household_id, lat, lng, radius_m) VALUES ($1::uuid, $2, $3, $4) RETURNING *, 0 AS visits",
            household_id,
            lat,
            lng,
            radius_m,
        )
        return _place(row)

    async def update_place(self, household_id: str, place_id: str, name: str | None, kind: str | None) -> bool:
        result = await self.pool.execute(
            "UPDATE places SET name = coalesce($3, name), kind = coalesce($4, kind) "
            "WHERE id = $2::uuid AND household_id = $1::uuid",
            household_id,
            place_id,
            name,
            kind,
        )
        return result.endswith("1")

    async def set_location(self, table: str, row_id: str, lat: float, lng: float, accuracy_m: float | None, place_id: str | None) -> None:
        assert table in ("memory_logs", "query_logs")
        await self.pool.execute(
            f"UPDATE {table} SET lat = $2, lng = $3, accuracy_m = $4, place_id = $5::uuid WHERE id = $1::uuid",
            row_id,
            lat,
            lng,
            accuracy_m,
            place_id,
        )

    async def unplaced_memories(self, device_ids: list[str]) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT id, lat, lng, accuracy_m FROM memory_logs "
            "WHERE device_id = ANY($1::text[]) AND lat IS NOT NULL AND place_id IS NULL ORDER BY created_at",
            device_ids,
        )
        return [dict(r) | {"id": str(r["id"])} for r in rows]

    async def located_memories(self, device_ids: list[str], hours: int) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT m.id, m.created_at, m.transcript, m.lat, m.lng, m.accuracy_m, m.place_id,
                   coalesce(string_agg(i.name, ', '), '') AS items
            FROM memory_logs m LEFT JOIN memory_items i ON i.memory_id = m.id
            WHERE m.device_id = ANY($1::text[]) AND m.lat IS NOT NULL
              AND m.created_at > now() - make_interval(hours => $2)
            GROUP BY m.id ORDER BY m.created_at
            """,
            device_ids,
            hours,
        )
        return [dict(r) | {"id": str(r["id"]), "place_id": str(r["place_id"]) if r["place_id"] else None} for r in rows]

    # ---------------------------------------------------------------- public stats

    async def add_public_stats(self, household_id: str, lat: float, lng: float, kind: str, items: list[str], day: datetime) -> None:
        cell_lat, cell_lng = cell_of(lat, lng)
        await self.pool.executemany(
            """
            INSERT INTO public_lost_stats (cell_lat, cell_lng, item, kind, household_hash, day, count)
            VALUES ($1, $2, $3, $4, $5, $6, 1)
            ON CONFLICT (cell_lat, cell_lng, item, kind, household_hash, day)
            DO UPDATE SET count = public_lost_stats.count + 1
            """,
            [(cell_lat, cell_lng, item, kind, household_hash(household_id), day.date()) for item in items],
        )

    async def public_cells(self) -> list[dict]:
        """Only cells where at least PUBLIC_MIN_HOUSEHOLDS distinct households contributed."""
        rows = await self.pool.fetch(
            """
            WITH recent AS (
                SELECT * FROM public_lost_stats WHERE day > current_date - $1::int
            ), eligible AS (
                SELECT cell_lat, cell_lng FROM recent GROUP BY cell_lat, cell_lng
                HAVING count(DISTINCT household_hash) >= $2
            )
            SELECT r.cell_lat, r.cell_lng, r.item,
                   sum(r.count) FILTER (WHERE r.kind = 'left') AS left_here,
                   sum(r.count) FILTER (WHERE r.kind = 'asked') AS asked_here
            FROM recent r JOIN eligible e USING (cell_lat, cell_lng)
            GROUP BY r.cell_lat, r.cell_lng, r.item
            """,
            PUBLIC_WINDOW_DAYS,
            PUBLIC_MIN_HOUSEHOLDS,
        )
        return [dict(r) for r in rows]


def _place(r) -> Place:
    return Place(
        id=str(r["id"]),
        household_id=str(r["household_id"]),
        name=r["name"],
        kind=r["kind"],
        lat=r["lat"],
        lng=r["lng"],
        radius_m=r["radius_m"],
        visits=r["visits"],
    )
