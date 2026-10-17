"""Items a household teaches the vision pipeline to recognise (mixed into MemoryDB).

A training item is a name ("keys"), other words for it ("car keys"), and photos of that
exact object, each with one box drawn around it ([x, y, w, h], 0..1, top-left origin: the
same box format as observations). The pipeline uses the names as its vocabulary and the
photos as a small dataset to fine-tune on or as visual prompts.

Unlike memories these are reference photos someone chose to add, so the 72-hour purge
leaves them alone; they go when someone deletes them, or with the household.
"""

from dataclasses import dataclass, field
from datetime import datetime

MAX_TRAINING_ITEMS = 50  # per household
MAX_SAMPLES_PER_ITEM = 30


@dataclass
class TrainingSample:
    id: str
    item_id: str
    bbox: list[float]
    width: int
    height: int
    created_at: datetime


@dataclass
class TrainingItem:
    id: str
    name: str
    aliases: list[str]
    created_at: datetime
    samples: list[TrainingSample] = field(default_factory=list)


class TrainingMixin:
    @staticmethod
    async def init_training_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_items (
                id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                household_id  uuid NOT NULL REFERENCES households (id) ON DELETE CASCADE,
                name          text NOT NULL,
                aliases       text[] NOT NULL DEFAULT '{}',
                created_at    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS training_items_name_idx ON training_items (household_id, lower(name))"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_samples (
                id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                item_id     uuid NOT NULL REFERENCES training_items (id) ON DELETE CASCADE,
                image       bytea NOT NULL,
                image_type  text NOT NULL,
                width       integer NOT NULL,
                height      integer NOT NULL,
                bbox        real[] NOT NULL,   -- [x, y, w, h], 0..1, top-left origin
                created_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS training_samples_item_idx ON training_samples (item_id)")

    async def training_items(self, household_id: str) -> list[TrainingItem]:
        items = await self.pool.fetch(
            "SELECT id, name, aliases, created_at FROM training_items WHERE household_id = $1::uuid ORDER BY lower(name)",
            household_id,
        )
        samples = await self.pool.fetch(
            """
            SELECT s.id, s.item_id, s.bbox, s.width, s.height, s.created_at
            FROM training_samples s JOIN training_items i ON i.id = s.item_id
            WHERE i.household_id = $1::uuid ORDER BY s.created_at
            """,
            household_id,
        )
        by_item = {
            str(r["id"]): TrainingItem(str(r["id"]), r["name"], list(r["aliases"]), r["created_at"]) for r in items
        }
        for s in samples:
            by_item[str(s["item_id"])].samples.append(
                TrainingSample(str(s["id"]), str(s["item_id"]), list(s["bbox"]), s["width"], s["height"], s["created_at"])
            )
        return list(by_item.values())

    async def count_training_items(self, household_id: str) -> int:
        return await self.pool.fetchval("SELECT count(*) FROM training_items WHERE household_id = $1::uuid", household_id)

    async def create_training_item(self, household_id: str, name: str, aliases: list[str]) -> TrainingItem:
        """Raises asyncpg.UniqueViolationError when the household already has that name."""
        r = await self.pool.fetchrow(
            "INSERT INTO training_items (household_id, name, aliases) VALUES ($1::uuid, $2, $3) "
            "RETURNING id, name, aliases, created_at",
            household_id,
            name,
            aliases,
        )
        return TrainingItem(str(r["id"]), r["name"], list(r["aliases"]), r["created_at"])

    async def update_training_item(self, household_id: str, item_id: str, name: str | None, aliases: list[str] | None) -> bool:
        status = await self.pool.execute(
            "UPDATE training_items SET name = coalesce($3, name), aliases = coalesce($4, aliases) "
            "WHERE id = $2::uuid AND household_id = $1::uuid",
            household_id,
            item_id,
            name,
            aliases,
        )
        return status.endswith(" 1")

    async def delete_training_item(self, household_id: str, item_id: str) -> bool:
        status = await self.pool.execute(
            "DELETE FROM training_items WHERE id = $2::uuid AND household_id = $1::uuid", household_id, item_id
        )
        return status.endswith(" 1")

    async def training_item_sample_count(self, household_id: str, item_id: str) -> int | None:
        """How many photos the item has, or None when it isn't this household's."""
        row = await self.pool.fetchrow(
            "SELECT (SELECT count(*) FROM training_samples s WHERE s.item_id = i.id) AS n "
            "FROM training_items i WHERE i.id = $2::uuid AND i.household_id = $1::uuid",
            household_id,
            item_id,
        )
        return row["n"] if row else None

    async def add_training_sample(
        self, item_id: str, image: bytes, image_type: str, width: int, height: int, bbox: list[float]
    ) -> TrainingSample:
        r = await self.pool.fetchrow(
            "INSERT INTO training_samples (item_id, image, image_type, width, height, bbox) "
            "VALUES ($1::uuid, $2, $3, $4, $5, $6) RETURNING id, item_id, bbox, width, height, created_at",
            item_id,
            image,
            image_type,
            width,
            height,
            bbox,
        )
        return TrainingSample(str(r["id"]), str(r["item_id"]), list(r["bbox"]), r["width"], r["height"], r["created_at"])

    async def delete_training_sample(self, household_id: str, sample_id: str) -> bool:
        status = await self.pool.execute(
            "DELETE FROM training_samples s USING training_items i "
            "WHERE s.id = $2::uuid AND s.item_id = i.id AND i.household_id = $1::uuid",
            household_id,
            sample_id,
        )
        return status.endswith(" 1")

    async def training_sample_image(self, sample_id: str) -> tuple[bytes, str] | None:
        row = await self.pool.fetchrow("SELECT image, image_type FROM training_samples WHERE id = $1::uuid", sample_id)
        return (bytes(row["image"]), row["image_type"]) if row else None
