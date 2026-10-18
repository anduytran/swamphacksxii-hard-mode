"""Caregiver review of photos the item detector is unsure about (mixed into MemoryDB).

The Teach page's "Needs a look" queue asks about two kinds of photo, both already in
memory_logs: observation frames whose detection confidence is in an unsure band, and memory
photos whose items name a taught item but carry no box. Each answer is one row here, keyed by
(training item, memory), so a photo is never asked about twice for the same item:
    yes / fixed   the photo became a training sample (sample_id)
    not           it isn't that item
Rows go with the memory (72-hour purge) or the item. (Rows with no decision, and the
note / suggested columns, are left from the removed Gemini box suggestions and ignored.)
"""

from dataclasses import dataclass
from datetime import datetime

REVIEW_DECISIONS = ("yes", "fixed", "not")


@dataclass
class Review:
    item_id: str
    memory_id: str
    decision: str | None
    bbox: list[float] | None


class ReviewMixin:
    pool: object  # asyncpg.Pool, from MemoryDB

    @staticmethod
    async def init_review_schema(conn) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_reviews (
                item_id     uuid NOT NULL REFERENCES training_items (id) ON DELETE CASCADE,
                memory_id   uuid NOT NULL REFERENCES memory_logs (id) ON DELETE CASCADE,
                decision    text,              -- yes | fixed | not; NULL = a cached suggestion only
                bbox        real[],            -- the suggested or chosen box, [x, y, w, h] 0..1
                note        text NOT NULL DEFAULT '',
                suggested   boolean NOT NULL DEFAULT false,
                sample_id   uuid,
                updated_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (item_id, memory_id)
            )
            """
        )

    async def review_detections(self, device_ids: list[str], low: float, high: float, hours: int) -> list[dict]:
        """Detections in [low, high) on frames from these devices that still have their photo."""
        rows = await self.pool.fetch(
            """
            SELECT m.id AS memory_id, m.created_at, d.label, d.confidence, d.bbox, d.image_width, d.image_height
            FROM memory_detections d JOIN memory_logs m ON m.id = d.memory_id
            WHERE m.device_id = ANY($1::text[]) AND m.image IS NOT NULL
              AND d.confidence >= $2 AND d.confidence < $3
              AND m.created_at > now() - make_interval(hours => $4)
            ORDER BY m.created_at DESC
            """,
            device_ids,
            low,
            high,
            hours,
        )
        return [
            {
                "memory_id": str(r["memory_id"]),
                "created_at": r["created_at"],
                "label": r["label"],
                "confidence": round(r["confidence"], 4),
                "bbox": [round(c, 5) for c in r["bbox"]],
                "width": r["image_width"],
                "height": r["image_height"],
            }
            for r in rows
        ]

    async def review_memory_photos(self, device_ids: list[str], hours: int) -> list[dict]:
        """Photos whose items were named (by the wearer or Gemini), with the labels any detector boxed."""
        rows = await self.pool.fetch(
            """
            SELECT m.id AS memory_id, m.created_at,
                   array_agg(DISTINCT i.name) AS names,
                   coalesce((SELECT array_agg(DISTINCT d.label) FROM memory_detections d WHERE d.memory_id = m.id),
                            '{}') AS boxed
            FROM memory_items i JOIN memory_logs m ON m.id = i.memory_id
            WHERE m.device_id = ANY($1::text[]) AND m.image IS NOT NULL
              AND m.created_at > now() - make_interval(hours => $2)
            GROUP BY m.id, m.created_at
            ORDER BY m.created_at DESC
            """,
            device_ids,
            hours,
        )
        return [
            {"memory_id": str(r["memory_id"]), "created_at": r["created_at"], "names": list(r["names"]), "boxed": list(r["boxed"])}
            for r in rows
        ]

    async def reviews(self, household_id: str) -> dict[tuple[str, str], Review]:
        rows = await self.pool.fetch(
            """
            SELECT r.item_id, r.memory_id, r.decision, r.bbox
            FROM training_reviews r JOIN training_items t ON t.id = r.item_id
            WHERE t.household_id = $1::uuid
            """,
            household_id,
        )
        return {
            (str(r["item_id"]), str(r["memory_id"])): Review(
                str(r["item_id"]), str(r["memory_id"]), r["decision"],
                [round(c, 5) for c in r["bbox"]] if r["bbox"] is not None else None,
            )
            for r in rows
        }

    async def decide_review(
        self, item_id: str, memory_id: str, decision: str, bbox: list[float] | None, sample_id: str | None
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO training_reviews (item_id, memory_id, decision, bbox, sample_id)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::uuid)
            ON CONFLICT (item_id, memory_id) DO UPDATE SET decision = EXCLUDED.decision,
                bbox = coalesce(EXCLUDED.bbox, training_reviews.bbox), sample_id = EXCLUDED.sample_id, updated_at = now()
            """,
            item_id,
            memory_id,
            decision,
            bbox,
            sample_id,
        )

    async def household_memory_image(self, memory_id: str, device_ids: list[str]) -> tuple[bytes, str] | None:
        """A memory's photo, only when one of these (the household's) devices took it."""
        row = await self.pool.fetchrow(
            "SELECT image, image_type FROM memory_logs WHERE id = $1::uuid AND device_id = ANY($2::text[]) AND image IS NOT NULL",
            memory_id,
            device_ids,
        )
        return (bytes(row["image"]), row["image_type"] or "image/jpeg") if row else None

    async def claim_review(self, item_id: str, memory_id: str, decision: str) -> bool:
        """Record the answer only if the photo hasn't been answered yet; False if it has (e.g. a
        double tap), so a photo can't become two training samples."""
        return bool(await self.pool.fetchval(
            """
            INSERT INTO training_reviews (item_id, memory_id, decision) VALUES ($1::uuid, $2::uuid, $3)
            ON CONFLICT (item_id, memory_id) DO UPDATE SET decision = EXCLUDED.decision, updated_at = now()
            WHERE training_reviews.decision IS NULL
            RETURNING true
            """,
            item_id,
            memory_id,
            decision,
        ))

    async def release_review(self, item_id: str, memory_id: str) -> None:
        """Undo a claim whose training sample couldn't be saved, so the photo can be answered again."""
        await self.pool.execute(
            "UPDATE training_reviews SET decision = NULL WHERE item_id = $1::uuid AND memory_id = $2::uuid AND sample_id IS NULL",
            item_id,
            memory_id,
        )
