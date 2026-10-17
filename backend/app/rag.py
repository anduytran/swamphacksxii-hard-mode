"""LastScene's retrieval-augmented generation layer: a small custom vector store on
TigerData (pgvector for meaning, Postgres full-text search for exact words).

    ingest    photo ──vision──▶ text ┐
              speech ──STT────▶ text ┴▶ chunks ──embed──▶ memory_chunks (vector + tsvector)
    retrieve  question ──▶ vector search  ┐
                       └─▶ keyword search ┴▶ reciprocal-rank fusion + recency ▶ top memories
    generate  top memories' chunks ──▶ grounded prompt ──▶ spoken answer

Everything the assistant "knows" about a memory is text in memory_chunks; the photo is
kept only so caregivers can look at it.
"""

from dataclasses import dataclass, field
from datetime import datetime

# Chunk kinds, one text per fact so each gets its own embedding and can match alone.
SPEECH = "speech"  # what the wearer said
SCENE = "scene"  # one-sentence description of the photo
ITEM = "item"  # "keys: hallway table by the front door"
TEXT = "text"  # words readable in the photo (labels, notes, screens)

RRF_K = 60  # standard reciprocal-rank-fusion constant
RECENCY_HALF_LIFE_HOURS = 24
# A recency bonus small enough to break ties between equally relevant memories
# (the newest spot for an item wins) but never to outrank a clearly better match.
RECENCY_WEIGHT = 0.5 / RRF_K
CANDIDATES_PER_SEARCH = 20


@dataclass
class ChunkHit:
    chunk_id: int
    memory_id: str
    kind: str
    content: str
    created_at: datetime
    distance: float | None = None  # cosine distance from the question (vector search)
    vector_rank: int | None = None  # 1 = closest
    keyword_rank: int | None = None  # 1 = best full-text match
    score: float = 0.0


@dataclass
class RetrievedMemory:
    memory_id: str
    created_at: datetime
    score: float
    matched: list[ChunkHit]  # chunks that the question actually hit
    chunks: list[ChunkHit] = field(default_factory=list)  # every chunk, for the prompt

    def _first(self, kind: str) -> str | None:
        return next((c.content for c in self.chunks or self.matched if c.kind == kind), None)

    @property
    def transcript(self) -> str:
        return self._first(SPEECH) or ""

    @property
    def scene(self) -> str | None:
        return self._first(SCENE)

    def facts(self, kind: str) -> list[str]:
        return [c.content for c in self.chunks or self.matched if c.kind == kind]


def item_chunk(name: str, location: str) -> str:
    return f"{name}: {location}" if location else name


def vision_chunks(caption: str | None, items: list[tuple[str, str]], visible_text: str | None) -> list[tuple[str, str]]:
    """Turn the image-to-text output into (kind, content) chunks."""
    chunks = []
    if caption:
        chunks.append((SCENE, caption))
    chunks += [(ITEM, item_chunk(name, location)) for name, location in items]
    if visible_text:
        chunks.append((TEXT, visible_text))
    return chunks


def fuse(
    vector_hits: list[ChunkHit], keyword_hits: list[ChunkHit], now: datetime, limit: int = 3
) -> list[RetrievedMemory]:
    """Reciprocal-rank fusion of both searches, grouped by memory, with a recency bonus.

    A chunk scores 1/(K + rank) for each search it appears in. A memory scores its best
    chunk plus a quarter of its other matching chunks (several facts agreeing is a
    stronger signal), plus a bonus that halves every RECENCY_HALF_LIFE_HOURS.
    """
    by_chunk: dict[int, ChunkHit] = {}
    for rank, hit in enumerate(vector_hits, 1):
        merged = by_chunk.setdefault(hit.chunk_id, hit)
        merged.vector_rank, merged.distance = rank, hit.distance
        merged.score += 1 / (RRF_K + rank)
    for rank, hit in enumerate(keyword_hits, 1):
        merged = by_chunk.setdefault(hit.chunk_id, hit)
        merged.keyword_rank = rank
        merged.score += 1 / (RRF_K + rank)

    by_memory: dict[str, list[ChunkHit]] = {}
    for hit in by_chunk.values():
        by_memory.setdefault(hit.memory_id, []).append(hit)

    memories = []
    for memory_id, hits in by_memory.items():
        hits.sort(key=lambda h: h.score, reverse=True)
        relevance = hits[0].score + 0.25 * sum(h.score for h in hits[1:])
        age_hours = max(0.0, (now - hits[0].created_at).total_seconds() / 3600)
        recency = RECENCY_WEIGHT * 0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS)
        memories.append(RetrievedMemory(memory_id, hits[0].created_at, relevance + recency, hits))
    memories.sort(key=lambda m: m.score, reverse=True)
    return memories[:limit]
