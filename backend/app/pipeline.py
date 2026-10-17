"""The RAG pipeline end to end: ingest a memory as text chunks, retrieve for a question.

See rag.py for the data model and ranking. Every step here is text: the photo is turned
into text by the vision model and only that text is embedded and searched.
"""

import logging
from datetime import datetime, timezone

from .ai import AIServices, ItemMention
from .db import MemoryDB
from .rag import CANDIDATES_PER_SEARCH, ITEM, SPEECH, RetrievedMemory, fuse, item_chunk, vision_chunks

log = logging.getLogger(__name__)


async def ingest_speech(db: MemoryDB, ai: AIServices, memory_id: str, transcript: str) -> None:
    """Index what was said right away, so the memory is searchable before the photo is."""
    [embedding] = await ai.embed_many([transcript])
    await db.add_chunks(memory_id, [(SPEECH, transcript, embedding)])


async def ingest_image(
    db: MemoryDB, ai: AIServices, memory_id: str, transcript: str, image: bytes, image_type: str
) -> list[ItemMention]:
    """Photo -> text -> chunks -> embeddings. Each stage is best effort: a failure
    only means fewer chunks; the memory itself is already saved and searchable."""
    try:
        items = await ai.extract_items(transcript)
    except Exception:
        log.exception("Item extraction failed for memory %s", memory_id)
        items = []
    try:
        scene = await ai.describe_scene(image, image_type, transcript)
    except Exception:
        log.exception("Image-to-text failed for memory %s", memory_id)
        scene = None

    if scene:
        # The camera fills in places the wearer didn't say ("I'm putting my keys down").
        seen_at = {i.name: i.location for i in scene.items if i.location}
        for item in items:
            if not item.location:
                item.location = seen_at.get(item.name, "")
        items = items or scene.items
        chunks = vision_chunks(scene.description, [(i.name, i.location) for i in items], scene.visible_text)
    else:
        chunks = [(ITEM, item_chunk(i.name, i.location)) for i in items]

    try:
        embeddings = await ai.embed_many([content for _, content in chunks])
        await db.add_chunks(memory_id, [(kind, content, emb) for (kind, content), emb in zip(chunks, embeddings)])
    except Exception:
        log.exception("Indexing image text failed for memory %s", memory_id)
    try:
        if scene:
            await db.set_scene(memory_id, scene.description)
        await db.add_items(memory_id, [(i.name, i.location) for i in items])
    except Exception:
        log.exception("Saving scene/items failed for memory %s", memory_id)
    return items


async def retrieve(
    db: MemoryDB, ai: AIServices, question: str, device_id: str | None, hours: int, limit: int = 3
) -> list[RetrievedMemory]:
    """Hybrid search: vector + keyword candidates, fused and grouped by memory, with every
    chunk of the winning memories attached for the prompt."""
    embedding = await ai.embed(question, query=True)
    vector = await db.vector_hits(device_id, embedding, hours, CANDIDATES_PER_SEARCH)
    keyword = await db.keyword_hits(device_id, question, hours, CANDIDATES_PER_SEARCH)
    memories = fuse(vector, keyword, datetime.now(timezone.utc), limit)
    chunks = await db.chunks_for([m.memory_id for m in memories])
    for m in memories:
        m.chunks = chunks.get(m.memory_id, [])
    return memories
