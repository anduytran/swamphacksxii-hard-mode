from datetime import datetime, timedelta, timezone

from app.rag import ITEM, SCENE, SPEECH, TEXT, ChunkHit, fuse, vision_chunks

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)


def hit(chunk_id, memory_id, hours_ago=1, kind=SPEECH, content="x"):
    return ChunkHit(chunk_id, memory_id, kind, content, NOW - timedelta(hours=hours_ago))


def test_vision_output_becomes_one_chunk_per_fact():
    chunks = vision_chunks("Keys on the hallway table.", [("keys", "hallway table"), ("wallet", "")], "SPARE")
    assert chunks == [(SCENE, "Keys on the hallway table."), (ITEM, "keys: hallway table"), (ITEM, "wallet"), (TEXT, "SPARE")]
    assert vision_chunks(None, [], "") == []


def test_chunks_found_by_both_searches_rank_first():
    vector = [hit(1, "a"), hit(2, "b")]
    keyword = [hit(2, "b")]
    ranked = fuse(vector, keyword, NOW)
    assert [m.memory_id for m in ranked] == ["b", "a"]
    (b_hit,) = ranked[0].matched
    assert (b_hit.vector_rank, b_hit.keyword_rank) == (2, 1)


def test_several_matching_chunks_strengthen_a_memory():
    vector = [hit(1, "a"), hit(2, "b"), hit(3, "b", kind=SCENE)]
    assert fuse(vector, [], NOW)[0].memory_id == "b"


def test_recency_breaks_ties_but_not_relevance():
    tie = fuse([hit(1, "old", hours_ago=50)], [hit(2, "new", hours_ago=1)], NOW)
    assert tie[0].memory_id == "new"  # same rank in its own search: the newer memory wins
    clear = fuse([hit(1, "old", hours_ago=50)], [hit(1, "old", hours_ago=50), hit(2, "new", hours_ago=1)], NOW)
    assert clear[0].memory_id == "old"  # matched by both searches beats merely recent


def test_limit_and_empty():
    assert fuse([], [], NOW) == []
    assert len(fuse([hit(i, str(i)) for i in range(10)], [], NOW, limit=3)) == 3
