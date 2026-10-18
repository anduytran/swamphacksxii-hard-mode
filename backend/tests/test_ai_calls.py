"""How many Gemini calls each step makes, and with what, counted on a fake google-genai client."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from google.genai import types

from app.ai import DIDNT_CATCH, AIServices, SceneDescription, template_answer
from app.config import Settings
from app.gemini import GeminiAI, least_thinking
from app.pipeline import ingest_image, ingest_speech
from app.rag import ChunkHit, RetrievedMemory

from .test_api import FakeDB


class FakeModels:
    def __init__(self):
        self.generated: list[dict] = []
        self.embedded: list[dict] = []
        self.parsed = None

    async def generate_content(self, *, model, contents, config=None):
        self.generated.append({"model": model, "contents": contents, "config": config})
        if config is not None and config.response_modalities == ["AUDIO"]:
            part = SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x00" * 10))
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        return SimpleNamespace(text="On the counter.", parsed=self.parsed)

    async def embed_content(self, *, model, contents, config=None):
        self.embedded.append({"model": model, "contents": contents, "config": config})
        n = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * config.output_dimensionality) for _ in range(n)])


def make_ai(**overrides) -> tuple[AIServices, FakeModels]:
    settings = Settings(gemini_api_key="test", embedding_dimensions=8, _env_file=None, **overrides)
    gemini = GeminiAI(settings)
    models = FakeModels()
    gemini.client = SimpleNamespace(aio=SimpleNamespace(models=models))
    ai = AIServices(settings)
    ai.gemini = gemini
    return ai, models


def texts(call) -> list[str]:
    return [c.parts[0].text for c in call["contents"]]


def test_a_memory_with_a_photo_costs_one_model_call_and_one_embedding_call():
    ai, models = make_ai()
    models.parsed = SceneDescription(
        description="Keys on a wooden counter.", items=[{"name": "keys", "location": "kitchen counter"}], visible_text="EXIT"
    )
    db = FakeDB()
    record = asyncio.run(db.insert("dev1", "I'm putting my keys down", b"jpeg", "image/jpeg"))

    items = asyncio.run(ingest_image(db, ai, record.id, "I'm putting my keys down", b"jpeg", "image/jpeg"))

    assert [(i.name, i.location) for i in items] == [("keys", "kitchen counter")]
    (scene_call,) = models.generated  # describe_scene only; no separate extract_items
    assert any(isinstance(c, types.Part) for c in scene_call["contents"])
    (embed_call,) = models.embedded  # all three chunks in one request, one vector each
    assert texts(embed_call) == [
        "title: none | text: Keys on a wooden counter.",
        "title: none | text: keys: kitchen counter",
        "title: none | text: EXIT",
    ]
    assert len(db.chunks) == 3


def test_without_vision_the_items_come_from_the_words():
    ai, models = make_ai(vision_provider="none")
    models.parsed = SimpleNamespace(items=[SimpleNamespace(name="keys", location="counter")])
    db = FakeDB()
    record = asyncio.run(db.insert("dev1", "my keys on the counter", b"jpeg", "image/jpeg"))
    asyncio.run(ingest_image(db, ai, record.id, "my keys on the counter", b"jpeg", "image/jpeg"))
    (extract_call,) = models.generated
    assert extract_call["contents"] == ["my keys on the counter"]


def test_embeddings_keep_their_prefixes_and_dimensions():
    ai, models = make_ai()
    [vector] = asyncio.run(ai.embed_many(["I put my keys down"]))
    question = asyncio.run(ai.embed("Where are my keys?", query=True))
    assert len(vector) == len(question) == 8
    assert texts(models.embedded[0]) == ["title: none | text: I put my keys down"]
    assert texts(models.embedded[1]) == ["task: search result | query: Where are my keys?"]
    assert all(c["config"].output_dimensionality == 8 for c in models.embedded)


def test_speech_ingest_is_one_embedding_call():
    ai, models = make_ai()
    db = FakeDB()
    record = asyncio.run(db.insert("dev1", "keys on the counter", None, "image/jpeg"))
    asyncio.run(ingest_speech(db, ai, record.id, "keys on the counter"))
    assert len(models.embedded) == 1 and not models.generated


def test_answer_and_extraction_think_as_little_as_possible_but_the_photo_does_not():
    ai, models = make_ai(gemini_model="gemini-3.8-flash")
    models.parsed = SimpleNamespace(items=[])
    asyncio.run(ai.answer("Where are my keys?", []))
    asyncio.run(ai.extract_items("Where are my keys?"))
    models.parsed = SceneDescription(description="", items=[], visible_text="")
    asyncio.run(ai.describe_scene(b"jpeg", "image/jpeg", "keys"))
    levels = [c["config"].thinking_config and c["config"].thinking_config.thinking_level for c in models.generated]
    assert levels == [types.ThinkingLevel.MINIMAL, types.ThinkingLevel.MINIMAL, None]


@pytest.mark.parametrize(
    "model, level, budget",
    [
        ("gemini-3.8-flash", types.ThinkingLevel.MINIMAL, None),
        ("gemini-3-pro-preview", types.ThinkingLevel.LOW, None),
        ("gemini-2.5-flash", None, 0),
        ("gemini-2.5-pro", None, 128),
    ],
)
def test_least_thinking(model, level, budget):
    config = least_thinking(model)
    assert (config.thinking_level, config.thinking_budget) == (level, budget)


@pytest.mark.parametrize("model", ["gemini-2.5-flash-lite", "gemini-2.0-flash", "something-else"])
def test_models_that_dont_think_are_left_alone(model):
    assert least_thinking(model) is None


def test_fixed_phrases_are_spoken_once():
    ai, models = make_ai()
    first = asyncio.run(ai.speak(DIDNT_CATCH))
    assert asyncio.run(ai.speak(DIDNT_CATCH)) == first
    asyncio.run(ai.speak("Your keys are on the counter."))
    asyncio.run(ai.speak("Your keys are on the counter."))
    assert len(models.generated) == 3  # once for the fixed phrase, twice for the answer


# ------------------------------------------------------------------ template answers

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def memory(minutes_ago: int, *items: str, score: float = 0.03) -> RetrievedMemory:
    at = NOW - timedelta(minutes=minutes_ago)
    chunks = [ChunkHit(i, f"m{minutes_ago}", "item", content, at) for i, content in enumerate(items)]
    return RetrievedMemory(f"m{minutes_ago}", at, score, chunks[:1], chunks)


def test_template_answer_from_the_one_clear_memory():
    memories = [memory(20, "keys: kitchen counter"), memory(90, "wallet: shelf")]
    assert template_answer("Where are my keys?", memories, NOW) == (
        "Your keys are on the kitchen counter, 20 minutes ago."
    )


@pytest.mark.parametrize(
    "item, said",
    [
        ("wallet: my coat pocket", "Your wallet is in your coat pocket, about 2 hours ago."),
        ("reading glasses: under the sofa", "Your reading glasses are under the sofa, about 2 hours ago."),
        ("phone: kitchen drawer", "Your phone is in the kitchen drawer, about 2 hours ago."),
    ],
)
def test_template_answer_wording(item, said):
    name = item.split(":")[0].split()[-1]
    assert template_answer(f"Where is my {name}?", [memory(120, item)], NOW) == said


@pytest.mark.parametrize(
    "question, memories",
    [
        ("Where are my keys?", []),
        ("Did I take my keys?", [memory(20, "keys: counter")]),
        ("Where are my keys?", [memory(20, "keys")]),  # no place
        ("Where are my glasses?", [memory(20, "keys: counter")]),  # not what was asked
        # A newer memory of the keys ranked lower: which one is right is the model's call.
        ("Where are my keys?", [memory(60, "keys: counter"), memory(5, "keys: hallway table")]),
        ("Where are my keys and wallet?", [memory(20, "keys: counter"), memory(30, "wallet: shelf")]),
    ],
)
def test_no_template_answer_when_unsure(question, memories):
    assert template_answer(question, memories, NOW) is None


def test_the_scene_call_asks_for_a_room_of_the_active_floor_plan():
    from app.ai import scene_schema
    from app.home_map import floor_plan

    plan = floor_plan("home")
    enum = scene_schema(plan).model_json_schema()["properties"]["room"]["enum"]
    assert enum == [r.id for r in plan.rooms] + ["unknown"]
    assert "grand_ballroom" in scene_schema(floor_plan("reitz_union")).model_json_schema()["properties"]["room"]["enum"]

    ai, models = make_ai(floor_plan="home")
    schema = scene_schema(plan)
    models.parsed = schema(description="Keys on a counter.", items=[], visible_text="", room="kitchen")
    db = FakeDB()
    record = asyncio.run(db.insert("dev1", "my keys", b"jpeg", "image/jpeg"))
    asyncio.run(ingest_image(db, ai, record.id, "my keys", b"jpeg", "image/jpeg"))

    (scene_call,) = models.generated  # still one call
    assert scene_call["config"].response_schema is schema
    assert "- kitchen: Kitchen (kitchen, counter" in scene_call["config"].system_instruction
    assert db.room_ids[record.id] == "kitchen"

    models.parsed = schema(description="Too dark.", items=[], visible_text="", room="unknown")
    assert asyncio.run(ai.describe_scene(b"jpeg", "image/jpeg", "keys")).room is None


def test_without_vision_no_room_is_stored():
    ai, models = make_ai(vision_provider="none")
    models.parsed = SimpleNamespace(items=[])
    db = FakeDB()
    record = asyncio.run(db.insert("dev1", "my keys", b"jpeg", "image/jpeg"))
    asyncio.run(ingest_image(db, ai, record.id, "my keys", b"jpeg", "image/jpeg"))
    assert record.id not in db.room_ids
