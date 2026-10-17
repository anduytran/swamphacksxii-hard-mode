"""The offline stand-ins used by scripts/simulate_hardware.py (no tokens, no network)."""

import asyncio
import importlib.util
import io
import sys
import wave
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "simulate_hardware", Path(__file__).resolve().parents[1] / "scripts" / "simulate_hardware.py"
)
sim = importlib.util.module_from_spec(spec)
sys.modules["simulate_hardware"] = sim  # dataclasses look their module up while loading
spec.loader.exec_module(sim)

run = asyncio.run


def test_fake_wav_is_playable_and_carries_its_transcript():
    data = sim.fake_wav("I'm putting my keys on the counter")
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16000)
    assert sim.wav_transcript(data) == "I'm putting my keys on the counter"


def test_item_extraction_rules():
    ai = sim.SimulatedAI()
    cases = {
        "I'm putting my keys on the kitchen counter": ("keys", "kitchen counter"),
        "My wallet is going in my coat pocket": ("wallet", "coat pocket"),
        "Putting the remote on the armrest": ("tv remote", "armrest"),
        "I'm putting my keys down": ("keys", ""),
        "Where did I put my glasses?": ("reading glasses", ""),
        "Have you seen my pill box?": ("pill box", ""),
        "Where's my wallet?": ("wallet", ""),
    }
    for text, expected in cases.items():
        (item,) = run(ai.extract_items(text))
        assert (item.name, item.location) == expected, text


def test_embeddings_match_on_the_item():
    ai = sim.SimulatedAI()
    keys = run(ai.embed("I'm putting my keys on the counter"))
    wallet = run(ai.embed("My wallet is in my coat pocket"))
    question = run(ai.embed("Where are my keys?", query=True))
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert len(keys) == 1536
    assert dot(question, keys) > dot(question, wallet)


def test_photo_is_a_valid_png():
    png = sim.scene_png((140, 98, 62), (220, 180, 60), seed=1)
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) < 200_000
