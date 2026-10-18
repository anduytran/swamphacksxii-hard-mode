"""The wake-phrase route: one clip, and the words decide between saving a memory and answering."""

import pytest

from app.intent import classify, strip_wake_phrase
from tests.test_api import CAREGIVER, DEVICE, client  # noqa: F401  (client is a fixture)

JPEG = b"\xff\xd8jpeg"


def voice(client, text, photo=True, headers=DEVICE):  # noqa: F811
    files = {"audio": ("v.wav", text.encode(), "audio/wav")}
    if photo:
        files["image"] = ("s.jpg", JPEG, "image/jpeg")
    return client.post("/api/v1/memory/voice", headers=headers, files=files)


def test_a_statement_is_saved_without_the_wake_phrase(client):  # noqa: F811
    resp = voice(client, "Hey Jarvis, I am putting my keys on the counter")
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "remember" and body["transcript"] == "I am putting my keys on the counter"
    (row,) = client.app_state.db.rows
    assert row.transcript == "I am putting my keys on the counter" and row.has_image


def test_a_question_is_answered_out_loud(client):  # noqa: F811
    voice(client, "I am putting my keys on the counter")
    resp = voice(client, "Hey Jarvis, where are my keys?", photo=False)
    assert resp.status_code == 200
    assert resp.headers["x-intent"] == "ask" and resp.headers["content-type"].startswith("audio/")
    assert len(client.app_state.db.rows) == 1  # the question isn't stored as a memory


def test_rules(client):  # noqa: F811
    assert voice(client, "", photo=True).status_code == 422  # nothing said
    assert voice(client, "my glasses are on the nightstand", photo=False).status_code == 422  # a memory needs its photo
    assert voice(client, "my glasses are on the nightstand", headers=CAREGIVER).status_code == 403  # devices only


@pytest.mark.parametrize(
    "said, intent",
    [
        ("I'm putting my keys on the counter", "remember"),
        ("Leaving my wallet by the door.", "remember"),
        ("Hey Jarvis. My keys are on the hallway table", "remember"),
        ("Keys in my coat pocket", "remember"),
        ("The pill box is in the kitchen drawer", "remember"),
        ("Hey Jarvis, where are my keys?", "ask"),
        ("where's my wallet", "ask"),
        ("Did I take my pills this morning", "ask"),
        ("I can't find my phone", "ask"),
        ("Have you seen my reading glasses", "ask"),
        ("Do you know where I put the charger", "ask"),
        ("Remind me where the umbrella is", "ask"),
        ("Last Scene, where are my keys?", "ask"),
        ("Last scene. I'm leaving my glasses on the nightstand", "remember"),
    ],
)
def test_classify(said, intent):
    assert classify(said) == intent


def test_strip_wake_phrase():
    assert strip_wake_phrase("Hey Jarvis, where are my keys?") == "where are my keys?"
    assert strip_wake_phrase("hey last scene glasses here") == "glasses here"
    assert strip_wake_phrase("Jarvis is my dog's name") == "Jarvis is my dog's name"  # only a leading wake phrase
    assert strip_wake_phrase("Last Scene, keys on the counter") == "keys on the counter"
    assert strip_wake_phrase("Lastscene where's my wallet") == "where's my wallet"
    assert strip_wake_phrase("Last seen, where are my glasses?") == "where are my glasses?"
    assert strip_wake_phrase("That was the last scene of the movie") == "That was the last scene of the movie"


# On-device speech-to-text (Moonshine/Vosk on the Pi): the wearable sends the words, not the recording.

def test_a_device_transcript_is_used_without_audio(client):  # noqa: F811
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, data={"transcript": "Last Scene, my wallet is in the car"},
                       files={"image": ("s.jpg", JPEG, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["intent"] == "remember" and resp.json()["transcript"] == "my wallet is in the car"


def test_a_device_transcript_works_for_the_buttons_too(client):  # noqa: F811
    resp = client.post("/api/v1/memory/log", headers=DEVICE, data={"transcript": "Keys on the hook"},
                       files={"image": ("s.jpg", JPEG, "image/jpeg")})
    assert resp.status_code == 200 and resp.json()["transcript"] == "Keys on the hook"
    resp = client.post("/api/v1/memory/query", headers=DEVICE, data={"transcript": "Where are my keys?"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("audio/")


def test_a_device_transcript_wins_over_audio(client):  # noqa: F811
    resp = client.post("/api/v1/memory/log", headers=DEVICE, data={"transcript": "Glasses by the sink"},
                       files={"audio": ("v.wav", b"something else", "audio/wav"), "image": ("s.jpg", JPEG, "image/jpeg")})
    assert resp.json()["transcript"] == "Glasses by the sink"


def test_neither_audio_nor_transcript_is_rejected(client):  # noqa: F811
    resp = client.post("/api/v1/memory/query", headers=DEVICE, data={"transcript": "  "})
    assert resp.status_code == 422
    resp = client.post("/api/v1/memory/log", headers=DEVICE, data={"transcript": "x" * 2001},
                       files={"image": ("s.jpg", JPEG, "image/jpeg")})
    assert resp.status_code == 413


def test_speech_to_text_gets_the_wake_phrase_and_teach_items_as_hints(client):  # noqa: F811
    from app.db_training import TrainingItem

    heard = []

    async def transcribe(filename, audio, content_type, hints=None):
        heard.append(hints)
        return "where are my swamphacks pins?"

    async def household_of_device(device_id):
        return ("h1", False)

    async def training_items(household_id):
        return [TrainingItem(id="i1", name="swamphacks pins", aliases=["pins"], samples=[], created_at=None)]

    client.app_state.ai.transcribe = transcribe
    client.app_state.db.household_of_device = household_of_device
    client.app_state.db.training_items = training_items
    client.post("/api/v1/memory/query", headers=DEVICE, files={"audio": ("q.wav", b"x", "audio/wav")})
    assert heard == [["Last Scene", "swamphacks pins", "pins"]]
    heard.clear()  # the device sent its own words: no transcription, so no hint lookup either
    client.post("/api/v1/memory/query", headers=DEVICE, data={"transcript": "where are my keys?"})
    assert heard == []
