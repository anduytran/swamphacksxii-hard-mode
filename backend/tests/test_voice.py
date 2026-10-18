"""The wake-phrase route: one clip, and the words decide between saving a memory and answering."""

import json
from urllib.parse import unquote

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
    # The wake phrase, the floor plan and its rooms (FLOOR_PLAN=home here), then the Teach items.
    assert heard[0][0] == "Last Scene" and heard[0][-2:] == ["swamphacks pins", "pins"]
    assert "Home" in heard[0] and "Kitchen" in heard[0]
    heard.clear()  # the device sent its own words: no transcription, so no hint lookup either
    client.post("/api/v1/memory/query", headers=DEVICE, data={"transcript": "where are my keys?"})
    assert heard == []


# ---------------------------------------------------------------- wake=true: confirm, words, then photos if needed

from app.ai import Command, PhotoCheck  # noqa: E402
from app.intent import has_wake_phrase  # noqa: E402


def wake_clip(client, said, photos=4, decipher=None, look=None):  # noqa: F811
    client.looked = []
    if decipher is not None:
        async def fake_decipher(words):
            client.deciphered = words
            if isinstance(decipher, Exception):
                raise decipher
            return decipher
        client.app_state.ai.decipher = fake_decipher

    async def fake_look(item, words, pics):
        client.looked.append((item, len(pics)))
        if isinstance(look, Exception):
            raise look
        return look

    client.app_state.ai.look_for = fake_look
    files = [("audio", ("v.wav", said.encode(), "audio/wav"))]
    files += [("image", (f"p{i}.jpg", b"\xff\xd8photo-%d" % i, "image/jpeg")) for i in range(photos)]
    return client.post("/api/v1/memory/voice", headers=DEVICE, data={"wake": "true"}, files=files)


def test_a_false_wake_is_dropped_before_anything_else(client):  # noqa: F811
    resp = wake_clip(client, "so the last thing I said was", decipher=Command(intent="remember", item="keys"))
    assert resp.status_code == 200 and resp.json()["intent"] == "ignored"
    assert client.app_state.db.rows == [] and not hasattr(client, "deciphered") and client.looked == []


def test_remember_sends_photos_only_after_the_words_decide(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, I'm putting my keys on the counter",
                     decipher=Command(intent="remember", item="keys"), look=PhotoCheck(visible=True, best_photo=2))
    body = resp.json()
    assert resp.status_code == 200 and body["intent"] == "remember" and body["photo"] == 2
    assert client.deciphered == "I'm putting my keys on the counter" and client.looked == [("keys", 4)]
    (row,) = client.app_state.db.rows
    assert client.app_state.db.images[row.id][0] == b"\xff\xd8photo-2"


def test_ask_says_so_when_the_item_is_in_view_now(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, where are my keys", decipher=Command(intent="ask", item="keys"),
                     look=PhotoCheck(visible=True, best_photo=1, where="on the desk next to the laptop."))
    assert resp.headers["x-intent"] == "ask" and resp.headers["x-answered-by"] == "camera"
    assert unquote(resp.headers["x-answer"]) == "Oh, I can see your keys right now, on the desk next to the laptop!"
    assert client.app_state.db.rows == []


def test_ask_answers_from_memories_when_the_item_isnt_in_view(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, where did I leave my keys", decipher=Command(intent="ask", item="keys"),
                     look=PhotoCheck(visible=False))
    assert resp.headers["x-intent"] == "ask" and resp.headers.get("x-answered-by") != "camera"
    assert client.looked == [("keys", 4)]


@pytest.mark.parametrize("intent, said", [("found", "Last Scene, I found my keys"), ("none", "Last Scene, never mind")])
def test_found_and_none_never_send_the_photos_to_a_model(client, intent, said):  # noqa: F811
    async def mark_found(scope, item, when):
        return True

    client.app_state.db.mark_found = mark_found
    wake_clip(client, said, decipher=Command(intent=intent, item="keys" if intent == "found" else ""),
              look=PhotoCheck(visible=True))
    assert client.looked == []


def test_found_marks_the_item_and_says_so(client):  # noqa: F811
    marked = []

    async def mark_found(scope, item, when):
        marked.append(item)
        return True

    client.app_state.db.mark_found = mark_found
    resp = wake_clip(client, "Last Scene, I found my Keys", decipher=Command(intent="found", item="Keys"))
    assert resp.headers["x-intent"] == "found" and resp.headers["content-type"].startswith("audio/")
    assert marked == ["keys"]


def test_not_a_request_does_nothing(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, never mind", decipher=Command(intent="none"))
    assert resp.json()["intent"] == "none" and client.app_state.db.rows == []


def test_when_the_word_model_is_busy_the_rules_decide(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, I'm putting my glasses on the table", decipher=RuntimeError("429"),
                     look=PhotoCheck(visible=True, best_photo=3))
    assert resp.json()["intent"] == "remember" and resp.json()["photo"] == 3
    resp = wake_clip(client, "Last Scene, where are my glasses?", decipher=RuntimeError("429"))
    assert resp.headers["x-intent"] == "ask"


def test_when_the_photo_check_fails_remember_keeps_the_first_photo(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, my wallet is in my bag", decipher=Command(intent="remember", item="wallet"),
                     look=TimeoutError("slow"))
    assert resp.json()["photo"] == 0


def test_an_out_of_range_photo_falls_back_to_the_first(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene, my wallet is in my bag", photos=2,
                     decipher=Command(intent="remember", item="wallet"), look=PhotoCheck(visible=True, best_photo=7))
    assert resp.json()["photo"] == 0


def test_the_old_one_photo_form_still_works(client):  # noqa: F811
    assert voice(client, "I am putting my keys on the counter").json()["intent"] == "remember"


@pytest.mark.parametrize("said, wake", [
    ("Last Scene, where are my keys?", True),
    ("Last scene: I put my pins down", True),
    ("Okay, Last Scene, did I lock the door", True),
    ("Hey Last Scene where's my phone", True),
    ("lastscene my keys", True),
    ("last seen my keys on the hook", True),
    ("I watched the last scene of the movie", False),
    ("where are my keys", False),
    ("", False),
])
def test_has_wake_phrase(said, wake):
    assert has_wake_phrase(said) is wake


# ---------------------------------------------------------------- slow command model: rules + saved item names

from app.intent import fallback_command, match_item  # noqa: E402

NAMES = {"keys": "keys", "reading glasses": "reading glasses", "glasses": "glasses",
         "swamphacks pins": "swamphacks pins", "pins": "swamphacks pins", "wallet": "wallet"}


@pytest.mark.parametrize("said, expected", [
    ("Last Scene, where did I put my key", ("ask", "keys")),
    ("I found my reading glasses", ("found", "reading glasses")),
    ("my pins are on the desk", ("remember", "swamphacks pins")),
    ("got my wallet back", ("found", "wallet")),
    ("where are the glasses", ("ask", "glasses")),
    ("did I find my keys?", ("ask", "keys")),  # a question, not "found"
    ("the weather is nice", ("remember", "")),
])
def test_fallback_command(said, expected):
    assert fallback_command(said, NAMES) == expected


def test_match_item_needs_whole_words():
    assert match_item("the keyboard is here", {"keys": "keys"}) == "" and match_item("", NAMES) == ""


def test_a_slow_model_still_marks_the_right_item_found_from_saved_names(client):  # noqa: F811
    from app.db_training import TrainingItem

    marked = []

    async def mark_found(scope, item, when):
        marked.append(item)
        return True

    async def household_of_device(device_id):
        return ("h1", False)

    async def household_device_ids(household_id):
        return ["dev1@clients"]

    async def last_seen(scope, item=None):
        return []

    async def training_items(household_id):
        return [TrainingItem(id="i1", name="swamphacks pins", aliases=["pins"], samples=[], created_at=None)]

    db = client.app_state.db
    db.mark_found, db.household_of_device, db.household_device_ids = mark_found, household_of_device, household_device_ids
    db.last_seen, db.training_items = last_seen, training_items
    resp = wake_clip(client, "Last Scene, I found my pins", decipher=TimeoutError("slow"))
    assert resp.headers["x-intent"] == "found" and marked == ["swamphacks pins"]
    assert client.looked == []  # found: still no photos sent


def test_a_model_answer_without_an_item_gets_one_from_saved_names(client):  # noqa: F811
    async def last_seen(scope, item=None):
        from types import SimpleNamespace
        return [SimpleNamespace(item="wallet")]

    client.app_state.db.last_seen = last_seen
    wake_clip(client, "Last Scene, where's the wallet", decipher=Command(intent="ask", item=""), look=PhotoCheck())
    assert client.looked == [("wallet", 4)]


def test_a_saved_item_name_beats_the_models_wording(client):  # noqa: F811
    from app.db_training import TrainingItem

    async def household_of_device(device_id):
        return ("h1", False)

    async def household_device_ids(household_id):
        return ["dev1@clients"]

    async def last_seen(scope, item=None):
        return []

    async def training_items(household_id):
        return [TrainingItem(id="i1", name="swamphacks pins", aliases=["pins"], samples=[], created_at=None)]

    db = client.app_state.db
    db.household_of_device, db.household_device_ids, db.last_seen, db.training_items = (
        household_of_device, household_device_ids, last_seen, training_items)
    wake_clip(client, "Last Scene, where did my pins go", decipher=Command(intent="ask", item="pin"), look=PhotoCheck())
    assert client.looked == [("swamphacks pins", 4)]  # the taught name, not the model's "pin"


# ---------------------------------------------------------------- the dashboard's Live view

def live(client, after=0):  # noqa: F811
    resp = client.get(f"/api/v1/dashboard/live?after={after}", headers=CAREGIVER)
    assert resp.status_code == 200
    return resp.json()["events"]


def test_live_shows_each_wake_command_with_its_decision_and_steps(client):  # noqa: F811
    wake_clip(client, "so the last thing I said was")
    wake_clip(client, "Last Scene, I'm putting my keys on the counter",
              decipher=Command(intent="remember", item="keys"), look=PhotoCheck(visible=True, best_photo=2))
    wake_clip(client, "Last Scene, where are my keys", decipher=Command(intent="ask", item="keys"),
              look=PhotoCheck(visible=True, best_photo=1, where="on the desk"))
    assert len(live(client)) == 2  # the false wake isn't recorded at all
    saved = live(client)[1]
    assert saved["kind"] == "wake" and saved["outcome"] == "saved" and saved["intent"] == "remember"
    assert saved["item"] == "keys" and saved["decided_by"] == "model" and saved["photos"] == 4
    assert saved["saved_photo"] == 2 and saved["image_url"] and saved["memory_id"]
    assert {"transcribe", "decide", "photos", "save"} <= set(saved["steps"]) and saved["total_ms"] >= 0
    (answered,) = live(client, after=1)  # only what's new
    assert answered["outcome"] == "answered" and answered["answered_by"] == "camera"
    assert answered["answer"] == "Oh, I can see your keys right now, on the desk!"


def test_live_shows_the_rules_deciding_and_errors(client):  # noqa: F811
    wake_clip(client, "Last Scene, where are my glasses?", decipher=RuntimeError("429"))
    voice(client, "my glasses are on the nightstand", photo=False)  # a memory needs its photo: 422
    error, asked = live(client)
    assert asked["decided_by"] == "rules" and asked["command_error"] == "RuntimeError" and asked["outcome"] == "answered"
    assert error["outcome"] == "error" and "photo" in error["error"]


def test_live_covers_the_buttons(client):  # noqa: F811
    from tests.test_api import log as button_log

    button_log(client)
    (event,) = live(client)
    assert event["kind"] == "button: remember" and event["outcome"] == "saved"


def test_live_is_only_for_the_household(client):  # noqa: F811
    wake_clip(client, "Last Scene, never mind", decipher=Command(intent="none"))
    client.app_state.db.household_devices = []
    assert live(client) == []
    assert client.get("/api/v1/dashboard/live", headers=DEVICE).status_code == 403


def test_a_failure_to_record_never_breaks_the_command(client):  # noqa: F811
    async def broken(device_id, event):
        raise RuntimeError("db down")

    client.app_state.db.add_voice_event = broken
    resp = wake_clip(client, "Last Scene, never mind", decipher=Command(intent="none"))
    assert resp.status_code == 200 and resp.json()["intent"] == "none"


# ---------------------------------------------------------------- only the words after "Last Scene"; defer

def test_only_the_words_after_the_wake_phrase_count(client):  # noqa: F811
    resp = wake_clip(client, "so I was telling her, Last Scene, I'm putting my keys on the counter",
                     decipher=Command(intent="remember", item="keys"), look=PhotoCheck(visible=True))
    assert resp.json()["intent"] == "remember" and client.deciphered == "I'm putting my keys on the counter"
    (row,) = client.app_state.db.rows
    assert row.transcript == "I'm putting my keys on the counter"


def test_the_wake_phrase_alone_does_nothing(client):  # noqa: F811
    resp = wake_clip(client, "Last Scene.", decipher=Command(intent="remember", item="keys"))
    assert resp.json()["intent"] == "none" and not hasattr(client, "deciphered")


def test_a_remember_replies_at_once_and_saves_the_best_photo_on_followup(client):  # noqa: F811
    async def fake_decipher(words):
        return Command(intent="remember", item="banana")

    looked = []

    async def fake_look(item, words, pics):
        looked.append((item, len(pics)))
        return PhotoCheck(visible=True, best_photo=2, where="on the table")

    client.app_state.ai.decipher, client.app_state.ai.look_for = fake_decipher, fake_look
    files = [("audio", ("v.wav", b"Last Scene, my banana is on the table", "audio/wav")),
             ("image", ("p0.jpg", b"\xff\xd8photo-0", "image/jpeg"))]
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files,
                       data={"wake": "true", "defer": "true", "client": '{"recorded_s": 3.1, "ended": "drop"}'})
    body = resp.json()
    assert body["intent"] == "remember" and body["photos_wanted"] and body["followup"]
    assert client.app_state.db.rows == [] and looked == []  # nothing saved or looked at before the reply
    event = live(client)[0]
    assert event["outcome"] == "saving" and "pending" not in event and event["device"]["recorded_s"] == 3.1

    photos = [("image", (f"p{i}.jpg", b"\xff\xd8photo-%d" % i, "image/jpeg")) for i in range(3)]
    done = client.post(f"/api/v1/memory/followup/{body['followup']}", headers=DEVICE, files=photos,
                       data={"timeline": '{"speech_end": 3.1, "reply": 4.0, "played": 4.2}'}).json()
    assert done["done"] and done["followup"] == "done" and done["saved_photo"] == 2 and looked == [("banana", 3)]
    (row,) = client.app_state.db.rows
    assert row.transcript == "my banana is on the table" and client.app_state.db.images[row.id][0] == b"\xff\xd8photo-2"
    assert client.app_state.db.chunks  # indexed too
    event = live(client)[0]
    assert event["outcome"] == "saved" and event["memory_id"] == row.id and event["image_url"]
    assert event["timeline"]["played"] == 4.2 and event["seen_where"] == "on the table"
    again = client.post(f"/api/v1/memory/followup/{body['followup']}", headers=DEVICE, files=photos)
    assert again.json() == {"done": False} and len(client.app_state.db.rows) == 1  # runs once
    assert client.post("/api/v1/memory/followup/999", headers=DEVICE).status_code == 404


def test_a_remember_whose_photo_check_fails_keeps_photo_1(client):  # noqa: F811
    async def fake_decipher(words):
        return Command(intent="remember", item="keys")

    async def fake_look(item, words, pics):
        raise TimeoutError

    client.app_state.ai.decipher, client.app_state.ai.look_for = fake_decipher, fake_look
    files = [("audio", ("v.wav", b"Last Scene, keys on the desk", "audio/wav"))]
    body = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true", "defer": "true"}).json()
    photos = [("image", (f"p{i}.jpg", b"\xff\xd8photo-%d" % i, "image/jpeg")) for i in range(3)]
    done = client.post(f"/api/v1/memory/followup/{body['followup']}", headers=DEVICE, files=photos).json()
    assert done["saved_photo"] == 0 and done["photo_error"] == "TimeoutError"
    nothing = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true", "defer": "true"}).json()
    failed = client.post(f"/api/v1/memory/followup/{nothing['followup']}", headers=DEVICE).json()
    assert failed["followup"].startswith("failed") and live(client)[0]["outcome"] == "error"


def test_a_deferred_answer_logs_its_question_on_followup(client):  # noqa: F811
    async def fake_decipher(words):
        return Command(intent="ask", item="keys")

    client.app_state.ai.decipher = fake_decipher
    voice(client, "I am putting my keys on the counter")
    files = [("audio", ("v.wav", b"Last Scene, where are my keys", "audio/wav"))]
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true", "defer": "true"})
    followup = resp.headers["x-followup"]
    assert unquote(resp.headers["x-heard"]) == "Last Scene, where are my keys"
    assert client.app_state.db.queries == []
    client.post(f"/api/v1/memory/followup/{followup}", headers=DEVICE)
    assert [q.question for q in client.app_state.db.queries] == ["where are my keys"]


def test_every_deferred_command_gets_an_id_for_the_device_timeline(client):  # noqa: F811
    async def none(words):
        return Command(intent="none")

    client.app_state.ai.decipher = none
    files = [("audio", ("v.wav", b"Last Scene, never mind", "audio/wav"))]
    body = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true", "defer": "true"}).json()
    assert body["intent"] == "none" and body["followup"]
    timeline = {"speech_end": 2.1, "sent": 2.4, "reply": 3.9, "played": 4.2, "note": "x"}
    resp = client.post(f"/api/v1/memory/followup/{body['followup']}", headers=DEVICE, data={"timeline": json.dumps(timeline)})
    assert resp.json() == {"done": False}  # no slow work for "never mind"
    assert live(client)[0]["timeline"] == {"speech_end": 2.1, "sent": 2.4, "reply": 3.9, "played": 4.2}


def test_a_sure_wake_counts_when_speech_to_text_misses_the_phrase(client):  # noqa: F811
    async def ask(words):
        return Command(intent="ask", item="keys")

    client.app_state.ai.decipher = ask
    files = [("audio", ("v.wav", b"where are my keys", "audio/wav"))]
    sure = client.post("/api/v1/memory/voice", headers=DEVICE, files=files,
                       data={"wake": "true", "client": '{"wake_score": 0.97}'})
    assert sure.headers["x-intent"] == "ask" and live(client)[0]["wake_trusted"] is True
    unsure = client.post("/api/v1/memory/voice", headers=DEVICE, files=files,
                         data={"wake": "true", "client": '{"wake_score": 0.7}'})
    assert unsure.json()["intent"] == "ignored"


def test_fillers_are_spoken_phrases_for_the_device_to_keep(client, monkeypatch):  # noqa: F811
    from app.routers import memory

    monkeypatch.setattr(memory, "FILLERS", ("Let me check.",))
    first = client.get("/api/v1/memory/fillers/0", headers=DEVICE)
    assert first.status_code == 200 and first.headers["content-type"].startswith("audio/")
    assert unquote(first.headers["x-text"]) == "Let me check."
    assert client.get("/api/v1/memory/fillers/1", headers=DEVICE).status_code == 404
    assert client.get("/api/v1/memory/fillers/0", headers=CAREGIVER).status_code == 403


def test_a_slow_answer_model_gives_way_to_the_memory_itself(client):  # noqa: F811
    import asyncio

    client.settings.answer_timeout_seconds, client.settings.template_answers = 0.2, False

    async def slow_answer(question, memories, language="en", context=None):
        await asyncio.sleep(5)
        return "too late"

    async def ask(words):
        return Command(intent="ask", item="banana")

    voice(client, "my banana is on the chair, the red one")
    client.app_state.ai.answer, client.app_state.ai.decipher = slow_answer, ask
    resp = wake_clip(client, "Last Scene, where's my banana, the red one", look=PhotoCheck())
    assert unquote(resp.headers["x-answer"]).startswith('I think you said "my banana is on the chair, the red one"')
    assert resp.headers["x-answered-by"] == "memory (model too slow)"


def test_a_slow_photo_check_never_holds_up_the_answer(client):  # noqa: F811
    import asyncio
    import time

    async def slow_look(item, words, pics):
        await asyncio.sleep(5)
        return PhotoCheck(visible=True, best_photo=0, where="on the desk")

    async def ask(words):
        return Command(intent="ask", item="keys")

    voice(client, "I am putting my keys on the counter")
    client.app_state.ai.decipher = ask
    files = [("audio", ("v.wav", b"Last Scene, where are my keys", "audio/wav")),
             ("image", ("p.jpg", b"\xff\xd8photo", "image/jpeg"))]
    client.app_state.ai.look_for = slow_look
    started = time.monotonic()
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true"})
    assert time.monotonic() - started < 2 and resp.headers["x-answered-by"] != "camera"
    assert "answered without it" in live(client)[0]["photo_error"]


def test_a_false_wake_leaves_nothing_on_the_server(client, caplog):  # noqa: F811
    resp = wake_clip(client, "my brother is acting weird today", decipher=Command(intent="remember", item="keys"))
    assert resp.json() == {"intent": "ignored"}  # not even the words come back
    assert live(client) == [] and client.app_state.db.rows == []
    assert "brother" not in caplog.text


def test_the_device_can_see_which_fillers_are_in_use(client):  # noqa: F811
    from app.ai import FILLERS

    assert client.get("/api/v1/memory/fillers", headers=DEVICE).json() == {"texts": list(FILLERS)} == {"texts": []}
    assert client.get("/api/v1/memory/fillers/0", headers=DEVICE).status_code == 404  # no fillers in use
