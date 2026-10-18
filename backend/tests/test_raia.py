"""raia frame context (app/raia.py): when a frame is sent, the requests from the swagger, the JSON context, and the
"LastScene: " memory it's saved as. From observations and from wake commands' photo checks."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from app.ai import Command, PhotoCheck
from app.config import Settings
from app.observations import Observation
from app.raia import Raia, chunks_for, context_from_photos, extract_contexts, parse_context, prompt
from tests.test_api import DEVICE, client  # noqa: F401  (client is a fixture)

SETTINGS = Settings(_env_file=None, raia_agent_key="agent-key", raia_org_key="org-key")
CONTEXT = {"visible": True, "where": "on the desk by the laptop", "surface": "wooden desk",
           "nearby": ["laptop", "blue mug"], "room": "office", "visible_text": "SwampHacks",
           "summary": "The pin is on the wooden desk next to the laptop."}


def run(coro):
    return asyncio.run(coro)


def obs(*objects, scene="hallway table"):
    return Observation.model_validate({
        "observation_id": str(uuid.uuid4()),
        "captured_at": "2026-10-18T04:12:00Z",
        "source": "auto",
        "pipeline": {"name": "lastscene-vision", "version": "0.3.0"},
        "image_size": {"width": 1280, "height": 720},
        "scene": {"id": "hall", "label": scene} if scene else None,
        "objects": [{"label": label, "confidence": conf, "bbox": [0.4, 0.5, 0.1, 0.1], "event": "placed"}
                    for label, conf in objects],
    })


class Record:
    def __init__(self, id):
        self.id = id


class FakeDB:
    def __init__(self, logged=("keys", "wallet", "pin")):
        self.logged = set(logged)
        self.now = datetime(2026, 10, 18, 4, 12, tzinfo=timezone.utc)
        self.claims: dict[tuple[str, str], datetime] = {}
        self.finished: dict[tuple[str, str], tuple] = {}
        self.memories: list[dict] = []
        self.chunks: list[tuple[str, str, str]] = []

    async def household_of_device(self, device_id):
        return ("house-1", False)

    async def household_device_ids(self, household_id):
        return ["cam@clients"]

    async def logged_items(self, scope, device_ids):
        assert scope == "house-1" and device_ids == ["cam@clients"]
        return self.logged

    async def claim_raia_context(self, scope, item, memory_id, cooldown_minutes):
        last = self.claims.get((scope, item))
        if last and last > self.now - timedelta(minutes=cooldown_minutes):
            return False
        self.claims[(scope, item)] = self.now
        return True

    async def finish_raia_context(self, scope, item, context, error, memory_id=None):
        self.finished[(scope, item)] = (json.loads(context) if context else None, error, memory_id)

    async def insert(self, device_id, transcript, image, image_type):
        self.memories.append({"id": f"raia-{len(self.memories)}", "device": device_id, "transcript": transcript,
                              "image": image, "scene": None, "items": []})
        return Record(self.memories[-1]["id"])

    def _memory(self, memory_id):
        return next(m for m in self.memories if m["id"] == memory_id)

    async def set_scene(self, memory_id, scene, room_id=None):
        self._memory(memory_id)["scene"] = scene

    async def add_items(self, memory_id, items):
        self._memory(memory_id)["items"] += items

    async def add_chunks(self, memory_id, chunks):
        self.chunks += [(memory_id, kind, text) for kind, text, _ in chunks]


class FakeAI:
    async def embed_many(self, texts):
        return [[1.0] for _ in texts]


class Api:
    """raia's external API as far as we use it, plus the signed upload URL."""

    def __init__(self, reply=None, user_exists=False, fail=None):
        self.reply = json.dumps(CONTEXT) if reply is None else reply
        self.user_exists, self.fail = user_exists, fail
        self.calls: list[tuple[str, str]] = []
        self.uploaded: bytes | None = None
        self.messages: list[dict] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.path))
        if req.url.host == "storage.example":
            assert req.headers["content-type"] == "image/jpeg"
            self.uploaded = req.content
            return httpx.Response(200)
        assert req.url.host == "api.raia2.com" and req.headers["agent-secret-key"] == "agent-key"
        assert "organization-secret-key" not in req.headers
        path = req.url.path.removeprefix("/external")
        if self.fail == path:
            return httpx.Response(500, json={"status": 500, "message": "Internal Server Error"})
        if path == "/users/search":
            assert req.url.params["fkId"].startswith("lastscene-")
            if self.user_exists:
                return httpx.Response(200, json={"user": {"id": "user-9"}, "conversationIds": []})
            return httpx.Response(404, json={"status": 404, "message": "User not found"})
        body = json.loads(req.content or b"{}")
        if path == "/users":
            assert body["fkId"].startswith("lastscene-") and body["firstName"] and body["lastName"]
            return httpx.Response(201, json={"id": "user-1", "firstName": "LastScene", "lastName": "Household"})
        if path == "/conversations":
            assert body["conversationUserId"] in {"user-1", "user-9"}
            return httpx.Response(201, json={"id": "conv-1"})
        if path == "/conversations/conv-1/message-files/upload-url":
            assert body["contentType"] == "image/jpeg" and body["fileName"].startswith("frame-")
            return httpx.Response(200, json={"signedUrl": "https://storage.example/put?sig=1",
                                             "storagePath": "conversations/conv-1/frame.jpg"})
        if path == "/conversations/conv-1/messages":
            self.messages.append(body)
            return httpx.Response(201, json={"id": "m-2", "message": self.reply, "conversationId": "conv-1",
                                             "senderRole": "assistant"})
        raise AssertionError(f"unexpected {req.method} {req.url}")


def raia(api: Api, settings=SETTINGS) -> Raia:
    return Raia(settings, http=httpx.AsyncClient(transport=httpx.MockTransport(api)))


def extract(db, api, observation, settings=SETTINGS, client=None):  # noqa: F811
    return run(extract_contexts(db, FakeAI(), client or raia(api), settings, "cam@clients", "mem-1",
                                observation, b"\xff\xd8jpeg", "image/jpeg"))


def from_photos(db, api, view, item="pin", photos=None, client=None):  # noqa: F811
    photos = photos or [(b"\xff\xd8one", "image/jpeg"), (b"\xff\xd8two", "image/jpeg")]
    return run(context_from_photos(db, FakeAI(), client or raia(api), SETTINGS, "cam@clients", item, view,
                                   photos, "ask"))


# ---------------------------------------------------------------- the JSON context and the saved memory

def test_an_observed_item_is_read_and_saved_as_a_lastscene_memory():
    db, api = FakeDB(), Api()
    assert extract(db, api, obs(("keys", 0.31))) == ["keys"]
    assert api.calls == [
        ("GET", "/external/users/search"),
        ("POST", "/external/users"),
        ("POST", "/external/conversations"),
        ("POST", "/external/conversations/conv-1/message-files/upload-url"),
        ("PUT", "/put"),
        ("POST", "/external/conversations/conv-1/messages"),
    ]
    assert api.uploaded == b"\xff\xd8jpeg"
    [message] = api.messages
    assert message["files"] == [{"fileName": "frame-20261018T041200.jpg", "storagePath": "conversations/conv-1/frame.jpg"}]
    assert "keys" in message["message"] and "JSON" in message["message"] and "x=0.40" in message["message"]

    [memory] = db.memories
    assert memory["transcript"] == "LastScene: The pin is on the wooden desk next to the laptop."
    assert memory["device"] == "cam@clients" and memory["image"] == b"\xff\xd8jpeg"
    assert memory["items"] == [("keys", "on the desk by the laptop")]  # template answers read this
    assert memory["scene"].startswith("LastScene: The pin is on the wooden desk")
    kinds = {kind: text for _, kind, text in db.chunks}
    assert kinds["item"] == "keys: on the desk by the laptop"
    assert "Next to: laptop, blue mug." in kinds["scene"] and "Room: office." in kinds["scene"]
    assert kinds["text"] == "SwampHacks"
    context, error, memory_id = db.finished[("house-1", "keys")]
    assert context == CONTEXT and error is None and memory_id == memory["id"]


def test_parse_context_normalises_json_and_falls_back_to_the_sentence():
    fenced = "```json\n" + json.dumps({**CONTEXT, "nearby": ["a", "b", "c", "d", "e", 7]}) + "\n```"
    assert parse_context(fenced)["nearby"] == ["a", "b", "c", "d"]
    assert parse_context('{"visible": false}') is None
    assert parse_context("Not visible.") is None and parse_context("") is None
    plain = parse_context("**On the desk**, next to a mug.")
    assert plain["summary"] == "On the desk, next to a mug." and plain["where"] == ""
    assert chunks_for("pin", plain)[0] == ("item", "pin: On the desk, next to a mug.")
    only_surface = parse_context('{"visible": true, "surface": "chair"}')
    assert only_surface["summary"] == "on the chair"


def test_prompt_asks_for_the_json_fields():
    text = prompt("pin", None, None)
    for field in ("visible", "where", "surface", "nearby", "room", "visible_text", "summary"):
        assert f'"{field}"' in text
    assert "box" not in text and "camera's location" not in text
    assert "kitchen" in prompt("pin", [0.1, 0.1, 0.2, 0.2], "kitchen")


def test_not_visible_saves_nothing():
    db, api = FakeDB(), Api(reply='{"visible": false}')
    assert extract(db, api, obs(("keys", 0.8))) == []
    assert db.finished[("house-1", "keys")] == (None, "not visible", None)
    assert db.memories == [] and db.chunks == []


# ---------------------------------------------------------------- when an observation is sent

def test_only_detections_above_the_threshold_are_sent():
    db, api = FakeDB(), Api()
    assert extract(db, api, obs(("keys", 0.30), ("wallet", 0.2))) == []
    assert api.calls == [] and db.claims == {}


def test_items_the_user_never_logged_are_not_sent():
    db, api = FakeDB(logged={"wallet"}), Api()
    assert extract(db, api, obs(("keys", 0.9))) == []
    assert api.calls == []


def test_an_item_is_read_at_most_once_per_cooldown():
    db, api = FakeDB(), Api(user_exists=True)
    client = raia(api)  # noqa: F811
    assert extract(db, api, obs(("keys", 0.8)), client=client) == ["keys"]
    db.now += timedelta(minutes=29)
    assert extract(db, api, obs(("keys", 0.8)), client=client) == []
    db.now += timedelta(minutes=2)
    assert extract(db, api, obs(("keys", 0.8)), client=client) == ["keys"]
    # The household's user is looked up once per process; one conversation per frame.
    assert api.calls.count(("GET", "/external/users/search")) == 1
    assert ("POST", "/external/users") not in api.calls
    assert api.calls.count(("POST", "/external/conversations")) == 2


def test_each_item_has_its_own_cooldown():
    db, api = FakeDB(), Api()
    assert extract(db, api, obs(("keys", 0.4), ("keys", 0.9), ("wallet", 0.5))) == ["keys", "wallet"]
    assert len(db.memories) == 2


def test_a_failure_is_recorded_and_never_raises():
    db, api = FakeDB(), Api(fail="/conversations/conv-1/messages")
    assert extract(db, api, obs(("keys", 0.8))) == []
    context, error, _ = db.finished[("house-1", "keys")]
    assert context is None and "500" in error
    assert db.memories == []


def test_off_without_the_agent_key():
    db, api = FakeDB(), Api()
    off = Settings(_env_file=None)
    assert extract(db, api, obs(("keys", 0.8)), settings=off, client=raia(api, off)) == []
    assert api.calls == []


# ---------------------------------------------------------------- wake commands' photo checks

def test_an_item_the_photo_check_sees_sends_the_best_photo():
    db, api = FakeDB(), Api()
    view = PhotoCheck(visible=True, best_photo=1, where="on the desk", confidence=0.8)
    memory_id = from_photos(db, api, view, item="Pin")
    assert memory_id == "raia-0" and api.uploaded == b"\xff\xd8two"
    assert "box" not in api.messages[0]["message"]
    assert db.memories[0]["transcript"].startswith("LastScene: ")
    assert db.memories[0]["items"] == [("pin", "on the desk by the laptop")]


def test_the_photo_check_must_see_the_item_above_the_threshold():
    db, api = FakeDB(), Api()
    assert from_photos(db, api, PhotoCheck(visible=True, confidence=0.30)) is None
    assert from_photos(db, api, PhotoCheck(visible=False, confidence=0.9)) is None
    assert from_photos(db, api, None) is None
    assert from_photos(db, api, PhotoCheck(visible=True, confidence=0.9), item="") is None
    assert from_photos(FakeDB(logged={"keys"}), api, PhotoCheck(visible=True, confidence=0.9)) is None
    assert api.calls == []


def test_a_photo_check_still_running_is_awaited():
    db, api = FakeDB(), Api()

    async def go():
        async def check():
            await asyncio.sleep(0.01)
            return PhotoCheck(visible=True, confidence=0.7)
        task = asyncio.create_task(check())
        return await context_from_photos(db, FakeAI(), raia(api), SETTINGS, "cam@clients", "pin", task,
                                         [(b"\xff\xd8one", "image/jpeg")], "ask")

    assert run(go()) == "raia-0"


def test_an_ask_with_the_pin_in_view_sends_it_after_the_answer(client):  # noqa: F811
    """Through POST /api/v1/memory/voice: the reply is spoken, then the photo goes to raia."""
    api = Api()
    db = client.app_state.db
    fake = FakeDB()
    for name in ("logged_items", "claim_raia_context", "finish_raia_context"):
        setattr(db, name, getattr(fake, name))
    db.household_of_device = fake.household_of_device
    db.household_device_ids = fake.household_device_ids
    client.app_state.raia = raia(api)
    client.settings.raia_min_confidence = 0.30

    async def ask(words):
        return Command(intent="ask", item="pin")

    async def look(item, words, photos):
        return PhotoCheck(visible=True, best_photo=0, where="on the desk", confidence=0.9)

    client.app_state.ai.decipher, client.app_state.ai.look_for = ask, look
    files = [("audio", ("v.wav", b"Last Scene, where is my pin", "audio/wav")),
             ("image", ("p.jpg", b"\xff\xd8pin-photo", "image/jpeg"))]
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true"})
    assert resp.status_code == 200 and resp.headers["x-intent"] == "ask"
    assert api.uploaded == b"\xff\xd8pin-photo"
    assert fake.finished[("house-1", "pin")][2] is not None
    saved = [r for r in db.rows if r.transcript.startswith("LastScene: ")]
    assert len(saved) == 1


# ---------------------------------------------------------------- logs

def test_each_prompt_is_logged_with_why_others_were_skipped(caplog):
    caplog.set_level("INFO", logger="lastscene.raia")
    db, api = FakeDB(logged={"keys"}), Api()
    client = raia(api)  # noqa: F811
    extract(db, api, obs(("keys", 0.8), ("umbrella", 0.6)), client=client)
    extract(db, api, obs(("keys", 0.8)), client=client)
    from_photos(db, api, PhotoCheck(visible=True, confidence=0.2), item="keys", client=client)
    lines = [r.getMessage() for r in caplog.records if r.name == "lastscene.raia"]
    assert sum(line.startswith("RAIA prompt sent: keys (0.80, observation mem-1)") for line in lines) == 1
    assert any(line.startswith("RAIA conversation conv-1") for line in lines)
    assert any(line.startswith("RAIA reply: keys (observation mem-1)") and '"surface": "wooden desk"' in line
               for line in lines)
    assert any(line.startswith("RAIA skip umbrella (0.60, observation): not one of the user's items") for line in lines)
    assert any(line.startswith("RAIA skip keys (0.80, observation mem-1): read in the last 30 min") for line in lines)
    assert any(line.startswith("RAIA skip keys (ask): photo check says visible=True, confidence 0.20") for line in lines)


def test_a_failed_prompt_is_logged_as_a_warning(caplog):
    caplog.set_level("INFO", logger="lastscene.raia")
    extract(FakeDB(), Api(fail="/conversations"), obs(("keys", 0.8)))
    failed = [r for r in caplog.records if r.getMessage().startswith("RAIA prompt failed: keys")]
    assert failed and failed[0].levelname == "WARNING" and "500" in failed[0].getMessage()
