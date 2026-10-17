import re
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import unquote

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app import auth, signing
from app.ai import ItemMention, SceneText, humanize_ago
from app.config import Settings, get_settings
from app.db import ItemSighting, MemoryRecord, QueryRecord
from app.rag import ChunkHit
from app.main import create_app

DOMAIN = "test.auth0.local"
AUDIENCE = "https://api.lastscene"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def make_token(sub: str, **claims) -> str:
    now = int(time.time())
    payload = {"sub": sub, "aud": AUDIENCE, "iss": f"https://{DOMAIN}/", "iat": now, "exp": now + 600}
    payload.update(claims)
    return jwt.encode(payload, KEY, algorithm="RS256")


DEVICE = {"Authorization": f"Bearer {make_token('dev1@clients', gty='client-credentials', scope='write:memories')}"}
CAREGIVER = {"Authorization": f"Bearer {make_token('auth0|carer')}"}


class FakeDB:
    def __init__(self):
        self.rows: list[MemoryRecord] = []
        self.images: dict[str, tuple[bytes, str]] = {}
        self.items: list[ItemSighting] = []
        self.queries: list[QueryRecord] = []
        self.chunks: list[ChunkHit] = []
        self.query_devices: list[str] = []

    async def insert(self, device_id, transcript, image, image_type):
        rec = MemoryRecord(str(len(self.rows)), device_id, datetime.now(timezone.utc), transcript, image is not None)
        self.rows.append(rec)
        if image is not None:
            self.images[rec.id] = (image, image_type)
        return rec

    async def get_image(self, record_id):
        return self.images.get(record_id)

    # Vector store stand-in: "vector" search returns the device's chunks newest first;
    # keyword search returns chunks sharing a word with the question.
    async def add_chunks(self, memory_id, chunks):
        rec = next(r for r in self.rows if r.id == memory_id)
        for kind, content, _embedding in chunks:
            self.chunks.append(ChunkHit(len(self.chunks), memory_id, kind, content, rec.created_at))

    def _device_chunks(self, device_ids):
        ids = {r.id for r in self.rows if device_ids is None or r.device_id in device_ids}
        return [c for c in self.chunks if c.memory_id in ids]

    async def vector_hits(self, device_ids, embedding, hours, limit):
        found = sorted(self._device_chunks(device_ids), key=lambda c: c.chunk_id, reverse=True)[:limit]
        return [ChunkHit(c.chunk_id, c.memory_id, c.kind, c.content, c.created_at, distance=0.5) for c in found]

    async def keyword_hits(self, device_ids, text, hours, limit):
        words = set(re.findall(r"[a-z]+", text.lower())) - {"where", "are", "my", "is", "the", "did", "i"}
        found = [c for c in self._device_chunks(device_ids) if words & set(re.findall(r"[a-z]+", c.content.lower()))]
        return [ChunkHit(c.chunk_id, c.memory_id, c.kind, c.content, c.created_at) for c in found[:limit]]

    async def chunks_for(self, memory_ids):
        out = {}
        for c in self.chunks:
            if c.memory_id in memory_ids:
                out.setdefault(c.memory_id, []).append(ChunkHit(c.chunk_id, c.memory_id, c.kind, c.content, c.created_at))
        return out

    # Households: the test caregiver's household holds the test device.
    household_devices = ["dev1@clients"]

    async def device_ids_for(self, user_sub):
        return list(self.household_devices)

    async def household_of_device(self, device_id):
        return None

    async def remember_pairing_code(self, device_id, code):
        self.pairing = (device_id, code)

    def _scoped(self, rows, device_ids):
        return [r for r in rows if device_ids is None or r.device_id in device_ids]

    async def recent(self, hours, device_ids=None):
        return sorted(self._scoped(self.rows, device_ids), key=lambda r: r.created_at, reverse=True)

    async def add_items(self, memory_id, items):
        rec = next(r for r in self.rows if r.id == memory_id)
        for name, location in items:
            self.items.append(ItemSighting(name, location, rec.id, rec.created_at, rec.transcript, rec.has_image, rec.scene))

    async def set_scene(self, memory_id, scene):
        next(r for r in self.rows if r.id == memory_id).scene = scene

    async def delete_older_than(self, hours):
        self.purged_hours = hours
        return 0

    async def item_sightings(self, hours, device_ids=None):
        mine = {r.id for r in self._scoped(self.rows, device_ids)}
        return sorted([s for s in self.items if s.memory_id in mine], key=lambda s: s.created_at, reverse=True)

    async def log_query(self, device_id, question, answer, item):
        self.queries.append(QueryRecord(datetime.now(timezone.utc), question, answer, item))
        self.query_devices.append(device_id)
        return str(len(self.queries))

    async def recent_queries(self, hours, device_ids=None):
        rows = [q for q, d in zip(self.queries, self.query_devices) if device_ids is None or d in device_ids]
        return sorted(rows, key=lambda q: q.created_at, reverse=True)


class FakeAI:
    def __init__(self):
        self.answered_with = None

    async def transcribe(self, filename, audio, content_type):
        return audio.decode()

    speech_media_type = "audio/wav"
    scene = None  # set to a SceneText to simulate Gemini image-to-text

    async def embed(self, text, *, query=False):
        return [0.1, 0.2]

    async def embed_many(self, texts):
        return [[0.1, 0.2] for _ in texts]

    async def describe_scene(self, image, content_type, transcript):
        return self.scene

    async def caregiver_summary(self, facts):
        self.summary_facts = facts
        return "Keys came up most this week."

    async def answer(self, question, memories):
        self.answered_with = (question, memories)
        return "On the kitchen counter."

    async def speak(self, text):
        return b"RIFF-fake-wav"

    async def extract_items(self, text):
        # Tiny stand-in for the Claude extractor: "... my <item> on the <place>" / "where are my <item>".
        words = text.lower().rstrip("?.").split()
        if "my" not in words:
            return []
        name = words[words.index("my") + 1]
        location = " ".join(words[words.index("the") + 1 :]) if "the" in words else ""
        return [ItemMention(name, location)]


@pytest.fixture
def client(monkeypatch):
    settings = Settings(auth0_domain=DOMAIN, auth0_audience=AUDIENCE, image_url_secret="test-secret", _env_file=None)
    signing = SimpleNamespace(key=KEY.public_key())
    monkeypatch.setattr(
        auth, "_jwks_client", lambda domain: SimpleNamespace(get_signing_key_from_jwt=lambda t: signing)
    )
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db, app.state.ai = FakeDB(), FakeAI()
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def log(client, text="I am putting my keys on the counter", headers=DEVICE):
    return client.post(
        "/api/v1/memory/log",
        headers=headers,
        files={"audio": ("a.wav", text.encode(), "audio/wav"), "image": ("i.jpg", b"\xff\xd8jpeg", "image/jpeg")},
    )


def test_log_stores_image_and_record(client):
    resp = log(client)
    assert resp.status_code == 200
    assert resp.json()["transcript"] == "I am putting my keys on the counter"
    (row,) = client.app_state.db.rows
    assert row.device_id == "dev1@clients"
    assert client.app_state.db.images[row.id] == (b"\xff\xd8jpeg", "image/jpeg")


def test_log_does_not_trust_image_content_type(client):
    client.post(
        "/api/v1/memory/log",
        headers=DEVICE,
        files={"audio": ("a.wav", b"hello", "audio/wav"), "image": ("x.html", b"<script>", "text/html")},
    )
    (row,) = client.app_state.db.rows
    assert client.app_state.db.images[row.id][1] == "image/jpeg"


def test_log_requires_device_token(client):
    assert log(client, headers={}).status_code == 401
    assert log(client, headers=CAREGIVER).status_code == 403
    no_scope = {"Authorization": f"Bearer {make_token('dev1@clients', gty='client-credentials')}"}
    assert log(client, headers=no_scope).status_code == 403


def test_rejects_bad_signature_and_audience(client):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode({"sub": "x@clients", "aud": AUDIENCE, "iss": f"https://{DOMAIN}/"}, other_key, "RS256")
    assert log(client, headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    wrong_aud = make_token("dev1@clients", gty="client-credentials", scope="write:memories", aud="other")
    assert log(client, headers={"Authorization": f"Bearer {wrong_aud}"}).status_code == 401


def test_query_returns_audio(client):
    log(client)
    resp = client.post(
        "/api/v1/memory/query", headers=DEVICE, files={"audio": ("q.wav", b"Where are my keys?", "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFF-fake-wav"
    assert unquote(resp.headers["x-answer"]) == "On the kitchen counter."
    question, memories = client.app_state.ai.answered_with
    assert question == "Where are my keys?" and len(memories) == 1


def test_history_for_caregiver(client):
    log(client)
    assert client.get("/api/v1/dashboard/history", headers=DEVICE).status_code == 403
    resp = client.get("/api/v1/dashboard/history", headers=CAREGIVER)
    assert resp.status_code == 200
    (entry,) = resp.json()
    assert entry["transcript"] == "I am putting my keys on the counter"

    image = client.get(entry["image_url"])  # no bearer token: the signature is the credential
    assert image.status_code == 200
    assert image.content == b"\xff\xd8jpeg"
    assert image.headers["content-type"] == "image/jpeg"


def test_image_link_rejects_bad_or_expired_signature(client):
    log(client)
    future = int(time.time()) + 60
    assert client.get(f"/api/v1/images/0?exp={future}&sig=deadbeef").status_code == 403
    expired = signing.image_url("test-secret", "0", -10)
    assert client.get(expired).status_code == 403
    other_record = signing.image_url("test-secret", "0", 60).replace("/images/0?", "/images/1?")
    assert client.get(other_record).status_code == 403


def test_humanize_ago():
    now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert humanize_ago(now - timedelta(seconds=10), now) == "just now"
    assert humanize_ago(now - timedelta(minutes=5), now) == "5 minutes ago"
    assert humanize_ago(now - timedelta(hours=2, minutes=10), now) == "about 2 hours ago"


def ask(client, text):
    return client.post("/api/v1/memory/query", headers=DEVICE, files={"audio": ("q.wav", text.encode(), "audio/wav")})


def test_items_page_shows_latest_spot_and_history(client):
    log(client, "I am putting my keys on the counter")
    log(client, "I am putting my keys on the hallway table")
    ask(client, "Where are my keys?")

    assert client.get("/api/v1/dashboard/items", headers=DEVICE).status_code == 403
    body = client.get("/api/v1/dashboard/items", headers=CAREGIVER).json()
    (keys,) = body["items"]
    assert keys["name"] == "keys"
    assert keys["location"] == "hallway table"
    assert keys["mentions"] == 2 and keys["times_asked"] == 1
    assert [p["location"] for p in keys["previous"]] == ["counter"]
    assert client.get(keys["image_url"]).status_code == 200


def test_insights_flags_questions_about_items_never_logged(client):
    log(client, "I am putting my keys on the counter")
    ask(client, "Where are my keys?")
    ask(client, "Where are my glasses?")
    ask(client, "Where are my glasses?")

    data = client.get("/api/v1/dashboard/insights", headers=CAREGIVER).json()
    assert data["memories_logged"] == 1
    assert data["questions_asked"] == 3
    assert data["items_tracked"] == 1
    assert data["unanswered_count"] == 2
    assert data["most_asked"][0] == {"name": "glasses", "count": 2}
    assert {q["item"]: q["answerable"] for q in data["recent_questions"]} == {"keys": True, "glasses": False}
    assert sorted(a["kind"] for a in data["activity"]) == ["memory", "question", "question", "question"]


def test_background_extraction_failure_does_not_break_logging(client):
    async def boom(text):
        raise RuntimeError("Claude is down")

    client.app_state.ai.extract_items = boom
    assert log(client).status_code == 200
    assert ask(client, "Where are my keys?").status_code == 200
    assert client.app_state.db.items == []
    assert client.app_state.db.queries[0].item is None  # question still logged, just untagged


def test_photo_description_fills_in_unsaid_location(client):
    client.app_state.ai.scene = SceneText(
        "Keys on a wooden table by the front door.",
        [ItemMention("keys", "table by the front door")],
        "",
    )
    log(client, "I am putting down my keys")  # no place said
    data = client.get("/api/v1/dashboard/items", headers=CAREGIVER).json()
    (keys,) = data["items"]
    assert keys["location"] == "table by the front door"
    assert keys["scene"] == "Keys on a wooden table by the front door."
    (entry,) = client.get("/api/v1/dashboard/history", headers=CAREGIVER).json()
    assert entry["scene"] == "Keys on a wooden table by the front door."


def test_enrichment_can_run_before_responding(client):
    settings = client.app.dependency_overrides[get_settings]()
    settings.enrich_in_background = False  # serverless mode
    log(client, "I am putting my keys on the counter")
    ask(client, "Where are my keys?")
    assert [i.name for i in client.app_state.db.items] == ["keys"]
    assert client.app_state.db.queries[0].item == "keys"


def test_caregiver_summary(client):
    assert client.get("/api/v1/dashboard/summary", headers=CAREGIVER).json() == {"summary": None}  # no data yet
    log(client, "I am putting my keys on the counter")
    ask(client, "Where are my glasses?")
    assert client.get("/api/v1/dashboard/summary", headers=DEVICE).status_code == 403
    body = client.get("/api/v1/dashboard/summary", headers=CAREGIVER).json()
    assert body["summary"] == "Keys came up most this week."
    assert "never logged: glasses" in client.app_state.ai.summary_facts


def test_cron_purge_requires_secret(client):
    settings = client.app.dependency_overrides[get_settings]()
    assert client.get("/api/v1/admin/purge").status_code == 401  # disabled without a secret
    settings.cron_secret = "s3cret"
    assert client.get("/api/v1/admin/purge", headers={"Authorization": "Bearer nope"}).status_code == 401
    resp = client.get("/api/v1/admin/purge", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200 and client.app_state.db.purged_hours == 72


def test_gemini_pcm_is_wrapped_as_playable_wav():
    import io
    import wave

    from app.gemini import pcm_to_wav

    wav = pcm_to_wav(b"\x00\x01" * 2400)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 2400)


def test_startup_survives_missing_settings_and_healthz_reports_them(monkeypatch):
    import app.main as main

    bare = Settings(_env_file=None)  # nothing configured, like a fresh Vercel project
    monkeypatch.setattr(main, "get_settings", lambda: bare)
    app = main.create_app()  # with lifespan
    with TestClient(app) as c:  # lifespan runs: must not raise
        body = c.get("/healthz").json()
    assert body["ok"] is False
    assert {"DATABASE_URL", "IMAGE_URL_SECRET", "GEMINI_API_KEY"} <= set(body["missing_settings"])
    assert "OPENAI_API_KEY" not in body["missing_settings"]  # not used with Gemini defaults


def test_memory_is_stored_as_text_chunks(client):
    client.app_state.ai.scene = SceneText(
        "Keys on a wooden table by the front door, next to a blue bowl.",
        [ItemMention("keys", "table by the front door")],
        "SPARE KEYS",
    )
    log(client, "I am putting down my keys")
    kinds = {c.kind: c.content for c in client.app_state.db.chunks}
    assert kinds == {
        "speech": "I am putting down my keys",
        "scene": "Keys on a wooden table by the front door, next to a blue bowl.",
        "item": "keys: table by the front door",
        "text": "SPARE KEYS",
    }


def test_question_is_answered_from_retrieved_chunks(client):
    log(client, "I am putting my keys on the counter")
    log(client, "I am putting my wallet on the shelf")
    ask(client, "Where are my keys?")
    question, memories = client.app_state.ai.answered_with
    assert memories[0].transcript == "I am putting my keys on the counter"  # keyword match wins
    assert {c.kind for c in memories[0].chunks} == {"speech", "item"}


def test_search_endpoint_explains_matches(client):
    log(client, "I am putting my keys on the counter")
    assert client.get("/api/v1/dashboard/search?q=keys", headers=DEVICE).status_code == 403
    body = client.get("/api/v1/dashboard/search?q=where%20are%20my%20keys", headers=CAREGIVER).json()
    (top,) = body["results"]
    matched = [c for c in top["chunks"] if c["matched"]]
    assert matched and all(c["vector_rank"] or c["keyword_rank"] for c in matched)
    assert client.get(top["image_url"]).status_code == 200


def test_home_map_places_items_in_rooms_and_traces_moves(client):
    log(client, "I am putting my keys on the kitchen counter")
    log(client, "I am putting my keys on the hallway table")
    log(client, "I am putting my wallet on the moon")  # no known room
    ask(client, "Where are my keys?")
    ask(client, "Where are my keys?")

    assert client.get("/api/v1/dashboard/map", headers=DEVICE).status_code == 403
    data = client.get("/api/v1/dashboard/map", headers=CAREGIVER).json()
    items = {i["name"]: i for i in data["items"]}
    assert items["keys"]["room"] == "hallway" and items["keys"]["times_asked"] == 2
    assert items["wallet"]["room"] == "unknown"
    rooms = {r["id"]: r for r in data["rooms"]}
    assert rooms["hallway"]["lost"] == 2 and rooms["kitchen"]["putdowns"] == 1
    (trip,) = data["trips"]  # only keys moved between rooms
    assert trip["name"] == "keys" and trip["rooms"] == ["Kitchen", "Hallway"]
    assert len(trip["path"]) == len(trip["timestamps"]) > 2
    assert trip["timestamps"] == sorted(trip["timestamps"])


def test_dashboard_only_shows_the_callers_household(client):
    log(client, "I am putting my keys on the counter")
    client.app_state.db.household_devices = []  # caregiver belongs to no household with this device
    assert client.get("/api/v1/dashboard/history", headers=CAREGIVER).json() == []
    assert client.get("/api/v1/dashboard/items", headers=CAREGIVER).json()["items"] == []
    assert client.get("/api/v1/dashboard/search?q=keys", headers=CAREGIVER).json()["results"] == []


def test_device_pairing_code_is_remembered(client):
    client.post(
        "/api/v1/memory/log",
        headers={**DEVICE, "X-Pairing-Code": "ABCD-1234"},
        files={"audio": ("a.wav", b"I am putting my keys down", "audio/wav"), "image": ("i.jpg", b"\xff\xd8", "image/jpeg")},
    )
    assert client.app_state.db.pairing == ("dev1@clients", "ABCD-1234")
