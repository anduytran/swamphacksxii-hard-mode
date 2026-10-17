import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import unquote

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app import auth, signing
from app.ai import humanize_ago
from app.config import Settings, get_settings
from app.db import MemoryRecord
from app.main import create_app

DOMAIN = "test.auth0.local"
AUDIENCE = "https://api.memory-anchor"
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

    async def insert(self, device_id, transcript, embedding, image, image_type):
        rec = MemoryRecord(str(len(self.rows)), device_id, datetime.now(timezone.utc), transcript, image is not None)
        self.rows.append(rec)
        if image is not None:
            self.images[rec.id] = (image, image_type)
        return rec

    async def get_image(self, record_id):
        return self.images.get(record_id)

    async def nearest(self, device_id, embedding, lookback_hours, limit=3):
        return [r for r in self.rows if r.device_id == device_id][:limit]

    async def recent(self, hours):
        return sorted(self.rows, key=lambda r: r.created_at, reverse=True)


class FakeAI:
    def __init__(self):
        self.answered_with = None

    async def transcribe(self, filename, audio, content_type):
        return audio.decode()

    async def embed(self, text):
        return [0.1, 0.2]

    async def answer(self, question, memories):
        self.answered_with = (question, memories)
        return "On the kitchen counter."

    async def speak(self, text):
        return b"RIFF-fake-wav"


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
