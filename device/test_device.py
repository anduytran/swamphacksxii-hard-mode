"""Network logic of the wearable client, against a fake Auth0 + API. No hardware needed.

Run from the repo root:  backend/.venv/bin/python -m pytest device
"""

import httpx
import pytest

from lastscene_device import ApiClient, Config, Spool


def make_client(handler, tmp_path=None):
    cfg = Config(
        api_url="https://api.test",
        auth0_domain="tenant.auth0.test",
        auth0_audience="https://api.lastscene",
        client_id="id",
        client_secret="secret",
        spool_dir=tmp_path or "/tmp",
    )
    return ApiClient(cfg, httpx.Client(transport=httpx.MockTransport(handler)))


class FakeServer:
    def __init__(self):
        self.tokens_issued = 0
        self.uploads = []
        self.reject_token = None  # a token the API should answer 401 to
        self.offline = False
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "tenant.auth0.test":
            self.tokens_issued += 1
            return httpx.Response(200, json={"access_token": f"tok{self.tokens_issued}", "expires_in": 86400})
        if self.offline:
            raise httpx.ConnectError("no wifi", request=request)
        if request.headers["authorization"] == f"Bearer {self.reject_token}":
            return httpx.Response(401, json={"detail": "revoked"})
        self.uploads.append(request.url.path)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "nope"})
        if request.url.path.endswith("/query"):
            return httpx.Response(200, content=b"RIFFwav", headers={"x-answer": "On%20the%20counter."})
        return httpx.Response(200, json={"id": "1", "transcript": "keys on the counter"})


def test_token_is_cached_between_calls():
    server = FakeServer()
    api = make_client(server)
    api.log_memory(b"wav", b"jpg")
    api.log_memory(b"wav", b"jpg")
    assert server.tokens_issued == 1
    assert server.uploads == ["/api/v1/memory/log"] * 2


def test_revoked_token_is_refreshed_once():
    server = FakeServer()
    api = make_client(server)
    api.token()
    server.reject_token = "tok1"
    assert api.log_memory(b"wav", b"jpg")["transcript"] == "keys on the counter"
    assert server.tokens_issued == 2


def test_ask_returns_audio_and_decoded_answer():
    api = make_client(FakeServer())
    audio, answer = api.ask(b"wav")
    assert audio == b"RIFFwav"
    assert answer == "On the counter."


def test_spool_keeps_memories_while_offline_and_uploads_later(tmp_path):
    server = FakeServer()
    api = make_client(server)
    spool = Spool(tmp_path / "spool")
    spool.add(b"wav1", b"jpg1")
    spool.add(b"wav2", b"jpg2")

    server.offline = True
    assert spool.flush(api) == 0
    assert len(spool.pending()) == 2

    server.offline = False
    assert spool.flush(api) == 2
    assert spool.pending() == []


def test_spool_drops_memories_the_server_rejects(tmp_path):
    server = FakeServer()
    server.status = 422  # e.g. no speech detected: retrying would never succeed
    spool = Spool(tmp_path / "spool")
    spool.add(b"silence", b"jpg")
    assert spool.flush(make_client(server)) == 0
    assert spool.pending() == []


def test_spool_retries_later_on_server_error(tmp_path):
    server = FakeServer()
    server.status = 503
    spool = Spool(tmp_path / "spool")
    spool.add(b"wav", b"jpg")
    assert spool.flush(make_client(server)) == 0
    assert len(spool.pending()) == 1
