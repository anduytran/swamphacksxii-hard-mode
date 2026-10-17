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


# ---------------------------------------------------------------- location + pairing

import json  # noqa: E402
import subprocess  # noqa: E402

import lastscene_device as dev  # noqa: E402

NMCLI_OUT = "AA\\:BB\\:CC\\:00\\:11\\:22:80\nAA\\:BB\\:CC\\:00\\:11\\:33:40\nAA\\:BB\\:CC\\:00\\:11\\:22:30\n"
IW_OUT = """BSS 12:34:56:78:9a:bc(on wlan0)
\tsignal: -48.00 dBm
\tSSID: Home
BSS de:ad:be:ef:00:01(on wlan0)
\tsignal: -71.00 dBm
"""


def fake_run(outputs):
    def run(cmd, **kwargs):
        out = outputs.get(cmd[0])
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out or "")
    return run


def test_nmcli_scan_parses_escaped_bssids_and_keeps_strongest(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": NMCLI_OUT}))
    aps = dev.scan_wifi()
    assert aps == [{"bssid": "aa:bb:cc:00:11:22", "signal": -60}, {"bssid": "aa:bb:cc:00:11:33", "signal": -80}]


def test_falls_back_to_iw_without_networkmanager(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": FileNotFoundError("nmcli"), "iw": IW_OUT}))
    assert dev.scan_wifi() == [{"bssid": "12:34:56:78:9a:bc", "signal": -48}, {"bssid": "de:ad:be:ef:00:01", "signal": -71}]


def test_whereabouts_by_source(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", fake_run({"nmcli": NMCLI_OUT}))
    cfg = Config("https://api.test", "t", "a", "id", "s", location_source="wifi", fixed_lat=29.65, fixed_lng=-82.32)
    fields = dev.whereabouts(cfg)
    assert json.loads(fields["wifi"])[0]["bssid"] == "aa:bb:cc:00:11:22"
    assert fields["lat"] == "29.65"  # fallback for when Wi-Fi gives no fix
    cfg.location_source = "fixed"
    assert "wifi" not in dev.whereabouts(cfg)
    cfg.location_source = "none"
    assert dev.whereabouts(cfg) == {}


def test_pairing_code_and_location_are_sent():
    seen = {}

    def handler(request):
        if request.url.host == "tenant.auth0.test":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        seen["code"] = request.headers.get("x-pairing-code")
        seen["body"] = request.content
        return httpx.Response(200, json={"id": "1"})

    api = make_client(handler)
    api.cfg.pairing_code = "ABCD-1234-EF56"
    api.log_memory(b"wav", b"jpg", {"wifi": '[{"bssid": "aa:bb:cc:00:11:22", "signal": -60}]'})
    assert seen["code"] == "ABCD-1234-EF56"
    assert b'name="wifi"' in seen["body"] and b"aa:bb:cc:00:11:22" in seen["body"]


def test_spooled_memory_keeps_where_it_happened(tmp_path):
    server = FakeServer()
    bodies = []
    original = server.__call__

    def capture(request):
        if request.url.host != "tenant.auth0.test":
            bodies.append(request.content)
        return original(request)

    spool = Spool(tmp_path / "spool")
    spool.add(b"wav", b"jpg", {"lat": "29.65", "lng": "-82.32"})
    assert spool.flush(make_client(capture)) == 1
    assert b'name="lat"' in bodies[0] and b"29.65" in bodies[0]
