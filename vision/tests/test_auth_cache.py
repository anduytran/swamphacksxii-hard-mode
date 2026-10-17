from lastscene_bridge.auth import Auth0ClientCredentials, token_provider_from_env
from lastscene_bridge.uploader import uploader_from_env


def test_dylan_aliases(tmp_path):
    env = dict(API_URL="https://example.test", PAIRING_CODE="pair", DEVICE_CLIENT_ID="id",
               DEVICE_CLIENT_SECRET="secret", AUTH0_DOMAIN="example.test", AUTH0_AUDIENCE="api",
               LASTSCENE_TOKEN_CACHE=str(tmp_path / "token.json"))
    uploader = uploader_from_env(tmp_path / "queue", env)
    assert uploader.api_base == "https://example.test"
    assert uploader.pairing_code == "pair"
    assert uploader.tokens._body["client_id"] == "id"


def test_persistent_token_and_invalidation(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"access_token": "private-token", "expires_in": 3600}
    calls = []
    monkeypatch.setattr("lastscene_bridge.auth.requests.post", lambda *a, **k: calls.append(1) or Response())
    path = tmp_path / "token.json"
    def provider():
        return Auth0ClientCredentials("example.test", "id", "secret", "api", cache_path=path)
    assert provider().get() == "private-token"
    second = provider()
    assert second.get() == "private-token"
    assert len(calls) == 1
    assert "secret" not in path.read_text()
    second.invalidate()
    assert not path.exists()
    assert provider().get() == "private-token"
    assert len(calls) == 2
