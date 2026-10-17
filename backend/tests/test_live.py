"""Live conversation mode: the Gemini Live token endpoint and the memory lookup its tool calls."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

from google import genai
from google.genai import _api_client, types

from app.live import LIVE_API_VERSION, LIVE_SYSTEM_PROMPT, RateLimit, token_config
from app.routers.live import NOTHING_FOUND
from tests.test_api import CAREGIVER, DEVICE, client, log  # noqa: F401  (client is a fixture)


class FakeTokens:
    """Stands in for genai.Client().aio.auth_tokens; keeps the config it was asked for."""

    def __init__(self, fail=False):
        self.configs = []
        self.fail = fail

    async def create(self, *, config):
        if self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        self.configs.append(config)
        return types.AuthToken(name=f"auth_tokens/t{len(self.configs)}")


def fake_genai(client, **kw):  # noqa: F811
    tokens = FakeTokens(**kw)
    client.app_state.live_tokens.client = SimpleNamespace(aio=SimpleNamespace(auth_tokens=tokens))
    return tokens


def token(client, headers=DEVICE):  # noqa: F811
    return client.post("/api/v1/live/token", headers=headers)


def test_token_is_404_while_live_mode_is_off(client):  # noqa: F811
    tokens = fake_genai(client)
    assert token(client).status_code == 404
    assert tokens.configs == []  # nothing minted


def test_token_locks_model_persona_tools_and_voice(client):  # noqa: F811
    client.settings.live_enabled = True
    client.settings.gemini_tts_voice = "Puck"
    client.settings.gemini_api_key = "server-secret-key"
    tokens = fake_genai(client)
    resp = token(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "auth_tokens/t1" and body["model"] == "gemini-3.8-live"
    assert body["api_version"] == "v1alpha"
    assert "server-secret-key" not in resp.text

    (cfg,) = tokens.configs
    assert cfg.uses == 1
    assert cfg.lock_additional_fields is None  # unset = the whole constraint is locked
    assert 60 <= (cfg.expire_time - cfg.new_session_expire_time).total_seconds() <= 600
    assert datetime.fromisoformat(body["expires_at"]) == cfg.expire_time
    locked = cfg.live_connect_constraints
    assert locked.model == "gemini-3.8-live"
    assert locked.config.response_modalities == [types.Modality.AUDIO]
    assert locked.config.system_instruction == LIVE_SYSTEM_PROMPT
    assert locked.config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"
    (tool,) = locked.config.tools
    assert [f.name for f in tool.function_declarations] == ["recall_memories", "save_memory"]
    assert tool.function_declarations[0].parameters.required == ["question"]


def test_token_needs_a_device_and_fails_soft(client):  # noqa: F811
    client.settings.live_enabled = True
    fake_genai(client)
    assert token(client, headers=CAREGIVER).status_code == 403
    assert client.post("/api/v1/live/token").status_code == 401
    fake_genai(client, fail=True)  # e.g. the free-tier key is out of quota
    assert token(client).status_code == 503


def test_tokens_are_rate_limited_per_device(client):  # noqa: F811
    client.settings.live_enabled = True
    client.settings.live_tokens_per_hour = 2
    fake_genai(client)
    assert [token(client).status_code for _ in range(3)] == [200, 200, 429]
    resp = token(client)
    assert resp.status_code == 429 and int(resp.headers["retry-after"]) > 0


def test_rate_limit_window_slides():
    now = [0.0]
    limit = RateLimit(window_s=3600, clock=lambda: now[0])
    assert limit.retry_after("dev", 1) is None
    assert limit.retry_after("dev", 1) == 3600
    assert limit.retry_after("other", 1) is None  # per device
    now[0] = 3600
    assert limit.retry_after("dev", 1) is None


def test_the_sdk_sends_the_locked_setup_to_v1alpha():
    """The real google-genai request builder (no network): checks the field names end to end."""
    client = genai.Client(api_key="server-key", http_options=types.HttpOptions(api_version=LIVE_API_VERSION))
    sent = []

    async def fake_request(http_request, http_options=None, stream=False):
        sent.append(http_request)
        return _api_client.HttpResponse(headers={}, response_stream=['{"name": "auth_tokens/abc"}'])

    client.aio._api_client._async_request = fake_request
    settings = SimpleNamespace(gemini_live_model="gemini-3.8-live", gemini_tts_voice="Kore")
    result = asyncio.run(client.aio.auth_tokens.create(config=token_config(settings)))
    assert result.name == "auth_tokens/abc"
    (req,) = sent
    assert req.url.endswith("/v1alpha/auth_tokens")
    body = json.loads(json.dumps(req.data, default=str))
    assert body["uses"] == 1 and "fieldMask" not in body  # no field mask: everything is locked
    setup = body["bidiGenerateContentSetup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["systemInstruction"]["parts"][0]["text"] == LIVE_SYSTEM_PROMPT
    names = [f["name"] for f in setup["tools"][0]["functionDeclarations"]]
    assert names == ["recall_memories", "save_memory"]


def test_recall_returns_retrieved_memories_as_facts(client):  # noqa: F811
    log(client, "I am putting my keys on the counter")
    log(client, "I am putting my wallet on the shelf")
    resp = client.post("/api/v1/memory/recall", headers=DEVICE, json={"question": "Where are my keys?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "Where are my keys?"
    assert body["facts"][0].startswith("(just now) said \"I am putting my keys on the counter\"")
    assert "items: keys: counter" in body["facts"][0]
    assert body["text"].splitlines() == body["facts"]
    assert client.app_state.ai.answered_with is None  # no language model call


def test_recall_with_nothing_saved_and_its_rules(client):  # noqa: F811
    resp = client.post("/api/v1/memory/recall", headers=DEVICE, json={"question": "Where is my hat?"})
    assert resp.json() == {"question": "Where is my hat?", "facts": [], "text": NOTHING_FOUND}
    assert client.post("/api/v1/memory/recall", headers=DEVICE, json={"question": ""}).status_code == 422
    assert client.post("/api/v1/memory/recall", headers=CAREGIVER, json={"question": "keys?"}).status_code == 403


def test_recall_fails_soft_when_search_does(client):  # noqa: F811
    async def broken(text, *, query=False):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    client.app_state.ai.embed = broken
    assert client.post("/api/v1/memory/recall", headers=DEVICE, json={"question": "keys?"}).status_code == 503


def test_recall_is_rate_limited_per_device(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr("app.routers.live.RECALLS_PER_HOUR", 2)
    codes = [client.post("/api/v1/memory/recall", headers=DEVICE, json={"question": "where are my keys?"}).status_code
             for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
