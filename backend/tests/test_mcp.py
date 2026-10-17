"""The read-only MCP server at /mcp, driven by the MCP SDK's own client over HTTP (in
process, against the ASGI app) and by plain JSON-RPC posts. Fake data only."""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anyio
import httpx2
import pytest
from fastapi.testclient import TestClient
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

from app import auth, mcp_server
from app.config import Settings, get_settings
from app.db import ItemSighting, MemoryRecord, QueryRecord
from app.db_mcp import mcp_key_hash
from app.db_observations import LastSeen
from app.db_places import Household, Place
from app.main import create_app
from app.rag import ChunkHit
from tests.test_api import AUDIENCE, CAREGIVER, DEVICE, DOMAIN, KEY, FakeAI, FakeDB, make_token

MOM = "11111111-1111-4111-8111-111111111111"  # the test caregiver's household
OTHER = "22222222-2222-4222-8222-222222222222"  # someone else's
FAMILY = {"Authorization": f"Bearer {make_token('auth0|sibling')}"}  # a non-owner member of MOM
JSON_RPC = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class MCPFakeDB(FakeDB):
    households = {MOM: ("Mom's household", ["dev1@clients"]), OTHER: ("Not yours", ["dev2@clients"])}
    members = {(MOM, "auth0|carer"): "owner", (MOM, "auth0|sibling"): "member", (OTHER, "auth0|stranger"): "owner"}

    def __init__(self):
        super().__init__()
        self.keys: dict[str, dict] = {}
        self.rate: dict[str, int] = {}
        self.writes: list[str] = []

    async def households_for(self, sub):
        return [Household(h, self.households[h][0], False, role) for (h, s), role in self.members.items() if s == sub]

    async def member_role(self, household_id, sub):
        return self.members.get((household_id, sub))

    async def household_device_ids(self, household_id):
        return list(self.households[household_id][1])

    async def last_seen(self, scope, item=None):
        now = datetime.now(timezone.utc)
        found = [
            LastSeen("ls1", MOM, "wallet", "m-cam", "obs1", "vision", now - timedelta(hours=2), None,
                     "hall table", [0.1, 0.1, 0.2, 0.2], 0.42, 640, 480, True, None),
            LastSeen("ls2", OTHER, "wallet", "m-x", "obs2", "vision", now, None, "their desk", None, 0.9, 640, 480, True, None),
        ]
        return [ls for ls in found if ls.scope == scope and item in (None, ls.item)]

    async def places_for(self, household_id):
        return [Place("p1", MOM, "Home", "private", 29.65, -82.32, 50, visits=3)] if household_id == MOM else []

    async def training_items(self, household_id):
        return [SimpleNamespace(name="wallet", aliases=["billfold"])] if household_id == MOM else []

    # MCP keys and the rate counter
    async def create_mcp_key(self, household_id, sub, label):
        key_id, key = str(uuid.uuid4()), f"lsk_test{len(self.keys)}xyz"
        self.keys[key_id] = {"household_id": household_id, "sub": sub, "label": label, "hash": mcp_key_hash(key),
                             "hint": key[:8], "created_at": datetime.now(timezone.utc), "last_used_at": None, "revoked": False}
        return key_id, key

    async def mcp_keys(self, household_id):
        return [{"id": i, **k} for i, k in self.keys.items() if k["household_id"] == household_id and not k["revoked"]]

    async def revoke_mcp_key(self, household_id, key_id):
        k = self.keys.get(key_id)
        if not k or k["household_id"] != household_id or k["revoked"]:
            return False
        k["revoked"] = True
        return True

    async def mcp_key_caller(self, key_hash):
        for i, k in self.keys.items():
            if k["hash"] == key_hash and not k["revoked"] and (k["household_id"], k["sub"]) in self.members:
                return i, k["household_id"], k["sub"]
        return None

    async def mcp_rate_hit(self, subject, window_seconds):
        self.rate[subject] = self.rate.get(subject, 0) + 1
        return self.rate[subject]

    async def missing_grid(self, device_ids, days):  # FakeDB's counts every device
        mine = {r.id for r in self.rows if r.device_id in device_ids}
        items, self.items = self.items, [i for i in self.items if i.memory_id in mine]
        try:
            return await super().missing_grid(device_ids, days)
        finally:
            self.items = items

    # Anything that writes household data would land here; the MCP tools must never call it.
    async def add_items(self, *a):
        self.writes.append("add_items")


def seed(db: MCPFakeDB) -> None:
    now = datetime.now(timezone.utc)
    mine = MemoryRecord("m1", "dev1@clients", now - timedelta(hours=3), "I put my wallet in the kitchen drawer", True,
                        "A kitchen with an open drawer")
    older = MemoryRecord("m0", "dev1@clients", now - timedelta(hours=30), "My keys are on the hook", False)
    theirs = MemoryRecord("m2", "dev2@clients", now - timedelta(hours=1), "Their wallet is in the car", False)
    db.rows += [older, mine, theirs]
    db.items += [
        ItemSighting("wallet", "kitchen drawer", "m1", mine.created_at, mine.transcript, True, mine.scene),
        ItemSighting("keys", "hook", "m0", older.created_at, older.transcript, False),
        ItemSighting("wallet", "car", "m2", theirs.created_at, theirs.transcript, False),
    ]
    db.chunks += [
        ChunkHit(1, "m1", "speech", mine.transcript, mine.created_at),
        ChunkHit(2, "m1", "item", "wallet: kitchen drawer", mine.created_at),
        ChunkHit(3, "m0", "speech", older.transcript, older.created_at),
        ChunkHit(4, "m2", "speech", theirs.transcript, theirs.created_at),
    ]
    db.queries.append(QueryRecord(now - timedelta(hours=1), "Where is my wallet?", "In the kitchen drawer.", "wallet"))
    db.query_devices.append("dev1@clients")


@pytest.fixture
def app(monkeypatch):
    settings = Settings(auth0_domain=DOMAIN, auth0_audience=AUDIENCE, image_url_secret="test-secret", _env_file=None)
    signing = SimpleNamespace(key=KEY.public_key())
    monkeypatch.setattr(auth, "_jwks_client", lambda domain: SimpleNamespace(get_signing_key_from_jwt=lambda t: signing))
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db, app.state.ai = MCPFakeDB(), FakeAI()
    seed(app.state.db)
    # FakeDB.recent_queries ignores devices; scope it like the real one.
    queries = app.state.db.queries

    async def recent_queries(hours, device_ids=None):
        return [q for q, d in zip(queries, app.state.db.query_devices) if device_ids is None or d in device_ids]

    app.state.db.recent_queries = recent_queries
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def with_mcp(app, headers, fn, mode="legacy"):
    """Run fn(session) with the SDK's Streamable HTTP client connected to /mcp."""

    async def main():
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://testserver", headers=headers)
        async with http, Client(streamable_http_client("http://testserver/mcp", http_client=http), mode=mode) as session:
            return await fn(session)

    return anyio.run(main)


def call(app, name, args=None, headers=CAREGIVER):
    async def go(session):
        return await session.call_tool(name, args or {})

    return with_mcp(app, headers, go)


def rpc(client, method, params=None, headers=CAREGIVER, id=1):
    return client.post("/mcp", headers={**headers, **JSON_RPC, "MCP-Protocol-Version": "2025-06-18"},
                       json={"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})


# ----------------------------------------------------------------- protocol


@pytest.mark.parametrize("mode", ["legacy", "auto"])  # the initialize handshake, and the 2026 discover path
def test_initialize_and_list_tools(app, mode):
    async def go(session):
        return await session.list_tools()

    tools = {t.name: t for t in with_mcp(app, CAREGIVER, go, mode).tools}
    assert set(tools) == {"list_households", "search_memories", "item_last_seen", "timeline", "items", "missing_times", "places"}
    for t in tools.values():
        assert t.description and t.annotations.read_only_hint and not t.annotations.destructive_hint
    assert tools["search_memories"].input_schema["required"] == ["household_id", "query"]


def test_plain_json_rpc_initialize_and_tools_list(client):
    init = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "curl", "version": "1"}})
    assert init.status_code == 200 and init.headers["content-type"].startswith("application/json")
    result = init.json()["result"]
    assert result["serverInfo"]["name"] == "lastscene" and "read-only" in result["instructions"]
    listed = rpc(client, "tools/list", id=2)
    assert len(listed.json()["result"]["tools"]) == 7


# ----------------------------------------------------------------- tools


def test_list_households_shows_only_the_callers(app):
    r = call(app, "list_households")
    assert r.structured_content == {"households": [{"id": MOM, "name": "Mom's household", "role": "owner", "wearables": 1}]}


def test_search_memories_finds_the_households_memories_only(app):
    r = call(app, "search_memories", {"household_id": MOM, "query": "wallet"})
    results = r.structured_content["results"]
    assert results[0]["memory_id"] == "m1"
    assert {"kind": "item", "content": "wallet: kitchen drawer"} in results[0]["text"]
    assert all(x["memory_id"] != "m2" for x in results)  # the other household's wallet
    assert "image" not in json.dumps(r.structured_content)


def test_item_last_seen_combines_memories_and_camera(app):
    r = call(app, "item_last_seen", {"household_id": MOM, "item": "Wallet"}).structured_content
    assert r["item"] == "wallet"
    assert [m["location"] for m in r["memories"]] == ["kitchen drawer"]
    assert r["camera"] == [{"name": "wallet", "timestamp": r["camera"][0]["timestamp"], "scene": "hall table",
                            "confidence": 0.42, "maybe": True, "found_at": None}]


def test_timeline_lists_memories_and_questions_newest_first(app):
    r = call(app, "timeline", {"household_id": MOM}).structured_content
    assert [e["kind"] for e in r["entries"]] == ["question", "memory", "memory"]
    assert r["entries"][1]["items"] == [{"name": "wallet", "location": "kitchen drawer"}]
    since = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    r = call(app, "timeline", {"household_id": MOM, "since": since, "limit": 1}).structured_content
    assert len(r["entries"]) == 1 and r["more"]
    bad = call(app, "timeline", {"household_id": MOM, "since": "yesterday"})
    assert bad.is_error and "ISO 8601" in bad.content[0].text


def test_items_without_photo_links(app):
    r = call(app, "items", {"household_id": MOM}).structured_content
    by_name = {i["name"]: i for i in r["items"]}
    assert by_name["wallet"]["location"] == "kitchen drawer" and by_name["wallet"]["has_photo"]
    assert by_name["wallet"]["times_asked"] == 1
    assert "image_url" not in json.dumps(r) and r["taught"] == [{"name": "wallet", "aliases": ["billfold"]}]


def test_missing_times_counts(app):
    r = call(app, "missing_times", {"household_id": MOM, "days": 7}).structured_content
    assert r["days"] == 7
    assert sum(h["asked"] for h in r["hours"]) == 1 and sum(h["put_down"] for h in r["hours"]) == 2


def test_places(app):
    r = call(app, "places", {"household_id": MOM}).structured_content
    assert r["places"][0]["label"] and r["places"][0]["kind"] == "private" and r["places"][0]["is_home"]


def test_tools_write_nothing(app):
    for name, args in [("items", {}), ("timeline", {}), ("item_last_seen", {"item": "wallet"}), ("places", {})]:
        call(app, name, {"household_id": MOM, **args})
    assert app.state.db.writes == []


# ----------------------------------------------------------------- access


def test_requires_a_token(client):
    r = client.post("/mcp", headers=JSON_RPC, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    bad = {"Authorization": f"Bearer {make_token('auth0|carer', aud='https://someone-else')}"}
    assert rpc(client, "tools/list", headers=bad).status_code == 401


def test_device_tokens_are_refused(client):
    assert rpc(client, "tools/list", headers=DEVICE).status_code == 403


@pytest.mark.parametrize("tool", ["search_memories", "item_last_seen", "timeline", "items", "missing_times", "places"])
@pytest.mark.parametrize("household", [OTHER, "not-a-uuid", "33333333-3333-4333-8333-333333333333"])
def test_another_households_id_is_forbidden(app, tool, household):
    extra = {"search_memories": {"query": "wallet"}, "item_last_seen": {"item": "wallet"}}.get(tool, {})
    r = call(app, tool, {"household_id": household, **extra})
    assert r.is_error and r.content[0].text.endswith(
        "403 Forbidden: you are not a member of that household. Use an id from list_households.")
    assert "Their wallet" not in json.dumps(r.model_dump())


def test_rate_limited_per_caregiver(client, monkeypatch):
    monkeypatch.setattr(mcp_server, "RATE_LIMIT_PER_MINUTE", 3)
    codes = [rpc(client, "tools/list", id=i).status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
    assert rpc(client, "tools/list", headers=FAMILY).status_code == 200  # another caregiver has their own budget


def test_only_post_is_served(client):
    for method in ("get", "delete"):
        assert getattr(client, method)("/mcp", headers={**CAREGIVER, **JSON_RPC}).status_code == 405


def test_oversized_request_is_refused(client):
    r = rpc(client, "tools/call", {"name": "search_memories", "arguments": {"household_id": MOM, "query": "x" * 70_000}})
    assert r.status_code == 413


def test_tool_calls_are_logged_without_their_content(app, caplog):
    with caplog.at_level(logging.INFO, logger="lastscene.mcp"):
        call(app, "search_memories", {"household_id": MOM, "query": "the secret diary"})
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "lastscene.mcp")
    assert "tool=search_memories" in logged and MOM in logged and "auth0|carer" in logged
    assert "secret diary" not in logged


# ----------------------------------------------------------------- household MCP keys


def test_mcp_key_lifecycle(app, client):
    base = f"/api/v1/households/{MOM}/mcp-keys"
    assert client.post(base, headers=FAMILY, json={"label": "script"}).status_code == 403  # owners only
    made = client.post(base, headers=CAREGIVER, json={"label": "Gemini script"})
    assert made.status_code == 201
    key = made.json()["key"]
    listed = client.get(base, headers=FAMILY).json()["keys"]
    assert [k["label"] for k in listed] == ["Gemini script"] and key not in json.dumps(listed)  # shown once only

    with_key = {"Authorization": f"Bearer {key}"}
    assert call(app, "list_households", headers=with_key).structured_content["households"][0]["id"] == MOM
    assert call(app, "places", {"household_id": OTHER}, headers=with_key).is_error

    assert client.delete(f"{base}/{made.json()['id']}", headers=CAREGIVER).status_code == 200
    assert rpc(client, "tools/list", headers=with_key).status_code == 401
    assert rpc(client, "tools/list", headers={"Authorization": "Bearer lsk_made-up"}).status_code == 401


def test_a_key_only_sees_its_own_household(app, client):
    MCPFakeDB.members[(OTHER, "auth0|carer")] = "member"
    try:
        key = client.post(f"/api/v1/households/{MOM}/mcp-keys", headers=CAREGIVER, json={"label": "k"}).json()["key"]
        r = call(app, "list_households", headers={"Authorization": f"Bearer {key}"})
        assert [h["id"] for h in r.structured_content["households"]] == [MOM]
        assert call(app, "places", {"household_id": OTHER}, headers={"Authorization": f"Bearer {key}"}).is_error
        assert not call(app, "places", {"household_id": OTHER}).is_error  # the caregiver's own token can
    finally:
        del MCPFakeDB.members[(OTHER, "auth0|carer")]


def test_key_count_is_capped(client):
    base = f"/api/v1/households/{MOM}/mcp-keys"
    codes = [client.post(base, headers=CAREGIVER, json={"label": f"k{i}"}).status_code for i in range(6)]
    assert codes == [201] * 5 + [409]
