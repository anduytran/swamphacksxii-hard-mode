"""Read-only MCP (Model Context Protocol) server at /mcp, for Gemini or any MCP client.

Streamable HTTP, stateless, JSON responses: every POST is answered on its own, so it runs
on serverless hosts (Vercel) with no sessions to keep. See docs/MCP.md.

Callers authenticate with a caregiver's Auth0 access token (checked like the dashboard's)
or a household MCP key (lsk_..., made on the Household page). Tools only reach households
the caregiver belongs to (a key: only its own household), and nothing here writes data.
"""

import contextvars
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import Field
from starlette.types import Receive, Scope, Send

from .ai import AIServices
from .auth import bearer, get_principal, require_caregiver
from .config import Settings, get_settings
from .db import MemoryDB
from .db_mcp import KEY_PREFIX, mcp_key_hash
from .deps import get_ai, get_db
from .pipeline import retrieve
from .routers import dashboard, households

log = logging.getLogger("lastscene.mcp")

# Every MCP request (initialize, tools/list, each tool call) counts against the caller.
RATE_LIMIT_PER_MINUTE = 60
RATE_WINDOW_SECONDS = 60
# JSON-RPC requests are tiny; anything bigger is refused with 413 before it is parsed.
MAX_REQUEST_BYTES = 64 * 1024

INSTRUCTIONS = """\
LastScene is a memory aid: a wearable camera and microphone note where a person with memory
loss put things down, and family caregivers look it up. These tools read one household's
data (read-only). Call list_households first to get a household_id. Times are UTC ISO 8601.
Memories are kept for 72 hours; older activity survives only as hourly counts (missing_times).
The data is personal: use it to answer the caregiver's question and nothing else."""


@dataclass
class Caller:
    sub: str  # the caregiver (for a key: the member who made it)
    key_id: str | None = None  # set when authenticated with a household MCP key
    household_id: str | None = None  # a key's only household


@dataclass
class _Request:
    caller: Caller
    db: MemoryDB
    ai: AIServices
    settings: Settings


_current: contextvars.ContextVar[_Request] = contextvars.ContextVar("lastscene_mcp_request")

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
# MCPServer() calls logging.basicConfig(), which would give the root logger a handler and
# print every "lastscene" line twice (main.py gives that logger its own). Keep root as it was.
_root = logging.getLogger()
_root_handlers, _root_level = list(_root.handlers), _root.level
server = MCPServer(
    name="lastscene",
    title="LastScene",
    description="Read-only access to a LastScene household's memories, items and places.",
    instructions=INSTRUCTIONS,
    website_url="https://www.lastscene.photo",
    version="1",
)
_root.handlers[:] = _root_handlers
_root.setLevel(_root_level)

HouseholdId = Annotated[str, Field(description="A household id from list_households.", min_length=1, max_length=64)]


def _forbidden() -> ToolError:
    # Same answer for "no such household" and "not yours", so ids can't be probed.
    return ToolError("403 Forbidden: you are not a member of that household. Use an id from list_households.")


async def _household(tool: str, household_id: str) -> tuple[_Request, str, list[str]]:
    """The request, the household id and its devices, if the caller may see it. Logs the
    call: which tool, household and caregiver, never the arguments (they're personal)."""
    req = _current.get()
    try:
        hid = str(uuid.UUID(household_id))
    except ValueError:
        raise _forbidden() from None
    if req.caller.household_id not in (None, hid) or await req.db.member_role(hid, req.caller.sub) is None:
        log.warning("MCP tool=%s denied household=%s caregiver=%s key=%s", tool, hid, req.caller.sub, req.caller.key_id or "-")
        raise _forbidden()
    log.info("MCP tool=%s household=%s caregiver=%s key=%s", tool, hid, req.caller.sub, req.caller.key_id or "-")
    return req, hid, await req.db.household_device_ids(hid)


def _label(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


def _same_item(a: str, b: str) -> bool:
    """'wallet' finds 'brown wallet', and the other way round."""
    a, b = _label(a), _label(b)
    return bool(a and b) and (a in b or b in a)


def _when(raw: str | None, name: str) -> datetime | None:
    if not raw:
        return None
    try:
        t = datetime.fromisoformat(raw.strip())
    except ValueError:
        raise ToolError(f"{name} must be an ISO 8601 time, e.g. 2026-10-17T14:00:00Z") from None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


@server.tool(annotations=READ_ONLY)
async def list_households() -> dict[str, Any]:
    """List the households this caregiver belongs to: id, name, the caller's role and how
    many wearables it has. Call this first; every other tool takes a household_id."""
    req = _current.get()
    log.info("MCP tool=list_households caregiver=%s key=%s", req.caller.sub, req.caller.key_id or "-")
    result = []
    for h in await req.db.households_for(req.caller.sub):
        if req.caller.household_id in (None, h.id):
            devices = await req.db.household_device_ids(h.id)
            result.append({"id": h.id, "name": h.name, "role": h.role, "wearables": len(devices)})
    return {"households": result}


@server.tool(annotations=READ_ONLY)
async def search_memories(
    household_id: HouseholdId,
    query: Annotated[str, Field(min_length=1, max_length=300, description="What to look for, in plain words, e.g. 'wallet' or 'where did she put the car keys'.")],
    hours: Annotated[int, Field(ge=1, le=720, description="How far back to search, in hours. Capped at the 72-hour retention.")] = 72,
    limit: Annotated[int, Field(ge=1, le=10, description="Most memories to return.")] = 5,
) -> dict[str, Any]:
    """Search the household's memories by meaning and by words (the same hybrid search the
    wearable uses to answer spoken questions). Each result is one memory: when it happened
    and its text (what was said, what the camera saw, items and where they were put),
    best match first. No photos."""
    req, _, devices = await _household("search_memories", household_id)
    if not devices:  # nothing to search, so don't spend an embedding call
        return {"query": query, "results": []}
    window = min(hours, req.settings.retention_hours)
    memories = await retrieve(req.db, req.ai, query, devices, window, limit)
    return {
        "query": query,
        "window_hours": window,
        "results": [
            {
                "memory_id": m.memory_id,
                "timestamp": m.created_at.isoformat(),
                "score": round(m.score, 5),
                "text": [{"kind": c.kind, "content": c.content} for c in m.chunks],
            }
            for m in memories
        ],
    }


@server.tool(annotations=READ_ONLY)
async def item_last_seen(
    household_id: HouseholdId,
    item: Annotated[str, Field(min_length=1, max_length=80, description="The item, e.g. 'wallet' or 'keys'. Partial names match.")],
) -> dict[str, Any]:
    """Where and when one item was last seen. `memories`: the newest memories naming the
    item (from what the wearer said and the photo), each with where it was put. `camera`:
    the vision pipeline's latest sightings, with the detector's confidence (`maybe` = low
    confidence) and `found_at` if the wearer has since found it. Both newest first."""
    req, hid, devices = await _household("item_last_seen", household_id)
    sightings = [s for s in await req.db.item_sightings(req.settings.retention_hours, devices) if _same_item(s.name, item)]
    camera = [ls for ls in await req.db.last_seen(hid) if _same_item(ls.item, item)]
    return {
        "item": _label(item),
        "memories": [
            {"name": s.name, "location": s.location or None, "timestamp": s.created_at.isoformat(),
             "said": s.transcript or None, "scene": s.scene}
            for s in sightings[:5]
        ],
        "camera": [
            {"name": ls.item, "timestamp": ls.captured_at.isoformat(), "scene": ls.scene_label,
             "confidence": ls.confidence,
             "maybe": ls.confidence is not None and ls.confidence < req.settings.maybe_confidence,
             "found_at": ls.found_at.isoformat() if ls.found_at else None}
            for ls in camera[:5]
        ],
    }


@server.tool(annotations=READ_ONLY)
async def timeline(
    household_id: HouseholdId,
    since: Annotated[str | None, Field(description="Start, ISO 8601 (UTC if no offset). Default: 72 hours ago.")] = None,
    until: Annotated[str | None, Field(description="End, ISO 8601 (UTC if no offset). Default: now.")] = None,
    limit: Annotated[int, Field(ge=1, le=100, description="Most entries to return.")] = 20,
) -> dict[str, Any]:
    """What happened, newest first: memories (what was said, what the camera saw, items put
    down) and questions the wearer asked the device, with its answers. Only the last 72
    hours are kept."""
    req, _, devices = await _household("timeline", household_id)
    now = datetime.now(timezone.utc)
    start, end = _when(since, "since"), _when(until, "until") or now
    retention = req.settings.retention_hours
    hours = retention if start is None else max(1, min(retention, ceil((now - start).total_seconds() / 3600)))
    in_range = lambda t: (start is None or t >= start) and t <= end  # noqa: E731

    items: dict[str, list[dict]] = {}
    for s in await req.db.item_sightings(hours, devices):
        items.setdefault(s.memory_id, []).append({"name": s.name, "location": s.location or None})
    entries = [
        {
            "kind": "memory",
            "timestamp": r.created_at.isoformat(),
            "source": r.source,  # manual (the wearer pressed Remember) | auto (seen by the camera)
            "said": r.transcript or None,
            "scene": r.scene.removeprefix(dashboard.AUTO_PREFIX) if r.scene else None,
            "items": items.get(r.id, []),
            "_t": r.created_at,
        }
        for r in await req.db.recent(hours, devices)
        if in_range(r.created_at)
    ] + [
        {"kind": "question", "timestamp": q.created_at.isoformat(), "question": q.question, "answer": q.answer,
         "item": q.item, "_t": q.created_at}
        for q in await req.db.recent_queries(hours, devices)
        if in_range(q.created_at)
    ]
    entries.sort(key=lambda e: e["_t"], reverse=True)
    for e in entries:
        del e["_t"]
    return {"since": start.isoformat() if start else None, "until": end.isoformat(),
            "entries": entries[:limit], "more": len(entries) > limit}


@server.tool(annotations=READ_ONLY)
async def items(household_id: HouseholdId) -> dict[str, Any]:
    """Every item seen in the last 72 hours: where it was last put and when, how often it
    was mentioned, how often the wearer asked about it, and up to four earlier spots.
    `taught` lists the items the household taught the camera to recognise."""
    req, hid, devices = await _household("items", household_id)
    data = await dashboard.items(devices=devices, db=req.db, settings=req.settings)
    for i in data["items"]:  # small JSON: no photo links or boxes
        i["has_photo"] = bool(i.pop("image_url"))
        i.pop("detection")
    taught = await req.db.training_items(hid) if hasattr(req.db, "training_items") else []
    return {**data, "taught": [{"name": t.name, "aliases": t.aliases} for t in taught]}


@server.tool(annotations=READ_ONLY)
async def missing_times(
    household_id: HouseholdId,
    days: Annotated[int, Field(ge=7, le=90, description="How many days back to count.")] = 28,
) -> dict[str, Any]:
    """When things go missing: per UTC hour with any activity, how many questions the wearer
    asked (`asked`) and how many things were put down (`put_down`). Convert to the
    household's local time before talking about times of day. Counts only, no content, so
    it reaches back further than the 72-hour memories."""
    req, _, devices = await _household("missing_times", household_id)
    return await dashboard.missing_times(days=days, devices=devices, db=req.db)


@server.tool(annotations=READ_ONLY)
async def places(household_id: HouseholdId) -> dict[str, Any]:
    """The household's places, found from the wearable's position: label, name, kind
    (private | public), whether it's home, centre, radius and how many memories happened
    there."""
    req, hid, _ = await _household("places", household_id)
    return {"places": await households._places_view(req.db, hid)}


# ------------------------------------------------------------------- HTTP endpoint

router = APIRouter(tags=["mcp"])


async def mcp_caller(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
    db: MemoryDB = Depends(get_db),
) -> Caller:
    """A caregiver token (validated exactly like the dashboard's) or a household MCP key."""
    if creds is not None and creds.scheme.lower() == "bearer" and creds.credentials.startswith(KEY_PREFIX):
        found = await db.mcp_key_caller(mcp_key_hash(creds.credentials))
        if found is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked MCP key",
                                headers={"WWW-Authenticate": "Bearer"})
        key_id, household_id, sub = found
        return Caller(sub=sub, key_id=key_id, household_id=household_id)
    try:
        principal = await require_caregiver(await get_principal(creds, settings), settings)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            exc.headers = {"WWW-Authenticate": "Bearer"}
        raise
    return Caller(sub=principal.sub)


async def rate_limited(caller: Caller = Depends(mcp_caller), db: MemoryDB = Depends(get_db)) -> Caller:
    """Per caregiver, counted in the database so every serverless instance shares it."""
    if await db.mcp_rate_hit(f"mcp:{caller.sub}", RATE_WINDOW_SECONDS) > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"At most {RATE_LIMIT_PER_MINUTE} MCP requests a minute",
                            headers={"Retry-After": str(RATE_WINDOW_SECONDS)})
    return caller


class _MCPResponse(Response):
    """Hands the raw request to the MCP SDK once FastAPI has authenticated it."""

    def __init__(self, request: _Request):
        super().__init__()
        self.request = request

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A fresh stateless manager per request: no sessions to keep, and nothing tied to
        # an event loop that a serverless host may swap between requests.
        server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            max_request_body_size=MAX_REQUEST_BYTES,
            # Bearer auth, not cookies, so DNS rebinding can't borrow a browser's access.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        manager = server.session_manager
        token = _current.set(self.request)
        try:
            async with manager.run():
                await manager.handle_request(scope, receive, send)
        finally:
            _current.reset(token)


# POST only: stateless JSON mode has no sessions to DELETE, and a GET would hold an SSE
# stream (and a serverless function) open for nothing.
@router.post("/mcp", include_in_schema=False)
async def mcp_endpoint(
    request: Request,
    caller: Caller = Depends(rate_limited),
    db: MemoryDB = Depends(get_db),
    ai: AIServices = Depends(get_ai),
    settings: Settings = Depends(get_settings),
):
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"MCP requests are limited to {MAX_REQUEST_BYTES} bytes")
    return _MCPResponse(_Request(caller, db, ai, settings))
