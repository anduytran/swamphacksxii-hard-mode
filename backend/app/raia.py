"""Context from the frame with a raia agent (api.raia2.com), when the camera sees one of the user's items.

Two ways a frame shows one of the user's logged items (a Teach item or alias, or an item named in one of the
household's spoken memories) at RAIA_MIN_CONFIDENCE or more:
  * an observation from the vision pipeline (routers/observations.py): the detector's confidence and box;
  * a wake command's photo check (routers/memory.py, ask and remember): the photo model says the item is in
    one of the photos, with its own 0-1 confidence; that photo is sent.
Each item is read at most once every RAIA_COOLDOWN_MINUTES (claimed in Postgres, so two server instances never
both send it). The agent answers in JSON (surface, nearby, room, readable text, a one-line summary), saved as a
memory of its own, transcript "LastScene: <summary>", with the item and its place indexed like a spoken memory
("pin: on the table next to the laptop"), so "where's my pin?" finds it, often without the answer model.

Follow-ups that need the last answer ("point me in the right direction", routers/memory.py) continue in the raia
conversation that read the item's frame, so the agent answers with the photo in view (follow_up). Without one,
or when raia is slow, the stored context goes to the answer model with the last exchange (context_note).

The external API, Agent-Secret-Key header (https://api.raia2.com/api/external/docs):
    GET  /external/users/search?fkId=...                  the household's conversation user (404 = none yet)
    POST /external/users                                  create it (firstName, lastName, fkId)
    POST /external/conversations                          a conversation per frame, so frames don't mix
    POST /external/conversations/{id}/message-files/upload-url   signed URL + storagePath for the JPEG
    PUT  <signedUrl>                                      the JPEG itself
    POST /external/conversations/{id}/messages            processMessage: the prompt with files=[storagePath]
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

import httpx

from .ai import AIServices
from .config import Settings
from .db import MemoryDB
from .observations import DetectedObject, Observation, scope_for
from .rag import ITEM, SCENE, TEXT, item_chunk

# Under "lastscene" so main.py's handler shows the INFO lines on the host (Vercel). grep "RAIA".
log = logging.getLogger("lastscene.raia")

PREFIX = "LastScene: "  # the saved memory's transcript
RETRY_MINUTES = 2  # after a failed read (raia down, a bad key), the item may be tried again this soon
MAX_CONTEXT_CHARS = 300
MAX_NEARBY = 4
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


class RaiaError(Exception):
    def __init__(self, what: str, response: httpx.Response):
        super().__init__(f"raia {what}: {response.status_code} {response.text[:200]}")
        self.status_code = response.status_code


def norm(label: str) -> str:
    return " ".join(label.strip().lower().split())


def prompt(label: str, bbox: list[float] | None, scene_label: str | None) -> str:
    if bbox:
        x, y, w, h = bbox
        found = (f"A detector found their {label} inside the box x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f} "
                 f"(fractions of the image width and height, from the top left).")
    else:
        found = f"Their {label} should be somewhere in it."
    where = f" The camera's location is the {scene_label}." if scene_label else ""
    return (
        f"This photo is from a wearable camera that helps someone with memory loss find their things. {found}{where}\n"
        f"Answer with only this JSON object, no markdown or other text:\n"
        f'{{"visible": true or false (is a {label} really in the photo), '
        f'"where": "where the {label} is, 2 to 6 words starting with a preposition, like \\"on the desk by the laptop\\"", '
        f'"surface": "what it is on or in", "nearby": ["up to {MAX_NEARBY} things right next to it"], '
        f'"room": "the room or kind of place, or \\"\\"", "visible_text": "words readable near it, or \\"\\"", '
        f'"summary": "one short sentence saying where the {label} is"}}\n'
        f'If there is no {label}, answer {{"visible": false}}.'
    )


def _text(v, limit: int = 120) -> str:
    return " ".join(str(v).split())[:limit] if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""


def clean(reply: str) -> str | None:
    """A plain-text reply, tidied; None when it saw nothing."""
    text = " ".join(re.sub(r"[*_`#]", "", reply or "").split()).strip()
    if not text or text.lower().rstrip(".") in {"not visible", "none"}:
        return None
    return text[:MAX_CONTEXT_CHARS]


def parse_context(reply: str) -> dict | None:
    """The agent's JSON context, normalised; None when the item isn't visible. A reply that isn't JSON is kept
    as the summary, so a chattier agent still gets saved."""
    start, end = (reply or "").find("{"), (reply or "").rfind("}")
    try:
        raw = json.loads(reply[start:end + 1]) if start >= 0 and end > start else None
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        summary = clean(reply)
        return {"visible": True, "where": "", "surface": "", "nearby": [], "room": "", "visible_text": "",
                "summary": summary} if summary else None
    if raw.get("visible") is False:
        return None
    nearby = raw.get("nearby") if isinstance(raw.get("nearby"), list) else []
    ctx = {
        "visible": True,
        "where": _text(raw.get("where"), 80),
        "surface": _text(raw.get("surface"), 80),
        "nearby": [t for t in (_text(n, 60) for n in nearby) if t][:MAX_NEARBY],
        "room": _text(raw.get("room"), 60),
        "visible_text": _text(raw.get("visible_text"), 200),
        "summary": _text(raw.get("summary"), MAX_CONTEXT_CHARS),
    }
    if not ctx["summary"]:
        ctx["summary"] = ctx["where"] or (f"on the {ctx['surface']}" if ctx["surface"] else "")
    return ctx if ctx["summary"] else None


def place_of(ctx: dict) -> str:
    """Where the item is, as the item chunk and template answers want it ("on the desk by the laptop")."""
    return ctx["where"] or (f"on the {ctx['surface']}" if ctx["surface"] else ctx["summary"])


def chunks_for(label: str, ctx: dict) -> list[tuple[str, str]]:
    """Searchable text from the context, one fact per chunk (rag.py)."""
    place = place_of(ctx)
    detail = ctx["summary"]
    if ctx["nearby"]:
        detail += f" Next to: {', '.join(ctx['nearby'])}."
    if ctx["room"]:
        detail += f" Room: {ctx['room']}."
    out = [(ITEM, item_chunk(label, place)), (SCENE, f"{PREFIX}{detail}")]
    if ctx["visible_text"]:
        out.append((TEXT, ctx["visible_text"]))
    return out


class Raia:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self.s = settings
        self.http = http
        self._users: dict[str, str] = {}  # fkId -> conversation user id

    @property
    def configured(self) -> bool:
        return bool(self.s.raia_agent_key)

    def _client(self) -> httpx.AsyncClient:
        if self.http is None:
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(self.s.raia_timeout_seconds, connect=10.0))
        return self.http

    async def _call(self, what: str, method: str, path: str, ok=(200, 201), **kw) -> httpx.Response:
        resp = await self._client().request(
            method, f"{self.s.raia_api_url.rstrip('/')}{path}",
            headers={"Agent-Secret-Key": self.s.raia_agent_key}, **kw,
        )
        if resp.status_code not in ok:
            raise RaiaError(what, resp)
        return resp

    async def user_for(self, scope: str) -> str:
        """The household's conversation user, found by fkId or created."""
        fk = f"lastscene-{scope}"
        if fk in self._users:
            return self._users[fk]
        found = await self._call("find user", "GET", "/users/search", ok=(200, 400, 404), params={"fkId": fk})
        # The swagger says 404 for an unknown user; the live API answers 400 USER_NOT_FOUND (Oct 18).
        if found.status_code == 400 and "USER_NOT_FOUND" not in found.text:
            raise RaiaError("find user", found)
        if found.status_code == 200:
            user_id = found.json()["user"]["id"]
        else:
            created = await self._call("create user", "POST", "/users", json={
                "firstName": "LastScene", "lastName": "Household", "fkId": fk, "source": "lastscene",
            })
            user_id = created.json()["id"]
        self._users[fk] = user_id
        return user_id

    async def describe(
        self, scope: str, label: str, bbox: list[float] | None, scene_label: str | None,
        image: bytes, image_type: str, captured_at: datetime,
    ) -> tuple[str, str]:
        """(the agent's raw reply about where `label` is in the frame, the conversation's id)."""
        user_id = await self.user_for(scope)
        conv = await self._call("create conversation", "POST", "/conversations", json={
            "conversationUserId": user_id,
            "title": f"{label} seen {captured_at:%Y-%m-%d %H:%M} UTC"[:255],
            "context": "Frames from the LastScene wearable camera. Answer with the JSON object asked for.",
        })
        conv_id = conv.json()["id"]
        log.info("RAIA conversation %s (user %s) for %s", conv_id, user_id, label)
        file_name = f"frame-{captured_at:%Y%m%dT%H%M%S}.{EXTENSIONS.get(image_type, 'jpg')}"
        upload = (await self._call(
            "upload url", "POST", f"/conversations/{conv_id}/message-files/upload-url",
            json={"fileName": file_name, "contentType": image_type},
        )).json()
        put = await self._client().put(upload["signedUrl"], content=image, headers={"Content-Type": image_type})
        if put.status_code >= 400:
            raise RaiaError("upload", put)
        reply = (await self._call("message", "POST", f"/conversations/{conv_id}/messages", json={
            "message": prompt(label, bbox, scene_label),
            "files": [{"fileName": file_name, "storagePath": upload["storagePath"]}],
        })).json()
        return _reply_text(reply), conv_id

    async def follow_up(self, conversation_id: str, item: str, question: str,
                        previous: tuple[str, str] | None) -> str:
        """A wearer's follow-up about `item`, in the conversation that read its photo: a short spoken answer."""
        earlier = f'Just before, they asked "{previous[0]}" and heard "{previous[1]}". ' if previous else ""
        message = (
            f"Follow-up from the person wearing the camera, who has memory loss and is looking for their {item}. "
            f'{earlier}Now they say: "{question}". Answer from the photo and what you said about it, as one friendly '
            f"spoken sentence under 100 characters, in plain text, not JSON. Directions are from the camera's view: "
            f"its left is their left."
        )
        reply = (await self._call("follow-up", "POST", f"/conversations/{conversation_id}/messages",
                                  json={"message": message})).json()
        text = _reply_text(reply)
        ctx = parse_context(text) if text.lstrip().startswith("{") else None  # an agent that kept to JSON
        return (ctx["summary"] if ctx else clean(text)) or ""


def _reply_text(reply: dict) -> str:
    return (reply.get("fullAIResponse") or {}).get("outputText") or reply.get("message") or ""


def context_note(ctx: dict, seen_at: datetime | None = None) -> str:
    """The stored JSON context as a line for the answer model, so a follow-up has what the camera saw."""
    when = f" ({seen_at:%H:%M} UTC)" if seen_at else ""
    note = f"What the camera saw{when}: {ctx.get('summary', '')}"
    if ctx.get("nearby"):
        note += f" Next to: {', '.join(ctx['nearby'])}."
    if ctx.get("room"):
        note += f" Room: {ctx['room']}."
    if ctx.get("visible_text"):
        note += f" Readable nearby: {ctx['visible_text']}."
    return note


async def save_context(db: MemoryDB, ai: AIServices, device_id: str, label: str, ctx: dict,
                       image: bytes, image_type: str) -> str:
    """The context as a memory of its own: transcript "LastScene: <summary>", the frame, and its chunks."""
    record = await db.insert(device_id, f"{PREFIX}{ctx['summary']}", image, image_type)
    chunks = chunks_for(label, ctx)
    await db.set_scene(record.id, chunks[1][1])
    await db.add_items(record.id, [(label, place_of(ctx))])
    try:
        embeddings = await ai.embed_many([text for _, text in chunks])
        await db.add_chunks(record.id, [(kind, text, emb) for (kind, text), emb in zip(chunks, embeddings)])
    except Exception:  # the memory and its item are saved; only its search vectors are missing
        log.exception("RAIA context for %s not indexed, memory %s", label, record.id)
    return record.id


async def read_item(
    db: MemoryDB, ai: AIServices, raia: Raia, settings: Settings, device_id: str, scope: str, label: str,
    confidence: float, bbox: list[float] | None, scene_label: str | None, image: bytes, image_type: str,
    captured_at: datetime, via: str,
) -> str | None:
    """One item's frame to raia, if it isn't in its cooldown: the saved memory's id, or None."""
    if not await db.claim_raia_context(scope, label, None, settings.raia_cooldown_minutes):
        log.info("RAIA skip %s (%.2f, %s): read in the last %g min", label, confidence, via,
                 settings.raia_cooldown_minutes)
        return None
    log.info("RAIA prompt sent: %s (%.2f, %s) box %s, scope %s, device %s, captured %s",
             label, confidence, via, bbox, scope, device_id, captured_at.isoformat())
    started = time.perf_counter()
    try:
        reply, conversation_id = await raia.describe(scope, label, bbox, scene_label, image, image_type, captured_at)
    except Exception as exc:
        log.warning("RAIA prompt failed: %s (%s), after %.1f s: %s; trying again in %d min",
                    label, via, time.perf_counter() - started, exc, RETRY_MINUTES)
        await db.finish_raia_context(scope, label, None, str(exc)[:500])
        await db.retry_raia_context(scope, label, settings.raia_cooldown_minutes, RETRY_MINUTES)
        return None
    seconds = time.perf_counter() - started
    ctx = parse_context(reply)
    if ctx is None:
        log.info("RAIA reply: %s not visible (%s), %.1f s; nothing saved", label, via, seconds)
        await db.finish_raia_context(scope, label, None, "not visible")
        return None
    memory_id = await save_context(db, ai, device_id, label, ctx, image, image_type)
    await db.finish_raia_context(scope, label, json.dumps(ctx), None, memory_id, conversation_id)
    log.info("RAIA reply: %s (%s), %.1f s, saved memory %s: %s", label, via, seconds, memory_id, json.dumps(ctx))
    return memory_id


def candidates(obs: Observation, min_confidence: float) -> dict[str, DetectedObject]:
    """The most confident box of each label seen at min_confidence or more."""
    best: dict[str, DetectedObject] = {}
    for o in obs.objects:
        if o.confidence >= min_confidence and (o.label not in best or o.confidence > best[o.label].confidence):
            best[o.label] = o
    return best


async def extract_contexts(
    db: MemoryDB, ai: AIServices, raia: Raia, settings: Settings, device_id: str, memory_id: str,
    obs: Observation, image: bytes, image_type: str,
) -> list[str]:
    """For an observation, best effort, after the response: the labels whose context was saved. Never raises."""
    try:
        seen = candidates(obs, settings.raia_min_confidence)
        if not seen or not raia.configured:
            return []
        scope, device_ids = await scope_for(db, device_id)
        logged = await db.logged_items(scope, device_ids)
        saved = []
        for label, o in seen.items():
            if label not in logged:
                log.info("RAIA skip %s (%.2f, observation): not one of the user's items, memory %s",
                         label, o.confidence, memory_id)
                continue
            if await read_item(db, ai, raia, settings, device_id, scope, label, o.confidence, o.bbox,
                               obs.scene.label if obs.scene else None, image, image_type, obs.captured_at,
                               f"observation {memory_id}"):
                saved.append(label)
        return saved
    except Exception:
        log.exception("RAIA context extraction failed, memory %s", memory_id)
        return []


async def context_from_photos(
    db: MemoryDB, ai: AIServices, raia: Raia, settings: Settings, device_id: str, item: str, view,
    photos: list[tuple[bytes, str]], via: str,
) -> str | None:
    """For a wake command's photo check (`view`: a PhotoCheck, or the task still working it out), best effort:
    the saved memory's id, or None. Never raises."""
    try:
        if asyncio.isfuture(view):
            view = await view
        label = norm(item or "")
        if not raia.configured or not label or not photos:
            return None
        if view is None:
            log.info("RAIA skip %s (%s): no photo check", label, via)
            return None
        confidence = float(getattr(view, "confidence", 0.0) or 0.0)
        if not view.visible or confidence < settings.raia_min_confidence:
            log.info("RAIA skip %s (%s): photo check says visible=%s, confidence %.2f (needs %.2f)",
                     label, via, view.visible, confidence, settings.raia_min_confidence)
            return None
        scope, device_ids = await scope_for(db, device_id)
        if label not in await db.logged_items(scope, device_ids):
            log.info("RAIA skip %s (%.2f, %s): not one of the user's items", label, confidence, via)
            return None
        best = view.best_photo if 0 <= view.best_photo < len(photos) else 0
        image, image_type = photos[best]
        return await read_item(db, ai, raia, settings, device_id, scope, label, confidence, None, None,
                               image, image_type, datetime.now(timezone.utc), f"{via}, photo {best}")
    except Exception:
        log.exception("RAIA context from %s photos failed", via)
        return None
