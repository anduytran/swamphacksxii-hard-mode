"""Upload observations to LastScene with an on-disk queue, rate limit and safe retries.

Queue layout (survives restarts; flushed in time order, one item at a time):
    <queue_dir>/pending/<captured_ms>_<observation_id>/{observation.json, image.jpg}
    <queue_dir>/pending/<found_ms>_found-<uuid>/found.json     a queued "found it" (see queue_found)
    <queue_dir>/sent/...        after 200/201, plus response.json
    <queue_dir>/failed/...      after 422 (bad field), an unfixable 413, or a found that got 404, plus error.json
    <queue_dir>/superseded/...  single-label sightings captured at/before that label's "found it" (audit only, never sent)

Behavior per response:
    201 / 200   sent (200 = the server already had this observation_id: a safe duplicate)
    429         wait (Retry-After if given) and keep it at the head of the queue
    5xx, network error   keep queued, exponential backoff, stop this flush (keeps order)
    401         fetch a fresh token once and retry; if it fails again, stop (credentials problem)
    403         stop without a new token: the token is valid but lacks a scope (or the pairing code
                is wrong); refetching would only burn the Auth0 M2M token quota
    3xx         configuration error (wrong API base; requests would drop the bearer token on the
                redirect): stop, keep everything queued; the background thread stops retrying
    413         re-encode smaller (up to 3 times) and retry; else move to failed/
    404 (found) the server has no sighting of that label: nothing to mark, move to failed/
    422         move to failed/ with the server's message (needs a code or data fix)
Rate limit: at most one POST every `min_interval_s` (contract: one observation per 2 s per device).
"""

import datetime
import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)
FOUND_NAMESPACE = uuid.UUID("5b0b6f0e-4d0e-4a34-9d7c-1f0c3a4e2f10")  # found-<uuid5(label, found ms)>
CONFLICT_NAMESPACE = uuid.UUID("9a4c2d7e-61f3-4b8a-8e2d-3c5f7a9b1d04")  # new id after a 409


class RedirectError(RuntimeError):
    """The API answered 3xx: the configured base URL is not the final one."""


def normalize_api_base(api_base: str) -> str:
    """https://lastscene.photo 308-redirects to www, and requests drops Authorization on that hop (-> 401)."""
    parts = urlsplit(api_base.rstrip("/"))
    if parts.hostname == "lastscene.photo":
        fixed = urlunsplit(parts._replace(netloc="www.lastscene.photo" + (f":{parts.port}" if parts.port else "")))
        log.warning("API base %s redirects to www (bearer token would be dropped); using %s", api_base, fixed)
        return fixed
    return parts.geturl()


class Uploader:
    def __init__(self, api_base: str, pairing_code: str, token_provider, queue_dir, min_interval_s: float = 2.05,
                 timeout: float = 15, max_backoff_s: float = 60):
        if not api_base:
            raise RuntimeError("LASTSCENE_API_BASE is not set; refusing to guess an endpoint")
        self.api_base = normalize_api_base(api_base)
        self.pairing_code = pairing_code
        self.tokens = token_provider
        self.queue = Path(queue_dir)
        for sub in ("pending", "sent", "failed", "superseded"):
            (self.queue / sub).mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        self.max_backoff_s = max_backoff_s
        self._last_post = 0.0
        self._next_try = 0.0
        self._backoff = 1.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.config_error = None  # set on a 3xx; the background thread stops retrying
        self.http_status = {}     # item id -> last HTTP status (observation_id or found-<uuid>)
        self._found_ms = {}       # label -> latest found_at (ms) this process queued; in memory only
        self._token_rejected = False  # a fresh token got 401 too: no refetch per retry (Auth0 M2M quota)
        self.log = []  # (time, item id, status/text) for reporting
        self.on_result = None  # optional callable(summary, item_id, status, http_status) after each send

    # --- queue -----------------------------------------------------------------------
    def enqueue(self, observation: dict, image_bytes: bytes) -> str:
        oid = observation["observation_id"]
        captured_ms = _iso_to_ms(observation["captured_at"])
        if self._find(oid):
            return oid  # already queued, sent or superseded: retries reuse the same id
        labels = {o["label"] for o in observation["objects"]}
        stale = len(labels) == 1 and captured_ms <= self._found_ms.get(next(iter(labels)), -1)
        sub = "superseded" if stale else "pending"  # e.g. a frame from before "found it" that finished inference after it
        tmp = self.queue / sub / f".tmp_{oid}"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "observation.json").write_text(json.dumps(observation, indent=2), encoding="utf-8")
        (tmp / "image.jpg").write_bytes(image_bytes)
        tmp.rename(self.queue / sub / f"{captured_ms:014d}_{oid}")  # atomic: never a half-written item
        if stale:
            self.log.append((time.time(), oid, "superseded: captured before that item was found"))
        return oid

    def queue_found(self, item_label: str, found_at_iso: str) -> dict:
        """Queue "found it" behind everything already queued, and drop the sightings it makes stale.

        Under the flush lock (so no observation is mid-POST): every pending observation whose objects
        are ALL this label and that was captured at or before found_at moves to superseded/; uploaded
        after the found action, it would reopen the item server-side. Observations that also show
        other labels stay queued: they sort before the found action (captured earlier), so they are
        sent first and the found action still lands last. The found action itself is a durable queue
        item, sent by flush() with the same ordering, retries, backoff and spacing as observations.
        (A mixed-label sighting from before found_at that is enqueued after the found action was SENT
        still uploads, and the real backend's upsert reopens the item.)
        Returns {"id": "found-<uuid>", "superseded": [observation ids]}.
        """
        from lastscene_bridge.observation import item_label as normalize

        label = normalize(item_label)
        found_ms = _iso_to_ms(found_at_iso)
        fid = f"found-{uuid.uuid5(FOUND_NAMESPACE, f'{label}|{found_ms}')}"
        superseded = []
        with self._lock:
            self._found_ms[label] = max(found_ms, self._found_ms.get(label, found_ms))  # enqueue() checks it too
            if not self._find(fid):  # found first: a crash before the moves below leaves stale sightings
                tmp = self.queue / "pending" / f".tmp_{fid}"  # queued AHEAD of it (harmless), never a lost found
                tmp.mkdir(parents=True, exist_ok=True)
                (tmp / "found.json").write_text(json.dumps({"label": label, "found_at": found_at_iso}, indent=2),
                                                encoding="utf-8")
                tmp.rename(self.queue / "pending" / f"{found_ms:014d}_{fid}")
            for item in self.pending():
                if _is_found(item):
                    continue
                obs = json.loads((item / "observation.json").read_text(encoding="utf-8"))
                if all(o["label"] == label for o in obs["objects"]) and _iso_to_ms(obs["captured_at"]) <= found_ms:
                    shutil.move(str(item), str(self.queue / "superseded" / item.name))
                    superseded.append(_item_id(item))
        if superseded:
            self.log.append((time.time(), fid, f"found {label!r}: {len(superseded)} queued sighting(s) superseded"))
            log.info("found %r: %d queued sighting(s) moved to superseded/", label, len(superseded))
        return {"id": fid, "superseded": superseded}

    def pending(self) -> list[Path]:
        """Time order; on the same millisecond, observations go before a found action."""
        items = [p for p in (self.queue / "pending").iterdir() if p.is_dir() and not p.name.startswith(".tmp_")]
        return sorted(items, key=lambda p: (p.name.split("_", 1)[0], _is_found(p), p.name))

    def is_pending(self, item_id: str) -> bool:
        return any((self.queue / "pending").glob(f"*_{item_id}"))

    def _find(self, oid):
        for sub in ("pending", "sent", "failed", "superseded"):
            for p in (self.queue / sub).glob(f"*_{oid}"):
                return p
        return None

    # --- sending -------------------------------------------------------------------------
    def flush(self, max_items: int | None = None) -> list[tuple[str, str]]:
        """Send pending items in order, head first. Stops at the first transient failure (keeps order).

        The lock is taken per item, so queue_found() can slot in between two POSTs; the head is
        re-read every time, so a found action queued meanwhile is sent in its place in the order.
        """
        results = []
        while max_items is None or len(results) < max_items:
            with self._lock:
                head = self.pending()[:1]
                if not head:
                    break
                summary = _summary(head[0])
                self._sent_id = _item_id(head[0])
                status = self._send(head[0])
            results.append((_item_id(head[0]), status))
            if self.on_result:
                self.on_result(summary, self._sent_id, status, self.http_status.get(self._sent_id))
            if status in ("retry", "auth", "config"):
                break
        return results

    def _wait_rate_limit(self):
        wait = self._last_post + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Every API call: bearer token + pairing code, and no silent redirects (see RedirectError)."""
        url = self.api_base + path
        headers = {"Authorization": f"Bearer {self.tokens.get()}", "X-Pairing-Code": self.pairing_code}
        r = requests.request(method, url, headers=headers, timeout=self.timeout, allow_redirects=False, **kwargs)
        if 300 <= r.status_code < 400:
            raise RedirectError(f"{method} {url} answered {r.status_code} redirect to {r.headers.get('Location')!r}; "
                                "set LASTSCENE_API_BASE/API_URL to the final URL (the bearer token is not forwarded)")
        return r

    def _post(self, item: Path, image_bytes: bytes | None):
        self._wait_rate_limit()
        try:
            if image_bytes is None:  # found action
                found = json.loads((item / "found.json").read_text(encoding="utf-8"))
                return self._request("POST", f"/api/v1/items/{quote(found['label'], safe='')}/found",
                                     json={"found_at": found["found_at"]})
            obs_text = (item / "observation.json").read_text(encoding="utf-8")
            files = {"image": ("frame.jpg", image_bytes, "image/jpeg"),
                     "observation": (None, obs_text, "application/json")}
            return self._request("POST", "/api/v1/observations", files=files)
        finally:
            self._last_post = time.monotonic()

    def _send(self, item: Path) -> str:
        oid = _item_id(item)
        found = _is_found(item)
        image = None if found else (item / "image.jpg").read_bytes()
        refreshed = reassigned = False
        shrinks = rate_limited = 0
        while True:
            try:
                r = self._post(item, image)
            except (requests.ConnectionError, requests.Timeout) as exc:
                return self._transient(oid, f"network: {type(exc).__name__}")
            except requests.HTTPError as exc:  # the Auth0 token request itself was refused
                self.log.append((time.time(), oid, f"token request failed: {exc}"))
                self._transient(oid, "token request failed")
                return "auth"
            except RedirectError as exc:
                self.config_error = str(exc)
                self.log.append((time.time(), oid, f"configuration error, not retrying: {exc}"))
                log.error("%s", exc)
                return "config"
            except requests.RequestException as exc:  # e.g. a truncated chunked body: don't kill the flush thread
                return self._transient(oid, f"network: {type(exc).__name__}")
            self.http_status[oid] = r.status_code
            if r.status_code in (200, 201):
                (item / "response.json").write_text(r.text, encoding="utf-8")
                shutil.move(str(item), str(self.queue / "sent" / item.name))
                self._backoff = 1.0
                self.config_error = None
                self._token_rejected = False
                self.log.append((time.time(), oid, f"{r.status_code} sent"))
                return "duplicate" if r.status_code == 200 and not found else "sent"
            if r.status_code == 401 and not refreshed and not self._token_rejected:  # expired/revoked: one fresh fetch
                self.tokens.invalidate()
                refreshed = True
                continue
            if r.status_code in (401, 403):  # 403: valid token, missing scope/pairing; a new token won't help
                self._token_rejected = self._token_rejected or r.status_code == 401  # until a send succeeds or it expires
                why = "credentials rejected" if r.status_code == 401 else "forbidden (scope or pairing code)"
                self.log.append((time.time(), oid, f"{r.status_code} {why}: {r.text[:200]}"))
                self._transient(oid, why)  # back off instead of hammering Auth0 or the API
                return "auth"
            if r.status_code == 429:
                rate_limited += 1
                if rate_limited > 3:
                    return self._transient(oid, "still rate limited")
                retry_after = _float(r.headers.get("Retry-After"), self.min_interval_s)
                self.log.append((time.time(), oid, f"429 rate limited, waiting {retry_after:.1f}s"))
                time.sleep(min(retry_after, self.max_backoff_s))
                continue
            if r.status_code == 413 and not found and shrinks < 3:  # server limit tighter than expected: shrink (bbox is normalized)
                image = _shrink(image)
                (item / "image.jpg").write_bytes(image)
                _update_image_size(item, image)
                shrinks += 1
                continue
            if r.status_code == 409 and not found and not reassigned:  # id taken by another device: new id, once
                item, oid = self._reassign_id(item, oid)
                self._sent_id = oid
                reassigned = True
                continue
            if found and r.status_code == 404:  # no last-seen row for that label: nothing to mark, don't retry
                (item / "error.json").write_text(json.dumps({"status": 404, "body": r.text}), encoding="utf-8")
                shutil.move(str(item), str(self.queue / "failed" / item.name))
                self.log.append((time.time(), oid, f"404 found not recorded, server has no sighting: {r.text[:200]}"))
                log.warning("found %s: server has no sighting of that label (404); not retried", oid)
                return "no_sighting"
            if 400 <= r.status_code < 500:
                (item / "error.json").write_text(json.dumps({"status": r.status_code, "body": r.text}), encoding="utf-8")
                shutil.move(str(item), str(self.queue / "failed" / item.name))
                self.log.append((time.time(), oid, f"{r.status_code} failed: {r.text[:200]}"))
                return "failed"
            return self._transient(oid, f"server {r.status_code}")

    def _reassign_id(self, item: Path, oid: str) -> tuple[Path, str]:
        """409: another device already used this observation_id. Derive a new one (stable across retries
        and restarts: from the old id + this device's pairing code), rewrite the item, log it."""
        new = str(uuid.uuid5(CONFLICT_NAMESPACE, f"{oid}|{self.pairing_code}"))
        obs = json.loads((item / "observation.json").read_text(encoding="utf-8"))
        obs["observation_id"] = new
        (item / "observation.json").write_text(json.dumps(obs, indent=2), encoding="utf-8")
        target = item.with_name(f"{item.name.split('_', 1)[0]}_{new}")
        item.rename(target)
        self.log.append((time.time(), oid, f"409 observation_id used by another device; resending as {new}"))
        log.warning("observation %s: id used by another device (409); resending as %s", oid, new)
        return target, new

    def _transient(self, oid, why):
        self.log.append((time.time(), oid, f"retry later ({why}), backoff {self._backoff:.0f}s"))
        self._next_try = time.monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, self.max_backoff_s)
        return "retry"

    # --- background flushing ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, poll_s: float = 1.0):
        def loop():
            while not self._stop.is_set():
                if not self.config_error and self.pending() and time.monotonic() >= self._next_try:
                    self.flush()
                self._stop.wait(poll_s)
        self._thread = threading.Thread(target=loop, name="lastscene-uploader", daemon=True)
        self._thread.start()

    def stop(self, drain_timeout_s: float = 30):
        deadline = time.monotonic() + drain_timeout_s
        while self.pending() and time.monotonic() < deadline and not self.config_error:
            if time.monotonic() >= self._next_try:
                if any(s in ("retry", "auth", "config") for _, s in self.flush()):
                    break
            time.sleep(0.2)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # --- read-only / manual contract endpoints (not queued) ------------------------------------
    def last_seen(self, item_label: str | None = None) -> requests.Response:
        from lastscene_bridge.observation import item_label as normalize
        item_label = quote(normalize(item_label), safe="") if item_label else None
        return self._request("GET", f"/api/v1/items/{item_label}/last-seen" if item_label else "/api/v1/items/last-seen")

    def delete_observation(self, observation_id: str) -> requests.Response:
        return self._request("DELETE", f"/api/v1/observations/{observation_id}")


def _item_id(item: Path) -> str:
    return item.name.split("_", 1)[1]


def _summary(item: Path) -> str:
    """For per-post log lines: "keys:placed, wallet:seen" or "found keys"."""
    try:
        if _is_found(item):
            return "found " + json.loads((item / "found.json").read_text(encoding="utf-8"))["label"]
        obs = json.loads((item / "observation.json").read_text(encoding="utf-8"))
        return ", ".join(f"{o['label']}:{o['event']}" for o in obs["objects"])
    except (OSError, ValueError, KeyError):
        return "?"


def _is_found(item: Path) -> bool:
    return _item_id(item).startswith("found-")


def _iso_to_ms(stamp: str) -> int:
    """Exact integer ms (no float rounding), so captured_at <= found_at compares cleanly."""
    when = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (when - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)) // datetime.timedelta(milliseconds=1)


def _float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _shrink(image_bytes: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def _update_image_size(item: Path, image_bytes: bytes):
    """Keep image_size truthful after re-encoding (bboxes are normalized, so they stay valid)."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    obs = json.loads((item / "observation.json").read_text(encoding="utf-8"))
    obs["image_size"] = {"width": img.shape[1], "height": img.shape[0]}
    (item / "observation.json").write_text(json.dumps(obs, indent=2), encoding="utf-8")


def uploader_from_env(queue_dir, env=None):
    import os

    from lastscene_bridge.auth import token_provider_from_env

    env = env if env is not None else os.environ
    return Uploader(env.get("LASTSCENE_API_BASE") or env.get("API_URL") or "https://www.lastscene.photo", env.get("LASTSCENE_PAIRING_CODE") or env.get("PAIRING_CODE", ""),
                    token_provider_from_env(env), queue_dir)
