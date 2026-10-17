r"""Step-by-step smoke test of the DEPLOYED integration (Auth0 -> observations -> last-seen -> found).

Run it once Dylan has shared the wearable's device credentials (env file outside the repo):

    # make one observation without sending it (writes <queue>\outbox\<observation_id>.json + .jpg)
    python -m lastscene_bridge.run --source <clip or stream> --preset m960d --fast --dry-run --queue <scratch>\q
    # then, from vision\
    python scripts\live_smoke.py --env-file %USERPROFILE%\.lastscene\bridge.env                      (step 1 only)
    python scripts\live_smoke.py <scratch>\q\outbox\<id>.json --env-file ...                          (steps 1-4)
    python scripts\live_smoke.py <...>.json --env-file ... --found                                    (+ found)
    python scripts\live_smoke.py <...>.json --env-file ... --found --reopen-test                      (+ reopen bug check)
    python scripts\live_smoke.py <...>.json --env-file ... --found --reopen-test --dry-run            (print, send nothing)

Steps (each prints PASS/FAIL, the HTTP status and one reason; the first FAIL stops the run and says
what to request from whom):
    1 auth        Auth0 client-credentials token; prints iss/aud/scope/gty/exp from the JWT payload
                  (decoded, NOT verified; the token and secret are never printed), then one read-only
                  GET to check the backend accepts it.
    2 post        POST /api/v1/observations with the dry-run JSON + the .jpg beside it (201, or 200 duplicate).
    3 last-seen   GET /api/v1/items/{label}/last-seen: bbox, captured_at, confidence, maybe, and the signed
                  photo (a JPEG whose decoded size equals image_size).
    4 dashboard   the manual check to do in the browser.
    --found       POST /api/v1/items/{label}/found {"found_at": now}, then last-seen must show found_at.
    --reopen-test WRITES A SYNTHETIC TEST RECORD TO PRODUCTION and deletes it again: the same JPEG under a new
                  observation_id, captured between the stored sighting and found_at. If last-seen reopens
                  (found_at -> null), the backend's upsert_last_seen bug is reproduced. Cleanup always runs
                  (DELETE, re-check, re-POST found if the rebuild dropped found_at).

Exit code 0 only if every requested step passed. HTTP goes through `requests` directly with
allow_redirects=False (a redirect would drop the bearer token), independent of the bridge's uploader,
so this tests the wire contract itself. Environment: the bridge's names and aliases (see
lastscene_bridge/auth.py): LASTSCENE_API_BASE / API_URL (default https://www.lastscene.photo; the apex
lastscene.photo is rewritten to www), LASTSCENE_PAIRING_CODE / PAIRING_CODE, AUTH0_DOMAIN,
AUTH0_AUDIENCE, AUTH0_CLIENT_ID / DEVICE_CLIENT_ID, AUTH0_CLIENT_SECRET / DEVICE_CLIENT_SECRET.
"""

import argparse
import base64
import copy
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lastscene_bridge.auth import token_provider_from_env  # noqa: E402
from lastscene_bridge.observation import _NAMESPACE  # noqa: E402  (same namespace as real observation ids)
from lastscene_bridge.run import load_env_file  # noqa: E402

DEFAULT_BASE = "https://www.lastscene.photo"
DEVICE_SCOPE = "write:memories"
DEVICE_AUDIENCE = "https://api.lastscene"
BBOX_TOL = 1e-4          # backend stores real (float32) and rounds bbox to 5 digits
CONF_TOL = 1e-4          # ... and confidence to 4
MAYBE_CONFIDENCE = 0.5   # backend default (MAYBE_CONFIDENCE); "maybe" = confidence below it
DASHBOARD_HOURS = 24     # dashboard history window (DASHBOARD_HISTORY_HOURS)
MIN_POST_SPACING_S = 2.05  # same spacing as the bridge; the backend allows one observation per 2 s
TIMEOUT_S = 45           # Vercel cold start + JWKS + DB connect + embedding can be slow


class StepFailed(Exception):
    pass


# ------------------------------------------------------------------ pure helpers (unit-tested)

def normalize_base(url: str | None) -> str:
    """https:// by default, no trailing slash or /api/v1, and lastscene.photo -> www.lastscene.photo
    (the apex 308-redirects to www and requests drops the Authorization header on that hop)."""
    url = (url or "").strip() or DEFAULT_BASE
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if path.endswith("/api/v1"):
        path = path[: -len("/api/v1")]
    netloc = parts.netloc.lower()
    if parts.hostname == "lastscene.photo":
        netloc = "www.lastscene.photo" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def decode_jwt_claims(token: str) -> dict:
    """The JWT payload, base64url-decoded WITHOUT verifying the signature (for display only)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT (expected header.payload.signature)")
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    if not isinstance(claims, dict):
        raise ValueError("JWT payload is not a JSON object")
    return claims


def claim_problems(claims: dict, audience: str | None, now: float | None = None) -> list[str]:
    """What the backend's require_device would reject (backend/app/auth.py), as readable reasons."""
    now = time.time() if now is None else now
    problems = []
    scopes = str(claims.get("scope", "")).split()
    if DEVICE_SCOPE not in scopes:
        problems.append(f"scope has no {DEVICE_SCOPE} (got {' '.join(scopes) or 'none'})")
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    if audience and audience not in auds:
        problems.append(f"aud is {aud!r}, expected {audience!r}")
    if claims.get("gty") != "client-credentials" and not str(claims.get("sub", "")).endswith("@clients"):
        problems.append("not a machine-to-machine token (gty is not client-credentials)")
    if "exp" in claims and float(claims["exp"]) <= now:
        problems.append("token already expired")
    return problems


def parse_time(value: str) -> datetime:
    """ISO 8601 with Z or an offset -> aware UTC datetime. Naive times are refused (no guessing)."""
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00").replace("z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"time without a timezone: {value!r}")
    return dt.astimezone(timezone.utc)


def iso_ms(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def same_instant(a: str, b: str) -> bool:
    return parse_time(a) == parse_time(b)


def bbox_matches(a, b, tol: float = BBOX_TOL) -> bool:
    return a is not None and b is not None and len(a) == len(b) == 4 and all(
        abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def pick_reopen_captured_at(stored: datetime, found: datetime) -> datetime:
    """A capture time strictly between the stored sighting and found_at, on a whole millisecond
    (the bridge's precision): the case of a sighting queued offline before 'found it'."""
    lo, hi = stored.astimezone(timezone.utc), found.astimezone(timezone.utc)
    mid = lo + (hi - lo) / 2
    mid = mid.replace(microsecond=mid.microsecond // 1000 * 1000)
    if not lo < mid < hi:
        raise ValueError(f"no whole millisecond strictly between {iso_ms(lo)} and {iso_ms(hi)}")
    return mid


def reopen_observation_id(observation_id: str) -> str:
    """Stable per original observation, so a leftover from an interrupted run can be found and deleted."""
    return str(uuid.uuid5(_NAMESPACE, f"{observation_id}|smoke-reopen"))


def last_seen_problems(obs: dict, got: dict) -> list[str]:
    """Differences between the observation we sent and the backend's last-seen record for it."""
    problems = []
    if got.get("observation_id") != obs["observation_id"]:
        problems.append(f"observation_id is {got.get('observation_id')}, sent {obs['observation_id']}")
    label = got.get("label")
    sent = max((o for o in obs["objects"] if o["label"] == label), key=lambda o: o["confidence"], default=None)
    if sent is None:
        return problems + [f"label {label!r} is not in the observation"]
    if not bbox_matches(sent["bbox"], got.get("bbox")):
        problems.append(f"bbox {got.get('bbox')} != sent {sent['bbox']} (tol {BBOX_TOL})")
    if not got.get("captured_at") or not same_instant(obs["captured_at"], got["captured_at"]):
        problems.append(f"captured_at {got.get('captured_at')} != sent {obs['captured_at']}")
    if got.get("confidence") is None or abs(float(got["confidence"]) - sent["confidence"]) > CONF_TOL:
        problems.append(f"confidence {got.get('confidence')} != raw detector confidence {sent['confidence']}")
    if got.get("image_size") != obs["image_size"]:
        problems.append(f"image_size {got.get('image_size')} != sent {obs['image_size']}")
    if not isinstance(got.get("maybe"), bool):
        problems.append(f"maybe is {got.get('maybe')!r}, expected a bool")
    return problems


def mask(secret: str) -> str:
    return "(not set)" if not secret else "*" * max(3, len(secret) - 2) + secret[-2:]


# ------------------------------------------------------------------ the run

class Smoke:
    def __init__(self, base: str, env: dict, obs: dict | None, jpeg: bytes | None, fresh_token: bool = False,
                 reopen_test: bool = False):
        self.base, self.env, self.obs, self.jpeg = base, env, obs, jpeg
        self.fresh_token, self.reopen_test = fresh_token, reopen_test
        self.pairing = env.get("LASTSCENE_PAIRING_CODE") or env.get("PAIRING_CODE", "")
        self.audience = env.get("AUTH0_AUDIENCE")
        self.label = obs["objects"][0]["label"] if obs else None
        self.http = requests.Session()
        self.token = None
        self.passed, self.failed = 0, 0
        self.post_status = None
        self.last_seen = None
        self.found_at = None
        self._last_obs_post = None

    # -------------------------------------------------------------- output
    def result(self, ok: bool, step: str, status, reason: str, ask: str | None = None):
        print(f"{'PASS' if ok else 'FAIL'}  {step:<14} HTTP {status if status is not None else '-':<4} {reason}")
        if ok:
            self.passed += 1
            return
        self.failed += 1
        if ask:
            print(f"      what to request from whom: {ask}")

    def fail(self, step, status, reason, ask):
        self.result(False, step, status, reason, ask)
        raise StepFailed(step)

    @staticmethod
    def note(text: str):
        print(f"      {text}")

    # -------------------------------------------------------------- http
    def url(self, path: str) -> str:
        return self.base + path

    def send(self, step: str, method: str, url: str, auth: bool = True, **kw) -> requests.Response:
        headers = kw.pop("headers", {})
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        shown = url.split("?")[0]  # the signed photo link's query works like a password: don't print it
        try:
            r = self.http.request(method, url, headers=headers, timeout=TIMEOUT_S, allow_redirects=False, **kw)
        except requests.RequestException as exc:
            self.fail(step, None, f"{method} {shown}: network error {type(exc).__name__}",
                      "Adam: check the network and the base URL (LASTSCENE_API_BASE / API_URL)")
        if 300 <= r.status_code < 400:
            self.fail(step, r.status_code, f"{method} {shown} redirects to {r.headers.get('Location')} "
                      "(requests would drop the bearer token on that hop)",
                      "Adam: set LASTSCENE_API_BASE to the final URL (https://www.lastscene.photo)")
        return r

    def post_observation(self, step: str, obs: dict) -> requests.Response:
        """One observation POST, spaced like the bridge and honoring one 429 Retry-After."""
        for attempt in range(2):
            if self._last_obs_post is not None:
                time.sleep(max(0.0, MIN_POST_SPACING_S - (time.monotonic() - self._last_obs_post)))
            headers = {"X-Pairing-Code": self.pairing} if self.pairing else {}
            r = self.send(step, "POST", self.url("/api/v1/observations"), headers=headers,
                          files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
                          data={"observation": json.dumps(obs)})
            self._last_obs_post = time.monotonic()
            if r.status_code != 429 or attempt:
                return r
            wait = min(10.0, float(r.headers.get("Retry-After") or 2))
            self.note(f"429 (another upload from this device within 2 s); waiting {wait:g} s, then one retry")
            time.sleep(wait)
        return r

    def get_last_seen(self, step: str) -> requests.Response:
        return self.send(step, "GET", self.url(f"/api/v1/items/{quote(self.label, safe='')}/last-seen"))

    def post_found(self, step: str, found_at: datetime) -> requests.Response:
        return self.send(step, "POST", self.url(f"/api/v1/items/{quote(self.label, safe='')}/found"),
                         json={"found_at": iso_ms(found_at)})

    def common_failure(self, step: str, r: requests.Response):
        """Statuses that mean the same thing on every authenticated route."""
        d = detail(r)
        if r.status_code == 401:
            self.fail(step, 401, f"backend rejected the token: {d}",
                      "Dylan: Vercel AUTH0_DOMAIN / AUTH0_AUDIENCE must match the token's iss / aud printed in step 1")
        if r.status_code == 403:
            self.fail(step, 403, d, "Dylan: authorize the M2M app for audience https://api.lastscene with "
                      "write:memories (then rerun with --fresh-token)" if "scope" in d.lower() else
                      "Dylan: these must be the Wearable M2M app's credentials, not the dashboard SPA's")
        if r.status_code >= 500:
            self.fail(step, r.status_code, f"server error: {d}",
                      "Dylan: check the Vercel function logs for this request (DB / Gemini / timeout)")

    # -------------------------------------------------------------- steps
    def step_auth(self):
        try:
            provider = token_provider_from_env(self.env)
        except (RuntimeError, ValueError) as exc:
            self.fail("1 auth", None, f"configuration: {exc}",
                      "Dylan: send DEVICE_CLIENT_ID + DEVICE_CLIENT_SECRET (the Wearable M2M app), AUTH0_DOMAIN, "
                      "AUTH0_AUDIENCE and PAIRING_CODE privately; Adam keeps them in an env file outside the repo")
        if self.fresh_token:
            provider.invalidate()
        try:
            self.token = provider.get()
        except requests.HTTPError as exc:
            r = exc.response
            d = detail(r, ("error_description", "error"))
            ask = {401: "Dylan: the client ID / secret pair is wrong; re-send the Wearable M2M app's credentials",
                   403: "Dylan: authorize the M2M app for audience https://api.lastscene (with write:memories) in Auth0",
                   429: "wait: Auth0 rate limit / M2M token quota reached"}.get(
                r.status_code, "Adam: check AUTH0_DOMAIN and AUTH0_AUDIENCE")
            self.fail("1 auth", r.status_code, f"Auth0 refused the token request: {d}", ask)
        except (requests.RequestException, KeyError, ValueError) as exc:
            self.fail("1 auth", None, f"token request failed: {type(exc).__name__}", "Adam: check AUTH0_DOMAIN")
        try:
            claims = decode_jwt_claims(self.token)
        except (ValueError, UnicodeDecodeError) as exc:
            self.fail("1 auth", None, f"token is not a readable JWT: {exc}",
                      "Dylan: the API must issue JWT access tokens (an audience is required)")
        exp = claims.get("exp")
        left = f" ({(float(exp) - time.time()) / 60:.0f} min left)" if exp else ""
        exp_txt = iso_ms(datetime.fromtimestamp(float(exp), timezone.utc)) if exp else "-"
        for k in ("iss", "aud", "scope", "gty"):
            self.note(f"{k:<5} = {claims.get(k)}")
        self.note(f"exp   = {exp_txt}{left}")
        problems = claim_problems(claims, self.audience)
        if problems:
            if any(DEVICE_SCOPE in p for p in problems):
                ask = (f"Dylan: in Auth0 > APIs > {self.audience or DEVICE_AUDIENCE} > Machine to Machine "
                       f"Applications, authorize this client with {DEVICE_SCOPE}; then rerun with --fresh-token "
                       "(the cached token keeps its old scopes)")
            elif any("expired" in p for p in problems):
                ask = "Adam: rerun with --fresh-token"
            else:
                ask = f"Adam: set AUTH0_AUDIENCE={DEVICE_AUDIENCE} and use the Wearable M2M app's credentials"
            self.fail("1 auth", 200, "; ".join(problems), ask)
        self.result(True, "1 auth", 200, f"token has {DEVICE_SCOPE} for {claims.get('aud')}")
        # one read-only request: does the deployed backend accept this token?
        r = self.send("1 auth", "GET", self.url("/api/v1/items/last-seen"))
        self.common_failure("1 auth", r)
        if r.status_code != 200:
            self.fail("1 auth", r.status_code, f"GET /api/v1/items/last-seen: {detail(r)}",
                      "Dylan: check the deployed backend")
        items = r.json().get("items", [])
        self.result(True, "1 backend", 200, f"backend accepts the token ({len(items)} last-seen item(s) in this scope)")

    def step_post(self):
        if not self.pairing:
            self.fail("2 post", None, "no pairing code (LASTSCENE_PAIRING_CODE / PAIRING_CODE)",
                      "Dylan: send the wearable's PAIRING_CODE (the first code a device sends is the one kept)")
        r = self.post_observation("2 post", self.obs)
        self.post_status = r.status_code
        d = detail(r)
        if r.status_code == 201:
            body = r.json()
            self.result(True, "2 post", 201, f"stored as memory {body.get('memory_id')}, items {body.get('items')}")
            return
        if r.status_code == 200:
            body = r.json()
            self.result(True, "2 post", 200, f"duplicate: memory {body.get('memory_id')} already has this "
                        "observation_id (last-seen was not touched by this request)")
            return
        if r.status_code == 422 and "at or above confidence" in d:  # the floor, not a schema error on a confidence
            self.fail("2 post", 422, d, "Dylan: set MIN_DETECTION_CONFIDENCE=0.05 on Vercel and redeploy "
                      "(needs Adam+Dylan OK); real keys score 0.05-0.35")
        if r.status_code == 422 and "future" in d:
            self.fail("2 post", 422, d, "Adam: sync this machine's clock (captured_at is ahead of the server)")
        if r.status_code == 409:
            self.fail("2 post", 409, d, "Adam: this observation_id belongs to another device; make a new "
                      "observation with these credentials")
        if r.status_code == 413:
            self.fail("2 post", 413, d, "Adam: the JPEG is over 2 MiB; use the bridge's (re-encoded) image")
        self.common_failure("2 post", r)
        self.fail("2 post", r.status_code, d, "Adam: the payload breaks the contract; compare it with "
                  "backend/app/observations.py (and send Dylan this message)")

    def step_last_seen(self):
        r = self.get_last_seen("3 last-seen")
        if r.status_code == 404:
            why = ("POST and GET both resolve the scope with scope_for(): the household once the device is claimed, "
                   "else the device itself. After a 201 they disagree only if the device was claimed (or moved) "
                   "between the two calls, or the deployed backend differs from this checkout"
                   if self.post_status == 201 else
                   "the earlier copy of this observation was stored while the device was unclaimed (device scope, "
                   "invisible after the claim) or half-stored (embedding failure); a duplicate never writes last-seen")
            self.fail("3 last-seen", 404, f"{detail(r)}: {why}",
                      "Adam: send a NEW observation (live camera) after the claim; Dylan: check household_devices for "
                      "this client ID and the item_last_seen scope")
        self.common_failure("3 last-seen", r)
        if r.status_code != 200:
            self.fail("3 last-seen", r.status_code, detail(r), "Dylan: check the deployed backend")
        got = r.json()
        rid = reopen_observation_id(self.obs["observation_id"])
        if got.get("observation_id") == rid:  # an interrupted --reopen-test left its synthetic record behind
            if not self.reopen_test:
                self.fail("3 last-seen", 200, f"last-seen is the synthetic test observation {rid} that an "
                          "interrupted --reopen-test left behind",
                          "Adam: rerun with --found --reopen-test (it deletes the leftover first)")
            self.note(f"last-seen is the synthetic test observation {rid} from an interrupted run; deleting it")
            gone = self.send("3 last-seen", "DELETE", self.url(f"/api/v1/observations/{rid}"))
            if gone.status_code != 200:
                self.common_failure("3 last-seen", gone)
                self.fail("3 last-seen", gone.status_code, f"DELETE leftover test observation {rid}: {detail(gone)}",
                          f"Dylan: delete observation {rid} by hand (DELETE /api/v1/observations/{rid})")
            return self.step_last_seen()  # rebuilt from the remaining sightings; step 5 posts found again
        self.last_seen = got
        if got.get("observation_id") != self.obs["observation_id"] and got.get("captured_at") and \
                parse_time(got["captured_at"]) > parse_time(self.obs["captured_at"]):
            self.fail("3 last-seen", 200, f"last-seen is a newer sighting ({got.get('observation_id')} at "
                      f"{got['captured_at']}), so this upload can't be checked against it",
                      "Adam: rerun with a fresher observation (newest capture time wins)")
        problems = last_seen_problems(self.obs, got)
        if problems:
            self.fail("3 last-seen", 200, "; ".join(problems), "Dylan: the stored record differs from what was "
                      "sent (send him this line)")
        expected_maybe = got["confidence"] < MAYBE_CONFIDENCE
        if got["maybe"] != expected_maybe:
            self.note(f"WARN maybe={got['maybe']} but confidence {got['confidence']} vs default "
                      f"MAYBE_CONFIDENCE {MAYBE_CONFIDENCE} says {expected_maybe} (configured differently?)")
        self.result(True, "3 last-seen", 200, f"bbox {got['bbox']}, captured_at {got['captured_at']}, "
                    f"confidence {got['confidence']}, maybe {got['maybe']}, found_at {got.get('found_at')}")
        self.check_photo(got)

    def check_photo(self, got: dict):
        link = got.get("image_url")
        if not link:
            self.fail("3 photo", 200, "last-seen has no image_url", "Dylan: the last-seen frame was not stored")
        if urlsplit(link).netloc != urlsplit(self.base).netloc:
            self.note(f"WARN photo link host {urlsplit(link).netloc} differs from {urlsplit(self.base).netloc}")
        r = self.send("3 photo", "GET", link, auth=False)  # signed link: no bearer token to other URLs
        if r.status_code != 200:
            self.fail("3 photo", r.status_code, f"signed photo link: {detail(r)}",
                      "Dylan: check IMAGE_URL_SECRET on Vercel (403 = signature/expiry, 404 = image missing)")
        data = r.content
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data[:3] == b"\xff\xd8\xff" else None
        if img is None:
            self.fail("3 photo", 200, f"not a JPEG ({r.headers.get('Content-Type')}, {len(data)} bytes)",
                      "Dylan: the stored last-seen image is not the uploaded JPEG")
        size = {"width": img.shape[1], "height": img.shape[0]}
        if size != got.get("image_size"):
            self.fail("3 photo", 200, f"decoded JPEG is {size}, last-seen says {got.get('image_size')}",
                      "Adam + Dylan: image_size must describe the uploaded image (bbox is normalized to it)")
        same = "identical to the uploaded bytes" if data == self.jpeg else "differs from the uploaded bytes"
        self.result(True, "3 photo", 200, f"JPEG {size['width']}x{size['height']}, {len(data)} bytes, {same}")

    def step_dashboard_hint(self):
        got = self.last_seen
        x, y, w, h = got["bbox"]
        W, H = got["image_size"]["width"], got["image_size"]["height"]
        client = self.env.get("AUTH0_CLIENT_ID") or self.env.get("DEVICE_CLIENT_ID") or "(LASTSCENE_DEVICE_TOKEN)"
        print(f"MANUAL 4 dashboard    open {self.base} and check by eye:")
        self.note(f"- Household: this wearable is claimed with Client ID {client} + pairing code {mask(self.pairing)}")
        self.note(f"- Items / Timeline: the photo of {got['label']!r} with its box at normalized [x, y, w, h] = "
                  f"[{x}, {y}, {w}, {h}] (pixels {round(x * W)},{round(y * H)} {round(w * W)}x{round(h * H)} "
                  f"of {W}x{H}){', with a Maybe chip' if got['maybe'] else ''}")
        age = datetime.now(timezone.utc) - parse_time(got["captured_at"])
        if age > timedelta(hours=DASHBOARD_HOURS):
            self.note(f"WARN captured_at {got['captured_at']} is {age.total_seconds() / 3600:.0f} h old: the Timeline "
                      f"only shows the last {DASHBOARD_HOURS} h, so it will not be there (use a live capture)")
        if self.post_status == 200:
            self.note("- step 2 was a duplicate: if the dashboard doesn't show it, the first copy was sent before "
                      "the claim and stays device-scoped (send a new observation)")

    def step_found(self):
        now = datetime.now(timezone.utc)
        when = now.replace(microsecond=now.microsecond // 1000 * 1000)  # whole ms, as sent
        r = self.post_found("5 found", when)
        if r.status_code == 404:
            self.fail("5 found", 404, f"{detail(r)}: no last-seen row in this device's scope",
                      "Dylan: check the device's household claim (see step 3)")
        self.common_failure("5 found", r)
        if r.status_code != 200:
            self.fail("5 found", r.status_code, detail(r), "Dylan: check POST /items/{label}/found")
        r = self.get_last_seen("5 found")
        self.common_failure("5 found", r)
        got = r.json() if r.status_code == 200 else {}
        if r.status_code != 200 or not got.get("found_at") or not same_instant(got["found_at"], iso_ms(when)):
            self.fail("5 found", r.status_code, f"last-seen found_at is {got.get('found_at')}, sent {iso_ms(when)}",
                      "Dylan: mark_found did not persist found_at")
        self.found_at, self.last_seen = when, got
        self.result(True, "5 found", 200, f"found_at {got['found_at']} is set on last-seen")

    def step_reopen(self):
        print("WARNING  --reopen-test writes a SYNTHETIC test observation to production (same JPEG, new id, "
              "made-up capture time) and deletes it again right after.")
        stored = parse_time(self.last_seen["captured_at"])
        try:
            cap = pick_reopen_captured_at(stored, self.found_at)
        except ValueError as exc:
            self.fail("6 reopen", None, str(exc), "Adam: rerun; found_at must be later than the stored sighting")
        rid = reopen_observation_id(self.obs["observation_id"])
        test = copy.deepcopy(self.obs)
        test["observation_id"], test["captured_at"] = rid, iso_ms(cap)
        for o in test["objects"]:
            o["track_id"] = "smoke-reopen-test"
        self.note(f"test observation {rid} captured_at {iso_ms(cap)} (between {iso_ms(stored)} and "
                  f"found_at {iso_ms(self.found_at)})")
        try:
            r = self.post_observation("6 reopen", test)
            if r.status_code == 200:  # leftover of an interrupted run: delete it and post a fresh copy
                self.note("the test id already exists (earlier run not cleaned up); deleting it and posting again")
                self.send("6 reopen", "DELETE", self.url(f"/api/v1/observations/{rid}"))
                self.restore_found()
                r = self.post_observation("6 reopen", test)
            if r.status_code != 201:
                self.common_failure("6 reopen", r)
                self.fail("6 reopen", r.status_code, f"test observation not stored: {detail(r)}",
                          "Dylan: see step 2's hints for this status")
            r = self.get_last_seen("6 reopen")
            got = r.json() if r.status_code == 200 else {}
            if r.status_code == 200 and got.get("found_at") is None:
                self.result(False, "6 reopen", 200, "BUG REPRODUCED: pre-found sighting reopened the item "
                            f"(backend upsert_last_seen); last-seen is now {got.get('observation_id')} with found_at null",
                            "Dylan: in backend/app/db_observations.py upsert_last_seen, add `AND (item_last_seen.found_at "
                            "IS NULL OR EXCLUDED.captured_at > item_last_seen.found_at)` to the WHERE")
            elif r.status_code == 200:
                self.result(True, "6 reopen", 200, f"found_at {got['found_at']} survived a sighting captured before it")
            else:
                self.result(False, "6 reopen", r.status_code, f"GET last-seen: {detail(r)}",
                            "Dylan: check the deployed backend")
        finally:
            self.cleanup_reopen(rid)

    def cleanup_reopen(self, rid: str):
        try:
            r = self.send("6 cleanup", "DELETE", self.url(f"/api/v1/observations/{rid}"))
            if r.status_code not in (200, 404):
                self.fail("6 cleanup", r.status_code, f"DELETE test observation {rid}: {detail(r)}",
                          f"Dylan: delete observation {rid} by hand (DELETE /api/v1/observations/{rid})")
            self.note(f"DELETE test observation: HTTP {r.status_code} ({'deleted' if r.status_code == 200 else 'none'})")
            self.restore_found()
        except StepFailed:
            pass

    def restore_found(self):
        r = self.get_last_seen("6 cleanup")
        got = r.json() if r.status_code == 200 else {}
        if got.get("observation_id") != self.obs["observation_id"]:
            self.fail("6 cleanup", r.status_code, f"last-seen is {got.get('observation_id')} after the DELETE, "
                      f"expected {self.obs['observation_id']}", "Dylan: rebuild_last_seen picked another sighting")
        if got.get("found_at") and same_instant(got["found_at"], iso_ms(self.found_at)):
            self.result(True, "6 cleanup", 200, "test record deleted; found_at survived the rebuild")
            return
        self.note(f"found_at did NOT survive the rebuild after DELETE (now {got.get('found_at')}): "
                  "rebuild_last_seen deletes and re-inserts the row (backend/app/db_observations.py:210-232); "
                  "Dylan: keep found_at there")
        r = self.post_found("6 cleanup", self.found_at)
        check = self.get_last_seen("6 cleanup")
        got = check.json() if check.status_code == 200 else {}
        if r.status_code != 200 or not got.get("found_at") or not same_instant(got["found_at"], iso_ms(self.found_at)):
            self.fail("6 cleanup", r.status_code, f"could not restore found_at (now {got.get('found_at')})",
                      f"Adam: POST /api/v1/items/{self.label}/found with found_at {iso_ms(self.found_at)} again")
        self.result(True, "6 cleanup", 200, f"test record deleted; found_at re-posted ({got['found_at']})")


def detail(r: requests.Response, keys=("detail",)) -> str:
    try:
        body = r.json()
        for k in keys:
            if isinstance(body, dict) and body.get(k):
                v = body[k]
                return (v if isinstance(v, str) else json.dumps(v))[:300]
    except ValueError:
        pass
    return (r.text or r.reason or "").strip()[:300]


def load_observation(path: Path) -> tuple[dict, bytes]:
    """The dry-run JSON plus the .jpg beside it, checked locally before anything is sent."""
    obs = json.loads(path.read_text(encoding="utf-8"))
    jpeg = path.with_suffix(".jpg").read_bytes()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{path.with_suffix('.jpg')} is not a decodable image")
    if obs.get("image_size") != {"width": img.shape[1], "height": img.shape[0]}:
        raise ValueError(f"image_size {obs.get('image_size')} != JPEG {img.shape[1]}x{img.shape[0]}")
    if not obs.get("objects"):
        raise ValueError("observation has no objects")
    parse_time(obs["captured_at"])
    return obs, jpeg


def print_plan(base: str, env: dict, obs: dict | None, args):
    domain = env.get("AUTH0_DOMAIN") or "(AUTH0_DOMAIN not set)"
    pairing = env.get("LASTSCENE_PAIRING_CODE") or env.get("PAIRING_CODE", "")
    print("DRY RUN: nothing is sent. Planned requests (all with allow_redirects=False):")
    print(f"  1  POST {domain}/oauth/token  client_credentials, audience {env.get('AUTH0_AUDIENCE')} "
          "(skipped if a cached token is still valid; secret never printed)")
    print(f"     GET  {base}/api/v1/items/last-seen  Authorization: Bearer <token>")
    if not obs:
        return
    label = quote(obs["objects"][0]["label"], safe="")
    print(f"  2  POST {base}/api/v1/observations  X-Pairing-Code: {mask(pairing)}  multipart image=frame.jpg, "
          f"observation={obs['observation_id']} captured_at {obs['captured_at']}")
    print(f"  3  GET  {base}/api/v1/items/{label}/last-seen, then GET <image_url> (signed, no token)")
    print("  4  (manual dashboard check)")
    if args.found:
        print(f"  5  POST {base}/api/v1/items/{label}/found  {{\"found_at\": <now UTC>}}; GET last-seen")
    if args.reopen_test:
        rid = reopen_observation_id(obs["observation_id"])
        print(f"  6  POST {base}/api/v1/observations  SYNTHETIC {rid}, captured_at between the stored sighting "
              f"and found_at; GET last-seen; DELETE {base}/api/v1/observations/{rid}; GET last-seen; "
              "POST found again if the rebuild dropped found_at")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation", nargs="?", type=Path,
                   help="dry-run observation JSON (<queue>/outbox/<id>.json, the .jpg beside it); omit = step 1 only")
    p.add_argument("--env-file", help="KEY=VALUE file with the credentials (outside the repo; never commit it)")
    p.add_argument("--found", action="store_true", help="also POST found and check found_at")
    p.add_argument("--reopen-test", action="store_true",
                   help="needs --found: writes and deletes a synthetic test record on production")
    p.add_argument("--fresh-token", action="store_true", help="ignore the cached Auth0 token (after a scope change)")
    p.add_argument("--dry-run", action="store_true", help="print the planned requests, send nothing")
    args = p.parse_args(argv)
    if args.reopen_test and not args.found:
        p.error("--reopen-test requires --found")
    if (args.found or args.reopen_test) and not args.observation:
        p.error("--found and --reopen-test need the observation path")

    if args.env_file:
        load_env_file(args.env_file)
    env = dict(os.environ)
    raw = env.get("LASTSCENE_API_BASE") or env.get("API_URL") or DEFAULT_BASE
    base = normalize_base(raw)
    if base != raw.rstrip("/"):
        print(f"note: API base {raw} -> {base}")
    obs = jpeg = None
    if args.observation:
        try:
            obs, jpeg = load_observation(args.observation)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FAIL  0 local        HTTP -    {exc}")
            print("      what to request from whom: Adam: rerun lastscene_bridge.run --dry-run and pass "
                  "<queue>/outbox/<id>.json (the .jpg must sit beside it)")
            return 1
    if args.dry_run:
        print_plan(base, env, obs, args)
        return 0

    smoke = Smoke(base, env, obs, jpeg, fresh_token=args.fresh_token, reopen_test=args.reopen_test)
    steps = [smoke.step_auth]
    if obs:
        steps += [smoke.step_post, smoke.step_last_seen, smoke.step_dashboard_hint]
    if args.found:
        steps.append(smoke.step_found)
    if args.reopen_test:
        steps.append(smoke.step_reopen)
    try:
        for step in steps:
            step()
    except StepFailed:
        pass
    print(f"\n{smoke.passed} passed, {smoke.failed} failed" + ("" if smoke.failed else ": all requested steps passed"))
    return 0 if smoke.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
