r"""Local mock of the LastScene backend + Auth0, written from the integration contract (Oct 17, 2026).

Used by the tests; also runnable by hand to watch real uploads land:
    python tests\mock_lastscene.py --port 8765 --save-dir data\mock_lastscene
then point the bridge at it:
    $env:LASTSCENE_API_BASE="http://127.0.0.1:8765"; $env:LASTSCENE_PAIRING_CODE="PAIR-1234"
    $env:AUTH0_DOMAIN="http://127.0.0.1:8765"; $env:AUTH0_CLIENT_ID="mock-client"
    $env:AUTH0_CLIENT_SECRET="mock-secret"; $env:AUTH0_AUDIENCE="https://lastscene.photo/api"

It is deliberately strict (unknown fields, image_size that doesn't match the image, bboxes
outside 0-1, wrong event names are all 422) so it catches adapter mistakes. It is NOT
Dylan's implementation; his real backend may differ.

Found / last-seen follow the real backend (backend/app/db_observations.py upsert_last_seen and
mark_found, routers/observations.py item_found), including its reopen bug, so tests catch it:
    found       404 when the label has no last-seen row; else sets found_at (body optional, default now)
    a new sighting replaces last-seen and CLEARS found_at whenever stored captured_at <= new
                captured_at; found_at is never consulted (a stale queued sighting reopens the item)
    a duplicate observation_id (200) does not touch last-seen; DELETE rebuilds it with found_at NULL
"""

import argparse
import datetime
import email.parser
import email.policy
import json
import re
import threading
import time
import uuid
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

CLIENT_ID, CLIENT_SECRET, AUDIENCE = "mock-client", "mock-secret", "https://lastscene.photo/api"
PAIRING_CODE = "PAIR-1234"
TOP_FIELDS = {"observation_id", "captured_at", "source", "pipeline", "image_size", "scene", "objects", "location"}
EVENTS = {"placed", "seen", "moved", "picked_up"}


class MockState:
    def __init__(self, max_image_bytes=2 * 1024 * 1024, rate_limit_s=2.0, save_dir=None):
        self.max_image_bytes = max_image_bytes
        self.rate_limit_s = rate_limit_s
        self.save_dir = Path(save_dir) if save_dir else None
        self.tokens = set()
        self.observations = {}   # observation_id -> {"obs":..., "image":..., "memory_id":..., "received":...}
        self.last_seen = {}      # label -> record
        self.found = {}          # label -> found_at
        self.post_times = []     # monotonic times of accepted-for-processing observation POSTs
        self.last_post = {}      # pairing code -> monotonic time
        self.forced = []         # status codes to force on the next observation POSTs (fault injection)
        self.forced_found = []   # same, for POST /items/{label}/found
        self.redirect = None     # set to a URL: every /api/ request answers 308 with that Location
        self.token_requests = 0  # successful + failed POST /oauth/token
        self.arrivals = []       # ("observation", id) / ("found", label), in the order the server applied them
        self.lock = threading.Lock()


def _bad(field, why):
    return 422, {"error": f"{field}: {why}"}


def validate(obs, image_bytes):
    """Contract checks. Returns None if valid, else (status, body)."""
    unknown = set(obs) - TOP_FIELDS
    if unknown:
        return _bad(",".join(sorted(unknown)), "unknown field")
    for f in ("observation_id", "captured_at", "source", "pipeline", "image_size", "objects"):
        if f not in obs:
            return _bad(f, "required")
    try:
        uuid.UUID(str(obs["observation_id"]))
    except ValueError:
        return _bad("observation_id", "must be a UUID")
    cap = obs["captured_at"]
    if not isinstance(cap, str) or not re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)$", cap):
        return _bad("captured_at", "must be ISO 8601 UTC")
    when = datetime.datetime.fromisoformat(cap.replace("Z", "+00:00"))
    if when > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5):
        return _bad("captured_at", "more than 5 minutes in the future")
    if obs["source"] != "auto":
        return _bad("source", 'must be "auto"')
    pipe = obs["pipeline"]
    if not isinstance(pipe, dict) or any(not isinstance(pipe.get(k), str) for k in ("name", "version", "detector", "matcher")):
        return _bad("pipeline", "needs string name, version, detector, matcher")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None or not (image_bytes[:2] == b"\xff\xd8" or image_bytes[:8] == b"\x89PNG\r\n\x1a\n"):
        return _bad("image", "must be JPEG or PNG")
    size = obs["image_size"]
    if not isinstance(size, dict) or size.get("width") != img.shape[1] or size.get("height") != img.shape[0]:
        return _bad("image_size", f"must match the uploaded image ({img.shape[1]}x{img.shape[0]})")
    if "scene" in obs:
        s = obs["scene"]
        score = s.get("match_score") if isinstance(s, dict) else None  # backend Scene: label/match_score optional
        if not isinstance(s, dict) or not isinstance(s.get("id"), str) or len(s["id"]) > 80 \
                or len(s.get("label") or "") > 120 or (score is not None and not 0 <= score <= 1):
            return _bad("scene", "needs id (<= 80 chars); label <= 120 chars; match_score 0-1 if present")
    if "location" in obs:  # backend ObservationLocation: optional Wi-Fi scan and/or position
        loc = obs["location"]
        aps = loc.get("wifi", []) if isinstance(loc, dict) else None
        if not isinstance(loc, dict) or set(loc) - {"wifi", "lat", "lng", "accuracy_m"} or not isinstance(aps, list) \
                or len(aps) > 60 or any(not isinstance(a, dict) or not isinstance(a.get("bssid"), str)
                                        or not -130 <= a.get("signal", 1) <= 0 for a in aps) \
                or (loc.get("lat") is not None and not -90 <= loc["lat"] <= 90) \
                or (loc.get("lng") is not None and not -180 <= loc["lng"] <= 180):
            return _bad("location", "wifi: up to 60 {bssid, signal -130..0}; lat -90..90, lng -180..180")
    objs = obs["objects"]
    if not isinstance(objs, list) or not objs:
        return _bad("objects", "list with at least one entry")
    for i, o in enumerate(objs):
        extra = set(o) - {"label", "confidence", "bbox", "track_id", "event"}
        if extra:
            return _bad(f"objects[{i}].{','.join(sorted(extra))}", "unknown field")
        if not isinstance(o.get("label"), str) or o["label"] != o["label"].lower() or not o["label"]:
            return _bad(f"objects[{i}].label", "lowercase string")
        if not isinstance(o.get("confidence"), (int, float)) or not 0 <= o["confidence"] <= 1:
            return _bad(f"objects[{i}].confidence", "number 0-1")
        b = o.get("bbox")
        if (not isinstance(b, list) or len(b) != 4 or any(not isinstance(v, (int, float)) for v in b)
                or any(v < 0 or v > 1 for v in b) or b[0] + b[2] > 1.0001 or b[1] + b[3] > 1.0001):
            return _bad(f"objects[{i}].bbox", "[x, y, w, h] normalized 0-1")
        if "track_id" in o and not isinstance(o["track_id"], str):
            return _bad(f"objects[{i}].track_id", "string")
        if o.get("event") not in EVENTS:
            return _bad(f"objects[{i}].event", f"one of {sorted(EVENTS)}")
    return None


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body=None, headers=None):
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _authorized(self):
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] not in state.tokens:
                self._send(401, {"error": "invalid or missing token"})
                return False
            if self.headers.get("X-Pairing-Code") != PAIRING_CODE:
                self._send(403, {"error": "unknown pairing code"})
                return False
            return True

        def do_POST(self):
            self.path = unquote(self.path)
            if self.path == "/oauth/token":
                state.token_requests += 1
                body = json.loads(self._body() or b"{}")
                ok = (body.get("grant_type") == "client_credentials" and body.get("client_id") == CLIENT_ID
                      and body.get("client_secret") == CLIENT_SECRET and body.get("audience") == AUDIENCE)
                if not ok:
                    return self._send(401, {"error": "access_denied"})
                token = uuid.uuid4().hex
                state.tokens.add(token)
                return self._send(200, {"access_token": token, "expires_in": 3600, "token_type": "Bearer"})
            if self._redirected():
                return
            if self.path == "/api/v1/observations":
                return self._observation()
            m = re.match(r"^/api/v1/items/([^/]+)/found$", self.path)
            if m:
                return self._found(" ".join(m.group(1).strip().lower().split()))
            self._send(404, {"error": "not found"})

        def _redirected(self):
            if state.redirect and self.path.startswith("/api/"):
                self._body()
                self._send(308, {"error": "moved"}, {"Location": state.redirect + self.path})
                return True
            return False

        def _found(self, label):
            raw = self._body()
            if not self._authorized():
                return
            body = json.loads(raw) if raw else {}
            try:
                when = (_parse(body["found_at"]) if body.get("found_at")
                        else datetime.datetime.now(datetime.timezone.utc))
            except (TypeError, ValueError):
                return self._send(422, {"error": "found_at: invalid datetime"})
            with state.lock:
                if state.forced_found:
                    code = state.forced_found.pop(0)
                    return self._send(code, {"error": f"forced {code}"})
                if label not in state.last_seen:  # mark_found updated no row
                    return self._send(404, {"error": f"No sighting of {label!r}"})
                state.last_seen[label]["found_at"] = when.isoformat()
                state.found[label] = body
                state.arrivals.append(("found", label))
            return self._send(200, {"label": label, "found_at": when.isoformat()})

        def _observation(self):
            raw = self._body()
            if not self._authorized():
                return
            with state.lock:
                if state.forced:
                    code = state.forced.pop(0)
                    return self._send(code, {"error": f"forced {code}"}, {"Retry-After": "1"} if code == 429 else None)
                now = time.monotonic()
                last = state.last_post.get(PAIRING_CODE)
                if last is not None and now - last < state.rate_limit_s:
                    return self._send(429, {"error": "rate limited"}, {"Retry-After": f"{state.rate_limit_s - (now - last):.2f}"})
                state.last_post[PAIRING_CODE] = now
                state.post_times.append(now)
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\n\r\n" + raw)
            parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()} if msg.is_multipart() else {}
            if "image" not in parts or "observation" not in parts:
                return self._send(422, {"error": "multipart needs image and observation parts"})
            image = parts["image"].get_payload(decode=True)
            if len(image) > state.max_image_bytes:
                return self._send(413, {"error": "image over limit"})
            try:
                obs = json.loads(parts["observation"].get_payload(decode=True))
            except ValueError:
                return self._send(422, {"error": "observation: invalid JSON"})
            problem = validate(obs, image)
            if problem:
                return self._send(*problem)
            oid = obs["observation_id"]
            with state.lock:
                if oid in state.observations:
                    rec = state.observations[oid]
                    return self._send(200, {"memory_id": rec["memory_id"], "items": rec["items"], "duplicate": True})
                memory_id = f"srv_{uuid.uuid4().hex[:10]}"
                items = sorted({o["label"] for o in obs["objects"]})
                state.observations[oid] = {"obs": obs, "image": image, "memory_id": memory_id, "items": items}
                state.arrivals.append(("observation", oid))
                for o in obs["objects"]:
                    if o["event"] in ("placed", "seen", "moved"):
                        prev = state.last_seen.get(o["label"])
                        # real upsert_last_seen: WHERE stored.captured_at <= new.captured_at, SET found_at = NULL
                        if prev is None or _parse(prev["captured_at"]) <= _parse(obs["captured_at"]):
                            state.last_seen[o["label"]] = {
                                "label": o["label"], "memory_id": memory_id, "observation_id": oid,
                                "captured_at": obs["captured_at"], "scene": obs.get("scene"), "bbox": o["bbox"],
                                "confidence": o["confidence"], "image_url": f"/media/{oid}.jpg",
                                "image_size": obs["image_size"], "source": "auto", "found_at": None}
            if state.save_dir:
                _save(state.save_dir, obs, image)
            self._send(201, {"memory_id": memory_id, "items": items})

        def do_GET(self):
            self.path = unquote(self.path)
            m = re.match(r"^/media/([0-9a-f-]+)\.jpg$", self.path)
            if m and m.group(1) in state.observations:
                data = state.observations[m.group(1)]["image"]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if self._redirected() or not self._authorized():
                return
            if self.path == "/api/v1/items/last-seen":
                return self._send(200, {"items": sorted(state.last_seen.values(), key=lambda r: r["captured_at"],
                                                        reverse=True)})
            m = re.match(r"^/api/v1/items/([^/]+)/last-seen$", self.path)
            if m:
                rec = state.last_seen.get(m.group(1))
                return self._send(200, rec) if rec else self._send(404, {"error": "never seen"})
            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if self._redirected() or not self._authorized():
                return
            m = re.match(r"^/api/v1/observations/([0-9a-f-]+)$", self.path)
            if not m or m.group(1) not in state.observations:
                return self._send(404, {"error": "no such observation"})
            with state.lock:
                gone = state.observations.pop(m.group(1))
                for label in gone["items"]:  # roll back last-seen to the previous observation of that label
                    candidates = [(r["obs"]["captured_at"], oid, r) for oid, r in state.observations.items()
                                  if label in r["items"]]
                    if not candidates:
                        state.last_seen.pop(label, None)
                    else:
                        _, oid, r = max(candidates)
                        o = next(x for x in r["obs"]["objects"] if x["label"] == label)
                        state.last_seen[label] = {**state.last_seen.get(label, {}), "memory_id": r["memory_id"],
                                                  "observation_id": oid, "captured_at": r["obs"]["captured_at"],
                                                  "bbox": o["bbox"], "confidence": o["confidence"],
                                                  "image_url": f"/media/{oid}.jpg", "image_size": r["obs"]["image_size"],
                                                  "found_at": None}  # real rebuild_last_seen re-inserts the row
            self._send(200, {"deleted": m.group(1)})

    return Handler


def _parse(stamp):
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _save(save_dir, obs, image):
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = save_dir / obs["observation_id"]
    stem.with_suffix(".json").write_text(json.dumps(obs, indent=2))
    stem.with_suffix(".jpg").write_bytes(image)
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    w, h = obs["image_size"]["width"], obs["image_size"]["height"]
    for o in obs["objects"]:  # draw from the NORMALIZED bbox, as Dylan's dashboard would
        x, y, bw, bh = o["bbox"]
        cv2.rectangle(img, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (0, 0, 255), 3)
        cv2.putText(img, f"{o['label']} {o['confidence']:.2f} auto/{o['event']}", (int(x * w), max(24, int(y * h) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(stem) + "_dashboard.jpg", img)


def serve(port=0, **kwargs):
    """Start in a background thread. Returns (server, state, base_url)."""
    state = MockState(**kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state, f"http://127.0.0.1:{server.server_address[1]}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--save-dir", default="data/mock_lastscene")
    args = p.parse_args()
    server, state, url = serve(args.port, save_dir=args.save_dir)
    print(f"mock LastScene + Auth0 at {url} (pairing code {PAIRING_CODE}); saving to {args.save_dir}. Ctrl+C to stop.")
    seen = 0
    try:
        while True:
            time.sleep(1)
            if len(state.observations) != seen:
                seen = len(state.observations)
                print(f"{seen} observations; last-seen: {json.dumps(state.last_seen, indent=1)[:600]}")
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
