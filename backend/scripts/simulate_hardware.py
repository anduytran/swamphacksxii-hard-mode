"""Simulate three days of the LastScene wearable, end to end, without spending tokens.

What is real:
  * the wearable's client code (device/lastscene_device.py: ApiClient, offline Spool)
  * the multipart WAV + PNG uploads, the FastAPI routes, validation and enrichment flow
  * TigerData: memories, photos, items and questions are written to DATABASE_URL, so the
    dashboard (locally or at lastscene.photo) shows the result
  * the dashboard's Items / Insights / Timeline computations

What is simulated (no API call that uses a token is made):
  * every AI stage -> SimulatedAI: deterministic transcription, hashed bag-of-words
    embeddings, rule-based item extraction, canned image-to-text output, template
    answers, a beep for speech. No Gemini, Claude or OpenAI calls.
  * the RAG layer is real: memories become text chunks in memory_chunks, and questions
    go through the same vector + keyword search and fusion as production.
  * Auth0 -> device/caregiver checks are bypassed in-process; no token is requested.
  * time -> events are backdated over the last ~72 h so the history looks lived-in.

All rows use device id "sim-wearable@clients"; each run replaces the previous run's rows.

    cd backend && .venv/bin/python scripts/simulate_hardware.py          # run the scenario
    cd backend && .venv/bin/python scripts/simulate_hardware.py --clean  # remove simulated rows
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import struct
import sys
import tempfile
import wave
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "device"))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.ai import ItemMention, SceneText, humanize_ago  # noqa: E402
from app.auth import Principal, require_caregiver, require_device  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.geo import Location  # noqa: E402
from app.main import create_app  # noqa: E402
from app.db_places import household_hash  # noqa: E402
from app.routers.dashboard import household_devices  # noqa: E402
from lastscene_device import ApiClient, Config, Spool  # noqa: E402

DEVICE_ID = "sim-wearable@clients"
PAIRING_CODE = "SIM-DEMO-0001"

# Wi-Fi positioning, simulated: each spot is "heard" as two access points. Real places in
# Gainesville; the home is a made-up address.
SPOTS = {
    "home": (29.64752, -82.33604, ["5a:11:00:00:00:01", "5a:11:00:00:00:02"]),
    "grocery": (29.65190, -82.34330, ["5a:22:00:00:00:01", "5a:22:00:00:00:02"]),
    "bus stop": (29.64650, -82.34780, ["5a:33:00:00:00:01", "5a:33:00:00:00:02"]),
    "library": (29.65160, -82.32360, ["5a:44:00:00:00:01", "5a:44:00:00:00:02"]),
}
PUBLIC_SPOTS = ("grocery", "bus stop", "library")


def wifi_scan(spot: str) -> dict:
    lat, lng, bssids = SPOTS[spot]
    return {"wifi": json.dumps([{"bssid": b, "signal": -55 - 10 * i} for i, b in enumerate(bssids)])}


class SimGeo:
    """Drop-in for app.geo.Geolocator: no Google call, the access points pick the spot."""

    by_bssid = {b: (lat, lng) for lat, lng, bs in SPOTS.values() for b in bs}

    async def locate(self, aps):
        spot = self.by_bssid.get(aps[0].bssid) if aps else None
        return Location(*spot, 20, "wifi") if spot else None


# ---------------------------------------------------------------- fake media

def fake_wav(transcript: str, seconds: float = 0.6) -> bytes:
    """A real 16 kHz mono WAV of silence, with the spoken words in a LIST/INFO 'ICMT'
    chunk so the simulated speech-to-text can 'hear' them."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    text = transcript.encode() + b"\x00"
    if len(text) % 2:
        text += b"\x00"
    info = b"INFO" + b"ICMT" + struct.pack("<I", len(text)) + text
    data = buf.getvalue() + b"LIST" + struct.pack("<I", len(info)) + info
    return data[:4] + struct.pack("<I", len(data) - 8) + data[8:]


def wav_transcript(data: bytes) -> str:
    i = data.find(b"ICMT")
    if i < 0:
        return ""
    (size,) = struct.unpack("<I", data[i + 4 : i + 8])
    return data[i + 8 : i + 8 + size].rstrip(b"\x00").decode()


def beep_wav(seconds: float = 0.3, hz: int = 660) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * n / 16000))) for n in range(int(16000 * seconds))))
    return buf.getvalue()


def scene_png(surface: tuple[int, int, int], item: tuple[int, int, int], seed: int, w: int = 320, h: int = 240) -> bytes:
    """A small 'photo': wall, a surface, and the item on it. Unique per memory."""
    rows = []
    wall = (40 + seed * 7 % 30, 38, 36)
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if y > h * 0.55:
                px = surface
            else:
                px = wall
            if (x - w * 0.5) ** 2 / 900 + (y - h * 0.58) ** 2 / 220 < 1:
                px = item
            row += bytes(px)
        rows.append(bytes(row))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------- simulated AI

STOPWORDS = {"i", "im", "i'm", "my", "the", "a", "an", "is", "are", "on", "in", "by", "to", "of", "where",
             "did", "put", "putting", "going", "leaving", "have", "you", "seen", "s", "it", "and", "down", "next"}
ALIASES = {"glasses": "reading glasses", "remote": "tv remote", "pills": "pill box"}
PLACE = r"(?:on|in|into|by|under|inside|at|next to)"


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


@dataclass
class SimulatedAI:
    """Drop-in for app.ai.AIServices. Deterministic and offline."""

    clock: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scenes: dict[str, SceneText] = field(default_factory=dict)
    dims: int = 1536
    speech_media_type: str = "audio/wav"

    async def transcribe(self, filename, audio, content_type) -> str:
        return wav_transcript(audio)

    async def embed(self, text: str, *, query: bool = False) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z']+", text.lower()):
            if word in STOPWORDS:
                continue
            h = int(hashlib.sha1(_stem(word).encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def extract_items(self, text: str) -> list[ItemMention]:
        t = text.lower().strip().rstrip("?.!")
        t = t.replace("where's", "where is")
        m = re.search(rf"\b(?:my|the) ([a-z ]+?) (?:is |are )?(?:going |)?{PLACE} (?:the |my )?([a-z ]+)$", t)
        if m and not t.startswith(("where", "have you")):
            return [ItemMention(ALIASES.get(m.group(1), m.group(1)), m.group(2))]
        m = re.search(r"\b(?:my|the) ([a-z ]+?)(?: down)?$", t)
        if m:
            name = m.group(1).removesuffix(" down")
            return [ItemMention(ALIASES.get(name, name), "")]
        return []

    async def describe_scene(self, image: bytes, content_type: str, transcript: str):
        return self.scenes.get(hashlib.sha1(image).hexdigest())

    async def answer(self, question: str, memories) -> str:
        asked = await self.extract_items(question)
        name = asked[0].name if asked else ""
        words = {_stem(w) for w in name.split()} - STOPWORDS
        # Like the real prompt: of the retrieved memories about this item, use the latest.
        for m in sorted(memories, key=lambda m: m.created_at, reverse=True):
            haystack = {_stem(w) for w in re.findall(r"[a-z]+", f"{m.transcript} {m.scene or ''}".lower())}
            if words and words <= haystack:
                where = f" The camera saw: {m.scene}" if m.scene else ""
                return f"You said “{m.transcript}” {humanize_ago(m.created_at, self.clock)}.{where}"
        return f"I don't have a note about your {name or 'that'} yet. Tell me next time you put it down."

    async def speak(self, text: str) -> bytes:
        return beep_wav()

    async def caregiver_summary(self, facts: str) -> str:
        return f"[Simulated summary — the real one is written by Gemini] {facts}"


# ---------------------------------------------------------------- scenario

COLORS = {
    "kitchen counter": (225, 222, 214), "nightstand": (120, 84, 58), "coat pocket": (52, 62, 84),
    "charger by the couch": (88, 74, 96), "case on the dresser": (150, 110, 80),
    "hallway table": (140, 98, 62), "armrest": (110, 60, 50), "bathroom shelf": (200, 210, 214),
    "kitchen drawer": (170, 150, 120), "dining table": (130, 90, 55),
    "bus stop": (90, 96, 104), "checkout": (200, 200, 190),
}
ITEM_COLORS = {"keys": (220, 180, 60), "reading glasses": (30, 30, 30), "wallet": (90, 50, 30),
               "phone": (20, 20, 24), "hearing aids": (230, 200, 180), "tv remote": (40, 40, 40),
               "pill box": (120, 170, 220), "umbrella": (25, 25, 30), "sunglasses": (60, 40, 20)}


@dataclass
class Event:
    hours_ago: float
    kind: str  # "log" | "ask" | "log-offline"
    text: str
    item: str = ""
    place: str = ""  # where the camera shows it (drives the photo + scene)
    scene: str = ""  # what Gemini vision would say
    visible: str = ""  # words readable in the photo
    spot: str = "home"  # where the wearer is (drives the Wi-Fi scan)


SCENARIO = [
    Event(70, "log", "I'm putting my keys on the kitchen counter", "keys", "kitchen counter", "Keys on a white kitchen counter beside the kettle."),
    Event(66, "log", "Putting my reading glasses on the nightstand", "reading glasses", "nightstand", "Reading glasses folded on a wooden nightstand next to a lamp."),
    Event(60, "log", "My wallet is going in my coat pocket", "wallet", "coat pocket", "A brown wallet in the right pocket of a navy coat on the hook."),
    Event(52, "log", "I'm leaving my phone on the charger by the couch", "phone", "charger by the couch", "Phone plugged in on the side table by the grey couch."),
    Event(47, "log", "Putting my hearing aids in the case on the dresser", "hearing aids", "case on the dresser", "Hearing aid case, open, on top of the bedroom dresser."),
    Event(44, "ask", "Where are my keys?"),
    # Out and about: public places (only counted publicly if the household opts in).
    Event(40, "log", "I'm leaving my umbrella at the bus stop", "umbrella", "bus stop", "Black umbrella leaning on the bench in the bus shelter.", spot="bus stop"),
    # Nothing said about where: Gemini vision fills the location in from the photo.
    Event(30, "log", "I'm putting my keys down", "keys", "hallway table", "Keys on the wooden hallway table by the front door, next to a blue bowl."),
    Event(28, "ask", "Where did I put my glasses?"),
    Event(25, "log", "Putting the remote on the armrest", "tv remote", "armrest", "TV remote on the left armrest of the red armchair."),
    Event(22, "log", "Putting my sunglasses on the shelf by the checkout", "sunglasses", "checkout", "Sunglasses on the shelf beside the grocery checkout.", spot="grocery"),
    Event(21, "ask", "Where did I leave my umbrella?", spot="grocery"),
    Event(20, "ask", "Have you seen my pill box?"),  # never logged before this
    Event(18, "log", "My glasses are going on the bathroom shelf", "reading glasses", "bathroom shelf", "Reading glasses on the glass shelf above the bathroom sink."),
    Event(12, "ask", "Where are my keys?"),
    Event(9, "log", "Putting my pill box in the kitchen drawer", "pill box", "kitchen drawer", "Blue weekly pill box in the top kitchen drawer by the fridge.", "MON TUE WED THU FRI SAT SUN"),
    Event(6, "log", "I'm leaving my wallet on the dining table", "wallet", "dining table", "Wallet on the dining table beside the fruit bowl and a handwritten note.", "Dentist Tuesday 3pm"),
    Event(4, "ask", "Where's my wallet?"),
    Event(3, "log", "Putting my keys on the hallway table", "keys", "hallway table", "Keys in the blue bowl on the hallway table."),
    Event(2, "ask", "Where are my glasses?"),
    # Wi-Fi drops: the wearable spools this memory and uploads it once it's back online.
    Event(1, "log-offline", "I'm putting my phone on the nightstand", "phone", "nightstand", "Phone face down on the nightstand next to the reading glasses case."),
    Event(0.3, "ask", "Where is my phone?"),
]


# ---------------------------------------------------------------- runner

def offline_api(cfg: Config) -> ApiClient:
    def no_network(request):
        raise httpx.ConnectError("Wi-Fi is down", request=request)

    api = ApiClient(cfg, httpx.Client(transport=httpx.MockTransport(no_network)))
    api._token, api._expires_at = "simulated", float("inf")
    return api


PUBLIC_LOSSES = [  # (spot, what each synthetic household says there)
    ("bus stop", "I'm leaving my umbrella at the bus stop"),
    ("bus stop", "Where did I leave my umbrella?"),
    ("library", "Putting my water bottle on the table by the door"),
    ("grocery", "Putting my sunglasses on the shelf by the checkout"),
    ("grocery", "Where are my keys?"),
]


def seed_public_households(client, db, run, who, cfg) -> None:
    """Three other opted-in households so the public map clears its 3-household threshold.
    They go through the real device endpoints, place clustering and consent rules."""
    print("\nSeeding 3 synthetic households for the public map (opted in, public places marked)…")
    for n in range(3):
        device = f"sim-public-{n}@clients"
        household = run(db.create_household, f"sim-public-{n}", f"Synthetic household {n + 1}", None)
        run(db.remember_pairing_code, device, "SIM-PUBLIC")
        run(db.claim_device, household, device, "SIM-PUBLIC", None)
        run(db.set_share_public, household, True)
        who["device"] = device
        api = ApiClient(cfg, client)
        api._token, api._expires_at = "simulated", float("inf")
        photo = scene_png((120, 120, 120), (200, 60, 60), 100 + n)
        for spot in ("home", *PUBLIC_SPOTS):  # visit each spot once so its place exists
            api.log_memory(fake_wav("Just checking in"), photo, wifi_scan(spot))
        for place in run(db.places_for, household):  # mark the shops/stops public
            spot = next(k for k, (lat, lng, _) in SPOTS.items() if abs(lat - place.lat) < 1e-4 and abs(lng - place.lng) < 1e-4)
            if spot in PUBLIC_SPOTS:
                run(db.update_place, household, place.id, spot.title(), "public")
        for spot, text in PUBLIC_LOSSES[: 3 + n]:  # households differ a little
            if text.startswith("Where"):
                api.ask(fake_wav(text), wifi_scan(spot))
            else:
                api.log_memory(fake_wav(text), photo, wifi_scan(spot))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="only remove simulated rows")
    parser.add_argument("--out", default=None, help="folder for the answer WAVs (default: a temp folder)")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="also seed 3 synthetic opted-in households so the real public map shows spots "
        "(off by default: the dashboard's public map has its own Simulated toggle instead)",
    )
    args = parser.parse_args()

    if not get_settings().database_url:
        sys.exit("DATABASE_URL is not set in backend/.env")

    app = create_app(use_lifespan=False)
    sim = SimulatedAI()
    app.state.ai = sim  # every AI stage now runs offline
    app.state.geo = SimGeo()  # Wi-Fi positioning without calling Google
    who = {"device": DEVICE_ID}
    app.dependency_overrides[require_device] = lambda: Principal(
        sub=who["device"], claims={"gty": "client-credentials"}, scopes={"write:memories"}
    )
    app.dependency_overrides[require_caregiver] = lambda: Principal(sub="sim-caregiver", claims={})
    # The script's own dashboard checks look at the simulated wearer's data.
    app.dependency_overrides[household_devices] = lambda: [DEVICE_ID]

    out = Path(args.out or tempfile.mkdtemp(prefix="lastscene-sim-"))
    out.mkdir(parents=True, exist_ok=True)

    with TestClient(app, base_url="http://sim") as client:
        client.get("/api/v1/dashboard/history")  # connects + migrates the schema
        db = app.state.db
        run = client.portal.call

        q_deleted = run(db.pool.execute, "DELETE FROM query_logs WHERE device_id LIKE 'sim-%'")
        m_deleted = run(db.pool.execute, "DELETE FROM memory_logs WHERE device_id LIKE 'sim-%'")
        old = [str(r["id"]) for r in run(db.pool.fetch, "SELECT id FROM households WHERE created_by LIKE 'sim-public-%'")]
        run(db.pool.execute, "DELETE FROM public_lost_stats WHERE household_hash = ANY($1::text[])", [household_hash(h) for h in old])
        run(db.pool.execute, "DELETE FROM households WHERE created_by LIKE 'sim-public-%'")
        run(db.pool.execute, "DELETE FROM device_pairing WHERE device_id LIKE 'sim-public-%'")
        print(f"Removed previous simulated rows: {m_deleted.split()[-1]} memories, {q_deleted.split()[-1]} questions, {len(old)} synthetic households")
        if args.clean:
            run(db.pool.execute, "DELETE FROM device_pairing WHERE device_id = $1", DEVICE_ID)
            return

        cfg = Config(api_url="http://sim", auth0_domain="unused.invalid", auth0_audience="unused",
                     client_id="unused", client_secret="unused", spool_dir=out / "spool", pairing_code=PAIRING_CODE)
        api = ApiClient(cfg, client)  # the real wearable client, talking to the in-process app
        api._token, api._expires_at = "simulated", float("inf")  # never asks Auth0
        spool = Spool(cfg.spool_dir)

        now = datetime.now(timezone.utc)
        print(f"\nSimulating {len(SCENARIO)} wearable events over the last 72 hours…\n")
        for n, ev in enumerate(SCENARIO, 1):
            when = now - timedelta(hours=ev.hours_ago)
            sim.clock = when
            stamp = when.astimezone().strftime("%a %I:%M %p")
            if ev.kind in ("log", "log-offline"):
                photo = scene_png(COLORS[ev.place], ITEM_COLORS[ev.item], n)
                sim.scenes[hashlib.sha1(photo).hexdigest()] = SceneText(ev.scene, [ItemMention(ev.item, ev.place)], ev.visible)
                wav = fake_wav(ev.text)
                where = wifi_scan(ev.spot)
                if ev.kind == "log-offline":
                    try:
                        offline_api(cfg).log_memory(wav, photo, where)
                    except httpx.TransportError:
                        spool.add(wav, photo, where)
                        print(f"{stamp}  📴 offline — spooled “{ev.text}” ({len(spool.pending())} waiting)")
                    sent_before = {r["id"] for r in run(db.pool.fetch, "SELECT id FROM memory_logs WHERE device_id = $1", DEVICE_ID)}
                    print(f"{stamp}  📶 back online — uploaded {spool.flush(api)} spooled memory")
                    rows = run(db.pool.fetch, "SELECT id FROM memory_logs WHERE device_id = $1", DEVICE_ID)
                    memory_id = next(str(r["id"]) for r in rows if r["id"] not in sent_before)
                else:
                    memory_id = api.log_memory(wav, photo, where)["id"]
                    print(f"{stamp}  🎙  REMEMBER “{ev.text}”" + (f"  📍 {ev.spot}" if ev.spot != "home" else ""))
                run(db.pool.execute, "UPDATE memory_logs SET created_at = $2 WHERE id = $1::uuid", memory_id, when)
                chunks = run(db.pool.fetch, "SELECT kind, content FROM memory_chunks WHERE memory_id = $1::uuid ORDER BY id", memory_id)
                for c in chunks:
                    print(f"           ▸ {c['kind']:<6} {c['content']}")
            else:
                audio, answer = api.ask(fake_wav(ev.text), wifi_scan(ev.spot))
                (out / f"answer-{n:02d}.wav").write_bytes(audio)
                run(db.pool.execute, """
                    UPDATE query_logs SET created_at = $2
                    WHERE id = (SELECT id FROM query_logs WHERE device_id = $1 ORDER BY created_at DESC LIMIT 1)""",
                    DEVICE_ID, when)
                print(f"{stamp}  ❓ ASK      “{ev.text}”\n           → {answer}")

        print("\nWhat the caregiver dashboard now shows:\n")
        items = client.get("/api/v1/dashboard/items").json()["items"]
        for it in items:
            print(f"  • {it['name']:<16} {it['location'] or '?':<24} put down {it['mentions']}×, asked {it['times_asked']}×")
        ins = client.get("/api/v1/dashboard/insights").json()
        print(f"\n  Insights: {ins['memories_logged']} things put down, {ins['questions_asked']} questions, "
              f"{ins['unanswered_count']} about items not logged first, {ins['items_tracked']} items tracked")
        print(f"  Most asked: {', '.join(f'{r['name']} ({r['count']})' for r in ins['most_asked'])}")
        print(f"  Timeline (24h): {len(client.get('/api/v1/dashboard/history').json())} memories")
        print("\nRetrieval check (same hybrid search the wearable uses):\n")
        for q in ["Where are my keys?", "When is the dentist?", "Where are my pills?"]:
            res = client.get("/api/v1/dashboard/search", params={"q": q, "limit": 2}).json()["results"]
            print(f"  “{q}”")
            for r in res:
                hits = [f"{c['kind']}(meaning #{c['vector_rank'] or '-'}, words #{c['keyword_rank'] or '-'})" for c in r["chunks"] if c["matched"]]
                said = next(c["content"] for c in r["chunks"] if c["kind"] == "speech")
                print(f"     {r['score']:.4f}  “{said}”  ← {', '.join(hits)}")
        if args.public_demo:
            seed_public_households(client, db, run, who, cfg)
        who["device"] = DEVICE_ID

        claimed = run(db.household_of_device, DEVICE_ID)
        print("\nHousehold & places:")
        if claimed:
            places = run(db.places_for, claimed[0])
            print(f"  The simulated wearable belongs to your household; it now has {len(places)} places.")
        else:
            print("  The simulated wearable isn't in a household yet, so only you can claim it:")
            print("  Dashboard → Household → Add a wearable →")
            print(f"     Client ID: {DEVICE_ID.removesuffix('@clients')}    Pairing code: {PAIRING_CODE}")
        cells = client.get("/api/v1/public/lost-map").json()["cells"]
        print(f"\nPublic map: {len(cells)} anonymous spots (each with 3+ opted-in households)")
        for c in cells:
            print(f"  ({c['lat']:.4f}, {c['lng']:.4f})  " + ", ".join(f"{i['name']} ×{i['left'] + i['asked']}" for i in c["items"]))
        print(f"\nAnswer audio saved in {out}")
        print("Remove these rows any time with:  .venv/bin/python scripts/simulate_hardware.py --clean")


if __name__ == "__main__":
    main()
