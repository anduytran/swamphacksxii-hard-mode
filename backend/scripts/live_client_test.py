"""Test the live wearable client path end to end against the deployed API, with the device's own credentials.

    .venv/bin/python scripts/live_client_test.py --image pin.jpg [--item pin] [--env-file ../device/device.env]

1. A passive sighting, as the camera Pi posts it: POST /api/v1/observations (the item at 0.87, event placed).
2. The silent raia read it triggers (the server does it after the response): raia_contexts and the
   "LastScene: ..." memory, read from DATABASE_URL (backend/.env), read-only. Skipped without --db-check.
3. A wake ask, as the audio Pi sends it with on-device speech-to-text: "Last Scene, where is my <item>?".
4. "Last Scene, which way is my <item>?": every ask about an item is answered with its last 5 LastScene
   locations and where it is from the wearer as context, so it says which way without a follow-up.

Writes real data for the device's household: one observation, a raia memory, two voice events. Prints no secrets.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def token() -> str:
    resp = httpx.post(f"https://{os.environ['AUTH0_DOMAIN']}/oauth/token", timeout=15, json={
        "grant_type": "client_credentials", "client_id": os.environ["DEVICE_CLIENT_ID"],
        "client_secret": os.environ["DEVICE_CLIENT_SECRET"], "audience": os.environ["AUTH0_AUDIENCE"]})
    resp.raise_for_status()
    return resp.json()["access_token"]


def step(n: int, title: str) -> None:
    print(f"\n== {n}. {title}", flush=True)


def observation(item: str, confidence: float) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "observation_id": str(uuid.uuid4()),
        "captured_at": now.isoformat(),
        "source": "auto",
        "pipeline": {"name": "live-client-test", "version": "1", "detector": "test"},
        "image_size": {"width": 768, "height": 432},
        "objects": [{"label": item, "confidence": confidence, "bbox": [0.42, 0.33, 0.21, 0.36], "event": "placed"}],
    }


async def raia_row(item: str, since: datetime, wait_s: float) -> dict | None:
    """The household's raia read of `item` claimed after `since`, once it's finished (read-only)."""
    import asyncpg

    from app.config import Settings

    deadline = time.monotonic() + wait_s
    conn = await asyncpg.connect(Settings().database_url, timeout=15)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        while time.monotonic() < deadline:
            row = await conn.fetchrow(
                "SELECT claimed_at, finished_at, context, error, memory_id, conversation_id FROM raia_contexts "
                "WHERE item = $1 AND claimed_at >= $2 ORDER BY claimed_at DESC LIMIT 1", item, since)
            if row and row["finished_at"]:
                out = dict(row)
                if row["memory_id"]:
                    out["transcript"] = await conn.fetchval("SELECT transcript FROM memory_logs WHERE id = $1",
                                                            row["memory_id"])
                return out
            await asyncio.sleep(2)
        return None
    finally:
        await conn.close()


def voice(http: httpx.Client, said: str, image: bytes) -> httpx.Response:
    """As the wearable sends a wake command: defer=true (the exchange is recorded before the reply), then the
    follow-up call with its id once the answer has been heard."""
    resp = http.post("/api/v1/memory/voice", data={"transcript": said, "wake": "true", "defer": "true"},
                     files=[("image", ("photo.jpg", image, "image/jpeg"))])
    if followup := resp.headers.get("x-followup"):
        done = http.post(f"/api/v1/memory/followup/{followup}", data={"timeline": "{}"})
        print(f"   follow-up call {followup}: {done.status_code}")
    return resp


def show_answer(resp: httpx.Response, out: Path) -> None:
    print(f"   {resp.status_code}, intent {resp.headers.get('x-intent')}, answered by {resp.headers.get('x-answered-by')}")
    if resp.headers.get("x-answer"):
        print(f"   answer: {unquote(resp.headers['x-answer'])}")
    if resp.headers.get("content-type", "").startswith("audio/"):
        out.write_bytes(resp.content)
        print(f"   speech saved: {out} ({len(resp.content) / 1024:.0f} KB)")
    elif resp.status_code >= 400:
        print(f"   {resp.text[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="a JPEG with the item in it")
    ap.add_argument("--item", default="pin")
    ap.add_argument("--env-file", type=Path, default=ROOT / "device" / "device.env")
    ap.add_argument("--no-db-check", action="store_true", help="don't read raia's result from DATABASE_URL")
    ap.add_argument("--raia-wait", type=float, default=60, help="seconds to wait for the raia read (default 60)")
    ap.add_argument("--out", type=Path, default=Path("."), help="where the answers' speech is saved")
    args = ap.parse_args()

    load_env(args.env_file)
    api = os.environ.get("API_URL", "https://www.lastscene.photo").rstrip("/")
    image = args.image.read_bytes()
    print(f"API {api}, item {args.item!r}, photo {args.image} ({len(image) / 1024:.0f} KB)")
    headers = {"Authorization": f"Bearer {token()}"}
    if os.environ.get("PAIRING_CODE"):
        headers["X-Pairing-Code"] = os.environ["PAIRING_CODE"]
    http = httpx.Client(base_url=api, headers=headers, timeout=60)
    started = datetime.now(timezone.utc)

    step(1, "Passive sighting (as the camera Pi posts it)")
    obs = observation(args.item, 0.87)
    for attempt in range(5):
        resp = http.post("/api/v1/observations", data={"observation": json.dumps(obs)},
                         files={"image": ("frame.jpg", image, "image/jpeg")})
        if resp.status_code != 429:  # the camera Pi shares this device's 2 s upload limit
            break
        time.sleep(float(resp.headers.get("retry-after", 2)))
    print(f"   {resp.status_code} {resp.text[:200]}")
    if resp.status_code not in (200, 201):
        return 1
    seen = http.get(f"/api/v1/items/{args.item}/last-seen")
    if seen.status_code == 200:
        body = seen.json()
        print(f"   last seen: {body['captured_at']} at confidence {body['confidence']}, photo kept: {bool(body['image_url'])}")

    step(2, "Silent raia read (after the response; read-only check)")
    if args.no_db_check:
        print("   skipped")
    else:
        row = asyncio.run(raia_row(args.item, started, args.raia_wait))
        if row is None:
            print(f"   no read within {args.raia_wait:g} s: in its 30 min cooldown, or not one of the household's items "
                  "(grep RAIA in the Vercel logs)")
        elif row["error"]:
            print(f"   failed: {row['error']}")
        else:
            print(f"   {(row['finished_at'] - row['claimed_at']).total_seconds():.1f} s, conversation {row['conversation_id']}")
            print(f"   JSON: {row['context']}")
            print(f"   memory: {row.get('transcript')}")

    step(3, f'Ask: "Last Scene, where is my {args.item}?"')
    show_answer(voice(http, f"Last Scene, where is my {args.item}?", image), args.out / "answer-1.wav")

    step(4, f'Ask: "Last Scene, which way is my {args.item}?"')
    show_answer(voice(http, f"Last Scene, which way is my {args.item}?", image), args.out / "answer-2.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
