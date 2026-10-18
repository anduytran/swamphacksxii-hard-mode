"""Test the location data on live observation POSTs, with the device's own credentials, then clean up.

    .venv/bin/python scripts/live_location_test.py --image frame.jpg [--lat 29.6465 --lng -82.3475]

1. A sighting with its own position ("location": {"lat", "lng", "accuracy_m"}).
2. A sighting with no location: it should take the wearer's last known position (step 1's, within 10 min).
3. (--wifi) A sighting with only a Wi-Fi scan, resolved on the server (needs 2+ access points it knows).
Each is read back from DATABASE_URL (read-only): lat, lng, accuracy and the household place it was assigned.
Uses labels "geo test 1", "geo test 2"... (not your items, so no raia read; each its own, so none merge) and
deletes every test sighting at the end.
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

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_client_test import ROOT, load_env, token  # noqa: E402

def observation(label: str, location: dict | None) -> dict:
    obs = {
        "observation_id": str(uuid.uuid4()),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "auto",
        "pipeline": {"name": "live-location-test", "version": "1", "detector": "test"},
        "image_size": {"width": 768, "height": 432},
        "objects": [{"label": label, "confidence": 0.9, "bbox": [0.4, 0.3, 0.2, 0.3], "event": "placed"}],
    }
    if location:
        obs["location"] = location
    return obs


def post(http: httpx.Client, obs: dict, image: bytes) -> httpx.Response:
    for _ in range(6):  # the camera Pi shares this device's 2 s upload limit
        resp = http.post("/api/v1/observations", data={"observation": json.dumps(obs)},
                         files={"image": ("frame.jpg", image, "image/jpeg")})
        if resp.status_code != 429:
            return resp
        time.sleep(float(resp.headers.get("retry-after", 2)))
    return resp


async def located(memory_id: str, wait_s: float = 30) -> dict | None:
    """The memory's position and place once the server has resolved it (after its response)."""
    import asyncpg

    from app.config import Settings

    conn = await asyncpg.connect(Settings().database_url, timeout=15)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            row = await conn.fetchrow(
                "SELECT m.lat, m.lng, m.accuracy_m, p.name, p.kind, p.id AS place_id FROM memory_logs m "
                "LEFT JOIN places p ON p.id = m.place_id WHERE m.id = $1::uuid", memory_id)
            if row and row["lat"] is not None:
                return dict(row)
            await asyncio.sleep(2)
        return None
    finally:
        await conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--lat", type=float, default=29.6465)  # Reitz Union, UF
    ap.add_argument("--lng", type=float, default=-82.3475)
    ap.add_argument("--wifi", help='JSON scan to try too: [{"bssid": "...", "signal": -60}, ...]')
    ap.add_argument("--env-file", type=Path, default=ROOT / "device" / "device.env")
    args = ap.parse_args()

    load_env(args.env_file)
    image = args.image.read_bytes()
    api = os.environ.get("API_URL", "https://www.lastscene.photo").rstrip("/")
    http = httpx.Client(base_url=api, headers={"Authorization": f"Bearer {token()}"}, timeout=60)
    cases = [("own position", {"lat": args.lat, "lng": args.lng, "accuracy_m": 20.0}),
             ("no location (borrows the last position)", None)]
    if args.wifi:
        cases.append(("Wi-Fi scan only", {"wifi": json.loads(args.wifi)}))
    sent, ok = [], True
    try:
        for n, (title, location) in enumerate(cases, 1):
            obs = observation(f"geo test {n}", location)
            resp = post(http, obs, image)
            print(f"\n== {n}. {title}: POST {resp.status_code} {resp.text[:160]}")
            if resp.status_code not in (200, 201):
                ok = False
                continue
            sent.append(obs["observation_id"])
            where = asyncio.run(located(resp.json()["memory_id"]))
            if where is None:
                ok = False
                print("   not located within 30 s")
            else:
                place = f'{where["name"] or "(unnamed)"} ({where["kind"]})' if where["place_id"] else "none (device not in a household)"
                print(f'   located: {where["lat"]:.5f}, {where["lng"]:.5f} ±{where["accuracy_m"] or 0:.0f} m; place: {place}')
    finally:
        for oid in sent:
            gone = http.delete(f"/api/v1/observations/{oid}")
            print(f"cleanup: DELETE observation {oid[:8]}… {gone.status_code}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
