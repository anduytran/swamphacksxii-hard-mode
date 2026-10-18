"""Places recognized from Wi-Fi fingerprints: "here or elsewhere" with no outside service.

Every save keeps the Wi-Fi scan it was made with (memory_logs.wifi, query_logs.wifi, item_history.wifi; deleted
with the save). A household names a place once by voice ("Last Scene, this is the Reitz Union", "we're in the
kitchen"): the scan becomes that place's fingerprint (wifi_places), refreshed whenever a later scan clearly matches
it. Two scans are compared by the access points they share, weighted by signal strength, so it works indoors, room
by room where the access points differ, and where Google or BeaconDB have no data (UF's buildings, Oct 18).

    fingerprint: {bssid: signal dBm}
    similarity:  sum over shared access points of the weaker weight / sum over all of the stronger weight,
                 weight = signal + 100 (clamped to 1..70): 0 = nothing in common, 1 = the same scan
"""

import json
import re

from .geo import AccessPoint, fixed_ap, parse_wifi

HERE = 0.3  # scans at least this similar were taken in the same place
MATCH = 0.3  # a scan belongs to a named place at least this similar (the best one wins)
LEARN = 0.5  # a scan this similar to a named place refreshes its fingerprint
MIN_APS = 3  # fewer access points than this aren't a fingerprint (faker often hears just one)
KEEP_APS = 40  # a place keeps its strongest access points
UNHEARD_DBM = -85  # a known access point missing from a new scan fades toward this


def fingerprint(aps: list[AccessPoint]) -> dict[str, int]:
    return {a.bssid.lower(): int(a.signal_dbm if a.signal_dbm is not None else -80)
            for a in aps if a.bssid and fixed_ap(a.bssid.lower())}


def fixed(scan: dict[str, int] | None) -> dict[str, int]:
    """A stored scan or fingerprint without hotspots (geo.fixed_ap): ones saved before they were left out."""
    return {k: v for k, v in (scan or {}).items() if fixed_ap(k)}


def from_wifi_json(raw: str | list | None) -> dict[str, int]:
    """A device's scan (the JSON string it sends, or the parsed list) as a fingerprint."""
    if isinstance(raw, list):
        raw = json.dumps(raw)
    return fingerprint(parse_wifi(raw))


def usable(scan: dict | None) -> bool:
    return len(fixed(scan)) >= MIN_APS


def _weight(dbm: int) -> float:
    return float(min(70, max(1, dbm + 100)))


def similarity(a: dict[str, int], b: dict[str, int]) -> float:
    a, b = fixed(a), fixed(b)
    if not a or not b:
        return 0.0
    shared = sum(min(_weight(a[k]), _weight(b[k])) for k in a.keys() & b.keys())
    total = sum(max(_weight(a.get(k, -100)), _weight(b.get(k, -100))) for k in a.keys() | b.keys())
    return shared / total if total else 0.0


def merge(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """A place's fingerprint after another scan there: shared access points average (new counts 30 %), new ones
    join, ones not heard this time fade, and only the strongest KEEP_APS stay."""
    old, new, out = fixed(old), fixed(new), {}
    for k in old.keys() | new.keys():
        if k in old and k in new:
            out[k] = round(0.7 * old[k] + 0.3 * new[k])
        elif k in new:
            out[k] = new[k]
        else:
            out[k] = round(0.8 * old[k] + 0.2 * UNHEARD_DBM)
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:KEEP_APS])


def best_place(places: list[dict], scan: dict[str, int]) -> tuple[dict, float] | None:
    """The named place this scan belongs to, with its similarity, or None."""
    if not usable(scan):
        return None
    scored = [(p, similarity(p["aps"], scan)) for p in places]
    best = max(scored, key=lambda ps: ps[1], default=None)
    return best if best and best[1] >= MATCH else None


def clean_name(raw: str) -> str:
    name = " ".join(re.sub(r"[^\w' -]", " ", raw).split())
    return name[:60]


def display(name: str) -> str:
    """"reitz union" -> "Reitz Union"; keeps what the household capitalised itself."""
    return name if any(c.isupper() for c in name) else " ".join(w[:1].upper() + w[1:] for w in name.split())


async def name_place(db, scope: str, name: str, scan: dict[str, int], room_id: str | None = None) -> dict:
    """Save `scan` as the fingerprint of the place called `name` (merged into it if it exists). `room_id`: the
    floor-plan room the name is, so saves there land in that room on the map."""
    existing = next((p for p in await db.wifi_places(scope) if p["name"].lower() == name.lower()), None)
    aps = merge(existing["aps"], scan) if existing else dict(sorted(scan.items(), key=lambda kv: -kv[1])[:KEEP_APS])
    return await db.save_wifi_place(scope, name, aps, existing["id"] if existing else None, room_id)


async def room_of_scan(db, scope: str, scan: dict[str, int]) -> str | None:
    """The floor-plan room a scan was taken in: the room of the named place it matches, if that place is one."""
    if not usable(scan) or not hasattr(db, "wifi_places"):
        return None
    found = best_place([p for p in await db.wifi_places(scope) if p.get("room_id")], scan)
    return found[0]["room_id"] if found else None


async def learn(db, scope: str, scan: dict[str, int]) -> None:
    """A scan that clearly matches a named place refreshes its fingerprint (access points change over time)."""
    if not usable(scan) or not hasattr(db, "wifi_places"):
        return
    found = best_place(await db.wifi_places(scope), scan)
    if found and found[1] >= LEARN:
        place = found[0]
        await db.save_wifi_place(scope, place["name"], merge(place["aps"], scan), place["id"])
