"""Where a memory happened: Wi-Fi positioning, places, and public statistics.

The wearable (a Pi Zero 2 W, no GPS) sends the Wi-Fi access points it can hear. The
server turns them into coordinates with a geolocation service (Google when a key is set,
then BeaconDB, which is free and needs no key), so no key lives on the device and results
can be cached. The scan itself is kept only with its save, for recognising the household's
named places (wifi_places.py), and goes with it in the retention purge.

Positions are then grouped into the household's places (home, a café, a park). Places
start private; only public places in households that opted in add to the public map.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .config import Settings
from .db_places import Place

log = logging.getLogger(__name__)

MIN_ACCESS_POINTS = 2  # the Google API needs at least two to answer
MAX_ACCESS_POINTS = 15
DEFAULT_PLACE_RADIUS_M = 75


@dataclass
class Location:
    lat: float
    lng: float
    accuracy_m: float | None
    source: str  # wifi | device


@dataclass
class AccessPoint:
    bssid: str
    signal_dbm: int | None = None


def parse_wifi(raw: str | None) -> list[AccessPoint]:
    """The device's scan, as JSON: [{"bssid": "aa:bb:..", "signal": -61}, ...]."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return []
    aps = []
    for e in entries if isinstance(entries, list) else []:
        bssid = str(e.get("bssid", "")).lower() if isinstance(e, dict) else ""
        if len(bssid) == 17 and bssid.count(":") == 5:
            signal = e.get("signal")
            aps.append(AccessPoint(bssid, int(signal) if isinstance(signal, (int, float)) else None))
    aps.sort(key=lambda a: a.signal_dbm if a.signal_dbm is not None else -200, reverse=True)
    return aps[:MAX_ACCESS_POINTS]


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance (haversine)."""
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def usage_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


class Geolocator:
    """Wi-Fi scan -> coordinates: Google's Geolocation API behind a spend guard, then BeaconDB.

    Google is asked only when GOOGLE_GEOLOCATION_API_KEY is set, and every call is first
    counted against GEOLOCATION_MONTHLY_LIMIT in the database; once the month's budget is used
    (or the database can't be reached), no Google call is made. BeaconDB (beacondb.net, the
    community successor to Mozilla Location Service) takes the same request, is free and needs
    no key, so it answers without a key, past the budget, and when Google finds nothing.
    Answers are cached in the database, so a spot the wearer visits often is looked up once.
    """

    URL = "https://www.googleapis.com/geolocation/v1/geolocate"
    # BeaconDB asks clients to identify themselves.
    USER_AGENT = "LastScene/1.0 (+https://www.lastscene.photo)"

    def __init__(self, settings: Settings, db=None):
        self.key = settings.google_geolocation_api_key
        self.limit = settings.geolocation_monthly_limit
        self.beacondb_url = settings.beacondb_url
        self.cache_days = settings.geolocation_cache_days
        self.db = db  # without one there is no budget to check, so Google is never called
        self._cache: dict[str, Location] = {}
        self._warned_month: str | None = None

    @property
    def google_enabled(self) -> bool:
        return bool(self.key) and self.limit > 0

    @property
    def providers(self) -> list[str]:
        return [name for name, on in (("google", self.google_enabled), ("beacondb", bool(self.beacondb_url))) if on]

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    async def locate(self, aps: list[AccessPoint]) -> Location | None:
        if not self.enabled or len(aps) < MIN_ACCESS_POINTS:
            return None
        # The three strongest access points identify a spot well enough to reuse an answer.
        cache_key = hashlib.sha256("|".join(sorted(a.bssid for a in aps[:3])).encode()).hexdigest()
        if cached := self._cache.get(cache_key):
            return cached
        db_ok = self.db is not None
        try:
            if db_ok and (hit := await self.db.cached_position(cache_key, self.cache_days)):
                loc = Location(*hit, "wifi")
                self._cache[cache_key] = loc
                return loc
        except Exception:
            log.exception("Geolocation cache unavailable; skipping Google")
            db_ok = False  # the budget can't be trusted either
        body = {
            "considerIp": False,  # the server's IP says nothing about the wearer
            "wifiAccessPoints": [
                {"macAddress": a.bssid, **({"signalStrength": a.signal_dbm} if a.signal_dbm is not None else {})}
                for a in aps
            ],
        }
        loc = None
        if db_ok and await self._google_allowed():
            loc = await self._ask("Google", self.URL, body, params={"key": self.key})
        if loc is None and self.beacondb_url:
            loc = await self._ask("BeaconDB", self.beacondb_url, body)
        if loc is None:
            return None
        self._cache[cache_key] = loc
        if db_ok:
            try:
                await self.db.cache_position(cache_key, loc.lat, loc.lng, loc.accuracy_m)
            except Exception:
                log.exception("Could not cache a geolocation answer")
        return loc

    async def _google_allowed(self) -> bool:
        """Reserve one call of this month's Google budget; False without a key or a working check."""
        if not self.google_enabled or self.db is None:
            return False
        month = usage_month()
        try:
            if await self.db.reserve_geolocation_call(month, self.limit):
                return True
        except Exception:
            log.exception("Geolocation spend guard unavailable; skipping Google")
            return False  # fail closed: no budget check, no paid call
        if self._warned_month != month:
            log.warning("Google geolocation budget for %s used up (%d calls); using BeaconDB only", month, self.limit)
            self._warned_month = month
        return False

    async def _ask(self, name: str, url: str, body: dict, params: dict | None = None) -> Location | None:
        try:
            async with httpx.AsyncClient(timeout=10, headers={"User-Agent": self.USER_AGENT}) as client:
                resp = await client.post(url, params=params, json=body)
        except httpx.HTTPError:
            log.exception("%s geolocation request failed", name)
            return None
        if resp.status_code == 404:
            return None  # no estimate for these access points
        if resp.status_code != 200:
            log.warning("%s geolocation returned %s: %s", name, resp.status_code, resp.text[:200])
            return None
        try:
            data = resp.json()
            return Location(float(data["location"]["lat"]), float(data["location"]["lng"]), data.get("accuracy"), "wifi")
        except (ValueError, KeyError, TypeError):
            log.warning("%s geolocation answer unreadable: %s", name, resp.text[:200])
            return None


async def resolve_location(
    geo: Geolocator, wifi: str | None, lat: float | None, lng: float | None, accuracy_m: float | None
) -> Location | None:
    """Wi-Fi scan first; a position the device sent itself (e.g. a fixed home) as fallback."""
    if loc := await geo.locate(parse_wifi(wifi)):
        return loc
    if lat is not None and lng is not None and -90 <= lat <= 90 and -180 <= lng <= 180:
        return Location(lat, lng, accuracy_m, "device")
    return None


def nearest_place(places: list[Place], loc: Location) -> Place | None:
    best, best_d = None, None
    for p in places:
        d = distance_m(p.lat, p.lng, loc.lat, loc.lng)
        if d <= max(p.radius_m, loc.accuracy_m or 0) and (best_d is None or d < best_d):
            best, best_d = p, d
    return best


async def assign_place(db, household_id: str, loc: Location) -> Place:
    """The household place this position belongs to, creating a private one if new."""
    places = await db.places_for(household_id)
    if place := nearest_place(places, loc):
        return place
    radius = max(DEFAULT_PLACE_RADIUS_M, min(loc.accuracy_m or 0, 250))
    return await db.create_place(household_id, loc.lat, loc.lng, radius)


def place_label(place: Place, home_id: str | None, index: int) -> str:
    if place.name:
        return place.name
    if place.id == home_id:
        return "Home"
    return f"Place {index}"


def home_place_id(places: list[Place]) -> str | None:
    """The most-visited place is home until someone names the places themselves."""
    visited = [p for p in places if p.visits]
    if not visited or any((p.name or "").lower() == "home" for p in places):
        return next((p.id for p in places if (p.name or "").lower() == "home"), None)
    return max(visited, key=lambda p: p.visits).id


async def locate_and_record(
    db, device_id: str, table: str, row_id: str, loc: Location | None
) -> tuple[Place | None, bool]:
    """Store a memory's/question's position and place. Returns (place, counts_publicly)."""
    if loc is None:
        return None, False
    household = await db.household_of_device(device_id)
    place = await assign_place(db, household[0], loc) if household else None
    await db.set_location(table, row_id, loc.lat, loc.lng, loc.accuracy_m, place.id if place else None)
    public = bool(household and household[1] and place and place.kind == "public")
    return place, public


async def record_public(db, device_id: str, loc: Location, kind: str, items: list[str]) -> None:
    household = await db.household_of_device(device_id)
    if household and items:
        await db.add_public_stats(household[0], loc.lat, loc.lng, kind, items, datetime.now(timezone.utc))


async def place_unplaced(db, household_id: str, device_ids: list[str]) -> int:
    """Group memories that were located before their device joined a household."""
    count = 0
    for m in await db.unplaced_memories(device_ids):
        place = await assign_place(db, household_id, Location(m["lat"], m["lng"], m["accuracy_m"], "wifi"))
        await db.set_location("memory_logs", m["id"], m["lat"], m["lng"], m["accuracy_m"], place.id)
        count += 1
    return count
