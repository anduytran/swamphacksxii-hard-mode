"""Where the wearer is, compared with an item's LastScene: a sentence for spoken answers.

An item's LastScene is the newest place it was recorded: its newest sighting (db_history), else the memory its
last-seen frame came from, else the newest memory that names it, among those with a Wi-Fi scan or a position.

Wi-Fi fingerprints first (wifi_places.py): the item's scan against the wearer's (this request's, else their latest),
and both against the household's named places. That answers "here or elsewhere" indoors and where Google and
BeaconDB have no data:

    "Its LastScene is here at Reitz Union."     the scans match (or fall in the same named place)
    "Its LastScene is at Home, not here."       the item's scan is a named place the wearer isn't in
    "Its LastScene is somewhere else, not where you are now."

Without two scans, coordinates (Google or BeaconDB, geo.py) and the household's map places:

    "Its LastScene is here at Reitz Union."     same place, or within SAME_PLACE_M (or the fixes' accuracy)
    "Its LastScene is at Home, about 1.3 kilometers from you."
"""

from dataclasses import dataclass
from datetime import datetime

from .config import Settings
from .geo import Location, distance_m, home_place_id, nearest_place, place_label, resolve_location
from .observations import scope_for
from .wifi_places import HERE, best_place, display, from_wifi_json, similarity, usable

SAME_PLACE_M = 75.0


@dataclass
class ItemPosition:
    lat: float
    lng: float
    accuracy_m: float | None
    place_id: str | None
    at: datetime


def spoken_distance(meters: float) -> str:
    if meters < 1000:
        step = 10 if meters < 100 else 50
        return f"{max(step, round(meters / step) * step):.0f} meters"
    return f"{meters / 1000:.1f} kilometers"


def compare(item: ItemPosition, here: Location, here_place: str | None, labels: dict[str, str]) -> str:
    """The sentence: is the item's LastScene here, or how far away (and at which place)."""
    meters = distance_m(item.lat, item.lng, here.lat, here.lng)
    near = max(SAME_PLACE_M, (item.accuracy_m or 0) + (here.accuracy_m or 0))
    name = labels.get(item.place_id) if item.place_id else None
    if (item.place_id and item.place_id == here_place) or meters <= near:
        return f"Its LastScene is here at {name}." if name else "Its LastScene is right around here."
    if name:
        return f"Its LastScene is at {name}, about {spoken_distance(meters)} from you."
    return f"Its LastScene is about {spoken_distance(meters)} from you."


def compare_scans(item_scan: dict[str, int], here_scan: dict[str, int], places: list[dict]) -> str:
    """The sentence from Wi-Fi fingerprints alone."""
    here_place, item_place = best_place(places, here_scan), best_place(places, item_scan)
    here_name = display(here_place[0]["name"]) if here_place else None
    item_name = display(item_place[0]["name"]) if item_place else None
    if similarity(item_scan, here_scan) >= HERE or (here_name and here_name == item_name):
        name = here_name or item_name
        return f"Its LastScene is here at {name}." if name else "Its LastScene is right around here."
    if item_name:
        return f"Its LastScene is at {item_name}, not here."
    return "Its LastScene is somewhere else, not where you are now."


async def wearer_position(geo, db, device_ids: list[str], wifi: str | None, lat: float | None, lng: float | None,
                          accuracy_m: float | None, settings: Settings) -> Location | None:
    if loc := await resolve_location(geo, wifi, lat, lng, accuracy_m):
        return loc
    if settings.location_reuse_minutes > 0 and hasattr(db, "recent_position"):
        if found := await db.recent_position(device_ids, settings.location_reuse_minutes):
            return Location(found[0], found[1], found[2], "recent")
    return None


async def lastscene_note(db, geo, settings: Settings, device_id: str, item: str, wifi: str | None = None,
                         lat: float | None = None, lng: float | None = None,
                         accuracy_m: float | None = None) -> str | None:
    """The sentence comparing where the wearer is with the item's LastScene, or None if either isn't known."""
    item = " ".join((item or "").lower().split())
    if not item or not hasattr(db, "lastscene_position"):
        return None
    scope, device_ids = await scope_for(db, device_id)
    found = await db.lastscene_position(scope, device_ids, item)
    if found is None:
        return None
    item_scan = found.get("wifi") or {}
    here_scan = from_wifi_json(wifi)
    if not usable(here_scan) and hasattr(db, "recent_scan"):
        here_scan = await db.recent_scan(device_ids, settings.location_reuse_minutes) or {}
    if usable(item_scan) and usable(here_scan):
        places = await db.wifi_places(scope) if hasattr(db, "wifi_places") else []
        return compare_scans(item_scan, here_scan, places)
    if found.get("lat") is None:
        return None
    here = await wearer_position(geo, db, device_ids, wifi, lat, lng, accuracy_m, settings)
    if here is None:
        return None
    labels: dict[str, str] = {}
    here_place = None
    household = await db.household_of_device(device_id)
    if household:
        places = await db.places_for(household[0])
        home = home_place_id(places)
        labels = {p.id: place_label(p, home, i) for i, p in enumerate(places, 1)}
        if spot := nearest_place(places, here):
            here_place = spot.id
    return compare(ItemPosition(found["lat"], found["lng"], found["accuracy_m"], found["place_id"], found["at"]),
                   here, here_place, labels)
