"""Where the wearer is, compared with an item's LastScene position: a sentence for spoken answers.

An item's LastScene is the newest place it was recorded: its newest located sighting (db_history), else the
memory its last-seen frame came from, else the newest located memory that names it. The wearer's position is
this request's Wi-Fi scan (or device position), else their last known one (LOCATION_REUSE_MINUTES). Both go
through the household's places (geo.py), so the sentence uses the place's name:

    "Its LastScene is here at Reitz Union."          same place, or within SAME_PLACE_M (or the fixes' accuracy)
    "Its LastScene is at Home, about 1.3 kilometers from you."
"""

from dataclasses import dataclass
from datetime import datetime

from .config import Settings
from .geo import Location, distance_m, home_place_id, nearest_place, place_label, resolve_location
from .observations import scope_for

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
    return compare(ItemPosition(*found), here, here_place, labels)
