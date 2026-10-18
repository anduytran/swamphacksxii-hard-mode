"""Where the wearer is, compared with an item's LastScene: a sentence for spoken answers.

An item's LastScene is the newest place it was recorded: its newest sighting (db_history), else the memory its
last-seen frame came from, else the newest memory that names it, among those with a Wi-Fi scan or a position.

Wi-Fi fingerprints first (wifi_places.py): the item's scan against the wearer's (this request's, else their latest),
and both against the household's named places. That answers "here or elsewhere" indoors and where Google and
BeaconDB have no data:

Positions (Google, then BeaconDB, geo.py) give the distance and compass direction from the wearer; the rooms the
two fingerprints match on the floor plan give the floor difference (relative_to_item, Relative.sentence):

    "Its LastScene is here at meeting room 2360."   the scans match (or fall in the same named place)
    "Its LastScene is in meeting room 2360, about 40 meters north-east of you, one floor up."
    "Its LastScene is at Home, about 1.4 kilometers south-west of you."
    "Its LastScene is somewhere else, not where you are now."   scans differ, no names, no positions
"""

import math
from dataclasses import dataclass
from datetime import datetime

from .config import Settings
from .geo import Location, distance_m, home_place_id, nearest_place, place_label, resolve_location
from .observations import scope_for
from .home_map import floor_plan
from .wifi_places import HERE, best_place, display, from_wifi_json, room_of_scan, similarity, usable

SAME_PLACE_M = 75.0  # a map place's radius
HERE_M = 30.0  # by positions alone, "here" within this or the two fixes' accuracy, whichever is more


COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
FLOORS = {1: "one floor up", 2: "two floors up", 3: "three floors up", -1: "one floor down", -2: "two floors down",
          -3: "three floors down"}


def bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Degrees clockwise from north, from the first point toward the second."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compass(degrees: float) -> str:
    return COMPASS[round(degrees / 45) % 8]


@dataclass
class Relative:
    """Where an item's LastScene is from the wearer: for the spoken sentence and the dashboard."""
    here: bool  # the same spot: the scans match, the same named place, or within the fixes' accuracy
    meters: float | None  # between the two positions (Google / BeaconDB), when both are known
    bearing: float | None  # degrees from north, wearer -> item
    floors: int | None  # the item's floor minus the wearer's, when both rooms are known
    item_room: str | None  # floor-plan room names, from the surveyed fingerprints
    wearer_room: str | None
    item_place: str | None  # a named place or map place, when there's no room
    wearer_place: str | None
    item_at: datetime
    accuracy_m: float | None  # the two fixes' accuracy together
    wearer: Location | None = None  # the wearer's position, for distances to earlier sightings

    @property
    def compass(self) -> str | None:
        return compass(self.bearing) if self.bearing is not None else None

    def sentence(self) -> str:
        where = self.item_room or self.item_place
        if self.here and not self.floors:
            return f"Its LastScene is here at {spoken_place(where)}." if where else "Its LastScene is right around here."
        parts = []
        if self.meters is not None and self.meters >= HERE_M:
            parts.append(f"about {spoken_distance(self.meters)} {self.compass} of you")
        if self.floors:
            parts.append(FLOORS.get(self.floors, f"{abs(self.floors)} floors {'up' if self.floors > 0 else 'down'}"))
        if where:
            lead = f"Its LastScene is {'in' if self.item_room else 'at'} {spoken_place(where)}"
            return f"{lead}, {', '.join(parts)}." if parts else f"{lead}, not here."
        if parts:
            return f"Its LastScene is {', '.join(parts)}."
        return "Its LastScene is somewhere else, not where you are now."


def spoken_place(name: str) -> str:
    """"Meeting Room 2360" -> "meeting room 2360"; "Grand Ballroom" -> "the Grand Ballroom"; a named place as is."""
    if name.startswith("Meeting Room "):
        return "meeting room " + name.removeprefix("Meeting Room ")
    if name.startswith(("Home", "The ")) or not name[:1].isupper():
        return name
    return f"the {name}" if " " in name or name in {"Gallery", "Auditorium"} else name


def spoken_distance(meters: float) -> str:
    if meters < 1000:
        step = 10 if meters < 100 else 50
        return f"{max(step, round(meters / step) * step):.0f} meters"
    return f"{meters / 1000:.1f} kilometers"


async def wearer_position(geo, db, device_ids: list[str], wifi: str | None, lat: float | None, lng: float | None,
                          accuracy_m: float | None, settings: Settings, minutes: float | None = None) -> Location | None:
    if loc := await resolve_location(geo, wifi, lat, lng, accuracy_m):
        return loc
    minutes = settings.location_reuse_minutes if minutes is None else minutes
    if minutes > 0 and hasattr(db, "recent_position"):
        if found := await db.recent_position(device_ids, minutes):
            return Location(found[0], found[1], found[2], "recent")
    return None


def _named_position(named: list[dict], scan: dict[str, int]) -> Location | None:
    found = best_place(named, scan) if usable(scan) else None
    if found and found[0].get("lat") is not None:
        return Location(found[0]["lat"], found[0]["lng"], found[0].get("accuracy_m"), "named place")
    return None


async def relative_to_item(db, geo, settings: Settings, scope: str, device_ids: list[str], item: str,
                           wifi: str | None = None, lat: float | None = None, lng: float | None = None,
                           accuracy_m: float | None = None, minutes: float | None = None) -> Relative | None:
    """Where the item's LastScene is from the wearer: this request's scan or position, else their latest within
    `minutes` (LOCATION_REUSE_MINUTES). None when either side has neither a scan nor a position."""
    item = " ".join((item or "").lower().split())
    if not item or not hasattr(db, "lastscene_position"):
        return None
    found = await db.lastscene_position(scope, device_ids, item)
    if found is None:
        return None
    minutes = settings.location_reuse_minutes if minutes is None else minutes
    item_scan = found.get("wifi") or {}
    here_scan = from_wifi_json(wifi)
    if not usable(here_scan) and hasattr(db, "recent_scan"):
        here_scan = await db.recent_scan(device_ids, minutes) or {}
    here = await wearer_position(geo, db, device_ids, wifi, lat, lng, accuracy_m, settings, minutes)
    named = await db.wifi_places(scope) if hasattr(db, "wifi_places") else []
    # No position from a service: a scan that matches a named place is where that place was named.
    if here is None and (spot := _named_position(named, here_scan)):
        here = spot
    if found.get("lat") is None and (spot := _named_position(named, item_scan)):
        found = {**found, "lat": spot.lat, "lng": spot.lng, "accuracy_m": spot.accuracy_m}
    if here is None and not (usable(item_scan) and usable(here_scan)):
        return None

    plan = floor_plan(settings.floor_plan)
    rooms = {}
    for side, scan in (("item", item_scan), ("wearer", here_scan)):
        room_id = await room_of_scan(db, scope, scan) if usable(scan) else None
        rooms[side] = plan.room(room_id) if room_id else None
    places = {side: (display(f[0]["name"]) if (f := best_place(named, scan)) else None)
              for side, scan in (("item", item_scan), ("wearer", here_scan))}

    labels: dict[str, str] = {}
    here_place = None
    household = await db.household_of_device(device_ids[0]) if device_ids else None
    if household and hasattr(db, "places_for"):
        map_places = await db.places_for(household[0])
        home = home_place_id(map_places)
        labels = {p.id: place_label(p, home, i) for i, p in enumerate(map_places, 1)}
        if here and (spot := nearest_place(map_places, here)):
            here_place = spot.id

    meters = bearing_deg = None
    near = HERE_M
    if here is not None and found.get("lat") is not None:
        meters = distance_m(here.lat, here.lng, found["lat"], found["lng"])
        bearing_deg = bearing(here.lat, here.lng, found["lat"], found["lng"])
        near = max(HERE_M, (found.get("accuracy_m") or 0) + (here.accuracy_m or 0))
    same_scan = usable(item_scan) and usable(here_scan) and similarity(item_scan, here_scan) >= HERE
    same_named = bool(places["item"]) and places["item"] == places["wearer"]
    same_map = bool(found.get("place_id")) and found.get("place_id") == here_place and not usable(item_scan)
    if usable(item_scan) and usable(here_scan):
        is_here = same_scan or same_named  # fingerprints decide when both sides have one
    else:
        is_here = same_map or (meters is not None and meters <= near)
    floors = (rooms["item"].level - rooms["wearer"].level) if rooms["item"] and rooms["wearer"] else None
    return Relative(
        here=is_here, meters=meters, bearing=bearing_deg, floors=floors or None,
        item_room=rooms["item"].name if rooms["item"] else None,
        wearer_room=rooms["wearer"].name if rooms["wearer"] else None,
        item_place=places["item"] or labels.get(found.get("place_id") or ""),
        wearer_place=places["wearer"] or labels.get(here_place or ""),
        item_at=found["at"],
        accuracy_m=((found.get("accuracy_m") or 0) + (here.accuracy_m or 0)) if here and found.get("lat") is not None else None,
        wearer=here,
    )


async def lastscene_note(db, geo, settings: Settings, device_id: str, item: str, wifi: str | None = None,
                         lat: float | None = None, lng: float | None = None,
                         accuracy_m: float | None = None) -> str | None:
    """The sentence comparing where the wearer is with the item's LastScene, or None if either isn't known."""
    scope, device_ids = await scope_for(db, device_id)
    rel = await relative_to_item(db, geo, settings, scope, device_ids, item, wifi, lat, lng, accuracy_m)
    return rel.sentence() if rel else None


@dataclass
class ItemContext:
    """What an answer about an item is given: its last LastScene locations and where it is from the wearer."""
    prompt: str  # for the answer model
    sentence: str | None  # the geolocation sentence, said when the model doesn't answer (template, timeout)


def _ago(then: datetime, now: datetime) -> str:
    minutes = max(0, round((now - then).total_seconds() / 60))
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes / 60
    if hours < 10:
        return f"{hours:.1f} h ago".replace(".0 h", " h")
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


async def item_context(db, geo, settings: Settings, device_id: str, item: str, wifi: str | None = None,
                       lat: float | None = None, lng: float | None = None,
                       accuracy_m: float | None = None) -> ItemContext | None:
    """The item's last few LastScene locations (db_history: newest first, the first being where it is now) with
    where each is from the wearer, and the geolocation summary (relative_to_item). None when nothing is known."""
    from datetime import timezone

    item = " ".join((item or "").lower().split())
    if not item:
        return None
    import asyncio

    scope, device_ids = await scope_for(db, device_id)

    async def recent():
        return await db.sightings([scope], item) if hasattr(db, "sightings") else []

    rel, sightings = await asyncio.gather(
        relative_to_item(db, geo, settings, scope, device_ids, item, wifi, lat, lng, accuracy_m), recent())
    if rel is None and not sightings:
        return None
    labels: dict[str, str] = {}
    household = await db.household_of_device(device_id)
    if household and hasattr(db, "places_for"):
        map_places = await db.places_for(household[0])
        home = home_place_id(map_places)
        labels = {p.id: place_label(p, home, i) for i, p in enumerate(map_places, 1)}
    now = datetime.now(timezone.utc)
    lines = []
    for i, s in enumerate(sightings, 1):
        ctx = s.context or {}
        what = ctx.get("where") or ctx.get("summary") or (f"on the {s.scene_label}" if s.scene_label else "seen by the camera")
        bits = [what]
        if s.confidence is not None and s.confidence < settings.maybe_confidence:
            bits.append(f"maybe, only {s.confidence:.0%} sure")
        if ctx.get("nearby"):
            bits.append("next to " + ", ".join(ctx["nearby"]))
        if s.place_id and labels.get(s.place_id):
            bits.append(f"at {labels[s.place_id]}")
        if rel and rel.wearer and s.lat is not None:
            meters = distance_m(rel.wearer.lat, rel.wearer.lng, s.lat, s.lng)
            bits.append("where the wearer is" if meters < HERE_M else
                        f"{spoken_distance(meters)} {compass(bearing(rel.wearer.lat, rel.wearer.lng, s.lat, s.lng))} "
                        f"of the wearer")
        lines.append(f"{i}. {_ago(s.captured_at, now)}: {'; '.join(bits)}")
    parts = []
    if lines:
        parts.append(f"The {item}'s last LastScene locations (seen by the camera), newest first:\n"
                     + "\n".join(lines))
    if rel:
        accuracy = f" (positions within about {rel.accuracy_m:.0f} m)" if rel.accuracy_m else ""
        rooms = ""
        if rel.wearer_room or rel.item_room:
            rooms = f" The wearer is in {rel.wearer_room or 'an unknown room'}; the {item} is in {rel.item_room or 'an unknown room'}."
        said = rel.sentence().replace("Its LastScene", f"The {item}'s LastScene", 1)
        parts.append(f"Where it is from the wearer now: {said}{rooms}{accuracy}")
    return ItemContext("\n\n".join(parts), rel.sentence() if rel else None)
