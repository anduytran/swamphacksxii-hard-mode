"""An indoor map for the dashboard: where items are, where they get lost, how they move.

The wearable has no GPS, so places are words ("hallway table", "charger by the couch").
Each location is matched to a room on a floor plan by keywords, with no model calls.
Coordinates are abstract floor-plan units (x right, y down), drawn by deck.gl in an
orthographic view. Two plans ship: a one-floor home and the Reitz Union (the SwampHacks
venue); FLOOR_PLAN picks one.
"""

from dataclasses import dataclass

from .reitz_union import REITZ_ROOMS, REITZ_UNKNOWN
from .rooms import Room


# A one-floor home, 100 x 70 units. Earlier keywords win, so specific places
# ("hallway table", "kitchen drawer") come before generic ones ("table", "drawer").
HOME_ROOMS = (
    Room("hallway", "Hallway", 0, 30, 60, 12, ("hallway", "hall", "front door", "entry", "door", "coat", "hook", "shoe")),
    Room("kitchen", "Kitchen", 0, 0, 35, 30, ("kitchen", "counter", "fridge", "kettle", "stove", "pantry")),
    Room("bathroom", "Bathroom", 40, 42, 20, 28, ("bathroom", "sink", "shower", "toilet", "medicine cabinet", "mirror")),
    Room("bedroom", "Bedroom", 0, 42, 40, 28, ("bedroom", "nightstand", "bedside", "bed", "dresser", "wardrobe", "closet", "pillow")),
    Room("dining", "Dining", 35, 0, 25, 30, ("dining", "fruit bowl")),
    Room("living", "Living room", 60, 0, 40, 40, ("living", "couch", "sofa", "armrest", "armchair", "tv", "coffee table", "fireplace", "charger")),
    Room("study", "Study", 60, 40, 40, 30, ("study", "office", "desk", "bookshelf", "computer", "printer")),
)

@dataclass(frozen=True)
class FloorPlan:
    id: str
    name: str
    rooms: tuple[Room, ...]
    unknown: Room

    @property
    def match_order(self) -> list[Room]:
        """Rooms before corridors, so "the hallway outside the ballroom" finds the ballroom."""
        return sorted(self.rooms, key=lambda r: r.kind == "hall")

    def room(self, room_id: str) -> Room:
        return next((r for r in (*self.rooms, self.unknown) if r.id == room_id), self.unknown)


UNKNOWN = Room("unknown", "Elsewhere", 106, 0, 18, 70, ())  # away from the plan, or unmatched
PLANS = {
    "home": FloorPlan("home", "Home", HOME_ROOMS, UNKNOWN),
    "reitz_union": FloorPlan("reitz_union", "Reitz Union", REITZ_ROOMS, REITZ_UNKNOWN),
}
HOME = PLANS["home"]


def floor_plan(plan_id: str) -> FloorPlan:
    return PLANS.get(plan_id, HOME)


def room_for(*texts: str | None, plan: FloorPlan = HOME) -> Room:
    """The room a location (falling back to the scene description) refers to."""
    for text in texts:
        if not text:
            continue
        lowered = f" {text.lower()} "
        for room in plan.match_order:
            if any(f" {kw}" in lowered for kw in room.keywords):
                return room
    return plan.unknown
