"""An indoor map for the dashboard: where items are, where they get lost, how they move.

The wearable has no GPS, so places are words ("hallway table", "charger by the couch").
Each location is matched to a room on a default floor plan by keywords, with no model
calls. Coordinates are abstract floor-plan units (x right, y down), drawn by deck.gl in
an orthographic view. Adjust ROOMS to match the real home.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    x: float
    y: float
    w: float
    h: float
    keywords: tuple[str, ...]

    @property
    def polygon(self) -> list[list[float]]:
        return [[self.x, self.y], [self.x + self.w, self.y], [self.x + self.w, self.y + self.h], [self.x, self.y + self.h]]

    def slot(self, index: int, count: int) -> list[float]:
        """Evenly spaced spots for the items currently in the room, so none overlap."""
        cols = max(1, round(count**0.5 + 0.49))
        rows = -(-count // cols)
        col, row = index % cols, index // cols
        return [
            round(self.x + self.w * (0.15 + 0.7 * (col + 0.5) / cols), 2),
            round(self.y + self.h * (0.3 + 0.55 * (row + 0.5) / rows), 2),
        ]

    def spot(self, key: str) -> list[float]:
        """A stable point inside the room for an item, so dots don't stack."""
        digest = hashlib.sha1(key.encode()).digest()
        fx, fy = digest[0] / 255, digest[1] / 255
        return [round(self.x + self.w * (0.2 + 0.6 * fx), 2), round(self.y + self.h * (0.3 + 0.55 * fy), 2)]


# A one-floor home, 100 x 70 units. Earlier keywords win, so specific places
# ("hallway table", "kitchen drawer") come before generic ones ("table", "drawer").
ROOMS = (
    Room("hallway", "Hallway", 0, 30, 60, 12, ("hallway", "hall", "front door", "entry", "door", "coat", "hook", "shoe")),
    Room("kitchen", "Kitchen", 0, 0, 35, 30, ("kitchen", "counter", "fridge", "kettle", "stove", "pantry")),
    Room("bathroom", "Bathroom", 40, 42, 20, 28, ("bathroom", "sink", "shower", "toilet", "medicine cabinet", "mirror")),
    Room("bedroom", "Bedroom", 0, 42, 40, 28, ("bedroom", "nightstand", "bedside", "bed", "dresser", "wardrobe", "closet", "pillow")),
    Room("dining", "Dining", 35, 0, 25, 30, ("dining", "fruit bowl")),
    Room("living", "Living room", 60, 0, 40, 40, ("living", "couch", "sofa", "armrest", "armchair", "tv", "coffee table", "fireplace", "charger")),
    Room("study", "Study", 60, 40, 40, 30, ("study", "office", "desk", "bookshelf", "computer", "printer")),
)
UNKNOWN = Room("unknown", "Elsewhere", 106, 0, 18, 70, ())  # away from home, or unmatched
BY_ID = {r.id: r for r in (*ROOMS, UNKNOWN)}


def room_for(*texts: str | None) -> Room:
    """The room a location (falling back to the scene description) refers to."""
    for text in texts:
        if not text:
            continue
        lowered = f" {text.lower()} "
        for room in ROOMS:
            if any(f" {kw}" in lowered for kw in room.keywords):
                return room
    return UNKNOWN
