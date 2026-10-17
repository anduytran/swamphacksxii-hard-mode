"""A room on a floor plan (see home_map)."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    x: float  # x, y, w, h: the rectangle items are placed in (the whole room, unless shaped)
    y: float
    w: float
    h: float
    keywords: tuple[str, ...]
    shape: tuple[tuple[float, float], ...] = ()  # outline, when the room isn't a rectangle
    kind: str = "room"  # room | hall (a level's corridors: drawn under the rooms, matched last)
    level: int = 0  # floor, for plans with several (drawn stacked, ground at the bottom)

    @property
    def polygon(self) -> list[list[float]]:
        if self.shape:
            return [[x, y] for x, y in self.shape]
        return [[self.x, self.y], [self.x + self.w, self.y], [self.x + self.w, self.y + self.h], [self.x, self.y + self.h]]

    def slot(self, index: int, count: int) -> list[float]:
        """Evenly spaced spots for the items currently in the room, so none overlap."""
        cols = 1 if self.w * 2 < self.h else max(1, round(count**0.5 + 0.49))  # tall, narrow: one column
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
