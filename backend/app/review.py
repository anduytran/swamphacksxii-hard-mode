"""Which photos the Teach page's "Needs a look" queue asks the caregiver about.

Two sources, newest first:
- detection: the vision pipeline boxed a taught item with confidence in the unsure band
  (REVIEW_CONFIDENCE_MIN..MAX): "Is this the keys?" with the detector's box.
- memory: a memory photo whose items name a taught item (or one of its other words) but
  that no detector boxed; the card asks Gemini for a box when it's shown.
A photo already answered for that item (db_review) is never asked again.
"""

from dataclasses import dataclass
from datetime import datetime

from .db_review import Review
from .db_training import TrainingItem


@dataclass
class Candidate:
    item: TrainingItem
    memory_id: str
    kind: str  # detection | memory
    captured_at: datetime
    bbox: list[float] | None
    confidence: float | None
    width: int | None
    height: int | None
    suggested: bool  # a box suggestion was already asked for (bbox may still be None)
    note: str


def _words(text: str) -> list[str]:
    # "Keys" / "key": a trailing plural s doesn't make it a different item.
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in text.lower().split()]


def mentions(item: TrainingItem, name: str) -> bool:
    """Whether an item name from a memory ("car keys") is this taught item ("keys"): one of its
    words for it appears in the name as whole words."""
    said = _words(name)
    for word in [item.name, *item.aliases]:
        want = _words(word)
        if want and any(said[i : i + len(want)] == want for i in range(len(said) - len(want) + 1)):
            return True
    return False


def candidates(
    items: list[TrainingItem], detections: list[dict], photos: list[dict], reviews: dict[tuple[str, str], Review]
) -> list[Candidate]:
    out: dict[tuple[str, str], Candidate] = {}
    for d in detections:
        for item in items:
            key = (item.id, d["memory_id"])
            if not mentions(item, d["label"]) or (key in reviews and reviews[key].decision):
                continue
            if key not in out or (d["confidence"] or 0) > (out[key].confidence or 0):  # one card per frame
                out[key] = Candidate(item, d["memory_id"], "detection", d["created_at"], d["bbox"], d["confidence"],
                                     d["width"], d["height"], False, "")
    for p in photos:
        for item in items:
            key = (item.id, p["memory_id"])
            if key in out or (key in reviews and reviews[key].decision):
                continue
            if any(mentions(item, label) for label in p["boxed"]):  # a detector already boxed it
                continue
            if any(mentions(item, n) for n in p["names"]):
                cached = reviews.get(key)
                out[key] = Candidate(item, p["memory_id"], "memory", p["created_at"], cached.bbox if cached else None, None,
                                     None, None, bool(cached and cached.suggested), cached.note if cached else "")
    return sorted(out.values(), key=lambda c: c.captured_at, reverse=True)
