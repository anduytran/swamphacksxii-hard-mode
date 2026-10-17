"""Suggested boxes for Teach photos: Gemini finds the item, the caregiver confirms or redraws.

Gemini's detection format is box_2d = [ymin, xmin, ymax, xmax] on a 0-1000 scale; the app's
is [x, y, w, h] in 0..1 with a top-left origin (training samples and observations alike).
A suggestion is only ever a starting point for the BoxEditor, so every failure (vision off,
quota, a malformed answer) comes back as "no suggestion" and the caregiver draws by hand.
"""

import logging

from pydantic import BaseModel, Field

log = logging.getLogger("lastscene.suggest")

SUGGEST_SYSTEM_PROMPT = """\
You help a family teach a camera to recognise one household item: {name}{aliases}. \
Find that item in the photo.
- found: true only if the item is clearly in the photo.
- box_2d: a tight box around the item as [ymin, xmin, ymax, xmax], each 0 to 1000 \
(0,0 is the top-left corner). If several match, box the most visible one. [0, 0, 0, 0] when not found.
- quality: a short note on the photo only if it hurts teaching ("blurry", "too dark", \
"item cut off", "item very small"); otherwise an empty string."""


class BoxSuggestion(BaseModel):
    found: bool
    box_2d: list[float] = Field(default_factory=list)  # usually whole numbers, but 512.5 is fine too
    quality: str = ""


def box_2d_to_xywh(box_2d: list[float]) -> list[float] | None:
    """[ymin, xmin, ymax, xmax] on 0-1000 to [x, y, w, h] on 0..1, clamped to the photo.

    None for anything that isn't a usable box (wrong length, a sliver, nothing left after clamping).
    Swapped corners are put right, since a model sometimes returns them reversed."""
    if len(box_2d) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (min(1.0, max(0.0, float(v) / 1000)) for v in box_2d)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((xmin, xmax))
    y0, y1 = sorted((ymin, ymax))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]


async def suggest_box(ai, image: bytes, content_type: str, name: str, aliases: list[str]) -> tuple[list[float] | None, str]:
    """(box or None, quality note). Raises on API errors; the route turns those into "failed"."""
    from google.genai import types

    also = f" (also called {', '.join(aliases)})" if aliases else ""
    parsed = await ai.gemini.generate_json(
        SUGGEST_SYSTEM_PROMPT.format(name=name, aliases=also),
        [types.Part.from_bytes(data=image, mime_type=content_type), f"Where is the {name}?"],
        BoxSuggestion,
        fast=True,
    )
    if parsed is None:
        return None, ""
    note = " ".join(parsed.quality.split())[:60]
    return (box_2d_to_xywh(parsed.box_2d) if parsed.found else None), note
