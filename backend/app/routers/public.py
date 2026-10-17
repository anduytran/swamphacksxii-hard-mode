"""The public, anonymous map of where people commonly lose things. No login.

Only data from places a household marked public, in households that opted in, and only
for ~100 m cells where at least PUBLIC_MIN_HOUSEHOLDS households contributed. Item names
and counts only: no transcripts, photos, device ids, household ids or exact positions.
"""

from fastapi import APIRouter, Depends, Response

from ..db import MemoryDB
from ..db_places import PUBLIC_CELL_DEGREES, PUBLIC_MIN_HOUSEHOLDS, PUBLIC_WINDOW_DAYS
from ..deps import get_db

router = APIRouter(prefix="/api/v1/public", tags=["public"])


@router.get("/lost-map")
async def lost_map(response: Response, db: MemoryDB = Depends(get_db)):
    cells: dict[tuple[float, float], dict] = {}
    for r in await db.public_cells():
        cell = cells.setdefault(
            (r["cell_lat"], r["cell_lng"]),
            {"lat": r["cell_lat"], "lng": r["cell_lng"], "left": 0, "asked": 0, "items": []},
        )
        left, asked = int(r["left_here"] or 0), int(r["asked_here"] or 0)
        cell["left"] += left
        cell["asked"] += asked
        cell["items"].append({"name": r["item"], "left": left, "asked": asked})
    for cell in cells.values():
        cell["items"].sort(key=lambda i: i["left"] + i["asked"], reverse=True)
    response.headers["Cache-Control"] = "public, max-age=300"
    return {
        "cell_degrees": PUBLIC_CELL_DEGREES,
        "min_households": PUBLIC_MIN_HOUSEHOLDS,
        "window_days": PUBLIC_WINDOW_DAYS,
        "cells": sorted(cells.values(), key=lambda c: c["left"] + c["asked"], reverse=True),
    }
