from fastapi import Depends, Request

from .ai import AIServices
from .auth import Principal, require_caregiver, require_device
from .config import Settings, get_settings
from .db import MemoryDB
from .geo import Geolocator
from .limits import Limiter, per
from .raia import Raia


async def get_db(request: Request) -> MemoryDB:
    db = request.app.state.db
    # Connects (and creates the schema) on first use, which serverless hosts rely on.
    if ensure_ready := getattr(db, "ensure_ready", None):
        await ensure_ready()
    return db


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai


def get_raia(request: Request) -> Raia | None:
    return getattr(request.app.state, "raia", None)


def get_geo(request: Request) -> Geolocator:
    return request.app.state.geo


# ---------------------------------------------------------------- request limits (limits.py)

async def _take(request: Request, rules) -> None:
    db = await get_db(request)
    if hasattr(db, "take_slots"):  # the real database; test fakes without it aren't limited
        await Limiter(db).take(rules)


def device_limit(kind: str = "spoken"):
    """Per-device caps: "spoken" (memories, questions, live lookups) or "observations"
    (the vision pipeline's uploads). Over a cap, RateLimited becomes a 429 with Retry-After."""

    async def check(request: Request, device: Principal = Depends(require_device),
                    settings: Settings = Depends(get_settings)) -> Principal:
        if kind == "observations":
            rules = per(f"device:{device.sub}:observations", settings.device_observations_rpm)
        else:
            rules = per(f"device:{device.sub}:{kind}", settings.device_rpm, settings.device_rpd)
        await _take(request, rules)
        return device

    return check


async def caregiver_ai_limit(request: Request, caregiver: Principal = Depends(require_caregiver),
                             settings: Settings = Depends(get_settings)) -> Principal:
    """Per-user cap on dashboard requests that call a model (search, summary)."""
    await _take(request, per(f"caregiver:{caregiver.sub}:ai", settings.caregiver_ai_rpm))
    return caregiver
