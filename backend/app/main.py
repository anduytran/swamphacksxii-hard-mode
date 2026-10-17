import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .ai import AIServices
from .config import Settings, get_settings
from .db import MemoryDB
from .geo import Geolocator, usage_month
from .limits import Limiter, RateLimited, Rule
from .live import LiveTokens
from .mcp_server import router as mcp_router
from .routers import mcp_keys
from .routers import admin, dashboard, households, images, live, memory, observations, public, training

# The built dashboard (cd web && npm run build; Vercel builds it during each deploy).
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"
log = logging.getLogger("lastscene")
# Our own INFO lines (e.g. the per-stage timings of each spoken answer) reach the host's
# logs; nothing configures logging on Vercel, where they'd otherwise be dropped.
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


def missing_settings(s: Settings) -> list[str]:
    """Required settings that are empty. Shown by /healthz, since a missing env var on a
    host like Vercel otherwise just looks like a crash."""
    required = {
        "DATABASE_URL": s.database_url,
        "IMAGE_URL_SECRET": s.image_url_secret,
        "AUTH0_DOMAIN": s.auth0_domain,
        "AUTH0_AUDIENCE": s.auth0_audience,
        "AUTH0_SPA_CLIENT_ID": s.auth0_spa_client_id,
    }
    providers = {s.stt_provider, s.embedding_provider, s.llm_provider, s.tts_provider, s.vision_provider}
    if "gemini" in providers:
        required["GEMINI_API_KEY"] = s.gemini_api_key
    if providers & {"openai"}:
        required["OPENAI_API_KEY"] = s.openai_api_key
    if "claude" in providers:
        required["ANTHROPIC_API_KEY"] = s.anthropic_api_key
    return [name for name, value in required.items() if not value]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Never let startup fail: one bad setting would take down every route, /healthz
    # included. Problems are logged here and reported by /healthz instead.
    if missing := missing_settings(get_settings()):
        log.error("Missing settings: %s", ", ".join(missing))
    try:
        await app.state.db.ensure_ready()  # warm-up; get_db() retries on first use
    except Exception:
        log.exception("Database not reachable at startup")
    yield
    await app.state.db.close()


def create_app(use_lifespan: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LastScene API", lifespan=lifespan if use_lifespan else None)
    app.state.db = MemoryDB(settings)
    app.state.ai = AIServices(settings, limiter=Limiter(app.state.db))
    app.state.geo = Geolocator(settings, app.state.db)
    app.state.live_tokens = LiveTokens()

    @app.exception_handler(RateLimited)
    async def rate_limited(request: Request, exc: RateLimited):
        # Which limit, never whose: buckets name devices and caregivers.
        log.warning("Rate limited (%s) on %s", exc.why, request.url.path)
        return JSONResponse(
            status_code=429,
            content={"detail": "Busy right now; try again shortly", "retry_after": round(exc.retry_after)},
            headers={"Retry-After": str(round(exc.retry_after))},
        )

    app.include_router(memory.router)
    app.include_router(dashboard.router)
    app.include_router(images.router)
    app.include_router(admin.router)
    app.include_router(households.router)
    app.include_router(public.router)
    app.include_router(observations.router)
    app.include_router(training.router)
    app.include_router(live.router)
    app.include_router(mcp_router)  # /mcp, read-only, see docs/MCP.md
    app.include_router(mcp_keys.router)

    async def geolocation_budget(settings) -> dict | None:
        """Google calls used this month against the cap (counts only, never the key)."""
        if not settings.google_geolocation_api_key:
            return None
        budget = {"month": usage_month(), "limit": settings.geolocation_monthly_limit, "used": None}
        try:
            await asyncio.wait_for(app.state.db.ensure_ready(), 5)
            budget["used"] = await asyncio.wait_for(app.state.db.geolocation_calls(budget["month"]), 5)
        except Exception:
            pass  # health stays answerable when the database isn't
        return budget

    async def model_usage(settings) -> dict | None:
        """Calls today (since midnight Pacific, when Gemini's daily quotas reset) per model cap."""
        ai = app.state.ai
        caps = {r.bucket: r.limit for stage in ("text", "tts", "embed") for r in ai.rules(stage) if r.window == "day"}
        if not caps:
            return None
        start = Rule("x", 1, "day").span(datetime.now(timezone.utc))[0]
        try:
            await asyncio.wait_for(app.state.db.ensure_ready(), 5)
            used = await asyncio.wait_for(app.state.db.slot_usage([f"{b}:day" for b in caps], start), 5)
        except Exception:
            return None
        return {b: {"used_today": used.get(f"{b}:day", 0), "limit_per_day": cap} for b, cap in caps.items()}

    @app.get("/healthz")
    async def healthz():
        settings = get_settings()
        missing = missing_settings(settings)
        # Optional features: whether each is configured, never the value itself.
        return {
            "ok": not missing,
            "missing_settings": missing,
            "wifi_positioning": bool(settings.google_geolocation_api_key),
            "geolocation_budget": await geolocation_budget(settings),
            "model_usage": await model_usage(settings),
        }

    @app.get("/config.json")
    async def spa_config():
        # Public values the dashboard needs to start Auth0 Universal Login.
        s = get_settings()
        return {
            "domain": s.auth0_domain,
            "clientId": s.auth0_spa_client_id,
            "audience": s.auth0_audience,
        }

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
