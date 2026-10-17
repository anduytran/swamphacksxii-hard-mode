import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .ai import AIServices
from .config import Settings, get_settings
from .db import MemoryDB
from .geo import Geolocator
from .routers import admin, dashboard, households, images, memory, observations, public

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
log = logging.getLogger("lastscene")


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
    app.state.ai = AIServices(settings)
    app.state.geo = Geolocator(settings)
    app.include_router(memory.router)
    app.include_router(dashboard.router)
    app.include_router(images.router)
    app.include_router(admin.router)
    app.include_router(households.router)
    app.include_router(public.router)
    app.include_router(observations.router)

    @app.get("/healthz")
    async def healthz():
        missing = missing_settings(get_settings())
        return {"ok": not missing, "missing_settings": missing}

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
