from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .ai import AIServices
from .config import get_settings
from .db import MemoryDB
from .routers import admin, dashboard, images, memory

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not get_settings().image_url_secret:
        raise RuntimeError("IMAGE_URL_SECRET must be set")
    # Optional warm-up: get_db() also connects on first use, for hosts that skip lifespan.
    await app.state.db.ensure_ready()
    yield
    await app.state.db.close()


def create_app(use_lifespan: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LastScene API", lifespan=lifespan if use_lifespan else None)
    app.state.db = MemoryDB(settings)
    app.state.ai = AIServices(settings)
    app.include_router(memory.router)
    app.include_router(dashboard.router)
    app.include_router(images.router)
    app.include_router(admin.router)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

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
