from fastapi import Request

from .ai import AIServices
from .db import MemoryDB


async def get_db(request: Request) -> MemoryDB:
    db = request.app.state.db
    # Connects (and creates the schema) on first use, which serverless hosts rely on.
    if ensure_ready := getattr(db, "ensure_ready", None):
        await ensure_ready()
    return db


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai
