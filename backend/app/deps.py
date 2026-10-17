from fastapi import Request

from .ai import AIServices
from .db import MemoryDB
from .storage import ObjectStore


def get_db(request: Request) -> MemoryDB:
    return request.app.state.db


def get_store(request: Request) -> ObjectStore:
    return request.app.state.store


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai
