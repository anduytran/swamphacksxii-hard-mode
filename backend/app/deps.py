from fastapi import Request

from .ai import AIServices
from .db import MemoryDB


def get_db(request: Request) -> MemoryDB:
    return request.app.state.db


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai
