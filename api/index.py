"""Vercel entry point: the FastAPI app as one Python function.

The dashboard is built from web/ during the deploy (vercel.json buildCommand) and served
by the FastAPI app from web/dist, along with the API. vercel.json rewrites every path
here; FastAPI still sees the original URL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app  # noqa: E402,F401
