"""Vercel entry point: the FastAPI app as one Python function.

The dashboard is built from web/ and served by Vercel's CDN (vercel.json outputDirectory).
Requests that match no built file (/api/..., /healthz, /config.json) are rewritten here;
FastAPI still sees the original URL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app  # noqa: E402,F401
