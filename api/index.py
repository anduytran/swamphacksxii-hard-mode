"""Vercel entry point: serves the FastAPI app (API + dashboard) as one Python function.

vercel.json rewrites every path here; FastAPI still sees the original URL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app  # noqa: E402,F401
