"""Vercel serverless entrypoint. Vercel's Python runtime looks for an ASGI
`app` object in this file; everything else is delegated to the real
FastAPI app in api/main.py so local dev (uvicorn) and Vercel share the
same application code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402
