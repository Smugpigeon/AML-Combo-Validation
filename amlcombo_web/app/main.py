"""FastAPI application entrypoint — wires routers, middleware, and startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import Base, engine
from app.routers import (
    api_keys, auth, llm_keys, pages, patients, reports,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On first boot: ensure tables exist + storage dir is writable."""
    s = get_settings()
    log.info("Starting amlcombo_web env=%s domain=%s", s.APP_ENV, s.DOMAIN)

    # Create tables if missing (idempotent). In production you'd want
    # Alembic migrations; for MVP on a single VPS this is fine.
    Base.metadata.create_all(bind=engine)

    # Ensure storage dir exists
    s.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    yield

    log.info("Shutting down amlcombo_web")


app = FastAPI(
    title="AML Combo-Prediction Kit API",
    version="0.2.0",
    description=("Public API for the AML Combo-Prediction Kit. "
                  "Research use only — not a medical device."),
    docs_url="/openapi",     # Swagger UI
    redoc_url="/redoc",
    lifespan=lifespan,
)


# --- Middleware ---
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Static files (CSS) ---
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)),
              name="static")


# --- Routers ---
app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(api_keys.router)
app.include_router(llm_keys.router)
app.include_router(patients.router)
app.include_router(reports.router)


# --- Health check ---
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "version": app.version}


# --- Generic error handler ---
@app.exception_handler(Exception)
async def unhandled_exc(request, exc):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
