"""Pytest fixtures — spin up an in-memory SQLite app for fast tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Provide required env vars BEFORE importing app.config.
os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault(
    "JWT_SECRET",
    "test-secret-please-do-not-use-in-prod-3djKL2sd8f7akLsmd0sadkl2",
)
# Generate a fixed test Fernet key (url-safe b64 32 bytes) up-front.
os.environ.setdefault(
    "FERNET_KEY", "zFpc6u5YJ4DggRRsOAcffpg22oO_rPJt6LVo9VYsL5I=",
)
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# STORAGE_ROOT and KIT_ASSETS_ROOT must be writable/readable paths for the
# lifespan handler. Use tempdirs.
import tempfile as _tmp                                   # noqa: E402
_storage_tmp = _tmp.mkdtemp(prefix="amlcombo_test_store_")
_assets_tmp = _tmp.mkdtemp(prefix="amlcombo_test_assets_")
os.environ.setdefault("STORAGE_ROOT", _storage_tmp)
os.environ.setdefault("KIT_ASSETS_ROOT", _assets_tmp)

# Make `app` package importable (amlcombo_web/ root)
_here = Path(__file__).resolve()
_root = _here.parents[1]
sys.path.insert(0, str(_root))

import pytest                                           # noqa: E402
from fastapi.testclient import TestClient               # noqa: E402
from sqlalchemy import create_engine                    # noqa: E402
from sqlalchemy.pool import StaticPool                  # noqa: E402

from app import db as app_db                            # noqa: E402
from app.db import Base                                 # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _use_in_memory_sqlite():
    """Replace the module-level engine with an in-memory SQLite one for tests.

    StaticPool + check_same_thread=False lets the TestClient share the
    single :memory: database across threads.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Swap the module-level engine + sessionmaker
    app_db.engine = test_engine
    from sqlalchemy.orm import sessionmaker
    app_db.SessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=test_engine)
    yield


@pytest.fixture
def client():
    # Import here so env-var and engine patches are already applied
    from app.main import app
    with TestClient(app) as c:
        yield c
