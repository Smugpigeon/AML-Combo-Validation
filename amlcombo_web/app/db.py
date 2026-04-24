"""SQLAlchemy engine + session factory."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


T = TypeVar("T")


def safe_get(db: Session, model: type[T], pk: Any) -> Optional[T]:
    """Lookup by primary key, coercing string UUIDs to uuid.UUID.

    Works around SQLAlchemy 2.0 + SQLite's `Uuid` type, which rejects
    plain strings at bind time. Safe to use for both UUID and integer
    primary keys (falls back to plain `db.get` on non-UUID).
    """
    if isinstance(pk, str):
        try:
            pk = uuid.UUID(pk)
        except (ValueError, TypeError):
            return None
    return db.get(model, pk)


class Base(DeclarativeBase):
    pass


_settings = get_settings()
_engine_kwargs: dict = {"pool_pre_ping": True, "echo": False}
# pool_size/max_overflow only apply to connection-pooled dialects (not sqlite).
if not _settings.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs.update(pool_size=10, max_overflow=20)
engine = create_engine(_settings.DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False,
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency — yields a Session, closes after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    """Non-FastAPI context manager (for scripts / Celery tasks)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
