"""SQLAlchemy ORM models.

Schema philosophy:
  - UUID primary keys (so IDs can be shared publicly without revealing counts)
  - Timestamps with tz
  - Soft-delete via `revoked_at` / `deleted_at` (don't lose audit trail)
  - No PII beyond email — researchers self-identify, don't store patient PHI
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, BigInteger, Boolean, DateTime, ForeignKey, String, Text, Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship


# `Uuid` (SA 2.0) maps to native UUID on Postgres and CHAR(32) on SQLite,
# so the same model works in dev tests and prod.
PgUUID = Uuid

from app.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True,
                                       nullable=False)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    institution: Mapped[Optional[str]] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=_utcnow, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="user")
    llm_keys: Mapped[list["LLMKey"]] = relationship(back_populates="user")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="user")


class APIKey(Base):
    """A per-user API key granting programmatic access to /api/v1/*.

    We store only the sha256 hash of the key. The prefix (first 12 chars)
    is kept in plaintext so users can identify keys in their dashboard
    without ever showing the full key again after creation.
    """
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    prefix: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=_utcnow, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="api_keys")


class LLMKey(Base):
    """User-supplied LLM API key (BYOK) — encrypted with Fernet.

    Users provide their own OpenAI / Anthropic API keys if they want
    optional LLM-powered report enhancements. We store only the
    ciphertext; decryption happens in-process only at call time.
    """
    __tablename__ = "llm_keys"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)   # "openai" | "anthropic"
    label: Mapped[Optional[str]] = mapped_column(String(100))
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    # Store only last 4 chars of plaintext for display ("...abcd")
    key_suffix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=_utcnow, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="llm_keys")


class Submission(Base):
    """One patient submission → produces one set of report artifacts."""
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(PgUUID, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    patient_label: Mapped[str] = mapped_column(String(100), nullable=False)
    # Status machine: queued → running → done | failed
    status: Mapped[str] = mapped_column(String(32), default="queued",
                                         index=True, nullable=False)
    # Inputs stored as JSON (mutations, karyotype, labs, demographics)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Relative paths (under STORAGE_ROOT/<user_id>/<submission_id>/)
    rna_counts_path: Mapped[Optional[str]] = mapped_column(String(500))
    rna_full_path: Mapped[Optional[str]] = mapped_column(String(500))
    # Output artifacts
    report_md_path: Mapped[Optional[str]] = mapped_column(String(500))
    report_pdf_path: Mapped[Optional[str]] = mapped_column(String(500))
    report_html_path: Mapped[Optional[str]] = mapped_column(String(500))
    report_figure_path: Mapped[Optional[str]] = mapped_column(String(500))
    dna_summary_json: Mapped[Optional[dict]] = mapped_column(JSON)
    rna_outlier_json: Mapped[Optional[dict]] = mapped_column(JSON)
    # Predictions (summary — so dashboard can show without fetching MD)
    predicted_eln2017: Mapped[Optional[str]] = mapped_column(String(32))
    top_regimen_name: Mapped[Optional[str]] = mapped_column(String(200))
    # Progress + error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=_utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="submissions")


class UsageEvent(Base):
    """Append-only audit trail for rate limiting + future billing."""
    __tablename__ = "usage_events"

    # BigInteger + autoincrement needs a dialect hint on SQLite — use
    # `.with_variant(Integer, "sqlite")` so tests work without losing
    # 8-byte capacity in Postgres.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(
            __import__("sqlalchemy").Integer(), "sqlite",
        ),
        primary_key=True, autoincrement=True,
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PgUUID, ForeignKey("users.id", ondelete="CASCADE"),
    )
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # free-form metadata (IP, endpoint, duration, etc.)
    meta: Mapped[Optional[dict]] = mapped_column(JSON)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow,
                                          index=True, nullable=False)
