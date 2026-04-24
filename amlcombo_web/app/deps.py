"""FastAPI dependencies: current_user resolution from cookie OR API key."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import APIKey, User
from app.security import (
    api_key_matches, decode_session_token, hash_api_key,
)


def _user_from_session_cookie(request: Request, db: Session) -> Optional[User]:
    token = request.cookies.get("session")
    if not token:
        return None
    user_id = decode_session_token(token)
    if not user_id:
        return None
    try:
        user = db.get(User, _uuid.UUID(user_id))
    except (ValueError, TypeError):
        return None
    if user and user.is_active:
        return user
    return None


def _user_from_api_key(request: Request, db: Session) -> Optional[User]:
    auth_header = request.headers.get("Authorization", "")
    plaintext: Optional[str] = None
    if auth_header.lower().startswith("bearer "):
        plaintext = auth_header[7:].strip()
    elif "X-API-Key" in request.headers:
        plaintext = request.headers["X-API-Key"].strip()
    if not plaintext:
        return None

    prefix = plaintext[:16]
    key_hash = hash_api_key(plaintext)
    row = db.execute(
        select(APIKey).where(
            APIKey.prefix == prefix,
            APIKey.revoked_at.is_(None),
        )
    ).scalar_one_or_none()
    if row is None or not api_key_matches(plaintext, row.key_hash):
        return None

    # Update last_used_at asynchronously — best effort
    row.last_used_at = datetime.now(tz=timezone.utc)
    user = db.get(User, row.user_id)
    if user and user.is_active:
        return user
    return None


def current_user_optional(
    request: Request, db: Session = Depends(get_db),
) -> Optional[User]:
    """Return the current user if authenticated (via cookie or API key),
    else None. Used on public pages that behave differently when signed in."""
    user = _user_from_session_cookie(request, db)
    if user is None:
        user = _user_from_api_key(request, db)
    return user


def current_user(
    request: Request, db: Session = Depends(get_db),
) -> User:
    """Require an authenticated user. 401 otherwise."""
    user = current_user_optional(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def current_user_api(
    request: Request, db: Session = Depends(get_db),
) -> User:
    """Stricter variant: requires API-key auth only (refuses cookie)."""
    user = _user_from_api_key(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required (Authorization: Bearer ac_live_...)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
