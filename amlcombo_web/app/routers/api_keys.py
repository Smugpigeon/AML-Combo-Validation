"""Per-user platform API keys for programmatic access."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db, safe_get
from app.deps import current_user
from app.models import APIKey, User
from app.security import mint_api_key


router = APIRouter(prefix="/api/v1/keys", tags=["api-keys"])


class APIKeyOut(BaseModel):
    id: str
    prefix: str
    label: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]


class APIKeyCreatedResponse(BaseModel):
    id: str
    prefix: str
    label: Optional[str]
    created_at: datetime
    # The plaintext key — shown ONLY on creation, never again
    plaintext_key: str
    warning: str = ("Save this key now — it will NEVER be shown again. "
                     "If you lose it, revoke it and create a new one.")


class APIKeyCreateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=100,
                                   description="Human-readable label")


@router.post("", response_model=APIKeyCreatedResponse, status_code=201)
def create_api_key(
    req: APIKeyCreateRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    plaintext, prefix, key_hash = mint_api_key()
    row = APIKey(
        user_id=user.id, prefix=prefix, key_hash=key_hash,
        label=req.label,
    )
    db.add(row)
    db.commit()
    return APIKeyCreatedResponse(
        id=str(row.id), prefix=prefix, label=req.label,
        created_at=row.created_at, plaintext_key=plaintext,
    )


@router.get("", response_model=list[APIKeyOut])
def list_api_keys(
    include_revoked: bool = False,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    stmt = select(APIKey).where(APIKey.user_id == user.id)
    if not include_revoked:
        stmt = stmt.where(APIKey.revoked_at.is_(None))
    stmt = stmt.order_by(APIKey.created_at.desc())
    rows = db.execute(stmt).scalars().all()
    return [APIKeyOut(
        id=str(r.id), prefix=r.prefix, label=r.label,
        created_at=r.created_at, last_used_at=r.last_used_at,
        revoked_at=r.revoked_at,
    ) for r in rows]


@router.delete("/{key_id}", status_code=204)
def revoke_api_key(
    key_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = safe_get(db, APIKey, key_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Key not found")
    if row.revoked_at is None:
        row.revoked_at = datetime.now(tz=timezone.utc)
        db.commit()
    return None
