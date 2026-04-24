"""BYOK — users store their own LLM provider keys (encrypted at rest)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crypto import encrypt_llm_key
from app.db import get_db, safe_get
from app.deps import current_user
from app.models import LLMKey, User


router = APIRouter(prefix="/api/v1/llm-keys", tags=["llm-keys"])


_ALLOWED_PROVIDERS = {"openai", "anthropic"}


class LLMKeyCreateRequest(BaseModel):
    provider: str = Field(description="openai | anthropic")
    plaintext_key: str = Field(min_length=20, max_length=500,
                                description="Your provider API key")
    label: Optional[str] = Field(default=None, max_length=100)


class LLMKeyOut(BaseModel):
    id: str
    provider: str
    label: Optional[str]
    key_suffix: str     # "...abcd" — last 4 chars only, for display
    created_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]


@router.post("", response_model=LLMKeyOut, status_code=201)
def add_llm_key(
    req: LLMKeyCreateRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    provider = req.provider.lower().strip()
    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider must be one of {sorted(_ALLOWED_PROVIDERS)}",
        )
    # Sanity check: OpenAI keys start with sk-, Anthropic with sk-ant-
    k = req.plaintext_key.strip()
    if provider == "openai" and not k.startswith("sk-"):
        raise HTTPException(status_code=400,
                             detail="OpenAI keys start with 'sk-'")
    if provider == "anthropic" and not k.startswith("sk-ant-"):
        raise HTTPException(status_code=400,
                             detail="Anthropic keys start with 'sk-ant-'")

    ct = encrypt_llm_key(k)
    suffix = f"...{k[-4:]}"
    row = LLMKey(
        user_id=user.id, provider=provider, encrypted_key=ct,
        key_suffix=suffix, label=req.label,
    )
    db.add(row)
    db.commit()
    return LLMKeyOut(
        id=str(row.id), provider=provider, label=req.label,
        key_suffix=suffix, created_at=row.created_at,
        last_used_at=None, revoked_at=None,
    )


@router.get("", response_model=list[LLMKeyOut])
def list_llm_keys(
    include_revoked: bool = False,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    stmt = select(LLMKey).where(LLMKey.user_id == user.id)
    if not include_revoked:
        stmt = stmt.where(LLMKey.revoked_at.is_(None))
    stmt = stmt.order_by(LLMKey.created_at.desc())
    rows = db.execute(stmt).scalars().all()
    return [LLMKeyOut(
        id=str(r.id), provider=r.provider, label=r.label,
        key_suffix=r.key_suffix, created_at=r.created_at,
        last_used_at=r.last_used_at, revoked_at=r.revoked_at,
    ) for r in rows]


@router.delete("/{key_id}", status_code=204)
def revoke_llm_key(
    key_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = safe_get(db, LLMKey, key_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Key not found")
    if row.revoked_at is None:
        row.revoked_at = datetime.now(tz=timezone.utc)
        db.commit()
    return None
