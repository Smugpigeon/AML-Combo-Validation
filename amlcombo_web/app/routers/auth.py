"""Signup / login / logout — both JSON (for API clients) and form-post
(for web UI). Session established via HTTP-only cookie containing a JWT."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import current_user_optional
from app.models import User, UsageEvent
from app.security import (
    create_session_token, hash_password, verify_password,
)


router = APIRouter(prefix="/auth", tags=["auth"])


_EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")


def _validate_password(pw: str) -> None:
    s = get_settings()
    if len(pw) < s.MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {s.MIN_PASSWORD_LEN} chars",
        )


# ---------------------------------------------------------------------------
# JSON API (for programmatic signup / login)
# ---------------------------------------------------------------------------


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    full_name: str | None = None
    institution: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    institution: str | None
    created_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "AuthResponse":
        return cls(
            id=str(user.id), email=user.email, full_name=user.full_name,
            institution=user.institution, created_at=user.created_at,
        )


@router.post("/signup", response_model=AuthResponse)
def signup_json(
    req: SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    user = _do_signup(
        req.email, req.password, req.full_name, req.institution,
        response, db,
    )
    return AuthResponse.from_user(user)


@router.post("/login", response_model=AuthResponse)
def login_json(
    req: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    user = _do_login(req.email, req.password, response, db)
    return AuthResponse.from_user(user)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("session", httponly=True, secure=True, samesite="lax")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Form-based (for browser UI)
# ---------------------------------------------------------------------------


@router.post("/signup-form")
def signup_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
    institution: str = Form(""),
    db: Session = Depends(get_db),
):
    if not _EMAIL_RE.match(email):
        return RedirectResponse("/signup?error=bad_email", status_code=303)
    try:
        response = RedirectResponse("/dashboard", status_code=303)
        _do_signup(email, password, full_name or None,
                    institution or None, response, db)
        return response
    except HTTPException as e:
        return RedirectResponse(f"/signup?error={e.detail}", status_code=303)


@router.post("/login-form")
def login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        response = RedirectResponse("/dashboard", status_code=303)
        _do_login(email, password, response, db)
        return response
    except HTTPException:
        return RedirectResponse("/login?error=invalid_credentials",
                                 status_code=303)


@router.post("/logout-form")
def logout_form():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("session", httponly=True, samesite="lax")
    return response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _do_signup(email: str, password: str, full_name: str | None,
                institution: str | None, response: Response,
                db: Session) -> User:
    _validate_password(password)
    existing = db.execute(
        select(User).where(User.email == email.lower())
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        full_name=full_name,
        institution=institution,
    )
    db.add(user)
    db.flush()
    db.add(UsageEvent(user_id=user.id, event_type="signup"))
    db.commit()
    _set_session_cookie(response, str(user.id))
    return user


def _do_login(email: str, password: str, response: Response,
               db: Session) -> User:
    user = db.execute(
        select(User).where(User.email == email.lower())
    ).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    user.last_login_at = datetime.now(tz=timezone.utc)
    db.add(UsageEvent(user_id=user.id, event_type="login"))
    db.commit()
    _set_session_cookie(response, str(user.id))
    return user


def _set_session_cookie(response: Response, user_id: str) -> None:
    token = create_session_token(user_id)
    s = get_settings()
    response.set_cookie(
        key="session", value=token,
        httponly=True,
        secure=(s.APP_ENV != "dev"),
        samesite="lax",
        max_age=s.JWT_EXPIRATION_HOURS * 3600,
    )
