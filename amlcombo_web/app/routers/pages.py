"""Jinja2 HTML pages for the browser UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db, safe_get
from app.deps import current_user_optional, current_user
from app.models import APIKey, LLMKey, Submission, User


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
def landing(request: Request, user: User | None = Depends(current_user_optional)):
    return templates.TemplateResponse(
        "landing.html", {"request": request, "user": user},
    )


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request,
                 user: User | None = Depends(current_user_optional)):
    if user is not None:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        "signup.html", {"request": request,
                         "error": request.query_params.get("error")},
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request,
                user: User | None = Depends(current_user_optional)):
    if user is not None:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        "login.html", {"request": request,
                        "error": request.query_params.get("error")},
    )


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    subs = db.execute(
        select(Submission).where(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc()).limit(20)
    ).scalars().all()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": user, "submissions": subs},
    )


@router.get("/new", response_class=HTMLResponse)
def new_patient_page(request: Request, user: User = Depends(current_user)):
    return templates.TemplateResponse(
        "new_patient.html", {"request": request, "user": user},
    )


@router.get("/patient/{submission_id}", response_class=HTMLResponse)
def patient_detail(request: Request, submission_id: str,
                    user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    sub = safe_get(db, Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    # Also list the user's LLM keys (for the summary feature)
    llm_keys = db.execute(
        select(LLMKey).where(LLMKey.user_id == user.id,
                              LLMKey.revoked_at.is_(None))
    ).scalars().all()
    return templates.TemplateResponse(
        "patient_detail.html",
        {"request": request, "user": user, "sub": sub, "llm_keys": llm_keys},
    )


@router.get("/api-keys", response_class=HTMLResponse)
def api_keys_page(request: Request, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    keys = db.execute(
        select(APIKey).where(APIKey.user_id == user.id)
        .order_by(APIKey.created_at.desc())
    ).scalars().all()
    return templates.TemplateResponse(
        "api_keys.html",
        {"request": request, "user": user, "keys": keys},
    )


@router.get("/llm-keys", response_class=HTMLResponse)
def llm_keys_page(request: Request, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    keys = db.execute(
        select(LLMKey).where(LLMKey.user_id == user.id)
        .order_by(LLMKey.created_at.desc())
    ).scalars().all()
    return templates.TemplateResponse(
        "llm_keys.html",
        {"request": request, "user": user, "keys": keys},
    )


@router.get("/docs/api", response_class=HTMLResponse)
def api_docs(request: Request,
              user: User | None = Depends(current_user_optional)):
    return templates.TemplateResponse(
        "docs_api.html", {"request": request, "user": user},
    )
