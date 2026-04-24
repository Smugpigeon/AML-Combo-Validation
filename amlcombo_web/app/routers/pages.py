"""Jinja2 HTML pages + language toggle endpoint.

All page handlers share a context dict that includes `lang` and `t(key)`
(from `app.i18n`), so every template can use `{{ t("nav.home") }}` and
conditionally render by `{{ lang }}`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db, safe_get
from app.deps import current_user_optional, current_user
from app.i18n import SUPPORTED_LANGS, make_template_context
from app.models import APIKey, LLMKey, Submission, User


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


router = APIRouter(tags=["pages"])


def _ctx(request: Request, **extra):
    """Shortcut for building a template context with i18n preloaded."""
    return make_template_context(request, **extra)


@router.get("/", response_class=HTMLResponse)
def landing(request: Request, user: User | None = Depends(current_user_optional)):
    return templates.TemplateResponse(
        request, "landing.html", _ctx(request, user=user),
    )


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request,
                 user: User | None = Depends(current_user_optional)):
    if user is not None:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request, "signup.html",
        _ctx(request, user=None, error=request.query_params.get("error")),
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request,
                user: User | None = Depends(current_user_optional)):
    if user is not None:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request, "login.html",
        _ctx(request, user=None, error=request.query_params.get("error")),
    )


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    subs = db.execute(
        select(Submission).where(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc()).limit(20)
    ).scalars().all()
    return templates.TemplateResponse(
        request, "dashboard.html",
        _ctx(request, user=user, submissions=subs),
    )


@router.get("/new", response_class=HTMLResponse)
def new_patient_page(request: Request, user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    # Also pass the user's active LLM keys so the "smart paste" block can
    # show a key selector. If they have none, the block shows a link to
    # /llm-keys to add one.
    llm_keys = db.execute(
        select(LLMKey).where(LLMKey.user_id == user.id,
                              LLMKey.revoked_at.is_(None))
        .order_by(LLMKey.created_at.desc())
    ).scalars().all()
    return templates.TemplateResponse(
        request, "new_patient.html",
        _ctx(request, user=user, llm_keys=llm_keys),
    )


@router.get("/patient/{submission_id}", response_class=HTMLResponse)
def patient_detail(request: Request, submission_id: str,
                    user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    sub = safe_get(db, Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    llm_keys = db.execute(
        select(LLMKey).where(LLMKey.user_id == user.id,
                              LLMKey.revoked_at.is_(None))
    ).scalars().all()
    return templates.TemplateResponse(
        request, "patient_detail.html",
        _ctx(request, user=user, sub=sub, llm_keys=llm_keys),
    )


@router.get("/api-keys", response_class=HTMLResponse)
def api_keys_page(request: Request, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    keys = db.execute(
        select(APIKey).where(APIKey.user_id == user.id)
        .order_by(APIKey.created_at.desc())
    ).scalars().all()
    return templates.TemplateResponse(
        request, "api_keys.html",
        _ctx(request, user=user, keys=keys),
    )


@router.get("/llm-keys", response_class=HTMLResponse)
def llm_keys_page(request: Request, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    keys = db.execute(
        select(LLMKey).where(LLMKey.user_id == user.id)
        .order_by(LLMKey.created_at.desc())
    ).scalars().all()
    return templates.TemplateResponse(
        request, "llm_keys.html",
        _ctx(request, user=user, keys=keys),
    )


@router.get("/docs/api", response_class=HTMLResponse)
def api_docs(request: Request,
              user: User | None = Depends(current_user_optional)):
    return templates.TemplateResponse(
        request, "docs_api.html", _ctx(request, user=user),
    )


# --- Language toggle endpoint ---

@router.get("/set-lang/{lang}", include_in_schema=False)
def set_language(request: Request, lang: str):
    """Set the `lang` cookie and redirect back to where the user was."""
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail="Unsupported language")
    redirect_to = request.query_params.get("next", "/")
    # Only accept site-internal paths to avoid open-redirect abuse
    if not redirect_to.startswith("/") or redirect_to.startswith("//"):
        redirect_to = "/"
    response = RedirectResponse(redirect_to, status_code=303)
    response.set_cookie(
        key="lang", value=lang,
        max_age=60 * 60 * 24 * 365,       # 1 year
        httponly=False,                    # client-side JS may want to read it
        samesite="lax",
        path="/",
    )
    return response
