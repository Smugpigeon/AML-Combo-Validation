"""Download report artifacts (MD / HTML / PDF / figure) for a submission."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crypto import decrypt_llm_key
from app.db import get_db, safe_get
from app.deps import current_user
from app.llm import summarize_report
from app.models import LLMKey, Submission, UsageEvent, User


router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


def _get_submission_or_404(db: Session, submission_id: str,
                             user: User) -> Submission:
    row = safe_get(db, Submission, submission_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return row


def _serve_file(path_str: str | None, media_type: str, filename: str):
    if not path_str:
        raise HTTPException(status_code=404, detail="Artifact not generated yet")
    p = Path(path_str)
    if not p.exists():
        raise HTTPException(status_code=410,
                             detail="Artifact is missing from storage")
    return FileResponse(p, media_type=media_type, filename=filename)


@router.get("/{submission_id}/markdown")
def report_md(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = _get_submission_or_404(db, submission_id, user)
    return _serve_file(row.report_md_path, "text/markdown",
                        f"{row.patient_label}_report.md")


@router.get("/{submission_id}/pdf")
def report_pdf(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = _get_submission_or_404(db, submission_id, user)
    return _serve_file(row.report_pdf_path, "application/pdf",
                        f"{row.patient_label}_report.pdf")


@router.get("/{submission_id}/html")
def report_html(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = _get_submission_or_404(db, submission_id, user)
    return _serve_file(row.report_html_path, "text/html",
                        f"{row.patient_label}_report.html")


@router.get("/{submission_id}/figure")
def report_figure(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = _get_submission_or_404(db, submission_id, user)
    return _serve_file(row.report_figure_path, "image/png",
                        f"{row.patient_label}_dna_profile.png")


class SummaryRequest(BaseModel):
    llm_key_id: str   # which stored LLM key to use


class SummaryResponse(BaseModel):
    text: str
    model: str
    provider: str
    tokens_in: int | None
    tokens_out: int | None


@router.post("/{submission_id}/llm-summary", response_model=SummaryResponse)
def generate_summary(
    submission_id: str,
    req: SummaryRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Call user's LLM key to produce a plain-English summary of the report."""
    row = _get_submission_or_404(db, submission_id, user)
    if row.status != "done" or not row.report_md_path:
        raise HTTPException(status_code=400, detail="Report not ready")

    llm_key = safe_get(db, LLMKey, req.llm_key_id)
    if llm_key is None or llm_key.user_id != user.id:
        raise HTTPException(status_code=404, detail="LLM key not found")
    if llm_key.revoked_at is not None:
        raise HTTPException(status_code=400, detail="LLM key has been revoked")

    # Decrypt user's key (in-memory only)
    try:
        plaintext_key = decrypt_llm_key(llm_key.encrypted_key)
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Could not decrypt stored LLM key. "
            "Please revoke and re-add.",
        )

    # Load the markdown report
    md_path = Path(row.report_md_path)
    if not md_path.exists():
        raise HTTPException(status_code=410,
                             detail="Report markdown missing from storage")
    md_text = md_path.read_text(encoding="utf-8")

    # Call the provider
    try:
        resp = summarize_report(md_text, plaintext_key, llm_key.provider)
    except Exception as e:
        raise HTTPException(status_code=502,
                             detail=f"LLM provider error: {e}")

    # Update audit + last-used timestamp
    llm_key.last_used_at = datetime.now(tz=timezone.utc)
    db.add(UsageEvent(
        user_id=user.id, event_type="llm_summary_generated",
        meta={"submission_id": str(submission_id),
              "provider": llm_key.provider, "model": resp.model,
              "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out},
    ))
    db.commit()

    return SummaryResponse(
        text=resp.text, model=resp.model, provider=resp.provider,
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
    )
