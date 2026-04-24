"""Submit a patient → queue kit job → track status → retrieve report."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import decrypt_llm_key
from app.db import get_db, safe_get
from app.deps import current_user
from app.llm import parse_clinical_text
from app.models import LLMKey, Submission, UsageEvent, User
from app.tasks import run_patient_kit


router = APIRouter(prefix="/api/v1/patients", tags=["patients"])


class MutationIn(BaseModel):
    gene: str
    variant_type: Optional[str] = None
    vaf: Optional[float] = None
    is_ITD: bool = False
    is_TKD: bool = False
    allelic_ratio: Optional[float] = None
    is_biallelic: bool = False


class PatientInputJSON(BaseModel):
    patient_label: str = Field(max_length=100)
    mutations: list[MutationIn] = []
    karyotype_text: Optional[str] = None
    fusions: list[str] = []
    # Labs
    wbc: Optional[float] = None
    platelet: Optional[float] = None
    hemoglobin: Optional[float] = None
    ldh: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    albumin: Optional[float] = None
    creatinine: Optional[float] = None
    blast_pct_bm: Optional[float] = None
    blast_pct_pb: Optional[float] = None
    # Demographics
    age: Optional[float] = None
    sex: Optional[str] = None
    is_relapse: Optional[bool] = None
    prior_mds: Optional[bool] = None
    prior_chemo: Optional[bool] = None
    is_initial_diagnosis: Optional[bool] = None
    intent_comment: Optional[str] = None


class SubmissionOut(BaseModel):
    id: str
    patient_label: str
    status: str
    predicted_eln2017: Optional[str]
    top_regimen_name: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]


def _check_rate_limit(db: Session, user_id: uuid.UUID) -> None:
    s = get_settings()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    n_today = db.execute(
        select(func.count(Submission.id))
        .where(Submission.user_id == user_id)
        .where(Submission.created_at >= cutoff)
    ).scalar_one()
    if n_today >= s.DAILY_PATIENT_SUBMIT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(f"Daily submission limit reached "
                    f"({s.DAILY_PATIENT_SUBMIT_LIMIT}/day). "
                    f"Contact us to request a higher quota."),
        )


def _persist_upload(upload: UploadFile, dest_dir: Path, label: str) -> str:
    s = get_settings()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{label}_{upload.filename}"
    size = 0
    with open(dest, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > s.MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} upload exceeds limit "
                    f"({s.MAX_UPLOAD_BYTES} bytes)",
                )
            f.write(chunk)
    return str(dest)


# ---------------------------------------------------------------------------
# Smart paste — LLM-powered structured extraction from free text
# ---------------------------------------------------------------------------


class ParseRequest(BaseModel):
    """Client sends raw clinical text + picks which stored LLM key to use."""
    raw_text: str = Field(
        min_length=10, max_length=20000,
        description="Free-form clinical text to parse.",
    )
    llm_key_id: str = Field(description="UUID of a stored LLMKey row.")


class ParseResponse(BaseModel):
    parsed: dict
    confidence: dict
    warnings: list[str]
    model: str
    provider: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


def _check_parse_rate_limit(db: Session, user_id: uuid.UUID) -> None:
    """Don't let a runaway loop rack up LLM costs on the user's card.

    Limit: DAILY_PATIENT_SUBMIT_LIMIT × 3 parses per 24h (parsing is cheap
    and often retried, so give some headroom).
    """
    s = get_settings()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    limit = s.DAILY_PATIENT_SUBMIT_LIMIT * 3
    n_today = db.execute(
        select(func.count(UsageEvent.id))
        .where(UsageEvent.user_id == user_id)
        .where(UsageEvent.event_type == "llm_parse_used")
        .where(UsageEvent.ts >= cutoff)
    ).scalar_one()
    if n_today >= limit:
        raise HTTPException(
            status_code=429,
            detail=(f"Parse rate limit reached ({limit}/day). "
                    f"Your LLM charges still stand; contact us to raise this cap."),
        )


@router.post("/parse", response_model=ParseResponse)
def parse_patient_text(
    req: ParseRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Parse free-text clinical paste into structured patient fields.

    Uses the user's stored LLM key (BYOK). We never log the `raw_text`;
    only metadata (provider, model, token counts) lands in UsageEvent.
    """
    _check_parse_rate_limit(db, user.id)

    llm_key = safe_get(db, LLMKey, req.llm_key_id)
    if llm_key is None or llm_key.user_id != user.id:
        raise HTTPException(status_code=404, detail="LLM key not found")
    if llm_key.revoked_at is not None:
        raise HTTPException(status_code=400,
                             detail="That LLM key has been revoked")

    try:
        plaintext_key = decrypt_llm_key(llm_key.encrypted_key)
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail=("Could not decrypt the stored LLM key. "
                     "Please revoke it and re-add."),
        )

    try:
        result = parse_clinical_text(
            raw_text=req.raw_text,
            api_key=plaintext_key,
            provider=llm_key.provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")
    except Exception as e:  # httpx / provider errors
        raise HTTPException(status_code=502,
                             detail=f"LLM provider error: {e!s}")

    # Audit (metadata only — no raw_text, no LLM output)
    llm_key.last_used_at = datetime.now(tz=timezone.utc)
    db.add(UsageEvent(
        user_id=user.id, event_type="llm_parse_used",
        meta={
            "provider": llm_key.provider, "model": result.model,
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
            "text_length": len(req.raw_text),
            "warnings_count": len(result.warnings),
        },
    ))
    db.commit()

    return ParseResponse(
        parsed=result.parsed,
        confidence=result.confidence,
        warnings=result.warnings,
        model=result.model, provider=result.provider,
        tokens_in=result.tokens_in, tokens_out=result.tokens_out,
    )


# ---------------------------------------------------------------------------
# Submit — multipart form with JSON payload + file uploads
# ---------------------------------------------------------------------------


@router.post("", response_model=SubmissionOut, status_code=202)
def submit_patient(
    payload_json: str = Form(..., description="JSON string matching PatientInputJSON"),
    rna_counts: UploadFile = File(..., description="2-col CSV: symbol,count (5000-gene panel)"),
    rna_full: Optional[UploadFile] = File(None, description="Optional 2-col CSV: full transcriptome"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Submit a new patient for kit prediction. Returns 202 with a
    submission ID; poll GET /api/v1/patients/{id} for status."""
    _check_rate_limit(db, user.id)

    try:
        payload = PatientInputJSON.model_validate_json(payload_json)
    except Exception as e:
        raise HTTPException(status_code=400,
                             detail=f"Invalid payload_json: {e}")

    # Create submission row
    sub = Submission(
        user_id=user.id,
        patient_label=payload.patient_label,
        status="queued",
        input_json=payload.model_dump(),
    )
    db.add(sub)
    db.flush()

    # Save uploads
    s = get_settings()
    sub_dir = s.STORAGE_ROOT / str(user.id) / str(sub.id)
    sub.rna_counts_path = _persist_upload(rna_counts, sub_dir, "rna_counts")
    if rna_full is not None:
        sub.rna_full_path = _persist_upload(rna_full, sub_dir, "rna_full")

    db.add(UsageEvent(user_id=user.id, event_type="patient_submitted",
                       meta={"submission_id": str(sub.id)}))
    db.commit()

    # Enqueue Celery task
    run_patient_kit.delay(str(sub.id))

    return SubmissionOut(
        id=str(sub.id), patient_label=sub.patient_label, status=sub.status,
        predicted_eln2017=None, top_regimen_name=None,
        created_at=sub.created_at, started_at=None, finished_at=None,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# List my patients
# ---------------------------------------------------------------------------


@router.get("", response_model=list[SubmissionOut])
def list_patients(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    rows = db.execute(
        select(Submission)
        .where(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc())
        .limit(limit)
    ).scalars().all()
    return [SubmissionOut(
        id=str(r.id), patient_label=r.patient_label, status=r.status,
        predicted_eln2017=r.predicted_eln2017,
        top_regimen_name=r.top_regimen_name,
        created_at=r.created_at, started_at=r.started_at,
        finished_at=r.finished_at, error_message=r.error_message,
    ) for r in rows]


# ---------------------------------------------------------------------------
# Get one
# ---------------------------------------------------------------------------


@router.get("/{submission_id}", response_model=SubmissionOut)
def get_patient(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = safe_get(db, Submission, submission_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return SubmissionOut(
        id=str(row.id), patient_label=row.patient_label, status=row.status,
        predicted_eln2017=row.predicted_eln2017,
        top_regimen_name=row.top_regimen_name,
        created_at=row.created_at, started_at=row.started_at,
        finished_at=row.finished_at, error_message=row.error_message,
    )


@router.delete("/{submission_id}", status_code=204)
def delete_patient(
    submission_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    row = safe_get(db, Submission, submission_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    # Remove storage dir
    s = get_settings()
    sub_dir = s.STORAGE_ROOT / str(user.id) / str(row.id)
    if sub_dir.exists():
        shutil.rmtree(sub_dir, ignore_errors=True)
    db.delete(row)
    db.commit()
    return None
