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


class ParseFileResponse(ParseResponse):
    """Extends ParseResponse with file-type detection metadata."""
    is_rna_seq: bool = False
    rna_seq_gene_count: int = 0
    rna_seq_mean_value: float = 0.0
    file_name: str = ""
    file_bytes: int = 0


def _read_upload_as_text(file: UploadFile) -> str:
    """Read an uploaded CSV/TSV/TXT file as UTF-8 text.

    Tries several common encodings (UTF-8, UTF-8-BOM, GBK/GB18030 for
    Chinese hospital exports, Latin-1 as last resort). Caps at 2 MB; we
    read up to that, slice larger, and let the LLM-truncation kick in.
    """
    data = file.file.read(2 * 1024 * 1024 + 1)
    size = len(data)
    if size > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 2 MB smart-parse limit. "
                   "For RNA-Seq counts (which can be larger), upload via the "
                   "RNA-Seq file input below instead.",
        )
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400,
                         detail="Could not decode file as text.")


def _looks_like_rna_seq(text: str) -> tuple[bool, int, float]:
    """Detect if a pasted/uploaded CSV/TSV is an RNA-Seq counts file.

    Returns (is_rna_seq, gene_count, mean_value).

    Heuristic:
      - First non-empty line looks like a 2-col header matching
        `(symbol|gene|gene_id|gene_symbol) , (count|value|expression|tpm|...)`
      - Subsequent lines have exactly 2 cols with a non-empty string + numeric
      - At least 100 rows (a 5000-gene panel is the smallest typical format)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 100:
        return False, 0, 0.0

    # Detect delimiter from the first line (comma or tab)
    first = lines[0]
    delim = "," if "," in first else ("\t" if "\t" in first else None)
    if delim is None:
        return False, 0, 0.0

    header_cells = [c.strip().lower() for c in first.split(delim)]
    if len(header_cells) != 2:
        return False, 0, 0.0
    gene_col_names = {"symbol", "gene", "gene_id", "gene_symbol",
                       "genesymbol", "id", "hugo_symbol"}
    value_col_names = {"count", "counts", "value", "expression",
                        "tpm", "fpkm", "rpkm", "cpm", "reads", "raw_count",
                        "rawcount"}
    has_gene_col = header_cells[0] in gene_col_names
    has_value_col = header_cells[1] in value_col_names
    if not (has_gene_col and has_value_col):
        return False, 0, 0.0

    # Sample up to 1000 rows to confirm 2-col numeric format
    sample = lines[1:1001]
    n_valid = 0
    numeric_values: list[float] = []
    for row in sample:
        parts = row.split(delim)
        if len(parts) != 2:
            continue
        symbol, val = parts[0].strip(), parts[1].strip()
        if not symbol:
            continue
        try:
            numeric_values.append(float(val))
            n_valid += 1
        except ValueError:
            continue
    if n_valid < len(sample) * 0.8:
        return False, 0, 0.0

    mean_val = (sum(numeric_values) / len(numeric_values)
                if numeric_values else 0.0)
    # Total gene count including rows beyond the 1000-row sample
    total_genes = len(lines) - 1   # minus header
    return True, total_genes, mean_val


@router.post("/parse-file", response_model=ParseFileResponse)
async def parse_patient_file(
    file: UploadFile = File(...),
    llm_key_id: str = Form(...),
    patient_identifier: Optional[str] = Form(
        None,
        description="Identify which patient to extract if the file has many.",
    ),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Smart upload: accepts a CSV / TSV / TXT file.

    Three routing cases:
      1. Looks like RNA-Seq counts (`symbol,count` with 100+ rows) →
         skip LLM, return is_rna_seq=true. The client then auto-populates
         the RNA-Seq file input below.
      2. Otherwise → pass the text to the BYOK LLM to extract clinical
         fields. `patient_identifier` (if provided) tells the LLM which
         row to focus on in a multi-patient table.
    """
    _check_parse_rate_limit(db, user.id)

    llm_key = safe_get(db, LLMKey, llm_key_id)
    if llm_key is None or llm_key.user_id != user.id:
        raise HTTPException(status_code=404, detail="LLM key not found")
    if llm_key.revoked_at is not None:
        raise HTTPException(status_code=400,
                             detail="That LLM key has been revoked")

    text = _read_upload_as_text(file)
    file_bytes = len(text.encode("utf-8"))

    # Case 1: RNA-Seq detection
    is_rna, n_genes, mean_v = _looks_like_rna_seq(text)
    if is_rna:
        db.add(UsageEvent(
            user_id=user.id, event_type="llm_parse_file_rna_detected",
            meta={"file_name": file.filename or "", "n_genes": n_genes,
                  "file_bytes": file_bytes},
        ))
        db.commit()
        return ParseFileResponse(
            parsed={}, confidence={}, warnings=[
                f"File detected as RNA-Seq counts ({n_genes} genes). "
                f"Use it directly as the RNA-Seq file below — no AI call "
                f"needed (saves your LLM tokens)."
            ],
            model="(no LLM used)", provider="(file-type-detection)",
            tokens_in=None, tokens_out=None,
            is_rna_seq=True, rna_seq_gene_count=n_genes,
            rna_seq_mean_value=round(float(mean_v), 3),
            file_name=file.filename or "",
            file_bytes=file_bytes,
        )

    # Case 2: looks like clinical — hand to LLM
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
            raw_text=text, api_key=plaintext_key,
            provider=llm_key.provider,
            focus_patient=patient_identifier,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")
    except Exception as e:
        msg = str(e)
        hint = ""
        if "401" in msg or "Unauthorized" in msg or "invalid_api_key" in msg:
            hint = (f" Your {llm_key.provider} API key appears to be "
                     f"invalid or expired — re-add it in /llm-keys.")
        elif "429" in msg:
            hint = " Provider rate limit hit — wait a minute."
        raise HTTPException(status_code=400,
                             detail=f"LLM provider error: {msg}.{hint}")

    llm_key.last_used_at = datetime.now(tz=timezone.utc)
    db.add(UsageEvent(
        user_id=user.id, event_type="llm_parse_file_used",
        meta={
            "provider": llm_key.provider, "model": result.model,
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
            "file_bytes": file_bytes,
            "file_name": file.filename or "",
            "focus_patient_given": bool(patient_identifier),
            "warnings_count": len(result.warnings),
        },
    ))
    db.commit()

    return ParseFileResponse(
        parsed=result.parsed,
        confidence=result.confidence,
        warnings=result.warnings,
        model=result.model, provider=result.provider,
        tokens_in=result.tokens_in, tokens_out=result.tokens_out,
        is_rna_seq=False, rna_seq_gene_count=0,
        rna_seq_mean_value=0.0,
        file_name=file.filename or "",
        file_bytes=file_bytes,
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
        # NOTE: Use 400 not 502. Cloudflare intercepts origin 5xx codes
        # with its own HTML error page, which breaks the frontend's
        # JSON parsing. 400 passes through unchanged.
        msg = str(e)
        hint = ""
        if "401" in msg or "Unauthorized" in msg or "invalid_api_key" in msg:
            hint = (f" Your {llm_key.provider} API key appears to be "
                     f"invalid or expired — re-add it in /llm-keys.")
        elif "429" in msg:
            hint = " Provider rate limit hit — wait a minute and retry."
        elif "timed out" in msg.lower() or "timeout" in msg.lower():
            hint = " The LLM call timed out — try a shorter paste."
        raise HTTPException(status_code=400,
                             detail=f"LLM provider error: {msg}.{hint}")

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
