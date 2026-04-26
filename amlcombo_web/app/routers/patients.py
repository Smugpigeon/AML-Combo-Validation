"""Submit a patient → queue kit job → track status → retrieve report."""

from __future__ import annotations

import base64
import io
import json
import shutil
import uuid
import zipfile
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
    # Issues #4 + #9 + #13 — extended hotspot / ELN 2022 fields
    is_bzip: bool = False
    is_multi_hit: bool = False
    protein_codon: Optional[str] = None


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


@router.get("/demo-rna-counts.csv", include_in_schema=False)
def download_demo_rna_counts():
    """Serve a tiny demo RNA-Seq counts CSV so users can test the full
    pipeline without hunting for a real expression file. Generated
    deterministically from the kit's kept_genes (same 5000 panel the MLP
    uses), log-normal distributed counts — clinically meaningless but
    formally valid."""
    from fastapi.responses import StreamingResponse
    import joblib
    import numpy as np
    import io

    bundle = joblib.load(Path(get_settings().KIT_ASSETS_ROOT)
                          / "beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]
    rng = np.random.default_rng(17)
    counts = rng.lognormal(mean=4.0, sigma=1.2, size=len(kept))

    buf = io.StringIO()
    buf.write("symbol,count\n")
    for g, c in zip(kept, counts):
        buf.write(f"{g},{c:.1f}\n")
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition":
                 'attachment; filename="demo_rna_counts.csv"'},
    )


class ParseFileResponse(ParseResponse):
    """Extends ParseResponse with file-type detection metadata."""
    is_rna_seq: bool = False
    rna_seq_gene_count: int = 0
    rna_seq_mean_value: float = 0.0
    file_name: str = ""
    file_bytes: int = 0

    # When the user dropped a .zip with multiple files, server may have
    # extracted an RNA-Seq counts file from inside. We base64-encode it
    # in the response so the client can reconstitute it as a File and
    # populate the RNA-Seq input below — no second upload needed.
    rna_seq_attachment_b64: Optional[str] = None
    rna_seq_attachment_filename: Optional[str] = None

    # When zip contained multiple files, summary of what was found.
    zip_summary: list[dict] = []


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


def _try_decode(data: bytes) -> Optional[str]:
    """Best-effort decode of bytes → text. Returns None if we can't."""
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _try_extract_pdf_text(data: bytes) -> Optional[str]:
    """Extract text from a PDF bytes blob using pypdfium2 if available."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        pdf = pdfium.PdfDocument(data)
        chunks = []
        for page in pdf:
            try:
                tp = page.get_textpage()
                chunks.append(tp.get_text_range())
            except Exception:
                continue
        return "\n\n".join(c for c in chunks if c)
    except Exception:
        return None


def _try_extract_xlsx_as_csv(data: bytes) -> Optional[str]:
    """Read .xlsx → flatten first sheet to CSV-like text."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        return None
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        rows = []
        for row in ws.iter_rows(values_only=True):
            rows.append(",".join("" if c is None else str(c) for c in row))
        return "\n".join(rows)
    except Exception:
        return None


def _decode_zip_filename(info: zipfile.ZipInfo) -> str:
    """Decode a ZipInfo's filename, fixing Chinese-name garble from macOS/
    Windows zip tools that don't set the UTF-8 flag (bit 0x800).

    Per the ZIP spec, when bit 0x800 of `flag_bits` is set, the filename
    is UTF-8; otherwise CP437. macOS Finder and many Chinese Windows zip
    tools store the filename as raw GBK/UTF-8 bytes WITHOUT setting the
    flag — so Python's zipfile decodes them as CP437, producing mojibake
    like "μ┤ïΦ»òσÑùΣ┐┤" instead of "测试套件".

    Recovery: take Python's CP437-decoded string, re-encode back to
    bytes, then try UTF-8 / GBK / GB18030. Whichever decodes cleanly
    AND looks like real Chinese (i.e., contains CJK characters) wins.
    """
    name = info.filename
    # If the encoder set the UTF-8 flag, the name is already correct
    if info.flag_bits & 0x800:
        return name
    # ASCII-only names are safe as-is
    if all(ord(c) < 128 for c in name):
        return name
    # Recover the original bytes
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name  # Can't reach the raw bytes; give up
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            decoded = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        # Accept the candidate if it has any CJK chars (which is what we
        # were trying to recover) — otherwise fall through and keep
        # trying the next encoding.
        if any("一" <= ch <= "鿿" for ch in decoded):
            return decoded
    return name  # All decodings failed; return whatever Python gave us


def _extract_zip_files(zip_bytes: bytes,
                        max_files: int = 50,
                        max_total_bytes: int = 5 * 1024 * 1024,
                        ) -> list[tuple[str, bytes]]:
    """Pull (filename, raw_bytes) pairs out of a .zip archive.

    Caps:
      - max_files entries (avoids zip bombs)
      - max_total_bytes uncompressed (also avoids zip bombs)
    Skips dotfiles (e.g., __MACOSX/, .DS_Store) and directories.
    Fixes mojibake'd Chinese filenames via _decode_zip_filename().
    """
    out: list[tuple[str, bytes]] = []
    total = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400,
                             detail="File is not a valid zip archive")
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = _decode_zip_filename(info)
        # Skip macOS metadata + hidden files
        base = Path(name).name
        if base.startswith(".") or "__MACOSX" in name:
            continue
        # Skip very large single entries (5 MB cap per file)
        if info.file_size > 5 * 1024 * 1024:
            continue
        with zf.open(info) as f:
            data = f.read()
        total += len(data)
        if total > max_total_bytes:
            raise HTTPException(
                status_code=413,
                detail=(f"Zip uncompressed size exceeds {max_total_bytes // (1024*1024)} MB. "
                         f"Smart paste won't process beyond that — split into smaller archives."),
            )
        out.append((name, data))
        if len(out) >= max_files:
            break
    return out


_DOC_FILENAME_HINTS = (
    "readme", "说明", "指南", "教程", "帮助", "help", "guide",
    "manual", "instructions", "tutorial", "license",
    "test_kit", "测试套件", "测试包", "demo_guide",
    "expected_results",
)


def _looks_like_documentation(name: str, text: str) -> bool:
    """Heuristic: is this likely a usage doc (not patient data)?

    Combines filename hints (README.md, *指南*.md, etc.) with content
    sniffing (presence of phrases like "this is a test kit", "下载示例",
    "step 1:", URL-heavy, etc.). Used to skip docs that would otherwise
    bloat the LLM context with irrelevant content.
    """
    lower_name = name.lower()
    base = Path(lower_name).name
    for hint in _DOC_FILENAME_HINTS:
        if hint in base:
            return True
    # Content sniff: look for tutorial-style markers in the first 2 KB
    head = text[:2000].lower()
    doc_markers = (
        "## ", "### ", "step 1", "step 2", "first, ",
        "this guide", "this tutorial", "this kit",
        "本指南", "本教程", "本测试", "本套件",
        "如何使用", "怎么用", "instructions:",
    )
    n_markers = sum(1 for m in doc_markers if m in head)
    return n_markers >= 3   # Need 3+ tutorial markers to call it a doc


def _classify_file(name: str, data: bytes) -> tuple[str, Optional[str]]:
    """Decide what kind of file this is and return (kind, decoded_text).

    kinds:
      - "rna_seq"        → looks like RNA counts CSV/TSV
      - "clinical"       → patient text/CSV/TSV/PDF/XLSX/TXT — pass to LLM
      - "documentation"  → README / guide / how-to — skip from LLM context
      - "ignored"        → unsupported binary; skip
    """
    lower = name.lower()
    text: Optional[str] = None
    if lower.endswith(".pdf"):
        text = _try_extract_pdf_text(data)
    elif lower.endswith(".xlsx") or lower.endswith(".xlsm"):
        text = _try_extract_xlsx_as_csv(data)
    else:
        text = _try_decode(data)
    if text is None or not text.strip():
        return "ignored", None
    is_rna, _, _ = _looks_like_rna_seq(text)
    if is_rna:
        return "rna_seq", text
    if _looks_like_documentation(name, text):
        return "documentation", text
    return "clinical", text


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

    # Read raw bytes once; route on extension
    raw = file.file.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 2 MB smart-parse limit.",
        )
    fname_lower = (file.filename or "").lower()

    # ─── Case 0: ZIP archive — extract, classify each file, then run
    #            both RNA detection AND LLM parse on the assembled text ───
    if fname_lower.endswith(".zip"):
        members = _extract_zip_files(raw)
        zip_summary: list[dict] = []
        rna_seq_member: Optional[tuple[str, bytes, str]] = None  # (name, raw, text)
        clinical_chunks: list[str] = []

        for name, data in members:
            kind, text = _classify_file(name, data)
            zip_summary.append({"name": name, "size": len(data), "kind": kind})
            if kind == "rna_seq" and rna_seq_member is None:
                rna_seq_member = (name, data, text or "")
            elif kind == "clinical":
                # Prefix with filename so the LLM has context
                clinical_chunks.append(f"=== {name} ===\n{text}")
            # documentation / ignored: tracked in summary but NOT sent to
            # LLM — keeps tutorial-style text out of the patient context

        if not clinical_chunks and rna_seq_member is None:
            raise HTTPException(
                status_code=400,
                detail=("Zip archive contained no parseable files. "
                         "Supported: CSV, TSV, TXT, PDF, XLSX. Check the "
                         "summary in the response."),
            )

        # If only RNA-Seq found, return early
        if not clinical_chunks and rna_seq_member is not None:
            n_genes = len(rna_seq_member[2].splitlines()) - 1
            db.add(UsageEvent(
                user_id=user.id, event_type="llm_parse_zip_rna_only",
                meta={"file_name": file.filename, "members": len(members)},
            ))
            db.commit()
            return ParseFileResponse(
                parsed={}, confidence={}, warnings=[
                    f"Zip contained only an RNA-Seq counts file "
                    f"({rna_seq_member[0]}, {n_genes} genes). Will be used "
                    f"as the RNA-Seq input below."
                ],
                model="(no LLM used)", provider="(zip-extraction)",
                tokens_in=None, tokens_out=None,
                is_rna_seq=True,
                rna_seq_gene_count=n_genes,
                rna_seq_mean_value=0.0,
                file_name=file.filename or "",
                file_bytes=len(raw),
                rna_seq_attachment_b64=base64.b64encode(rna_seq_member[1]).decode(),
                rna_seq_attachment_filename=rna_seq_member[0],
                zip_summary=zip_summary,
            )

        # Otherwise: send clinical chunks to LLM, also stage RNA if present
        combined_text = "\n\n".join(clinical_chunks)

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
                raw_text=combined_text, api_key=plaintext_key,
                provider=llm_key.provider,
                focus_patient=patient_identifier,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Parse failed: {e}")
        except Exception as e:
            msg = str(e)
            hint = ""
            if "401" in msg or "Unauthorized" in msg:
                hint = (f" Your {llm_key.provider} API key appears to be "
                         f"invalid — re-add it in /llm-keys.")
            raise HTTPException(status_code=400,
                                 detail=f"LLM provider error: {msg}.{hint}")

        llm_key.last_used_at = datetime.now(tz=timezone.utc)
        # Per concern #4b — emit dedicated LLM audit event with PHI metadata
        # (no prompt content). Parse the PHI summary out of the warnings list
        # the LLM module surfaced.
        phi_warning_lines = [w for w in result.warnings
                              if "PHI detected" in w or "auto-redacted" in w]
        phi_categories: list[str] = []
        phi_high_count = 0
        for line in phi_warning_lines:
            # Extract count + categories from the standard message format
            import re
            m = re.search(r"(\d+) high-severity match", line)
            if m:
                phi_high_count = int(m.group(1))
            m = re.search(r"categories: ([^.]+)", line)
            if m:
                phi_categories = [c.strip() for c in m.group(1).split(",")]
        db.add(UsageEvent(
            user_id=user.id, event_type="llm_parse_zip_combined",
            meta={
                "provider": llm_key.provider, "model": result.model,
                "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "members": len(members),
                "rna_seq_in_zip": rna_seq_member is not None,
                "warnings_count": len(result.warnings),
            },
        ))
        db.add(UsageEvent(
            user_id=user.id, event_type="llm_audit",
            meta={
                "feature": "parse_clinical_text_zip",
                "provider": llm_key.provider, "model": result.model,
                "input_chars": len(combined_text),
                "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "phi_categories_detected": phi_categories,
                "phi_high_severity_count": phi_high_count,
                "phi_policy": "redact",
                "success": True,
            },
        ))
        db.commit()

        # Add a meta-warning summarizing what was processed
        synthetic_warning = (
            f"Zip processed: {len(zip_summary)} files. "
            f"Clinical text chunks: {len(clinical_chunks)}. "
            f"RNA-Seq counts: "
            f"{'found ' + rna_seq_member[0] if rna_seq_member else 'not found'}."
        )
        warnings_out = [synthetic_warning] + (result.warnings or [])

        rna_b64 = (base64.b64encode(rna_seq_member[1]).decode()
                    if rna_seq_member else None)
        rna_name = rna_seq_member[0] if rna_seq_member else None

        return ParseFileResponse(
            parsed=result.parsed,
            confidence=result.confidence,
            warnings=warnings_out,
            model=result.model, provider=result.provider,
            tokens_in=result.tokens_in, tokens_out=result.tokens_out,
            is_rna_seq=False,
            rna_seq_gene_count=(
                len(rna_seq_member[2].splitlines()) - 1
                if rna_seq_member else 0
            ),
            rna_seq_mean_value=0.0,
            file_name=file.filename or "",
            file_bytes=len(raw),
            rna_seq_attachment_b64=rna_b64,
            rna_seq_attachment_filename=rna_name,
            zip_summary=zip_summary,
        )

    # ─── Case 1+2: single-file path (CSV / TSV / TXT / PDF / XLSX) ───
    # Re-decode the raw we already read
    text = _try_decode(raw)
    if text is None and fname_lower.endswith(".pdf"):
        text = _try_extract_pdf_text(raw)
    if text is None and (fname_lower.endswith(".xlsx") or fname_lower.endswith(".xlsm")):
        text = _try_extract_xlsx_as_csv(raw)
    if text is None:
        raise HTTPException(status_code=400,
                             detail="Could not decode file as text.")
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
            # Stage the same bytes back for the client to reconstitute.
            # This is what makes "drag CSV → form below auto-fills" work
            # even when DataTransfer.items.add() fails on certain Safari
            # versions (we just rebuild the File from base64).
            rna_seq_attachment_b64=base64.b64encode(raw).decode(),
            rna_seq_attachment_filename=file.filename or "rna_counts.csv",
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
    rna_counts: Optional[UploadFile] = File(
        None, description="2-col CSV: symbol,count (5000-gene panel). "
                          "Optional only when use_demo_rna_seq=true."),
    rna_full: Optional[UploadFile] = File(None, description="Optional 2-col CSV: full transcriptome"),
    use_demo_rna_seq: bool = Form(
        False,
        description="If true, server fills in a built-in demo RNA-Seq counts "
                    "file when the user has none. The kit's Layer-3 prediction "
                    "becomes meaningless (synthetic input) — the report adds "
                    "an explicit caveat so the clinician knows.",
    ),
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

    if rna_counts is None and not use_demo_rna_seq:
        raise HTTPException(
            status_code=400,
            detail=("RNA-Seq counts file is required. Upload a real one OR "
                     "set use_demo_rna_seq=true to use a built-in demo "
                     "(Layer-3 prediction will be synthetic and the report "
                     "will say so)."),
        )

    # Inject a confidence note if running on demo data
    if rna_counts is None and use_demo_rna_seq:
        # Append a synthetic-data warning into intent_comment so the
        # downstream kit's confidence_notes reflect it.
        existing = payload.intent_comment or ""
        marker = "[SYNTHETIC RNA-SEQ — DEMO DATA — Layer-3 prediction not clinically valid]"
        if marker not in existing:
            payload.intent_comment = (existing + "\n" + marker).strip()

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
    if rna_counts is not None:
        sub.rna_counts_path = _persist_upload(rna_counts, sub_dir, "rna_counts")
    else:
        # Generate the demo CSV directly into the submission dir
        import joblib as _joblib
        import numpy as _np
        bundle = _joblib.load(s.KIT_ASSETS_ROOT
                              / "beataml_rna_preprocessor.joblib")
        kept = bundle["kept_genes"]
        rng = _np.random.default_rng(17)
        counts = rng.lognormal(mean=4.0, sigma=1.2, size=len(kept))
        sub_dir.mkdir(parents=True, exist_ok=True)
        demo_path = sub_dir / "rna_counts_DEMO.csv"
        with open(demo_path, "w") as fh:
            fh.write("symbol,count\n")
            for g, c in zip(kept, counts):
                fh.write(f"{g},{c:.1f}\n")
        sub.rna_counts_path = str(demo_path)
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
