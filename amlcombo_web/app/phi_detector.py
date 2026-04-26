"""PHI (Protected Health Information) detector for smart-paste inputs.

Per clinical-reviewer concern #4 (post-v0.3): Smart-paste rounds-trips
user-pasted text through third-party LLM APIs (OpenAI / Anthropic) over
international borders. The "please don't paste PHI" instruction in the UI
is a USER responsibility, not a technical safeguard. Real users in busy
clinical settings will paste raw chart text containing names, MRNs, and
DOBs.

This detector runs BEFORE any LLM dispatch and:
  1. Detects high-confidence PHI patterns (MRN, names with title, ID
     numbers, dates of birth)
  2. Returns a structured PHIDetectionResult listing every match with
     position + category, so the API can either:
       (a) reject the request (default for high-risk findings), or
       (b) auto-redact the text to "[REDACTED-MRN]" / "[REDACTED-NAME]"
           before forwarding to the LLM, or
       (c) require the user to acknowledge the risk via a separate
           confirmation flag

Coverage is deliberately conservative — we'd rather have false positives
(reject benign text) than false negatives (leak PHI). The kit is research
use only, so erring toward refusal is the right safety stance.

Patterns covered:
  - **MRN-like ID**: ≥6 digit number near keywords like 病历号 / 住院号 /
    MRN / hospital_no / patient_id (excluding the user's de-identified
    research ID like A-2026-001 which has dashes)
  - **Chinese full name**: 姓名 / Patient Name / 患者姓名 followed by 2-4
    Chinese characters
  - **Western full name with title**: Mr./Ms./Mrs./Dr. + capitalized first
    + last name
  - **DOB**: explicit "DOB:" or "出生日期" + a date
  - **National ID number** (China 18-digit citizen ID, US SSN xxx-xx-xxxx)
  - **Phone**: 11-digit Chinese mobile (1[3-9]xxxxxxxxx), US (xxx)xxx-xxxx
  - **Email** (any @ pattern with domain)
  - **Postal address**: contains 路 / 街 / 巷 / 室 / 号 along with city/省

NOT covered (intentionally — these are clinical, not personal):
  - Lab values, drug doses, dates of treatment events
  - Karyotypes (which look numeric)
  - Mutation VAFs / allelic ratios
  - Hospital department names
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern


class PHICategory(str, Enum):
    MRN = "MRN/hospital_id"
    NAME_CN = "Chinese full name"
    NAME_WESTERN = "Western full name with title"
    DOB = "Date of birth"
    NATIONAL_ID_CN = "Chinese citizen ID (18 digit)"
    SSN_US = "US Social Security Number"
    PHONE_CN = "Chinese mobile number"
    PHONE_US = "US phone number"
    EMAIL = "Email address"
    ADDRESS = "Postal address"


# Allowed de-identified research ID patterns — these contain digits but
# are explicitly NOT PHI (patient labels like "A-2026-001", "INST-2026-N").
DEIDENTIFIED_RESEARCH_ID_PATTERNS = [
    re.compile(r"^[A-Z]{1,5}-\d{4}-[A-Z0-9]{1,5}$"),       # A-2026-001
    re.compile(r"^[A-Z]+_\d+$"),                              # PATIENT_001
    re.compile(r"^TEST-\d+$"),                                # TEST-001
    re.compile(r"^SYN-\d+$"),                                 # synthetic
]


@dataclass
class PHIMatch:
    category: PHICategory
    matched_text: str
    start: int
    end: int
    severity: str  # "high" | "medium" | "low"


@dataclass
class PHIDetectionResult:
    has_phi: bool
    high_severity_count: int
    matches: list[PHIMatch] = field(default_factory=list)
    redacted_text: str = ""

    def categories_found(self) -> list[str]:
        return sorted({m.category.value for m in self.matches})


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------


# MRN: keyword followed by digits OR pure 6+ digit number near MRN-keyword
_MRN_KEYWORDS_CN = ["病历号", "住院号", "门诊号", "MRN", "病案号", "卡号"]
_MRN_KEYWORDS_EN = ["mrn", "hospital_no", "hospital no", "patient id",
                     "patient_id", "medical record"]


def _is_deidentified_research_id(s: str) -> bool:
    """Check if a token looks like a de-identified research ID and so
    should NOT be flagged as PHI."""
    s = s.strip().upper()
    return any(p.match(s) for p in DEIDENTIFIED_RESEARCH_ID_PATTERNS)


def _detect_mrn(text: str) -> list[PHIMatch]:
    """MRN-like IDs near MRN keywords. Skip de-identified research labels."""
    matches: list[PHIMatch] = []
    # Pattern: (keyword)[\s:=]*([digits or alphanumeric ID])
    keyword_pattern = "|".join(re.escape(k) for k in _MRN_KEYWORDS_CN + _MRN_KEYWORDS_EN)
    pat = re.compile(
        rf"(?i)({keyword_pattern})\s*[:：=\-]?\s*([A-Z0-9\-_]{{4,20}})"
    )
    for m in pat.finditer(text):
        candidate = m.group(2)
        if _is_deidentified_research_id(candidate):
            continue
        # Pure digits ≥ 6, or alphanumeric with no dash/underscore
        if re.match(r"^\d{6,}$", candidate) or re.match(r"^[A-Z]\d{4,}$", candidate, re.I):
            matches.append(PHIMatch(
                category=PHICategory.MRN,
                matched_text=m.group(0),
                start=m.start(), end=m.end(),
                severity="high",
            ))
    return matches


def _detect_name_cn(text: str) -> list[PHIMatch]:
    """姓名: 张三 / 患者姓名: 王xx / Patient Name: 李四 — 2-4 Chinese chars
    after a name-keyword."""
    pat = re.compile(
        r"(?:姓名|患者姓名|患者|name|patient name|患者名)[:：\s]*"
        r"([一-龥]{2,4})(?![一-龥])",
        re.IGNORECASE,
    )
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.NAME_CN,
            matched_text=m.group(0),
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


def _detect_name_western(text: str) -> list[PHIMatch]:
    """Mr./Ms./Mrs./Dr. + Capitalized First + Last."""
    pat = re.compile(
        r"\b(?:Mr|Ms|Mrs|Dr|Prof|Patient|Pt)\.?\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    )
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.NAME_WESTERN,
            matched_text=m.group(0),
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


def _detect_dob(text: str) -> list[PHIMatch]:
    """Date-of-birth keywords + a date. Plain dates (treatment dates) NOT
    flagged because they're clinically necessary."""
    pat = re.compile(
        r"(?i)(?:DOB|date of birth|出生日期|出生年月|生日)\s*[:：=]?\s*"
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    )
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.DOB,
            matched_text=m.group(0),
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


def _detect_national_id_cn(text: str) -> list[PHIMatch]:
    """China citizen ID: 18 digits ending in digit or X.
    Validates province prefix loosely (11-65 are valid province codes)."""
    pat = re.compile(
        r"(?<![\dXx])((?:1[1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4]|6[1-5])"
        r"\d{4}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
        r"\d{3}[\dXx])(?![\dXx])"
    )
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.NATIONAL_ID_CN,
            matched_text=m.group(1),
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


def _detect_ssn(text: str) -> list[PHIMatch]:
    """US SSN xxx-xx-xxxx. Excludes obvious placeholders (000-00-0000 etc)."""
    pat = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")
    matches = []
    for m in pat.finditer(text):
        s = m.group(1)
        if s.startswith("000") or s.split("-")[1] == "00" or s.endswith("0000"):
            continue
        matches.append(PHIMatch(
            category=PHICategory.SSN_US,
            matched_text=s,
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


def _detect_phone_cn(text: str) -> list[PHIMatch]:
    """China mobile: 1[3-9]xxxxxxxxx (11 digits). Excludes lab values that
    look like phone numbers — phones rarely appear without area context."""
    pat = re.compile(
        r"(?<!\d)(1[3-9]\d{9})(?!\d)"
    )
    matches = []
    for m in pat.finditer(text):
        # Sanity: skip if surrounded by lab-value cues (10^9/L, x10^9, etc.)
        ctx = text[max(0, m.start() - 20): min(len(text), m.end() + 20)]
        if re.search(r"10\^?9|×10|x10|10\*\*9|/L|μL|uL", ctx):
            continue
        matches.append(PHIMatch(
            category=PHICategory.PHONE_CN,
            matched_text=m.group(1),
            start=m.start(), end=m.end(),
            severity="medium",
        ))
    return matches


def _detect_phone_us(text: str) -> list[PHIMatch]:
    """(xxx)xxx-xxxx or xxx-xxx-xxxx, restricted to US-style area codes."""
    pat = re.compile(
        r"(?<!\d)(\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4})(?!\d)"
    )
    matches = []
    for m in pat.finditer(text):
        # Need context cue to reduce FP on lab numbers
        ctx = text[max(0, m.start() - 30): m.start()].lower()
        if "phone" in ctx or "tel" in ctx or "电话" in ctx or "tel:" in ctx:
            matches.append(PHIMatch(
                category=PHICategory.PHONE_US,
                matched_text=m.group(1),
                start=m.start(), end=m.end(),
                severity="medium",
            ))
    return matches


def _detect_email(text: str) -> list[PHIMatch]:
    pat = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.EMAIL,
            matched_text=m.group(0),
            start=m.start(), end=m.end(),
            severity="medium",
        ))
    return matches


def _detect_address(text: str) -> list[PHIMatch]:
    """Conservative: require ≥2 of {省/市/区, 路/街/巷, 号/室}."""
    pat = re.compile(
        r"(?:[一-龥]{2,8}(?:省|市|区|县))"
        r"[一-龥]*"
        r"(?:[一-龥]{2,10}(?:路|街|巷|大道))"
        r"[一-龥\d]*"
        r"(?:\d+号|\d+室)?"
    )
    matches = []
    for m in pat.finditer(text):
        matches.append(PHIMatch(
            category=PHICategory.ADDRESS,
            matched_text=m.group(0),
            start=m.start(), end=m.end(),
            severity="high",
        ))
    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_DETECTORS = [
    _detect_mrn,
    _detect_name_cn,
    _detect_name_western,
    _detect_dob,
    _detect_national_id_cn,
    _detect_ssn,
    _detect_phone_cn,
    _detect_phone_us,
    _detect_email,
    _detect_address,
]


def detect_phi(text: str) -> PHIDetectionResult:
    """Run all PHI detectors over the input text. Returns aggregated
    matches sorted by position + a redacted version of the text."""
    if not text:
        return PHIDetectionResult(has_phi=False, high_severity_count=0,
                                    matches=[], redacted_text="")

    all_matches: list[PHIMatch] = []
    for detector in _DETECTORS:
        all_matches.extend(detector(text))

    # De-duplicate overlapping matches (keep earliest start, longest span)
    all_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    deduped = []
    last_end = -1
    for m in all_matches:
        if m.start >= last_end:
            deduped.append(m)
            last_end = m.end

    # Build redacted text
    if deduped:
        parts = []
        cursor = 0
        for m in deduped:
            parts.append(text[cursor:m.start])
            parts.append(f"[REDACTED-{m.category.name}]")
            cursor = m.end
        parts.append(text[cursor:])
        redacted = "".join(parts)
    else:
        redacted = text

    high = sum(1 for m in deduped if m.severity == "high")

    return PHIDetectionResult(
        has_phi=bool(deduped),
        high_severity_count=high,
        matches=deduped,
        redacted_text=redacted,
    )


def assert_no_high_severity_phi(text: str) -> PHIDetectionResult:
    """Convenience: raise PHIDetected if any high-severity PHI present.
    Use as a hard gate before LLM dispatch when redaction is not desired."""
    result = detect_phi(text)
    if result.high_severity_count > 0:
        raise PHIDetected(result)
    return result


class PHIDetected(Exception):
    """Raised when high-severity PHI is found and the caller has chosen
    'reject' rather than 'redact'."""
    def __init__(self, result: PHIDetectionResult):
        self.result = result
        cats = ", ".join(result.categories_found())
        super().__init__(
            f"High-severity PHI detected ({result.high_severity_count} match(es)): "
            f"{cats}. Either redact and resubmit, or set acknowledge_phi=true "
            f"if you have institutional authorization to send this content "
            f"to a third-party LLM."
        )
