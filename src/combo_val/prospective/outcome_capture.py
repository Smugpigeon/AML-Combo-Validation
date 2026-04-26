"""Outcome-capture form schema for prospective validation.

After kit prediction is locked, the participating site fills these forms
at fixed time points to capture the patient's actual treatment + outcome:

  - day-30   : induction CR / CRi / refractory + adverse events
  - month-6  : MRD status + EFS-event-free
  - month-12 : OS + AE summary + transplant status (PRIMARY analysis trigger)
  - month-18 : OS + late events + 18-month survival landmark

Forms are stored as JSON in the same per-site directory as the locked
prediction. They reference the prediction_id for join (1:N — one
prediction may have multiple follow-up rows).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


CRStatus = Literal[
    "CR",          # complete remission per ELN 2017/2022
    "CRi",         # CR with incomplete count recovery
    "CRh",         # CR with partial hematologic recovery
    "PR",          # partial remission
    "MLFS",        # morphologic leukemia-free state
    "Refractory",  # failed induction
    "Death",       # induction-related death
    "Withdrawn",   # withdrew consent / lost to follow-up
    "Pending",     # too early to assess
]

MRDStatus = Literal[
    "MRD_neg",     # MRD-negative by appropriate modality
    "MRD_pos",     # MRD-positive
    "MRD_NA",      # not assessed (lab unable / not indicated)
    "Pending",
]

TransplantStatus = Literal[
    "received_alloSCT_CR1",
    "received_alloSCT_post_relapse",
    "planned_not_yet_received",
    "not_pursued_per_clinician",
    "patient_declined",
    "not_eligible",
    "Pending",
]


@dataclass
class Day30Outcome:
    prediction_id: str
    site_id: str
    patient_id: str
    assessment_date: str                     # ISO YYYY-MM-DD
    days_from_diagnosis: int
    actual_regimen_id: str                   # what was actually given (must be in REGIMEN_BY_ID or 'other')
    actual_regimen_free_text: str = ""       # required if actual_regimen_id == 'other'
    cr_status: CRStatus = "Pending"
    bone_marrow_blast_pct: Optional[float] = None
    adverse_events_grade_3plus: list[str] = field(default_factory=list)
    early_death: bool = False
    early_death_cause: str = ""
    notes: str = ""


@dataclass
class Month6Outcome:
    prediction_id: str
    site_id: str
    patient_id: str
    assessment_date: str
    days_from_diagnosis: int
    cr_status: CRStatus = "Pending"
    mrd_status: MRDStatus = "Pending"
    mrd_modality: str = ""                   # "NPM1_PCR" | "PML_RARA_PCR" | "MFC" | "NGS_MRD" etc
    mrd_log_reduction: Optional[float] = None
    relapsed: bool = False
    relapse_date: Optional[str] = None
    transplant_status: TransplantStatus = "Pending"
    alive: bool = True
    notes: str = ""


@dataclass
class Month12Outcome:
    """The PRIMARY analysis time point. Triggers external_validation pipeline."""
    prediction_id: str
    site_id: str
    patient_id: str
    assessment_date: str
    days_from_diagnosis: int
    alive: bool
    death_date: Optional[str] = None
    death_cause: str = ""
    cr_status_at_12mo: CRStatus = "Pending"
    mrd_status_at_12mo: MRDStatus = "Pending"
    relapsed_in_first_12mo: bool = False
    relapse_date: Optional[str] = None
    transplant_status: TransplantStatus = "Pending"
    transplant_date: Optional[str] = None
    follow_up_complete: bool = True
    notes: str = ""


@dataclass
class Month18Outcome:
    prediction_id: str
    site_id: str
    patient_id: str
    assessment_date: str
    days_from_diagnosis: int
    alive: bool
    death_date: Optional[str] = None
    relapsed_in_first_18mo: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


_OUTCOME_FILENAMES = {
    "day30": "day30_outcome.json",
    "month6": "month6_outcome.json",
    "month12": "month12_outcome.json",
    "month18": "month18_outcome.json",
}


def _patient_dir(runs_root: Path, site_id: str, patient_id: str) -> Path:
    safe_site = "".join(c if c.isalnum() or c in "_-" else "_" for c in site_id)
    safe_pt = "".join(c if c.isalnum() or c in "_-" else "_" for c in patient_id)
    return runs_root / "prospective" / safe_site / safe_pt


def write_outcome(runs_root: Path | str, timepoint: str, outcome) -> Path:
    """Write an outcome record to disk. Outcome must be one of the four
    @dataclass types above."""
    if timepoint not in _OUTCOME_FILENAMES:
        raise ValueError(f"Unknown timepoint {timepoint!r}; "
                          f"expected one of {list(_OUTCOME_FILENAMES)}")
    pt_dir = _patient_dir(Path(runs_root), outcome.site_id, outcome.patient_id)
    pt_dir.mkdir(parents=True, exist_ok=True)
    path = pt_dir / _OUTCOME_FILENAMES[timepoint]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "1.0",
            "timepoint": timepoint,
            "written_at": datetime.now(tz=timezone.utc).isoformat(),
            "outcome": asdict(outcome),
        }, f, ensure_ascii=False, indent=2)
    return path


def read_outcome(runs_root: Path | str, site_id: str, patient_id: str,
                  timepoint: str) -> Optional[dict]:
    pt_dir = _patient_dir(Path(runs_root), site_id, patient_id)
    path = pt_dir / _OUTCOME_FILENAMES[timepoint]
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("outcome")


# ---------------------------------------------------------------------------
# Cohort assembly for analysis
# ---------------------------------------------------------------------------


def assemble_cohort_for_analysis(runs_root: Path | str,
                                  site_ids: Optional[list[str]] = None,
                                  require_timepoint: str = "month12") -> list[dict]:
    """Walk the prospective/ tree and produce a list of (locked_prediction +
    outcome) joined dicts, suitable for feeding to external_validation.

    Args:
      runs_root: Path("runs") usually
      site_ids: list of sites to include; None = all
      require_timepoint: only include patients with this outcome filed
    """
    runs_root = Path(runs_root)
    base = runs_root / "prospective"
    if not base.exists():
        return []

    rows: list[dict] = []
    site_dirs = [p for p in base.iterdir() if p.is_dir()]
    if site_ids:
        safe_ids = ["".join(c if c.isalnum() or c in "_-" else "_" for c in s)
                    for s in site_ids]
        site_dirs = [p for p in site_dirs if p.name in safe_ids]

    for site_dir in site_dirs:
        for pt_dir in site_dir.iterdir():
            if not pt_dir.is_dir():
                continue
            outcome_path = pt_dir / _OUTCOME_FILENAMES[require_timepoint]
            if not outcome_path.exists():
                continue
            with open(outcome_path, "r", encoding="utf-8") as f:
                outcome = json.load(f).get("outcome", {})

            # Locked prediction — pick the latest (one per patient typically)
            locks = sorted(pt_dir.glob("*_locked_prediction.json"))
            if not locks:
                continue
            with open(locks[-1], "r", encoding="utf-8") as f:
                locked = json.load(f)

            # Compose external_validation cohort.csv-shaped row
            rows.append({
                "patient_id": locked.get("patient_id"),
                "site_id": locked.get("site_id"),
                "predicted_eln_2022": locked.get("predicted_eln_2022"),
                "predicted_top1_id": locked.get("predicted_top1_regimen_id"),
                "true_regimen_id": outcome.get("actual_regimen_id"),
                "true_cr": 1 if outcome.get("cr_status_at_12mo") in ("CR", "CRi", "CRh") else 0,
                "true_os_event": 0 if outcome.get("alive") else 1,
                "true_eln_class": locked.get("predicted_eln_2022"),  # placeholder; site can override
                "kit_commit_hash": locked.get("kit_commit_hash"),
                "model_checkpoint_sha256": locked.get("model_checkpoint_sha256"),
                "backbone": locked.get("backbone"),
            })

    return rows
