"""Match a new patient to AML regimens from REGIMEN_DB.

Logic:
  Hard eligibility (pass/fail):
    - required_all:  every biomarker must be 1
    - required_any:  at least one of these biomarkers must be 1 (if non-empty)
    - excluded_any:  none of these biomarkers may be 1
    - age in [age_min, age_max] (None = unbounded)
    - fitness matches (or regimen declares "any")
    - stage matches (or regimen declares "any")

  Soft score (for ranking among eligible regimens):
    score = evidence_weight(trial_phase)           # FDA=100, P3=80, P2=50, ...
          + 100 * outcome_cr_cri_rate              # higher-efficacy wins
          + 5 * #preferred_biomarkers_present
          + 5 * bonus_for_specific_match           # FLT3-targeted for FLT3-mut etc.

  Ties broken by: n_drugs desc (prefer triplets), then regimen_id.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from combo_val.clinical.regimen_db import (
    EVIDENCE_SCORE,
    REGIMEN_BY_ID,
    REGIMEN_DB,
    Regimen,
)


@dataclass
class MatchedRegimen:
    """One regimen matched to one patient, with full reasoning."""
    regimen: Regimen
    eligible: bool
    score: float
    biomarker_matches: list[str]                    # which required/preferred hit
    biomarker_violations: list[str]                 # which failed (if not eligible)
    age_match: bool
    fitness_match: bool
    stage_match: bool

    def as_summary(self) -> dict:
        return {
            "regimen_id": self.regimen.regimen_id,
            "name": self.regimen.name,
            "n_drugs": self.regimen.n_drugs,
            "drugs": list(self.regimen.drugs),
            "trial_name": self.regimen.trial_name,
            "trial_phase": self.regimen.trial_phase,
            "published_cr_cri_rate": self.regimen.outcome_cr_cri_rate,
            "published_median_os_months": self.regimen.outcome_median_os_months,
            "eligible": self.eligible,
            "score": round(self.score, 2),
            "biomarker_matches": self.biomarker_matches,
            "biomarker_violations": self.biomarker_violations,
            "cautions": list(self.regimen.cautions),
            "pmid": self.regimen.pmid,
            "nct_id": self.regimen.nct_id,
            # Issue #1 — surface clinical_tier so downstream renderers and
            # external consumers can audit ranking decisions.
            "clinical_tier": getattr(
                self.regimen, "clinical_tier", "experimental_triplet"
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bio_present(features: Mapping[str, float], name: str) -> bool:
    """Truthy check for a binary biomarker column."""
    if name not in features:
        return False
    v = features[name]
    if v is None:
        return False
    try:
        return float(v) > 0.5
    except (TypeError, ValueError):
        return bool(v)


def _infer_stage(features: Mapping[str, float]) -> str:
    """Infer 'newly_diagnosed' or 'relapsed_refractory' from patient fields."""
    is_relapse = _bio_present(features, "clin_is_relapse")
    return "relapsed_refractory" if is_relapse else "newly_diagnosed"


def _infer_fitness(features: Mapping[str, float]) -> str:
    """Return 'fit' or 'unfit' based on clin_fit_for_intensive + age."""
    # Our existing rule: fit if age ≤ 65. But real practice also uses ECOG,
    # comorbidities. For now mirror the training-schema rule.
    age = features.get("clin_age")
    fit_flag = features.get("clin_fit_for_intensive")
    if fit_flag is not None:
        try:
            if float(fit_flag) >= 0.5:
                return "fit"
        except (TypeError, ValueError):
            pass
    if age is not None:
        try:
            return "fit" if float(age) <= 65 else "unfit"
        except (TypeError, ValueError):
            pass
    return "fit"  # default


def _age(features: Mapping[str, float]) -> float | None:
    a = features.get("clin_age")
    if a is None:
        return None
    try:
        return float(a)
    except (TypeError, ValueError):
        return None


def _check_eligibility(regimen: Regimen,
                        features: Mapping[str, float],
                        patient_stage: str,
                        patient_fitness: str,
                        patient_age: float | None) -> tuple[bool, list[str], list[str]]:
    """Return (eligible, matched_bio_list, violation_list)."""
    matches: list[str] = []
    violations: list[str] = []

    # required_all: every biomarker must be present
    for bio in regimen.required_all:
        if _bio_present(features, bio):
            matches.append(f"required_all:{bio}=1")
        else:
            violations.append(f"required_all:{bio}=0")

    # required_any: if list is non-empty, at least one must be present
    if regimen.required_any:
        any_hits = [bio for bio in regimen.required_any if _bio_present(features, bio)]
        if any_hits:
            for h in any_hits:
                matches.append(f"required_any:{h}=1")
        else:
            violations.append(f"required_any:{'|'.join(regimen.required_any)}=0 (none present)")

    # excluded_any: none may be present
    for bio in regimen.excluded_any:
        if _bio_present(features, bio):
            violations.append(f"excluded_any:{bio}=1")

    # age
    age_match = True
    if regimen.age_min is not None and (patient_age is None or patient_age < regimen.age_min):
        age_match = False
        violations.append(f"age {patient_age} < min {regimen.age_min}")
    if regimen.age_max is not None and (patient_age is not None and patient_age > regimen.age_max):
        age_match = False
        violations.append(f"age {patient_age} > max {regimen.age_max}")

    # fitness
    fitness_match = True
    if regimen.fitness != "any" and regimen.fitness != patient_fitness:
        fitness_match = False
        violations.append(f"fitness mismatch: regimen={regimen.fitness}, patient={patient_fitness}")

    # stage
    stage_match = True
    if regimen.stage != "any" and regimen.stage != patient_stage:
        stage_match = False
        violations.append(f"stage mismatch: regimen={regimen.stage}, patient={patient_stage}")

    eligible = (
        not violations
        or all(not v.startswith(("required_all", "required_any", "excluded_any"))
               for v in violations) and not regimen.required_all and not regimen.required_any
    )
    # Simpler: no violations → eligible
    eligible = len(violations) == 0
    return eligible, matches, violations


def _clinical_tier_match_bonus(regimen: Regimen,
                                 fitness: str, stage: str) -> float:
    """Hard tier match: ensures SOC ranks above experimental for context.

    Per issue #1 — for fit + newly-Dx FLT3-ITD AML, the SOC is 7+3 + FLT3i
    (RATIFY / QUANTUM-First, peer-reviewed Phase 3 + FDA approved). It must
    rank above HMA-based 'experimental triplets' that report inflated CR
    rates from small abstracts.

    Returns:
      +200 — strong match (regimen tier matches patient context)
       -50 — clear mismatch (e.g., experimental triplet for fit newly-Dx)
         0 — neutral
    """
    tier = regimen.clinical_tier
    if tier == "first_line_intensive":
        if fitness == "fit" and stage == "newly_diagnosed":
            return 200.0
        return -50.0

    if tier == "first_line_unfit":
        if fitness == "unfit" and stage == "newly_diagnosed":
            return 200.0
        return -30.0

    if tier == "experimental_triplet":
        # Available for any newly-Dx with the right biomarker, but never
        # first-choice over peer-reviewed SOC. Sized to not exceed the
        # +200 SOC bonus — so SOC always wins, but experimental still
        # ranks ABOVE salvage when patient has the biomarker.
        return -50.0

    if tier == "salvage":
        if stage == "relapsed_refractory":
            return 200.0
        return -150.0

    if tier == "supportive":
        return -100.0

    return 0.0


def _score(regimen: Regimen,
            features: Mapping[str, float],
            matched_biomarkers: list[str],
            fitness: str = "any",
            stage: str = "any") -> float:
    """Rank-ordering score among eligible regimens.

    Design (post-issue #1):
      1. Clinical-tier match dominates (±200) — SOC for the patient
         context (fit/unfit × newly-Dx/R/R) wins by default.
      2. Evidence level (FDA / P3 / P2 / consensus) is the base
      3. Published CR rate adds 0-100, but is capped lower for low-evidence
         trials so a 95% Phase-2 abstract can't outrank a 60% Phase-3
      4. Biomarker-specificity (moderate)
      5. APL (PML-RARA) extra ATRA/ATO bonus
    """
    s = 0.0
    # 1. Tier match (the headline change in issue #1)
    s += _clinical_tier_match_bonus(regimen, fitness, stage)
    # 2. Evidence level
    s += float(EVIDENCE_SCORE.get(regimen.trial_phase, 0))
    # 3. CR rate, weighted by evidence quality (caps inflation from small
    #    abstracts)
    cr_weight = {
        "FDA": 100.0, "Phase3": 100.0, "Phase2": 60.0,
        "Phase1": 30.0, "consensus": 50.0,
    }.get(regimen.trial_phase, 50.0)
    s += cr_weight * regimen.outcome_cr_cri_rate
    # 4. Preferred biomarkers (soft)
    for pref in regimen.preferred:
        if _bio_present(features, pref):
            s += 5.0
    # 5. Target-specificity (moderate boost when the regimen explicitly
    #    targets a driver the patient has)
    has_targeted_match = False
    for bio in regimen.required_all:
        if _bio_present(features, bio):
            s += 15.0
            has_targeted_match = True
    for bio in regimen.required_any:
        if _bio_present(features, bio):
            s += 10.0
            has_targeted_match = True
            break
    # 5b. Targeted-FDA bonus: when the regimen is BOTH (a) FDA-approved
    #     for an SOC tier matching the patient, AND (b) targets a specific
    #     driver mutation the patient carries, it should beat a general
    #     SOC regimen. Example: AGILE (Aza+Ivo for IDH1-mut unfit) must
    #     rank above Ven+Aza for an IDH1-mut elderly unfit patient.
    if (has_targeted_match
            and regimen.trial_phase in ("FDA", "Phase3")
            and regimen.clinical_tier in ("first_line_intensive",
                                            "first_line_unfit")):
        s += 50.0
    # 6. APL biology rule
    if "ATRA" in regimen.drugs and _bio_present(features, "fusion_PML_RARA"):
        s += 30.0
    # 7. Triplet preference (small tiebreaker)
    if regimen.n_drugs >= 3:
        s += 2.0
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_patient(
    patient_features: Mapping[str, float],
    top_k: int = 5,
    include_ineligible: bool = False,
) -> list[MatchedRegimen]:
    """Match one patient (dict or Series of feature-column → value).

    Returns top-k eligible regimens by score, descending. If
    include_ineligible=True, also returns ineligible regimens (at the tail)
    with their violation list for transparency.
    """
    stage = _infer_stage(patient_features)
    fitness = _infer_fitness(patient_features)
    age = _age(patient_features)

    all_matches: list[MatchedRegimen] = []
    for regimen in REGIMEN_DB:
        eligible, matched_bio, violations = _check_eligibility(
            regimen, patient_features, stage, fitness, age,
        )
        s = (_score(regimen, patient_features, matched_bio,
                     fitness=fitness, stage=stage)
             if eligible else 0.0)
        all_matches.append(MatchedRegimen(
            regimen=regimen,
            eligible=eligible,
            score=s,
            biomarker_matches=matched_bio,
            biomarker_violations=violations,
            age_match=(
                (regimen.age_min is None or (age is not None and age >= regimen.age_min))
                and (regimen.age_max is None or (age is not None and age <= regimen.age_max))
            ),
            fitness_match=(regimen.fitness == "any" or regimen.fitness == fitness),
            stage_match=(regimen.stage == "any" or regimen.stage == stage),
        ))

    eligible = [m for m in all_matches if m.eligible]
    eligible.sort(key=lambda m: (-m.score, -m.regimen.n_drugs, m.regimen.regimen_id))

    if include_ineligible:
        ineligible = [m for m in all_matches if not m.eligible]
        return eligible[:top_k] + ineligible
    return eligible[:top_k]


def match_cohort(
    patient_features_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Match a full patient cohort at once. Returns a long table with
    (patient_id, rank, regimen_id, score, ...) rows."""
    rows: list[dict] = []
    for pid, feat_row in patient_features_df.iterrows():
        feat_dict = feat_row.to_dict()
        matches = match_patient(feat_dict, top_k=top_k)
        for rank, m in enumerate(matches, start=1):
            s = m.as_summary()
            s["patient_id"] = pid
            s["rank"] = rank
            rows.append(s)
    return pd.DataFrame(rows)
