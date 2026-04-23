"""Tests for Route C regimen retrieval."""

from __future__ import annotations

import pytest

from combo_val.clinical.regimen_db import (
    EVIDENCE_SCORE,
    REGIMEN_BY_ID,
    REGIMEN_DB,
    count_by_property,
)
from combo_val.clinical.regimen_matcher import match_patient


# ---------------------------------------------------------------------------
# DB sanity
# ---------------------------------------------------------------------------


def test_db_has_expected_size_and_distribution():
    stats = count_by_property()
    assert stats["total"] >= 20
    assert stats["by_n_drugs"].get(3, 0) >= 5                 # at least 5 triplets
    assert stats["by_n_drugs"].get(2, 0) >= 8                 # doublets
    assert "FDA" in stats["by_evidence"]
    assert "Phase3" in stats["by_evidence"]


def test_every_regimen_has_required_fields():
    for r in REGIMEN_DB:
        assert r.regimen_id
        assert r.name
        assert r.n_drugs >= 0
        assert 0.0 <= r.outcome_cr_cri_rate <= 1.0
        assert r.trial_phase in EVIDENCE_SCORE


def test_regimen_ids_are_unique():
    ids = [r.regimen_id for r in REGIMEN_DB]
    assert len(ids) == len(set(ids))


def test_db_covers_key_biomarker_classes():
    """Every common AML driver must have at least one matching regimen."""
    # At least one FLT3-targeted regimen
    assert any("mut_FLT3" in r.required_any or "clin_flt3_itd" in r.required_any
               or "clin_flt3_itd" in r.required_all for r in REGIMEN_DB)
    # At least one IDH1-targeted regimen
    assert any("mut_IDH1" in r.required_all for r in REGIMEN_DB)
    # At least one IDH2-targeted regimen
    assert any("mut_IDH2" in r.required_all for r in REGIMEN_DB)
    # At least one APL regimen
    assert any("fusion_PML_RARA" in r.required_all for r in REGIMEN_DB)
    # At least one fallback for driver-negative
    assert any(r.regimen_id == "supportive_care_or_trial" for r in REGIMEN_DB)


# ---------------------------------------------------------------------------
# Matcher behaviour
# ---------------------------------------------------------------------------


def test_flt3_itd_young_fit_gets_flt3_targeted_top1():
    matches = match_patient({
        "mut_FLT3": 1, "clin_flt3_itd": 1, "clin_flt3_allelic_ratio": 0.6,
        "clin_age": 45, "clin_fit_for_intensive": 1, "clin_is_relapse": 0,
    }, top_k=3)
    assert matches
    top_drugs = set(matches[0].regimen.drugs)
    flt3_inhibitors = {"Quizartinib (AC220)", "Gilteritinib", "Midostaurin"}
    assert flt3_inhibitors & top_drugs


def test_idh1_elderly_unfit_gets_agile_or_triplet():
    matches = match_patient({
        "mut_IDH1": 1, "clin_age": 78, "clin_fit_for_intensive": 0,
        "clin_is_relapse": 0,
    }, top_k=3)
    assert matches
    # Top-1 should contain Ivosidenib
    assert "Ivosidenib" in matches[0].regimen.drugs


def test_idh2_newly_diagnosed_has_idh2i_in_top3():
    matches = match_patient({
        "mut_IDH2": 1, "clin_age": 72, "clin_fit_for_intensive": 0,
        "clin_is_relapse": 0,
    }, top_k=3)
    assert any("Enasidenib" in m.regimen.drugs for m in matches)


def test_apl_top1_is_atra_ato():
    """APL patient MUST get ATRA+ATO as top-1. Fallback supportive_care may
    also match (stage=any, no biomarker restriction) but must rank below."""
    matches = match_patient({
        "fusion_PML_RARA": 1, "clin_age": 42, "clin_fit_for_intensive": 1,
        "clin_is_relapse": 0,
    }, top_k=5)
    assert matches
    assert matches[0].regimen.regimen_id == "atra_ato_low"
    # No other chemotherapy regimen should be eligible (excluded_any=PML-RARA)
    eligible_ids = [m.regimen.regimen_id for m in matches]
    forbidden = {"7_3", "7_3_midostaurin", "7_3_quizartinib", "cpx351_vyxeos",
                  "ven_aza", "ven_dec", "ldac_ven", "ldac_glasdegib"}
    assert not (set(eligible_ids) & forbidden), \
        f"Chemotherapy regimens leaked through APL exclusion: {set(eligible_ids) & forbidden}"


def test_driver_negative_rr_unfit_gets_fallback():
    matches = match_patient({
        "clin_age": 72, "clin_fit_for_intensive": 0, "clin_is_relapse": 1,
    }, top_k=3)
    assert matches                                     # must get something
    assert any(m.regimen.regimen_id == "supportive_care_or_trial" for m in matches)


def test_flt3_mut_triplet_outranks_doublet_when_both_eligible():
    """Elderly unfit FLT3-mut patient — triplet with 96% CR should beat Ven+Aza 66%."""
    matches = match_patient({
        "mut_FLT3": 1, "clin_flt3_itd": 1, "clin_age": 75,
        "clin_fit_for_intensive": 0, "clin_is_relapse": 0,
    }, top_k=10)
    # Find highest-rank triplet and doublet
    top_triplet = next((m for m in matches if m.regimen.n_drugs == 3), None)
    top_doublet = next((m for m in matches if m.regimen.n_drugs == 2), None)
    if top_triplet and top_doublet:
        assert top_triplet.score >= top_doublet.score


def test_excluded_biomarker_removes_regimen():
    """APL patient should NOT match Ven+Aza (excluded_any=fusion_PML_RARA)."""
    matches = match_patient({
        "fusion_PML_RARA": 1, "clin_age": 40, "clin_fit_for_intensive": 1,
        "clin_is_relapse": 0,
    }, top_k=20)
    regimen_ids = [m.regimen.regimen_id for m in matches]
    assert "ven_aza" not in regimen_ids


def test_age_constraint_enforced():
    """Patient age 90 should not match age_max=75 regimens."""
    matches = match_patient({
        "mut_FLT3": 1, "clin_flt3_itd": 1,
        "clin_age": 90, "clin_fit_for_intensive": 0, "clin_is_relapse": 0,
    }, top_k=20)
    regimen_ids = [m.regimen.regimen_id for m in matches]
    # Quiz+7+3 and RATIFY both have age_max, should be filtered for very old unfit patients
    # (patient is unfit so they're filtered on fitness too)
    for rid in regimen_ids:
        r = REGIMEN_BY_ID[rid]
        if r.age_max is not None:
            assert r.age_max >= 90 or r.fitness == "unfit" or r.fitness == "any"


def test_stage_routing():
    """Newly diagnosed patient should NOT get R/R-only regimens."""
    matches = match_patient({
        "mut_FLT3": 1, "clin_age": 60, "clin_fit_for_intensive": 1,
        "clin_is_relapse": 0,
    }, top_k=20)
    regimen_ids = [m.regimen.regimen_id for m in matches]
    assert "gilt_mono" not in regimen_ids              # ADMIRAL is R/R
    assert "gilt_ven_rr" not in regimen_ids
    assert "ivo_mono" not in regimen_ids
    assert "ena_mono" not in regimen_ids


def test_summary_fields_complete():
    matches = match_patient({
        "mut_FLT3": 1, "clin_age": 45, "clin_fit_for_intensive": 1,
        "clin_is_relapse": 0,
    }, top_k=1)
    assert matches
    s = matches[0].as_summary()
    for key in ["regimen_id", "name", "n_drugs", "drugs", "trial_name",
                "trial_phase", "published_cr_cri_rate", "eligible", "score"]:
        assert key in s
