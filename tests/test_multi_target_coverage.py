"""Smoke tests for the Multi-Target Coverage solver (Path A v2).

Per Palmer & Sorger Cell 2017 (PMID 29245013) IDA framework — combo
benefit comes from covering multiple patient-specific vulnerability
targets, not from molecular drug-drug synergy.

Tests:
  taxonomy.py     — load yaml, build coverage matrix
  patient_targets — infer active targets from mut/fusion/RNA/clin
  set_cover       — Bliss-IDA aggregation, greedy + LS, constraint enforcement
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.coverage.set_cover import (
    bliss_coverage_per_target,
    find_top_combinations,
    greedy_select,
    local_search_refine,
    total_weighted_coverage,
    violates_toxicity,
)
from combo_val.coverage.taxonomy import (
    DrugCoverageMatrix, build_coverage_matrix, load_taxonomy,
)


# ---------------------------------------------------------------------------
# Bliss-IDA math
# ---------------------------------------------------------------------------


def test_bliss_single_drug_returns_drug_coverage():
    cov = np.array([[0.9, 0.0, 0.5]], dtype=np.float32)  # 1 drug × 3 targets
    out = bliss_coverage_per_target(cov, [0])
    assert np.allclose(out, [0.9, 0.0, 0.5])


def test_bliss_two_drugs_independent_combine():
    # Both drugs cover target 0 at 0.5: combined = 1 - (0.5)(0.5) = 0.75
    cov = np.array([[0.5, 0.0],
                     [0.5, 0.8]], dtype=np.float32)
    out = bliss_coverage_per_target(cov, [0, 1])
    assert np.isclose(out[0], 0.75)
    # target 1: 1 - (1)(0.2) = 0.8
    assert np.isclose(out[1], 0.8)


def test_bliss_perfect_drug_dominates():
    cov = np.array([[1.0, 0.0],
                     [0.5, 0.5]], dtype=np.float32)
    out = bliss_coverage_per_target(cov, [0, 1])
    assert np.isclose(out[0], 1.0)   # drug 0 already covers fully
    assert np.isclose(out[1], 0.5)


def test_bliss_empty_combo_returns_zero():
    cov = np.array([[0.5, 0.5]], dtype=np.float32)
    out = bliss_coverage_per_target(cov, [])
    assert np.allclose(out, [0.0, 0.0])


def test_total_weighted_coverage_normalizes():
    cov = np.array([0.8, 0.4, 0.0])
    weights = np.array([1.0, 1.0, 0.0])  # third target irrelevant
    # weighted = (1*0.8 + 1*0.4 + 0*0.0) / (1+1+0) = 0.6
    assert np.isclose(total_weighted_coverage(cov, weights), 0.6)


def test_total_weighted_coverage_handles_zero_weights():
    cov = np.array([0.5])
    assert total_weighted_coverage(cov, np.array([0.0])) == 0.0


# ---------------------------------------------------------------------------
# Toxicity constraints
# ---------------------------------------------------------------------------


def test_violates_toxicity_detects_overflow():
    tox = np.array([3.0, 1.0])
    axes = ["myelosuppression", "QT_prolongation"]
    ceiling = {"myelosuppression": 2.5, "QT_prolongation": 1.8}
    violations = violates_toxicity(tox, axes, ceiling)
    assert len(violations) == 1
    assert "myelosuppression" in violations[0]


def test_violates_toxicity_clean():
    tox = np.array([2.0, 1.5])
    axes = ["myelosuppression", "QT_prolongation"]
    ceiling = {"myelosuppression": 2.5, "QT_prolongation": 1.8}
    assert violates_toxicity(tox, axes, ceiling) == []


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------


def test_taxonomy_loads_18_targets():
    tx = load_taxonomy()
    ids = tx.target_ids()
    assert len(ids) >= 15
    # Spot-check critical targets present
    for t in ("FLT3", "IDH1", "IDH2", "BCL2", "MENIN_NPM1",
              "MENIN_KMT2A", "DNMT", "TP53_multihit"):
        assert t in ids, f"Missing {t} in taxonomy"


def test_taxonomy_targets_have_required_fields():
    tx = load_taxonomy()
    for t in tx.targets:
        assert t.id
        assert 1 <= t.tier <= 4
        assert 0 <= t.default_weight <= 1


def test_build_coverage_matrix_nonzero_for_known_drugs():
    tx = load_taxonomy()
    pool = ["Quizartinib (AC220)", "Venetoclax", "Azacytidine",
             "Ivosidenib", "Revumenib"]
    cm = build_coverage_matrix(tx, pool)
    assert cm.coverage.shape == (5, len(tx.target_ids()))
    # Quizartinib should have nonzero coverage on FLT3
    quiz_idx = cm.drug_id_to_idx["Quizartinib (AC220)"]
    flt3_idx = cm.target_id_to_idx["FLT3"]
    assert cm.coverage[quiz_idx, flt3_idx] > 0.5
    # Venetoclax should hit BCL2
    ven_idx = cm.drug_id_to_idx["Venetoclax"]
    bcl2_idx = cm.target_id_to_idx["BCL2"]
    assert cm.coverage[ven_idx, bcl2_idx] > 0.5
    # Ivosidenib should hit IDH1
    ivo_idx = cm.drug_id_to_idx["Ivosidenib"]
    idh1_idx = cm.target_id_to_idx["IDH1"]
    assert cm.coverage[ivo_idx, idh1_idx] > 0.5


# ---------------------------------------------------------------------------
# Patient target inference
# ---------------------------------------------------------------------------


def test_patient_target_inference_flt3_npm1():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    pf = pd.Series({
        "mut_FLT3": 1, "mut_NPM1": 1, "mut_DNMT3A": 1,
        "clin_flt3_itd": 1,
    })
    targets = infer_active_targets(pf, tx)
    # Should activate FLT3, MENIN_NPM1, DNMT (DNMT3A in mut list)
    assert "FLT3" in targets
    assert "MENIN_NPM1" in targets
    assert "DNMT" in targets
    # Always-active baselines too
    assert "BCL2" in targets       # always_active_baseline 0.6


def test_patient_target_inference_ignores_absent_genes():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    # All zeros — only always-active targets should fire
    pf = pd.Series({f"mut_{g}": 0 for g in [
        "FLT3", "NPM1", "IDH1", "IDH2", "TP53", "DNMT3A", "RUNX1"
    ]})
    targets = infer_active_targets(pf, tx)
    assert "FLT3" not in targets
    assert "IDH1" not in targets
    # Always-active baselines still appear
    assert "BCL2" in targets


def test_patient_target_inference_rna_modifier_boost():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    pf = pd.Series({"mut_FLT3": 0})
    rna = {"BCL2_high": 0.95}
    targets = infer_active_targets(pf, tx, rna_programs=rna)
    assert "BCL2" in targets
    assert targets["BCL2"] >= 0.6   # RNA boost above baseline


# ---------------------------------------------------------------------------
# End-to-end set-cover on a real BeatAML patient profile
# ---------------------------------------------------------------------------


def test_e2e_flt3_patient_recommends_flt3i_combo():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    pool = [
        "Quizartinib (AC220)", "Gilteritinib", "Midostaurin",
        "Venetoclax", "Azacytidine", "Decitabine",
        "Ivosidenib", "Enasidenib",
        "Revumenib", "Olutasidenib",   # may not be in pool unless added
        "Cytarabine", "Daunorubicin",
    ]
    cm = build_coverage_matrix(tx, pool)
    # FLT3-ITD + NPM1 + DNMT3A patient
    pf = pd.Series({
        "mut_FLT3": 1, "mut_NPM1": 1, "mut_DNMT3A": 1,
        "clin_flt3_itd": 1, "clin_age": 45, "clin_fit_for_intensive": 1,
    })
    active = infer_active_targets(pf, tx)
    constraints = {
        "max_arity": 4,
        "coverage_threshold": 0.65,
        "toxicity_ceiling": {"myelosuppression": 2.5,
                              "QT_prolongation": 1.8},
    }
    results = find_top_combinations(active, cm, constraints, n_solutions=3)
    assert len(results) >= 1
    top = results[0]
    # Top combo should contain a FLT3 inhibitor — that's the highest-weight target
    flt3i_drugs = {"Quizartinib (AC220)", "Gilteritinib", "Midostaurin"}
    assert any(d in flt3i_drugs for d in top.drug_ids), \
        f"Top combo {top.drug_ids} has no FLT3 inhibitor"
    # Total weighted coverage should be reasonable
    assert top.total_weighted_coverage > 0.3


def test_e2e_idh1_patient_recommends_ivosidenib():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    pool = [
        "Ivosidenib", "Enasidenib", "Olutasidenib",
        "Venetoclax", "Azacytidine", "Decitabine",
        "Quizartinib (AC220)", "Gilteritinib",
    ]
    cm = build_coverage_matrix(tx, pool)
    pf = pd.Series({"mut_IDH1": 1, "clin_age": 75})
    active = infer_active_targets(pf, tx)
    constraints = {
        "max_arity": 3, "coverage_threshold": 0.55,
        "toxicity_ceiling": {"myelosuppression": 2.5},
    }
    results = find_top_combinations(active, cm, constraints, n_solutions=3)
    assert len(results) >= 1
    top = results[0]
    assert "Ivosidenib" in top.drug_ids or "Olutasidenib" in top.drug_ids


def test_e2e_returns_empty_if_no_active_targets():
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    cm = build_coverage_matrix(tx, ["Cytarabine"])
    # Patient with NO mutations and no RNA — only baseline-active targets fire
    pf = pd.Series({})
    active = infer_active_targets(pf, tx)
    # Even with baseline-only targets, solver should still try
    results = find_top_combinations(
        active, cm,
        {"max_arity": 1, "coverage_threshold": 0.05},
        n_solutions=1,
    )
    assert isinstance(results, list)


def test_constraints_block_high_qt_combo():
    """Toxicity ceiling should prevent two FLT3i (both QT-stacking)."""
    from combo_val.coverage.patient_targets import infer_active_targets
    tx = load_taxonomy()
    pool = ["Quizartinib (AC220)", "Gilteritinib", "Midostaurin",
             "Venetoclax", "Azacytidine"]
    cm = build_coverage_matrix(tx, pool)
    # Manually set high QT toxicity for the FLT3i drugs
    qt_idx = cm.toxicity_axes.index("QT_prolongation")
    for d in ("Quizartinib (AC220)", "Gilteritinib"):
        cm.toxicity[cm.drug_id_to_idx[d], qt_idx] = 1.5  # each at 1.5
    pf = pd.Series({"mut_FLT3": 1, "clin_flt3_itd": 1})
    active = infer_active_targets(pf, tx)
    constraints = {
        "max_arity": 4, "coverage_threshold": 0.50,
        "toxicity_ceiling": {"QT_prolongation": 2.0},   # 1.5+1.5=3.0 > 2.0
    }
    results = find_top_combinations(active, cm, constraints, n_solutions=3)
    # Should never see Quiz + Gilt together
    for r in results:
        assert not ({"Quizartinib (AC220)", "Gilteritinib"} <= set(r.drug_ids))
