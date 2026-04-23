"""Tests for path A clonal-coverage module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.combo.clonal_coverage import (
    CLONE_ARCHETYPES,
    build_drug_clone_coverage,
    build_patient_clone_matrix,
    combo_clone_coverage_bliss,
    diagnose,
    rank_top_combos_per_patient,
    score_combo_for_patient,
    score_named_regimens,
)


# ---------------------------------------------------------------------------
# Bliss aggregation math
# ---------------------------------------------------------------------------


def test_bliss_single_full_coverage():
    """Single drug fully covers → combo covers fully."""
    cov = np.array([[1.0, 0.0, 0.5]])   # 1 drug × 3 clones
    out = combo_clone_coverage_bliss(cov)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.5])


def test_bliss_two_partial_drugs():
    """Two drugs each 0.5 → combo ≈ 0.75."""
    cov = np.array([[0.5, 0.0], [0.5, 0.0]])
    out = combo_clone_coverage_bliss(cov)
    np.testing.assert_allclose(out, [0.75, 0.0])


def test_bliss_no_coverage():
    cov = np.zeros((3, 4))
    np.testing.assert_allclose(combo_clone_coverage_bliss(cov), [0.0, 0.0, 0.0, 0.0])


def test_bliss_commutative():
    """Order of drugs shouldn't matter."""
    cov1 = np.array([[1.0, 0.0, 0.3], [0.0, 0.8, 0.5]])
    cov2 = np.array([[0.0, 0.8, 0.5], [1.0, 0.0, 0.3]])
    np.testing.assert_allclose(
        combo_clone_coverage_bliss(cov1),
        combo_clone_coverage_bliss(cov2),
    )


# ---------------------------------------------------------------------------
# Patient clone decomposition
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_patients():
    # 4 patients with diverse mutation profiles
    return pd.DataFrame(
        {
            "patient_id": ["P_flt3", "P_idh1_npm1", "P_tp53", "P_wild"],
            "mut_FLT3":   [1, 0, 0, 0],
            "mut_NPM1":   [0, 1, 0, 0],
            "mut_IDH1":   [0, 1, 0, 0],
            "mut_IDH2":   [0, 0, 0, 0],
            "mut_TP53":   [0, 0, 1, 0],
            "mut_NRAS":   [0, 0, 0, 0],
            "mut_KRAS":   [0, 0, 0, 0],
            "mut_PTPN11": [0, 0, 0, 0],
            "mut_KMT2A":  [0, 0, 0, 0],
        }
    ).set_index("patient_id")


def test_patient_clone_matrix_mutation_driven(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    # FLT3 patient → FLT3_clone = 1.0
    assert pc.loc["P_flt3", "FLT3_clone"] == 1.0
    assert pc.loc["P_wild", "FLT3_clone"] == 0.0
    # IDH1+NPM1 patient → IDH1_clone + MENIN_HOX_clone = 1.0 each
    assert pc.loc["P_idh1_npm1", "IDH1_clone"] == 1.0
    assert pc.loc["P_idh1_npm1", "MENIN_HOX_clone"] == 1.0
    # TP53 patient → TP53_clone = 1.0
    assert pc.loc["P_tp53", "TP53_clone"] == 1.0


def test_patient_clone_matrix_default_clones(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    # Default clones present for everyone at their weight
    assert (pc["BCL2_dependent_clone"] == 0.5).all()
    assert (pc["proliferative_clone"] == 0.5).all()
    assert (pc["LSC_compartment"] == 0.3).all()


def test_wild_type_patient_only_default_clones(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    wild = pc.loc["P_wild"]
    driver_clones = [c for c, spec in CLONE_ARCHETYPES.items()
                     if "DEFAULT" not in spec["presence_markers"]]
    for c in driver_clones:
        assert wild[c] == 0.0, f"wild-type patient should have 0 in {c}"


# ---------------------------------------------------------------------------
# Drug-clone coverage
# ---------------------------------------------------------------------------


def test_drug_clone_coverage_flt3_drugs():
    # Known: Quizartinib covers FLT3_clone at 1.0
    cov = build_drug_clone_coverage(
        ["Quizartinib (AC220)", "Venetoclax", "SomeUnknownDrug"]
    )
    assert cov.loc["Quizartinib (AC220)", "FLT3_clone"] == 1.0
    assert cov.loc["Venetoclax", "BCL2_dependent_clone"] == 1.0
    # Unknown drug → all zeros
    assert (cov.loc["SomeUnknownDrug"] == 0.0).all()


def test_drug_clone_coverage_tp53_uncoverable():
    """No drug in our panel should cover TP53_clone (by design)."""
    cov = build_drug_clone_coverage(
        ["Quizartinib (AC220)", "Venetoclax", "Cytarabine",
         "Azacytidine", "Enasidenib", "Ivosidenib"]
    )
    assert (cov["TP53_clone"] == 0.0).all(), (
        "TP53_clone should be uncoverable by current drug panel"
    )


# ---------------------------------------------------------------------------
# End-to-end scoring
# ---------------------------------------------------------------------------


def test_score_combo_flt3_patient_prefers_flt3i_plus_venetoclax(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    cov = build_drug_clone_coverage(
        ["Venetoclax", "Quizartinib (AC220)", "Imatinib"]
    )
    # FLT3 patient
    patient_vec = pc.loc["P_flt3"].to_numpy()

    ven_quiz = cov.loc[["Venetoclax", "Quizartinib (AC220)"]].to_numpy()
    ven_imatinib = cov.loc[["Venetoclax", "Imatinib"]].to_numpy()

    s_ven_quiz, _ = score_combo_for_patient(patient_vec, ven_quiz)
    s_ven_imatinib, _ = score_combo_for_patient(patient_vec, ven_imatinib)
    assert s_ven_quiz > s_ven_imatinib, (
        "FLT3 patient must prefer Ven+Quiz over Ven+Imatinib"
    )


def test_rank_top_combos_flt3_patient_puts_flt3i_first(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients.iloc[[0]])   # just FLT3 patient
    drugs = [
        "Venetoclax", "Azacytidine", "Quizartinib (AC220)",
        "Gilteritinib", "Imatinib", "Nilotinib",
    ]
    cov = build_drug_clone_coverage(drugs)
    top = rank_top_combos_per_patient(pc, cov, arity=2, top_k=3)
    # Top pick for FLT3-mut should include an FLT3i (Quiz or Gilt)
    top1 = top.iloc[0]
    flt3i_names = {"Quizartinib (AC220)", "Gilteritinib"}
    combo_drugs = {top1["drug1"], top1["drug2"]}
    assert combo_drugs & flt3i_names, (
        f"top combo for FLT3 patient must include FLT3i; got {combo_drugs}"
    )


def test_arity_scaling_monotonic_for_complex_patient(synthetic_patients):
    """More drugs should never DECREASE best-achievable coverage."""
    pc = build_patient_clone_matrix(synthetic_patients.loc[["P_idh1_npm1"]])
    drugs = [
        "Venetoclax", "Azacytidine", "Ivosidenib", "Cytarabine",
        "Enasidenib", "Midostaurin", "Gilteritinib",
    ]
    cov = build_drug_clone_coverage(drugs)

    top_by_arity = {}
    for ar in [1, 2, 3, 4]:
        top = rank_top_combos_per_patient(pc, cov, arity=ar, top_k=1)
        top_by_arity[ar] = float(top.iloc[0]["score"])

    # Best score must be monotonically non-decreasing
    assert top_by_arity[1] <= top_by_arity[2] <= top_by_arity[3] <= top_by_arity[4]


def test_diagnose_reports_correct_counts(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    cov = build_drug_clone_coverage(
        ["Venetoclax", "SomeUnknownDrug", "Quizartinib (AC220)"]
    )
    diag = diagnose(pc, cov)
    assert diag.n_patients == 4
    assert diag.n_drugs == 3
    assert diag.drugs_annotated == 2   # Ven + Quiz
    assert diag.drugs_unannotated == 1


def test_named_regimens_scoring(synthetic_patients):
    pc = build_patient_clone_matrix(synthetic_patients)
    cov = build_drug_clone_coverage(
        ["Venetoclax", "Azacytidine", "Gilteritinib", "Ivosidenib"]
    )
    regimens = {
        "VenAzaGilt": ["Venetoclax", "Azacytidine", "Gilteritinib"],
        "VenAzaIvo":  ["Venetoclax", "Azacytidine", "Ivosidenib"],
    }
    scores = score_named_regimens(pc, cov, regimens)

    # FLT3 patient should score higher on VenAzaGilt (has Gilt) than on VenAzaIvo
    assert scores.loc["P_flt3", "VenAzaGilt"] > scores.loc["P_flt3", "VenAzaIvo"]
    # IDH1+NPM1 patient should score higher on VenAzaIvo (has Ivo) than on VenAzaGilt
    assert scores.loc["P_idh1_npm1", "VenAzaIvo"] > scores.loc["P_idh1_npm1", "VenAzaGilt"]


def test_tp53_patient_coverage_saturates_at_non_tp53_clones(synthetic_patients):
    """TP53-mut patient: even best combo cannot achieve score 1.0 (TP53_clone always uncovered)."""
    pc = build_patient_clone_matrix(synthetic_patients.loc[["P_tp53"]])
    drugs = list(["Venetoclax", "Azacytidine", "Gilteritinib", "Ivosidenib",
                 "Cytarabine", "Enasidenib"])
    cov = build_drug_clone_coverage(drugs)
    # Use ALL 6 drugs
    scores, _ = score_combo_for_patient(
        pc.iloc[0].to_numpy(),
        cov.to_numpy(),
    )
    # TP53_clone weight 1.0 uncovered → score capped below 1.0.
    # Expected ceiling: total weight = 1.0 (TP53) + 0.5 + 0.5 + 0.3 = 2.3
    # Covered weight = 0 (TP53) + 0.5 + 0.5 + 0.3 = 1.3 → 1.3/2.3 ≈ 0.565
    assert scores < 0.7, f"TP53 patient score should not saturate ; got {scores}"
