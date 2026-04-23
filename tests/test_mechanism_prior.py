"""Tests for the simplified mechanism-prior module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.combo.mechanism_prior import (
    BEATAML_TO_MECH_ID,
    MechPriorConfig,
    compute_combo_mech_scores,
    diagnostics,
    load_drug_mechanism_matrix,
    patient_target_deficit,
)


@pytest.fixture
def synthetic_patients():
    """5 synthetic patients with different mutation profiles."""
    df = pd.DataFrame(
        {
            "patient_id": ["P_flt3", "P_idh2", "P_flt3_npm1", "P_wild", "P_idh1"],
            "mut_FLT3":   [1, 0, 1, 0, 0],
            "mut_NPM1":   [0, 0, 1, 0, 0],
            "mut_IDH1":   [0, 0, 0, 0, 1],
            "mut_IDH2":   [0, 1, 0, 0, 0],
            "mut_TP53":   [0, 0, 0, 1, 0],
            "mut_DNMT3A": [1, 0, 0, 0, 0],
        }
    ).set_index("patient_id")
    return df


def test_load_drug_mechanism_matrix():
    m = load_drug_mechanism_matrix()
    assert len(m) >= 20   # v1 had 20, v2 expanded to 32; allow future growth
    assert "venetoclax" in m.index
    assert "quizartinib" in m.index
    assert "trametinib" in m.index       # v2 addition
    assert "ruxolitinib" in m.index      # v2 addition
    assert m.loc["venetoclax", "tgt_BCL2"] == 1.0
    assert m.loc["quizartinib", "tgt_FLT3"] == 1.0
    assert m.loc["trametinib", "tgt_RAS_MAPK"] == 1.0
    assert m.loc["ruxolitinib", "tgt_JAK_STAT"] == 1.0


def test_patient_target_deficit(synthetic_patients):
    deficit = patient_target_deficit(synthetic_patients)
    # FLT3 patient should have tgt_FLT3 == 1
    assert deficit.loc["P_flt3", "tgt_FLT3"] == 1.0
    # NPM1 patient (P_flt3_npm1) should have tgt_MENIN_HOX == 1 (NPM1 → MENIN)
    assert deficit.loc["P_flt3_npm1", "tgt_MENIN_HOX"] == 1.0
    # IDH1 → tgt_IDH1; IDH2 → tgt_IDH2
    assert deficit.loc["P_idh1", "tgt_IDH1"] == 1.0
    assert deficit.loc["P_idh2", "tgt_IDH2"] == 1.0
    # v2: P_wild has mut_TP53=1 so gets tgt_TP53_PATHWAY=1 (expanded mapping).
    # The synthetic fixture was written before the v2 expansion; the key
    # invariant is that P_wild has NO FLT3/IDH1/IDH2/MENIN deficits.
    assert deficit.loc["P_wild", "tgt_FLT3"] == 0.0
    assert deficit.loc["P_wild", "tgt_IDH1"] == 0.0
    assert deficit.loc["P_wild", "tgt_IDH2"] == 0.0
    assert deficit.loc["P_wild", "tgt_MENIN_HOX"] == 0.0


def test_compute_combo_mech_scores_shape(synthetic_patients):
    drug_ids = ["Venetoclax", "Quizartinib (AC220)", "Gilteritinib", "SomeUnknownDrug"]
    scores = compute_combo_mech_scores(synthetic_patients, drug_ids)
    assert scores.shape == (5, 4, 4)
    # Symmetric
    for p in range(5):
        assert np.allclose(scores[p], scores[p].T)


def test_flt3_patient_rewards_quizartinib_venetoclax(synthetic_patients):
    """FLT3 patient's mech score for (quizartinib, venetoclax) should be high."""
    drug_ids = ["Venetoclax", "Quizartinib (AC220)", "Gilteritinib", "SomeUnknownDrug"]
    scores = compute_combo_mech_scores(synthetic_patients, drug_ids)

    p_flt3 = 0  # row 0 = P_flt3
    ven_i = drug_ids.index("Venetoclax")
    quiz_i = drug_ids.index("Quizartinib (AC220)")
    unk_i = drug_ids.index("SomeUnknownDrug")

    # Quizartinib + Venetoclax should have high mech score for FLT3 patient
    # (FLT3 target covered by quizartinib)
    assert scores[p_flt3, ven_i, quiz_i] >= 1.0
    # Same patient + SomeUnknownDrug should have lower score
    assert scores[p_flt3, unk_i, unk_i] <= scores[p_flt3, ven_i, quiz_i]


def test_targetable_driverless_patient_scores_only_tiebreaker(synthetic_patients):
    """A patient with no FLT3/IDH/NPM1/KMT2A driver and no RAS/TP53/spliceosome
    mutation would have near-zero target coverage. Our fixture's P_flt3_npm1
    row has mut_FLT3=1 + mut_NPM1=1 — use a different patient."""
    # Build a minimal no-driver patient on the fly
    drugs = ["Venetoclax", "Quizartinib (AC220)", "Ivosidenib"]
    empty = synthetic_patients.loc[["P_flt3"]].copy() * 0
    empty.index = ["P_empty"]
    scores = compute_combo_mech_scores(
        empty, drugs, cfg=MechPriorConfig(toxicity_penalty_scale=0.0),
    )
    # All pair scores should come only from tiebreakers (annot + diversity)
    # bounded by ~0.05 with current weights.
    for i in range(3):
        for j in range(3):
            assert scores[0, i, j] <= 0.12


def test_diagnostics(synthetic_patients):
    drug_ids = ["Venetoclax", "Quizartinib (AC220)", "SomeUnknownDrug"]
    diag = diagnostics(synthetic_patients, drug_ids)
    assert diag["n_patients"] == 5
    # v2: P_wild has mut_TP53=1 so now counts as having a deficit (TP53_PATHWAY).
    # All 5 synthetic patients have at least one driver now.
    assert diag["n_patients_with_any_deficit"] == 5
    assert diag["n_drugs_with_mech_annotation"] == 2  # Ven + Quiz; not Unknown
    assert diag["target_axis_coverage"]["tgt_FLT3"] == 2   # P_flt3 + P_flt3_npm1
    assert diag["target_axis_coverage"]["tgt_MENIN_HOX"] == 1  # P_flt3_npm1 (NPM1)
    assert diag["target_axis_coverage"]["tgt_TP53_PATHWAY"] == 1  # P_wild (TP53)


def test_beataml_to_mech_id_keys_are_real_beataml_names():
    """Each key in BEATAML_TO_MECH_ID must look like a real BeatAML drug name."""
    # Basic sanity check: non-empty, properly cased
    for beat_id, mech_id in BEATAML_TO_MECH_ID.items():
        assert beat_id != ""
        assert mech_id != ""
        # BeatAML drugs are typically proper-case English names with possible parens
        assert beat_id[0].isupper() or beat_id.startswith(("17-", "5-"))


def test_mech_score_prefers_matching_drugs_over_unmatched(synthetic_patients):
    """IDH2 patient should score (Enasidenib, Venetoclax) > (Venetoclax alone pair)."""
    # Note: BEATAML_TO_MECH_ID maps BeatAML drug names → mech matrix IDs.
    # Enasidenib is in the 20-drug vocab.
    drug_ids = ["Enasidenib", "Venetoclax", "Cytarabine"]
    scores = compute_combo_mech_scores(synthetic_patients, drug_ids)
    p_idh2 = 1  # P_idh2
    ena_i = drug_ids.index("Enasidenib")
    ven_i = drug_ids.index("Venetoclax")
    cyt_i = drug_ids.index("Cytarabine")

    # For IDH2 patient, any pair containing Enasidenib covers tgt_IDH2
    score_ena_ven = scores[p_idh2, ena_i, ven_i]
    score_ven_cyt = scores[p_idh2, ven_i, cyt_i]
    assert score_ena_ven > score_ven_cyt
