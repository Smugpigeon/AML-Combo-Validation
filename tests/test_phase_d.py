"""Tests for Phase D — patient-side modeling realism.

  D.1 — RNA expression program scoring
  D.2 — Mutation cooperativity rules
  D.3 — Clinical feature weighting (constraint adjustment, applied in
         predict_for_patient — tested via integration)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.coverage.cooperativity import (
    COOPERATIVITY_RULES, apply_cooperativity_rules,
)
from combo_val.coverage.patient_targets import infer_active_targets
from combo_val.coverage.rna_programs import (
    PROGRAM_SIGNATURES, program_to_axis_modulation, score_programs,
)
from combo_val.coverage.taxonomy import load_taxonomy


# ---------------------------------------------------------------------------
# D.1 — RNA program scoring
# ---------------------------------------------------------------------------


def test_program_signatures_have_required_fields():
    for name, spec in PROGRAM_SIGNATURES.items():
        assert "up_genes" in spec or "down_genes" in spec
        assert "axis_modulated" in spec
        assert isinstance(spec.get("up_genes", []), list)


def test_score_empty_rna_returns_empty():
    out = score_programs(pd.Series([], dtype=float))
    assert out == {}


def test_score_high_bcl2_expression_drives_high_score():
    # Create a synthetic RNA Series where BCL2/BCL2L1/BCL2L11 are at top
    # of distribution. All other genes have random expression.
    rng = np.random.default_rng(0)
    genes = ["BCL2", "BCL2L1", "BCL2L11", "MCL1"] + \
            [f"GENE_{i}" for i in range(100)]
    expr = rng.lognormal(mean=4.0, sigma=1.2, size=len(genes))
    expr_s = pd.Series(expr, index=genes)
    # Force BCL2 family to top
    for g in ("BCL2", "BCL2L1", "BCL2L11"):
        expr_s[g] = expr_s.max() * 1.5
    scores = score_programs(expr_s)
    assert "BCL2_high" in scores
    assert scores["BCL2_high"] > 0.8


def test_score_skips_program_when_too_few_genes_present():
    """If fewer than 50% of program genes are in the RNA panel, skip."""
    expr = pd.Series([5.0, 6.0, 7.0], index=["GENE_X", "GENE_Y", "GENE_Z"])
    scores = score_programs(expr)
    # No program should score because none of the marker genes are present
    assert scores == {}


def test_program_to_axis_modulation_maps_correctly():
    scores = {"BCL2_high": 1.0, "MCL1_high": 0.5}
    mod = program_to_axis_modulation(scores)
    # BCL2_high score 1.0 → multiplier 1.5 on BCL2 axis
    assert "BCL2" in mod
    assert mod["BCL2"] == pytest.approx(1.5, rel=0.01)
    # MCL1_high score 0.5 → multiplier 1.0 (no change) on MCL1 axis
    assert "MCL1" in mod
    assert mod["MCL1"] == pytest.approx(1.0, rel=0.01)


def test_rna_modulation_integrated_with_infer_active_targets():
    """Pre-step: high BCL2 program → BCL2 axis weight should be higher
    than baseline-only weight."""
    tx = load_taxonomy()
    pf = pd.Series({})    # empty mut → only baseline-active axes fire
    baseline = infer_active_targets(pf, tx)
    boosted = infer_active_targets(pf, tx, rna_programs={"BCL2_high": 0.95})
    assert "BCL2" in baseline
    assert "BCL2" in boosted
    assert boosted["BCL2"] > baseline["BCL2"]


# ---------------------------------------------------------------------------
# D.2 — Mutation cooperativity
# ---------------------------------------------------------------------------


def test_cooperativity_rules_have_required_fields():
    for r in COOPERATIVITY_RULES:
        assert r.name
        assert r.description
        assert r.citation
        assert r.predicate
        assert r.modify
        assert r.rationale


def test_npm1_dnmt3a_boosts_menin_axis():
    """NPM1 + DNMT3A co-mut → MENIN_NPM1 axis weight ×1.25."""
    pf = {"mut_NPM1": 1, "mut_DNMT3A": 1}
    active = {"MENIN_NPM1": 0.6, "FLT3": 0.0, "BCL2": 0.4}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("NPM1_DNMT3A" in n for n in fired)
    assert new["MENIN_NPM1"] == pytest.approx(0.75, rel=0.01)   # 0.6 × 1.25


def test_tp53_multihit_suppresses_bcl2_axis():
    """TP53 multi-hit → BCL2 axis weight ×0.30 (Ven less effective)."""
    pf = {"mut_TP53": 1, "tp53_multihit": True}
    active = {"BCL2": 0.7, "TP53_multihit": 0.9}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("TP53" in n for n in fired)
    assert new["BCL2"] == pytest.approx(0.21, rel=0.01)   # 0.7 × 0.30


def test_flt3_itd_high_ar_activates_secondary_axes():
    pf = {"mut_FLT3": 1, "clin_flt3_itd": 1, "clin_flt3_allelic_ratio": 0.6}
    active = {"FLT3": 1.0}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("FLT3_ITD_high_AR" in n for n in fired)
    assert new.get("JAK_STAT", 0) >= 0.4
    assert new.get("MCL1", 0) >= 0.35


def test_flt3_itd_low_ar_does_NOT_activate_secondaries():
    pf = {"mut_FLT3": 1, "clin_flt3_itd": 1, "clin_flt3_allelic_ratio": 0.3}
    active = {"FLT3": 1.0}
    new, fired = apply_cooperativity_rules(pf, active)
    assert not any("FLT3_ITD_high_AR" in n for n in fired)
    assert "JAK_STAT" not in new


def test_cbf_aml_with_kit_priority():
    pf = {"mut_KIT": 1, "fusion_RUNX1_RUNX1T1": 1}
    active = {"KIT_RTK": 0.7}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("CBF_AML_KIT" in n for n in fired)
    assert new["KIT_RTK"] == 1.0


def test_apl_dominates_other_axes():
    pf = {"fusion_PML_RARA": 1}
    active = {"differentiation_block": 0.5, "BCL2": 0.7, "DNMT": 0.5}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("APL" in n for n in fired)
    assert new["differentiation_block"] == 1.0
    assert new["BCL2"] < 0.7    # suppressed
    assert new["DNMT"] < 0.5    # suppressed


def test_prior_hma_failure_suppresses_dnmt():
    pf = {"clin_prior_mds": 1}
    active = {"DNMT": 0.6}
    new, fired = apply_cooperativity_rules(pf, active)
    assert any("prior_HMA" in n for n in fired)
    assert new["DNMT"] == pytest.approx(0.24, rel=0.01)   # 0.6 × 0.4


def test_npm1_flt3_itd_dnmt3a_triplet_strongest_menin():
    """Triple-driver NPM1+FLT3-ITD+DNMT3A → +30% MENIN_NPM1 axis."""
    pf = {
        "mut_NPM1": 1, "mut_FLT3": 1, "mut_DNMT3A": 1,
        "clin_flt3_itd": 1, "clin_flt3_allelic_ratio": 0.6,
    }
    active = {"MENIN_NPM1": 0.6, "FLT3": 1.0}
    new, fired = apply_cooperativity_rules(pf, active)
    # Both NPM1+DNMT3A AND triplet rules should fire
    rule_names_fired = [n.split(":")[0] for n in fired]
    assert "NPM1_DNMT3A_MENIN_strong" in rule_names_fired
    assert "NPM1_FLT3_DNMT3A_triplet_MENIN" in rule_names_fired
    # MENIN_NPM1 should be doubly boosted: 0.6 × 1.25 × 1.30 ≈ 0.975
    assert new["MENIN_NPM1"] >= 0.95


def test_no_rules_fire_for_simple_cytogenetically_normal_patient():
    pf = {"mut_NPM1": 1}    # NPM1-mut alone
    active = {"MENIN_NPM1": 0.6, "BCL2": 0.5, "DNMT": 0.5}
    _, fired = apply_cooperativity_rules(pf, active)
    # None of the co-mut rules should fire
    assert fired == []


def test_cooperativity_safe_on_missing_features():
    pf = {}    # truly empty patient
    active = {"FLT3": 0.5}
    new, fired = apply_cooperativity_rules(pf, active)
    # No rules fire, no crash
    assert active == {"FLT3": 0.5}    # original unchanged
    assert new == active


# ---------------------------------------------------------------------------
# D — End-to-end via predict_for_patient (integration)
# ---------------------------------------------------------------------------


def test_d_integration_npm1_dnmt3a_patient_gets_menin_emphasis():
    """Phase D end-to-end: NPM1+DNMT3A patient → MENIN axis weight ↑
    in the multi_target_coverage output."""
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    from combo_val.clinical.kit_predict import _compute_multi_target_coverage

    kit = KitInput(
        patient_id="TEST",
        mutations=[
            MutationCall(gene="NPM1"),
            MutationCall(gene="DNMT3A"),
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6),
        ],
        karyotype_text="46,XX[20]",
        wbc=80.0, age=45, sex="female", is_initial_diagnosis=True,
    )
    driver_flags = {"FLT3_ITD": True, "NPM1": True, "DNMT3A": True}
    out = _compute_multi_target_coverage(kit, driver_flags)
    fired = out.get("cooperativity_rules_fired", [])
    rule_names = [n.split(":")[0] for n in fired]
    # Should fire at least the NPM1+DNMT3A rule and the triplet rule
    assert any("NPM1_DNMT3A_MENIN_strong" in r for r in rule_names)
    assert any("FLT3_ITD_high_AR_secondary" in r for r in rule_names)


def test_d3_age_80_caps_arity_at_2():
    """Phase D.3: age ≥ 80 → max_arity = 2."""
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    from combo_val.clinical.kit_predict import _compute_multi_target_coverage

    kit = KitInput(
        patient_id="TEST", age=82,
        mutations=[MutationCall(gene="NPM1")],
        is_initial_diagnosis=True,
    )
    out = _compute_multi_target_coverage(kit, {"NPM1": True})
    constraints = out.get("constraints", {})
    assert constraints.get("max_arity", 999) <= 2
    assert any("Very elderly" in adj or "≥80" in adj
                for adj in out.get("clinical_adjustments", []))
