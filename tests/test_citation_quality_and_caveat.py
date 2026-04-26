"""Tests for issues #10 (Layer-3 caveat prominence) and #11 (citation
quality validator)."""

from __future__ import annotations

import pytest

from combo_val.clinical.regimen_db import (
    REGIMEN_BY_ID,
    REGIMEN_DB,
    CitationQualityError,
    validate_first_line_citations,
)


# ---------------------------------------------------------------------------
# Issue #11 — Citation quality validator
# ---------------------------------------------------------------------------


def test_db_passes_citation_quality_audit_at_runtime():
    """All first-line regimens currently in the DB must pass the audit."""
    complaints = validate_first_line_citations(strict=False)
    assert complaints == [], (
        "First-line regimens with insufficient citation quality:\n"
        + "\n".join(f"  - {c}" for c in complaints)
    )


def test_validator_raises_strict():
    """When run in strict mode on a clean DB, it must NOT raise."""
    # Should not raise on the actual DB
    validate_first_line_citations(strict=True)


def test_validator_catches_missing_pmid():
    """Synthetic regimen lacking PMID for a first-line tier should be caught."""
    from combo_val.clinical.regimen_db import Regimen
    bad = Regimen(
        regimen_id="bogus_first_line",
        name="Fake first-line — abstract only",
        drugs=("X", "Y"),
        n_drugs=2,
        clinical_tier="first_line_intensive",
        trial_phase="Phase2",
        trial_n=21,  # too small AND no PMID
        outcome_cr_cri_rate=0.95,
        outcome_median_os_months=None,
        pmid=None,  # missing!
    )
    # Inject + validate
    REGIMEN_DB.append(bad)
    try:
        complaints = validate_first_line_citations(strict=False)
        assert any("missing PMID" in c for c in complaints)
        assert any("trial_n=21 < 50" in c for c in complaints)
        assert any("missing outcome_median_os_months" in c for c in complaints)
        with pytest.raises(CitationQualityError):
            validate_first_line_citations(strict=True)
    finally:
        REGIMEN_DB.pop()


def test_consensus_regimens_get_pmid_exemption():
    """7+3 (consensus) doesn't need PMID since it predates RCT-era; it
    needs a `notes` field instead."""
    seven_three = REGIMEN_BY_ID["7_3"]
    assert seven_three.trial_phase == "consensus"
    assert seven_three.notes  # must have non-empty notes
    # Validator should not flag it
    complaints = validate_first_line_citations(strict=False)
    assert not any("7_3 (" in c for c in complaints)


def test_apl_low_risk_has_real_os_data():
    """ATRA+ATO low-risk APL — median OS not reached in trial, but encoded
    as 120 months with note pointing to 5-yr update PMID 28732571."""
    apl = REGIMEN_BY_ID["atra_ato_low"]
    assert apl.outcome_median_os_months is not None
    assert apl.outcome_median_os_months > 0
    # Note should reference the 5-yr update
    assert apl.notes
    assert "5-yr" in apl.notes or "28732571" in apl.notes


def test_first_line_phase23_have_minimum_n():
    """Every Phase 2/3/FDA first-line regimen must have trial_n ≥ 50."""
    fail = []
    for r in REGIMEN_DB:
        if r.clinical_tier not in ("first_line_intensive", "first_line_unfit"):
            continue
        if r.trial_phase in ("Phase2", "Phase3", "FDA") and r.trial_n < 50:
            fail.append(f"{r.regimen_id}: n={r.trial_n}")
    assert not fail, f"Under-powered first-line: {fail}"


def test_quiz_ven_dec_triplet_is_NOT_first_line():
    """Per issue #1 + #11, the Quiz+Ven+Dec ASH 2024 abstract triplet
    must NOT be classified as first-line — it has no PMID + only n=21."""
    triplet = REGIMEN_BY_ID["quiz_ven_dec_triplet"]
    assert triplet.clinical_tier == "experimental_triplet"
    # Sanity check on the underlying issues
    assert triplet.trial_n < 50
    assert triplet.pmid is None  # abstract-only


# ---------------------------------------------------------------------------
# Issue #10 — Layer-3 caveat prominence
# ---------------------------------------------------------------------------


def test_layer3_prospective_banner_renders_above_section_7_1():
    """v0.5 Prospective Validation Phase banner must appear BEFORE §7.1
    so a clinician reading top-to-bottom hits the research-only framing
    before the predicted AUC numbers."""
    from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall
    from combo_val.clinical.patient_report import build_clinical_report_markdown

    kit = KitInput(
        patient_id="TEST-CAVEAT",
        mutations=[MutationCall(gene="NPM1")],
        karyotype_text="46,XX[20]",
        wbc=10.0, platelet=100.0, hemoglobin=10.0, ldh=300.0,
        age=50, sex="female", is_initial_diagnosis=True,
    )
    out = KitOutput(
        patient_id="TEST-CAVEAT",
        predicted_eln2017="Favorable",
        top_combinations=[{
            "rank": 1, "drug1": "Venetoclax", "drug2": "Azacitidine",
            "predicted_combo_auc": 88.2, "single_auc_d1": 92.0,
            "single_auc_d2": 105.3, "mech_score": 0.42,
            "both_mech_annotated": True, "clonal_coverage_score": 0.71,
            "layer3_backbone": "BaselineA-MLP",
        }],
        top_single_drugs=[],
        top_regimens=[],
        clonal_coverage={},
        dna_summary={"sample_qc": {"n_mutations_called": 1,
                                     "n_above_vaf_0_20": 1,
                                     "karyotype_parsed": True,
                                     "fusions_reported": 0}},
        rna_outlier={},
        driver_flags={"NPM1": True},
        fitness_flag="fit_for_intensive",
        cautions=[],
        confidence_notes=[],
        eln_2017={"category": "Favorable", "rationale": []},
        eln_2022={"category": "Favorable", "rationale": []},
    )
    md = build_clinical_report_markdown(kit, out)

    assert "前瞻性验证" in md or "Prospective Validation" in md
    assert "0.05" in md
    assert "第五节" in md or "Section 4" in md or "Layer 1" in md

    banner_pos = md.find("前瞻性验证")
    if banner_pos < 0:
        banner_pos = md.find("Prospective Validation")
    sec_71_pos = md.find("### 7.1 组合 AUC 预测")
    assert banner_pos > 0 and sec_71_pos > 0
    assert banner_pos < sec_71_pos, (
        f"Prospective banner must appear ABOVE §7.1. "
        f"banner_pos={banner_pos}, sec_71_pos={sec_71_pos}"
    )


def test_default_report_includes_layer3_with_prospective_banner():
    """v0.5 Prospective Validation Phase: Layer-3 IS visible by default
    but explicitly framed as research output requiring prospective
    outcome capture (not clinical decision)."""
    from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall
    from combo_val.clinical.patient_report import build_clinical_report_markdown

    kit = KitInput(patient_id="T", age=50, mutations=[MutationCall(gene="NPM1")])
    out = KitOutput(
        patient_id="T", predicted_eln2017="Favorable",
        top_combinations=[{
            "rank": 1, "drug1": "Venetoclax", "drug2": "Azacitidine",
            "predicted_combo_auc": 88.2, "single_auc_d1": 92.0,
            "single_auc_d2": 105.3, "mech_score": 0.42,
            "both_mech_annotated": True, "layer3_backbone": "MLP",
        }],
        top_single_drugs=[], top_regimens=[],
        clonal_coverage={}, dna_summary={}, rna_outlier={},
        driver_flags={}, fitness_flag="fit_for_intensive",
        cautions=[], confidence_notes=[],
        eln_2017={"category": "Favorable", "rationale": []},
        eln_2022={"category": "Favorable", "rationale": []},
    )
    md = build_clinical_report_markdown(kit, out)
    # Layer-3 heading IS present
    assert "七、Layer-3 ML 组合预测" in md
    # Prospective Validation banner explicitly flags this as research
    assert "Prospective Validation" in md or "前瞻性验证" in md
    # The Pearson r ≈ 0.05 caveat is still surfaced
    assert "0.05" in md
    # The clinical-decision pointer to §5 is preserved
    assert "第五节" in md or "Layer 1" in md


def test_prospective_phase_banner_uses_research_emoji():
    """v0.5: the banner uses 🔬 to flag research-only framing."""
    from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall
    from combo_val.clinical.patient_report import build_clinical_report_markdown

    kit = KitInput(patient_id="T", age=50, mutations=[MutationCall(gene="NPM1")])
    out = KitOutput(
        patient_id="T", predicted_eln2017="Favorable",
        top_combinations=[], top_single_drugs=[], top_regimens=[],
        clonal_coverage={}, dna_summary={}, rna_outlier={},
        driver_flags={}, fitness_flag="fit_for_intensive",
        cautions=[], confidence_notes=[],
        eln_2017={"category": "Favorable", "rationale": []},
        eln_2022={"category": "Favorable", "rationale": []},
    )
    md = build_clinical_report_markdown(kit, out)
    # Either the header banner or the §7 banner uses the research emoji
    assert "🔬" in md
    assert "RESEARCH ONLY" in md or "research" in md.lower() or "研究" in md
