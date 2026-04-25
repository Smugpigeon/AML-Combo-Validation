"""Tests for issues #5 (hyperleukocytic warning), #6 (baseline workup
checklist), #7 (MRD monitoring plan)."""

from __future__ import annotations

import pytest

from combo_val.clinical.kit_predict import _check_kit_cautions
from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall
from combo_val.clinical.patient_report import (
    _baseline_workup_section,
    _mrd_monitoring_section,
)


def _empty_kit_output(eln_2022: str = "Intermediate", fitness="fit_for_intensive",
                       driver_flags=None, top_regimens=None) -> KitOutput:
    return KitOutput(
        patient_id="TEST",
        predicted_eln2017="Intermediate",
        top_combinations=[],
        top_single_drugs=[],
        top_regimens=top_regimens or [],
        clonal_coverage={},
        dna_summary={},
        rna_outlier={},
        driver_flags=driver_flags or {},
        fitness_flag=fitness,
        cautions=[],
        confidence_notes=[],
        eln_2017={"category": "Intermediate", "rationale": []},
        eln_2022={"category": eln_2022, "rationale": []},
    )


# ---------------------------------------------------------------------------
# Issue #5 — Hyperleukocytic / leukostasis warning
# ---------------------------------------------------------------------------


def test_hyperleukocytic_emergency_at_wbc_above_100():
    """WBC > 100 → emergency caution."""
    kit = KitInput(patient_id="T", wbc=120.0)
    cautions = _check_kit_cautions(kit, driver_flags={})
    assert any("EMERGENCY" in c or "emergency" in c.lower()
                for c in cautions), \
        f"Expected hyperleukocytic emergency caution; got {cautions}"


def test_hyperleukocytic_warning_at_wbc_above_50():
    """WBC > 50 (but ≤ 100) → standard hyperleukocytic warning."""
    kit = KitInput(patient_id="T", wbc=75.0)
    cautions = _check_kit_cautions(kit, driver_flags={})
    relevant = [c for c in cautions if "Hyperleukocytic" in c or "hyperleukocytic" in c]
    assert relevant, f"Expected hyperleukocytic warning; got {cautions}"
    # Should mention hydroxyurea
    assert any("hydroxyurea" in c.lower() or "hu" in c.lower() for c in relevant)


def test_normal_wbc_no_hyperleuko_caution():
    """WBC ≤ 50 → no hyperleukocytic caution."""
    kit = KitInput(patient_id="T", wbc=30.0)
    cautions = _check_kit_cautions(kit, driver_flags={})
    assert not any("Hyperleukocytic" in c or "hyperleukocytic" in c
                    for c in cautions)


def test_wbc_unspecified_no_hyperleuko_caution():
    """No WBC → no caution (don't fabricate one)."""
    kit = KitInput(patient_id="T", wbc=None)
    cautions = _check_kit_cautions(kit, driver_flags={})
    assert not any("Hyperleukocytic" in c for c in cautions)


# ---------------------------------------------------------------------------
# Issue #6 — Baseline workup checklist
# ---------------------------------------------------------------------------


def test_baseline_workup_universal_items():
    """Universal items (CBC, CMP, ECG, hepatitis screen, etc.) for all patients."""
    kit = KitInput(patient_id="T", age=60)
    out = _empty_kit_output()
    md = _baseline_workup_section(kit, out)
    for item in ("ECG", "HBV", "HCV", "HIV", "CMP", "Coagulation"):
        assert item in md, f"Universal workup item missing: {item}"


def test_baseline_workup_anthracycline_eligible_gets_echo():
    """fit + non-APL → ECHO line shows."""
    kit = KitInput(patient_id="T", age=45)
    out = _empty_kit_output(fitness="fit_for_intensive")
    md = _baseline_workup_section(kit, out)
    assert "ECHO" in md
    assert "EF" in md or "LVEF" in md


def test_baseline_workup_unfit_no_echo_section():
    """Unfit patient — anthracycline unlikely; no ECHO subsection."""
    kit = KitInput(patient_id="T", age=80)
    out = _empty_kit_output(fitness="unfit")
    md = _baseline_workup_section(kit, out)
    # The Anthracycline subsection header should NOT appear
    assert "强化诱导" not in md or "ECHO" not in md.split("强化诱导")[1].split("FLT3 抑制剂")[0] \
        if "强化诱导" in md else True


def test_baseline_workup_flt3i_section_when_flt3_itd():
    """FLT3-ITD patient gets FLT3i-specific subsection (QTc, electrolytes)."""
    kit = KitInput(patient_id="T", age=45)
    out = _empty_kit_output(driver_flags={"FLT3_ITD": True})
    md = _baseline_workup_section(kit, out)
    assert "QTc" in md
    assert "FLT3" in md


def test_baseline_workup_ven_section_when_ven_in_top_regimens():
    """If a top-3 regimen contains Venetoclax → TLS labs section."""
    kit = KitInput(patient_id="T", age=70)
    top_regs = [{"drugs": ["Venetoclax", "Azacitidine"], "regimen_id": "ven_aza"}]
    out = _empty_kit_output(top_regimens=top_regs)
    md = _baseline_workup_section(kit, out)
    assert "Venetoclax" in md
    assert "TLS" in md
    assert "uric acid" in md.lower() or "尿酸" in md


def test_baseline_workup_lp_for_high_wbc():
    """Hyperleukocytic (WBC > 50) → LP + IT MTX subsection."""
    kit = KitInput(patient_id="T", age=45, wbc=80.0)
    out = _empty_kit_output()
    md = _baseline_workup_section(kit, out)
    assert "Lumbar puncture" in md or "LP" in md
    assert "methotrexate" in md.lower() or "MTX" in md


def test_baseline_workup_hla_for_intermediate_fit():
    """ELN 2022 ≥ Intermediate AND fit → HLA + transplant referral subsection."""
    kit = KitInput(patient_id="T", age=45)
    out = _empty_kit_output(eln_2022="Intermediate", fitness="fit_for_intensive")
    md = _baseline_workup_section(kit, out)
    assert "HLA" in md
    assert "transplant" in md.lower() or "移植" in md


def test_baseline_workup_no_hla_for_favorable():
    """ELN 2022 = Favorable → no HLA urgency subsection."""
    kit = KitInput(patient_id="T", age=45)
    out = _empty_kit_output(eln_2022="Favorable", fitness="fit_for_intensive")
    md = _baseline_workup_section(kit, out)
    # The HLA *prep subsection header* should not fire (universal workup
    # may still mention HLA elsewhere, but not the dedicated section).
    assert "HLA + transplant prep" not in md


def test_baseline_workup_fertility_for_young_patient():
    """Age ≤ 50 → fertility consult subsection."""
    kit = KitInput(patient_id="T", age=35)
    out = _empty_kit_output()
    md = _baseline_workup_section(kit, out)
    assert "fertility" in md.lower() or "生育" in md


def test_baseline_workup_no_fertility_for_elderly():
    kit = KitInput(patient_id="T", age=72)
    out = _empty_kit_output()
    md = _baseline_workup_section(kit, out)
    assert "生育" not in md and "fertility" not in md.lower()


# ---------------------------------------------------------------------------
# Issue #7 — MRD monitoring plan
# ---------------------------------------------------------------------------


def test_mrd_npm1mut_uses_npm1_pcr():
    """NPM1mut patient → NPM1 RT-qPCR as primary modality."""
    kit = KitInput(
        patient_id="T",
        mutations=[MutationCall(gene="NPM1")],
        karyotype_text="46,XX[20]",
    )
    out = _empty_kit_output(driver_flags={"NPM1": True})
    md = _mrd_monitoring_section(kit, out)
    assert "NPM1 RT-qPCR" in md or "NPM1mut" in md
    assert "ELN 2021" in md
    assert "33591443" in md  # PMID of consensus paper


def test_mrd_cbf_aml_uses_fusion_pcr():
    """CBF-AML (t(8;21)) → fusion transcript RT-qPCR."""
    kit = KitInput(
        patient_id="T",
        karyotype_text="46,XX,t(8;21)(q22;q22)[20]",
        fusions=["RUNX1-RUNX1T1"],
    )
    out = _empty_kit_output()
    md = _mrd_monitoring_section(kit, out)
    assert "RUNX1-RUNX1T1" in md or "RUNX1_RUNX1T1" in md
    assert "RT-qPCR" in md


def test_mrd_cbf_inv16():
    """inv(16) CBFB-MYH11 → CBFB-MYH11 fusion PCR."""
    kit = KitInput(
        patient_id="T",
        karyotype_text="46,XY,inv(16)(p13.1q22)[20]",
        fusions=["CBFB-MYH11"],
    )
    out = _empty_kit_output()
    md = _mrd_monitoring_section(kit, out)
    assert "CBFB-MYH11" in md or "CBFB_MYH11" in md


def test_mrd_apl_uses_pml_rara():
    """APL → PML-RARA RT-qPCR (distinct from other AML)."""
    kit = KitInput(
        patient_id="T",
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        fusions=["PML-RARA"],
    )
    out = _empty_kit_output()
    md = _mrd_monitoring_section(kit, out)
    assert "PML-RARA" in md or "APL" in md


def test_mrd_default_falls_back_to_flow():
    """No NPM1 / CBF / APL → multiparametric flow cytometry as fallback."""
    kit = KitInput(
        patient_id="T",
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="ASXL1")],
    )
    out = _empty_kit_output(driver_flags={})
    md = _mrd_monitoring_section(kit, out)
    assert "flow" in md.lower() or "MFC" in md or "流式" in md


def test_mrd_flt3_itd_adds_ngs_subsection():
    """FLT3-ITD patient gets an additional NGS-MRD section regardless of
    primary modality (NPM1 PCR if NPM1 co-mut, otherwise flow)."""
    kit = KitInput(
        patient_id="T",
        mutations=[
            MutationCall(gene="NPM1"),
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.5),
        ],
        karyotype_text="46,XX[20]",
    )
    out = _empty_kit_output(driver_flags={"NPM1": True, "FLT3_ITD": True})
    md = _mrd_monitoring_section(kit, out)
    # Primary: NPM1 PCR
    assert "NPM1 RT-qPCR" in md or "NPM1mut" in md
    # Plus FLT3-ITD-specific subsection
    assert "FLT3-ITD" in md
    assert "NGS-MRD" in md or "ClonoSEQ" in md


def test_mrd_lists_standard_timepoints():
    """Standard ELN 2021 timepoints (post-induction, post-cons 1-3, q3mo)."""
    kit = KitInput(
        patient_id="T",
        mutations=[MutationCall(gene="NPM1")],
        karyotype_text="46,XX[20]",
    )
    out = _empty_kit_output(driver_flags={"NPM1": True})
    md = _mrd_monitoring_section(kit, out)
    assert "诱导后" in md or "induction" in md.lower()
    assert "巩固" in md or "consolidation" in md.lower()
    assert "q3" in md.lower() or "3 mo" in md.lower() or "3mo" in md.lower()
