"""Tests for issue #8 — CYP3A4 / drug-interaction warnings."""

from __future__ import annotations

from combo_val.clinical.kit_schema import KitOutput
from combo_val.clinical.patient_report import _drug_interaction_warnings


def _kit_output_with(drugs: list[str]) -> KitOutput:
    """Make a KitOutput with a single top regimen containing the given drugs."""
    return KitOutput(
        patient_id="T",
        predicted_eln2017="Intermediate",
        top_combinations=[],
        top_single_drugs=[],
        top_regimens=[{
            "regimen_id": "test_regimen",
            "name": "Test regimen",
            "drugs": drugs,
            "trial_phase": "Phase3",
            "score": 100,
        }],
        clonal_coverage={},
        dna_summary={},
        rna_outlier={},
        driver_flags={},
        fitness_flag="fit_for_intensive",
        cautions=[],
        confidence_notes=[],
        eln_2017={},
        eln_2022={},
    )


def test_venetoclax_triggers_cyp3a4_warning():
    out = _kit_output_with(["Venetoclax", "Azacitidine"])
    warnings = _drug_interaction_warnings(out)
    assert any("Venetoclax" in w for w in warnings)
    cyp_w = next((w for w in warnings if "Venetoclax" in w), "")
    assert "CYP3A4" in cyp_w
    assert "posaconazole" in cyp_w.lower() or "azole" in cyp_w.lower()
    # Specific Ven dose adjustment numbers must be present
    assert "100 mg" in cyp_w


def test_quizartinib_triggers_qt_warning():
    out = _kit_output_with(["Cytarabine", "Daunorubicin", "Quizartinib (AC220)"])
    warnings = _drug_interaction_warnings(out)
    quiz_w = next((w for w in warnings if "Quizartinib" in w), "")
    assert quiz_w
    assert "QT" in quiz_w
    assert "black-box" in quiz_w.lower() or "Quiz" in quiz_w
    # Should suggest gilteritinib as alternative when QT risk too high
    assert "Gilteritinib" in quiz_w or "改用" in quiz_w


def test_midostaurin_triggers_specific_warning():
    out = _kit_output_with(["Cytarabine", "Daunorubicin", "Midostaurin"])
    warnings = _drug_interaction_warnings(out)
    mido_w = next((w for w in warnings if "Midostaurin" in w), "")
    assert mido_w
    assert "CYP3A4" in mido_w


def test_gilteritinib_warning_separate_from_quizartinib():
    out = _kit_output_with(["Gilteritinib"])
    warnings = _drug_interaction_warnings(out)
    gilt_w = next((w for w in warnings if "Gilteritinib" in w), "")
    assert gilt_w
    # Gilt has lower QT risk than Quiz — that should be reflected
    assert "QT" in gilt_w
    # Should not falsely apply Quiz black-box language
    assert "black-box" not in gilt_w.lower()


def test_ivosidenib_differentiation_syndrome_warning():
    out = _kit_output_with(["Azacitidine", "Ivosidenib"])
    warnings = _drug_interaction_warnings(out)
    ivo_w = next((w for w in warnings if "Ivosidenib" in w), "")
    assert ivo_w
    # Differentiation syndrome must be highlighted
    assert "differentiation syndrome" in ivo_w.lower() or "DS" in ivo_w
    # Dexamethasone is the canonical DS treatment
    assert "dexamethasone" in ivo_w.lower()


def test_enasidenib_differentiation_syndrome_warning():
    out = _kit_output_with(["Azacitidine", "Enasidenib"])
    warnings = _drug_interaction_warnings(out)
    ena_w = next((w for w in warnings if "Enasidenib" in w), "")
    assert ena_w
    assert "differentiation syndrome" in ena_w.lower() or "DS" in ena_w


def test_anthracycline_cardiotoxicity_warning():
    out = _kit_output_with(["Cytarabine", "Daunorubicin"])
    warnings = _drug_interaction_warnings(out)
    anth_w = next((w for w in warnings if "蒽环" in w or "Daunorubicin" in w), "")
    assert anth_w
    # Cumulative dose limits must be listed
    assert "550" in anth_w  # mg/m² Daunorubicin lifetime
    # ECHO/LVEF reference
    assert "ECHO" in anth_w or "LVEF" in anth_w


def test_no_warnings_for_supportive_only():
    """Supportive care has no targeted agents → no interaction warnings."""
    out = _kit_output_with(["Hydroxyurea", "Transfusion"])
    warnings = _drug_interaction_warnings(out)
    # Hydroxyurea isn't on our interaction list
    assert warnings == []


def test_no_warnings_when_no_top_regimens():
    out = _kit_output_with([])
    out.top_regimens = []
    warnings = _drug_interaction_warnings(out)
    assert warnings == []


def test_warnings_appear_in_full_report():
    """End-to-end — interactions must appear in §8 (cautions) of the report."""
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    from combo_val.clinical.patient_report import build_clinical_report_markdown

    kit = KitInput(
        patient_id="T",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6),
            MutationCall(gene="NPM1"),
        ],
        karyotype_text="46,XX[20]",
        wbc=20.0, platelet=80.0, hemoglobin=9.0, ldh=400.0,
        age=50, sex="female", is_initial_diagnosis=True,
    )
    out = _kit_output_with(["Cytarabine", "Daunorubicin", "Midostaurin"])
    out.driver_flags = {"FLT3_ITD": True, "NPM1": True}
    out.eln_2017 = {"category": "Intermediate", "rationale": []}
    out.eln_2022 = {"category": "Intermediate", "rationale": []}
    md = build_clinical_report_markdown(kit, out)
    # CYP3A4 caveat must be in §8 (用药警告)
    assert "Midostaurin" in md
    assert "CYP3A4" in md
    # Anthracycline cumulative dose
    assert "550" in md or "蒽环" in md
