"""Tests for the clinician-facing narrative report (Markdown + PDF)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from combo_val.clinical.kit_schema import KitInput, KitOutput, MutationCall
from combo_val.clinical.patient_report import (
    _FUSION_PROSE,
    _GENE_PROSE,
    _cytogenetic_narrative,
    _eln_rationale_prose,
    _executive_summary,
    _fusion_narrative,
    _mutation_narrative,
    build_clinical_report_markdown,
    export_clinical_report,
    render_markdown_to_html,
    render_markdown_to_pdf,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _flt3_npm1_kit() -> KitInput:
    return KitInput(
        patient_id="TEST-FLT3",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45),
            MutationCall(gene="NPM1", variant_type="missense", vaf=0.42),
        ],
        karyotype_text="46,XX[20]",
        fusions=[],
        wbc=95.0, platelet=32.0, hemoglobin=8.5, ldh=1240.0,
        age=45, sex="female", is_initial_diagnosis=True,
    )


def _apl_kit() -> KitInput:
    return KitInput(
        patient_id="TEST-APL",
        mutations=[],
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        fusions=["PML-RARA"],
        wbc=2.5, platelet=30.0, hemoglobin=9.0, ldh=320.0,
        age=38, sex="female", is_initial_diagnosis=True,
    )


def _tp53_kit() -> KitInput:
    return KitInput(
        patient_id="TEST-TP53",
        mutations=[
            MutationCall(gene="TP53", variant_type="missense", vaf=0.55),
            MutationCall(gene="ASXL1", variant_type="nonsense", vaf=0.42),
        ],
        karyotype_text="45,XY,-7,del(5)(q13q33)[18]",
        fusions=[],
        wbc=12.0, platelet=25.0, hemoglobin=7.8, ldh=850.0,
        age=72, sex="male", prior_mds=True, is_initial_diagnosis=True,
    )


def _mk_output(kit: KitInput, eln: str, driver_flags: dict | None = None,
               top_regimens: list[dict] | None = None) -> KitOutput:
    return KitOutput(
        patient_id=kit.patient_id,
        predicted_eln2017=eln,
        top_combinations=[{
            "rank": 1, "drug1": "Gilteritinib", "drug2": "Venetoclax",
            "predicted_combo_auc": 71.3, "mech_score": 1.04,
            "clonal_coverage_score": 0.67, "layer3_backbone": "test-mlp",
        }],
        top_single_drugs=[{"rank": 1, "drug": "Venetoclax", "predicted_auc": 85.0}],
        top_regimens=top_regimens or [{
            "rank": 1, "name": "Quizartinib + Venetoclax + Decitabine (Triplet)",
            "drugs": ["Quizartinib (AC220)", "Venetoclax", "Decitabine"],
            "trial_name": "ASH 2024 abstract",
            "trial_phase": "Phase2",
            "published_cr_cri_rate": 0.95,
            "pmid": None,
            "biomarker_matches": ["required_all:clin_flt3_itd=1"],
            "cautions": ["QT prolongation from quizartinib"],
        }],
        clonal_coverage={
            "n_clones_present": 2,
            "patient_clones": {"FLT3_clone": 1.0, "BCL2_dependent_clone": 0.5},
            "dominant_clones": ["FLT3_clone"],
            "top_triplets_by_coverage": [{
                "drugs": ["Cytarabine", "Gilteritinib", "Venetoclax"],
                "coverage_score": 0.82,
            }],
        },
        dna_summary={
            "sample_qc": {
                "n_mutations_called": len(kit.mutations or []),
                "n_above_vaf_0_20": len(kit.mutations or []),
                "karyotype_parsed": True,
                "fusions_reported": len(kit.fusions or []),
            },
            "cytogenetics": [
                {"finding": "Normal karyotype", "present": True,
                 "interpretation": "No ELN 2017 high-risk markers"},
            ],
        },
        driver_flags=driver_flags or {"FLT3_ITD": True, "NPM1": True},
        fitness_flag="fit_for_intensive",
        cautions=["High LDH — TLS risk"],
        confidence_notes=["Sample confidence: baseline"],
    )


# ---------------------------------------------------------------------------
# Corpus structure
# ---------------------------------------------------------------------------


def test_gene_prose_covers_common_targetable_genes():
    """The per-gene prose library must at minimum cover the FDA-targetable drivers."""
    must_have = {"FLT3", "NPM1", "IDH1", "IDH2", "TP53", "RUNX1", "ASXL1", "CEBPA"}
    assert must_have <= set(_GENE_PROSE.keys())


def test_fusion_prose_covers_four_major_aml_fusions():
    assert {"PML-RARA", "KMT2A-r", "CBFB-MYH11", "RUNX1-RUNX1T1"} <= set(_FUSION_PROSE.keys())


def test_flt3_prose_mentions_itd_and_tki_names():
    flt3 = _GENE_PROSE["FLT3"]
    assert "ITD" in flt3
    # All 3 FDA FLT3 inhibitors must be named
    for drug in ("Midostaurin", "Quizartinib", "Gilteritinib"):
        assert drug in flt3


def test_pml_rara_prose_forbids_seven_plus_three():
    """PML-RARA (APL) must warn against 7+3 to avoid mis-treatment."""
    assert "ATRA" in _FUSION_PROSE["PML-RARA"]
    assert "7+3" in _FUSION_PROSE["PML-RARA"]
    assert "不适当" in _FUSION_PROSE["PML-RARA"] or "不推荐" in _FUSION_PROSE["PML-RARA"]


# ---------------------------------------------------------------------------
# Narrative building blocks
# ---------------------------------------------------------------------------


def test_executive_summary_includes_age_sex_eln_topregimen():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    summary = _executive_summary(kit, out)
    assert "45" in summary
    assert "女性" in summary
    assert "Intermediate" in summary
    assert "Quizartinib" in summary  # top regimen name leaks through


def test_mutation_narrative_flt3_itd_shows_high_burden_flag():
    kit = _flt3_npm1_kit()
    text = _mutation_narrative(kit.mutations or [])
    assert "FLT3-ITD" in text
    assert "AR=0.62" in text
    assert "高负荷" in text  # AR >= 0.5 threshold


def test_mutation_narrative_empty_panel_gives_reassuring_message():
    text = _mutation_narrative([])
    assert "未检出" in text
    assert "lab" in text.lower()


def test_fusion_narrative_pml_rara_invokes_apl_prose():
    text = _fusion_narrative(["PML-RARA"])
    assert "APL" in text
    assert "ATRA" in text


def test_fusion_narrative_unknown_fusion_flags_for_review():
    text = _fusion_narrative(["NOVEL-FUSION-XYZ"])
    assert "NOVEL-FUSION-XYZ" in text
    assert "文献核查" in text or "literature" in text.lower()


def test_cytogenetic_narrative_handles_empty_karyotype():
    text = _cytogenetic_narrative([], None)
    assert "无" in text or "未" in text


def test_eln_rationale_apl_identifies_distinct_from_cbf():
    """Critical clinical distinction: APL is Favorable but NOT CBF-AML."""
    kit = _apl_kit()
    out = _mk_output(kit, eln="Favorable", driver_flags={})
    text = _eln_rationale_prose(kit, out)
    assert "PML-RARA" in text or "APL" in text
    assert "ATRA" in text
    # Must NOT conflate APL with CBF-AML
    assert "core-binding factor" not in text.lower()


def test_eln_rationale_tp53_adverse_names_allo_sct():
    kit = _tp53_kit()
    out = _mk_output(kit, eln="Adverse",
                     driver_flags={"TP53": True, "ASXL1": True})
    text = _eln_rationale_prose(kit, out)
    assert "Adverse" in text
    assert "TP53" in text
    assert "allo" in text.lower() or "SCT" in text


def test_eln_rationale_flt3_npm1_cohabit_intermediate():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate",
                     driver_flags={"FLT3_ITD": True, "NPM1": True})
    text = _eln_rationale_prose(kit, out)
    assert "NPM1" in text
    assert "FLT3" in text
    assert "Intermediate" in text


# ---------------------------------------------------------------------------
# Full report builder
# ---------------------------------------------------------------------------


def test_build_markdown_has_all_nine_sections():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md = build_clinical_report_markdown(kit, out)
    # All 9 top-level sections present
    for header in ("一、临床快报", "二、患者基本信息", "三、分子特征",
                   "四、治疗方案推荐", "五、模型辅助预测",
                   "六、用药警告", "七、质量控制",
                   "八、方法学背景", "九、关键参考文献"):
        assert header in md, f"Missing section: {header}"


def test_build_markdown_section_numbering_is_consistent():
    """Section 四's subsections must be 4.1/4.2/4.3, not 3.1/3.2/3.3."""
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md = build_clinical_report_markdown(kit, out)
    # Section 3 subsections (Molecular Profile)
    assert "### 3.1 核心驱动突变" in md
    assert "### 3.4 ELN 2017 风险分层" in md
    # Section 4 subsections (Treatment Recommendations) must be 4.x
    assert "### 4.1 首选方案" in md
    # Must NOT re-use 3.1 numbering for section 4
    assert md.count("### 3.1 首选方案") == 0


def test_build_markdown_references_major_trials():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md = build_clinical_report_markdown(kit, out)
    # Key guideline + trial citations must appear
    for trial_or_pmid in ("ELN 2017", "ELN 2022", "RATIFY", "VIALE-A",
                           "ADMIRAL", "AGILE", "27895058", "32813947"):
        assert trial_or_pmid in md, f"Missing reference: {trial_or_pmid}"


def test_build_markdown_includes_reviewer_checklist():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md = build_clinical_report_markdown(kit, out)
    assert "审核建议" in md or "Checklist" in md
    # A few checklist items should appear
    assert "- [ ]" in md


def test_build_markdown_marks_disclaimer_and_kit_version():
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md = build_clinical_report_markdown(kit, out)
    assert "v0.2" in md
    assert "research use only" in md.lower() or "辅助决策" in md
    assert "免责声明" in md


def test_build_markdown_is_deterministic_up_to_timestamp():
    """Same kit + output → identical MD except for the timestamp line."""
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    md1 = build_clinical_report_markdown(kit, out)
    md2 = build_clinical_report_markdown(kit, out)
    # Strip the 2 timestamp occurrences (header + footer)
    import re
    ts_pat = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")
    assert ts_pat.sub("TS", md1) == ts_pat.sub("TS", md2)


# ---------------------------------------------------------------------------
# Export integration (file side effects)
# ---------------------------------------------------------------------------


def test_export_clinical_report_writes_md_and_attempts_pdf(tmp_path):
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    paths = export_clinical_report(kit, out, tmp_path, also_render_pdf=True)

    # Markdown must always be written
    assert "markdown" in paths
    md_path = Path(paths["markdown"])
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("# AML")

    # HTML must be written (pandoc is available in the dev env)
    assert paths.get("html"), "HTML render should have succeeded"
    assert Path(paths["html"]).exists()

    # PDF might fail gracefully if no engine, but key must exist
    assert "pdf" in paths


def test_export_clinical_report_markdown_only(tmp_path):
    """also_render_pdf=False → only MD is produced."""
    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    paths = export_clinical_report(
        kit, out, tmp_path, also_render_pdf=False, also_render_html=False,
    )
    assert Path(paths["markdown"]).exists()
    assert "pdf" not in paths
    assert "html" not in paths


def test_render_markdown_to_html_produces_standalone_page(tmp_path):
    """HTML output must be self-contained and titled."""
    pytest.importorskip("subprocess")
    if not shutil.which("pandoc"):
        pytest.skip("pandoc not available in test environment")

    md = tmp_path / "r.md"
    md.write_text("# Hello\n\nParagraph with **bold** text.\n", encoding="utf-8")
    html = tmp_path / "r.html"
    render_markdown_to_html(md, html)
    assert html.exists()
    content = html.read_text(encoding="utf-8")
    # pandoc standalone HTML always includes a doctype + body
    assert "<!DOCTYPE" in content or "<html" in content.lower()
    assert "Hello" in content


def test_render_markdown_to_pdf_raises_helpful_error_when_no_engine(tmp_path, monkeypatch):
    """If no engine is available the error must mention all tried options."""
    md = tmp_path / "r.md"
    md.write_text("# Test\n", encoding="utf-8")

    # Force all engine probes to fail
    real_run = subprocess.run

    def fake_run(args, *a, **kw):
        if isinstance(args, list) and args and args[0] == "which":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        if isinstance(args, list) and args and "chrome" in str(args[0]).lower():
            return subprocess.CompletedProcess(args, 1, stdout="",
                                                stderr="mock: no chrome")
        return real_run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Also hide any chrome binaries by pointing the candidate list at a fake dir
    monkeypatch.setattr(
        "combo_val.clinical.patient_report._CHROME_CANDIDATES", [],
    )

    with pytest.raises(RuntimeError) as exc:
        render_markdown_to_pdf(md, tmp_path / "r.pdf")
    assert "PDF engine" in str(exc.value)


def test_export_clinical_report_survives_pdf_failure(tmp_path, monkeypatch):
    """When PDF rendering fails, MD is still written and pdf_error is reported."""
    from combo_val.clinical import patient_report as pr

    def boom(*a, **kw):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(pr, "render_markdown_to_pdf", boom)

    kit = _flt3_npm1_kit()
    out = _mk_output(kit, eln="Intermediate")
    paths = pr.export_clinical_report(
        kit, out, tmp_path, also_render_pdf=True, also_render_html=False,
    )
    assert Path(paths["markdown"]).exists()
    assert paths.get("pdf") == ""
    assert "simulated engine failure" in paths.get("pdf_error", "")
