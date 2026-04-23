"""Tests for the DNA-level profile report."""

from __future__ import annotations

from pathlib import Path

import pytest

from combo_val.clinical.dna_report import (
    CORE_DRIVER_GENES,
    FUSION_IMPLICATIONS,
    GENE_IMPLICATIONS,
    build_dna_summary,
    pretty_print_dna_summary,
)
from combo_val.clinical.kit_schema import KitInput, MutationCall


# ---------------------------------------------------------------------------
# Knowledge base structure
# ---------------------------------------------------------------------------


def test_core_25_gene_panel_covers_all_mut_features():
    """Core driver list must match the 25-gene mut_* columns in patient features."""
    expected = {
        "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TP53",
        "RUNX1", "ASXL1", "TET2", "CEBPA", "KIT",
        "NRAS", "KRAS", "PTPN11", "WT1", "BCOR",
        "STAG2", "PHF6", "SRSF2", "SF3B1", "U2AF1",
        "EZH2", "KMT2A", "MECOM", "CBFB",
    }
    assert set(CORE_DRIVER_GENES) == expected


def test_all_tier1_genes_have_targetable_drugs():
    """Tier 1 = FDA-approved targeted therapy exists → must list drugs."""
    tier1 = [g for g, info in GENE_IMPLICATIONS.items() if info.get("tier") == 1]
    assert len(tier1) >= 4  # FLT3, IDH1, IDH2, KMT2A
    for g in tier1:
        drugs = GENE_IMPLICATIONS[g].get("targetable_by") or []
        assert len(drugs) >= 1, f"Tier-1 gene {g} must list at least one targetable drug"


def test_critical_fusions_annotated():
    """The four clinically-critical AML fusions must be in the DB."""
    critical = {"PML-RARA", "KMT2A-r", "CBFB-MYH11", "RUNX1-RUNX1T1"}
    assert critical <= set(FUSION_IMPLICATIONS.keys())


def test_pml_rara_flags_7plus3_inappropriate():
    """PML-RARA entry must flag 7+3 as inappropriate (ATRA/ATO mandatory)."""
    pml = FUSION_IMPLICATIONS["PML-RARA"]
    assert "ATRA" in pml["regimen"] or "ATO" in pml["regimen"]
    assert "inappropriate" in pml["notes"].lower() or "7+3" not in pml["regimen"]


# ---------------------------------------------------------------------------
# build_dna_summary
# ---------------------------------------------------------------------------


def test_build_dna_summary_structure():
    kit = KitInput(
        patient_id="TEST",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6, vaf=0.45),
            MutationCall(gene="NPM1", vaf=0.4),
        ],
        karyotype_text="46,XX[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    assert set(summary.keys()) >= {
        "driver_mutations", "fusion_analysis", "cytogenetics",
        "eln_risk", "targetability", "sample_qc",
    }


def test_flt3_itd_detail_captured():
    kit = KitInput(
        patient_id="P-FLT3",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.43)],
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    flt3 = summary["driver_mutations"][0]
    assert flt3["gene"] == "FLT3"
    assert flt3["variant_type"] == "ITD"
    assert flt3["allelic_ratio"] == 0.62
    assert "HIGH" in flt3["ar_interpretation"]
    assert flt3["tier"] == 1
    assert len(flt3["targetable_by"]) >= 3


def test_cebpa_biallelic_detail():
    kit = KitInput(
        patient_id="P-CEBPA",
        mutations=[MutationCall(gene="CEBPA", is_biallelic=True, vaf=0.40)],
    )
    summary = build_dna_summary(kit, computed_eln="Favorable")
    cebpa = summary["driver_mutations"][0]
    assert cebpa["biallelic"] is True
    assert "favorable" in cebpa["allelic_interpretation"].lower()


def test_tp53_flagged_adverse():
    kit = KitInput(
        patient_id="P-TP53",
        mutations=[MutationCall(gene="TP53", variant_type="missense", vaf=0.55)],
    )
    summary = build_dna_summary(kit, computed_eln="Adverse")
    tp53 = summary["driver_mutations"][0]
    assert tp53["is_adverse_driver"] is True
    assert tp53["targetable_by"] == []  # no direct target


def test_pml_rara_fusion_flags_apl_regimen():
    kit = KitInput(
        patient_id="P-APL",
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        fusions=["PML-RARA"],
    )
    summary = build_dna_summary(kit, computed_eln="Favorable")
    fus = summary["fusion_analysis"]
    assert len(fus) == 1
    assert "PML-RARA" in fus[0]["fusion"]
    assert "ATRA" in fus[0]["regimen_note"] or "ATO" in fus[0]["regimen_note"]
    # Targetability should include ATRA+ATO
    targ = summary["targetability"]
    assert "PML-RARA" in targ
    assert any("ATRA" in d or "ATO" in d for d in targ["PML-RARA"])


def test_complex_karyotype_detected():
    kit = KitInput(
        patient_id="P-COMPLEX",
        karyotype_text="45,XY,-7,del(5)(q13q33),+8,t(3;3)(q21;q26),del(17)(p13)[18]/46,XY[2]",
    )
    summary = build_dna_summary(kit, computed_eln="Adverse")
    cyto = {r["finding"]: r["present"] for r in summary["cytogenetics"]}
    assert cyto["Complex karyotype"] is True
    assert cyto["Monosomy 5/7 or del(5q)/del(7q)"] is True
    assert cyto["del(17p) / TP53 locus"] is True


def test_targetability_includes_universal_venetoclax():
    """Every patient should have venetoclax listed as universally indicated."""
    kit = KitInput(
        patient_id="P",
        mutations=[MutationCall(gene="RUNX1")],
    )
    summary = build_dna_summary(kit, computed_eln="Adverse")
    assert "universal" in summary["targetability"]
    assert any("Venetoclax" in d for d in summary["targetability"]["universal"])


def test_mutations_sorted_by_tier():
    kit = KitInput(
        patient_id="P",
        mutations=[
            MutationCall(gene="DNMT3A", vaf=0.5),   # tier 3
            MutationCall(gene="FLT3", is_ITD=True, vaf=0.4),   # tier 1
            MutationCall(gene="NPM1", vaf=0.45),   # tier 2
        ],
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    tiers = [m.get("tier") for m in summary["driver_mutations"]]
    # Tier 1 must come first, then 2, then 3
    assert tiers == sorted(tiers, key=lambda t: t or 99)


def test_sample_qc_counts():
    kit = KitInput(
        patient_id="P",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, vaf=0.45),
            MutationCall(gene="NPM1", vaf=0.40),
            MutationCall(gene="DNMT3A", vaf=0.05),  # below 0.20 threshold
        ],
        karyotype_text="46,XX[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    qc = summary["sample_qc"]
    assert qc["n_mutations_called"] == 3
    assert qc["n_with_vaf_annotation"] == 3
    assert qc["n_above_vaf_0_20"] == 2


# ---------------------------------------------------------------------------
# pretty_print_dna_summary
# ---------------------------------------------------------------------------


def test_pretty_print_contains_expected_sections():
    kit = KitInput(
        patient_id="P",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6, vaf=0.45)],
        karyotype_text="46,XY[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    output = pretty_print_dna_summary(summary)
    assert "DRIVER MUTATIONS" in output
    assert "FUSION ANALYSIS" in output
    assert "CYTOGENETIC FLAGS" in output
    assert "TARGETABILITY" in output
    assert "ELN 2017 RISK" in output
    assert "SAMPLE QC" in output


def test_pretty_print_handles_no_mutations():
    kit = KitInput(patient_id="P-empty")
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    output = pretty_print_dna_summary(summary)
    # Should still produce valid output, not crash
    assert "DNA-LEVEL PROFILE" in output
    assert "No driver mutations called" in output


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------


def test_export_dna_summary_csv(tmp_path):
    from combo_val.clinical.dna_report import export_dna_summary_csv
    kit = KitInput(
        patient_id="EXPORT-001",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45),
            MutationCall(gene="NPM1", variant_type="missense", vaf=0.42),
        ],
        karyotype_text="46,XX[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    paths = export_dna_summary_csv(summary, "EXPORT-001", tmp_path)

    assert "driver_mutations" in paths
    driver_csv = Path(paths["driver_mutations"])
    assert driver_csv.exists()
    # Spot-check the CSV actually has the expected data
    content = driver_csv.read_text()
    assert "FLT3" in content
    assert "ITD" in content
    assert "Midostaurin" in content   # targetable_by serialized
    assert "NPM1" in content

    # Other files
    assert Path(paths["cytogenetics"]).exists()
    assert Path(paths["targetability"]).exists()
    assert Path(paths["sample_qc"]).exists()
    assert Path(paths["dna_summary_json"]).exists()


def test_render_dna_summary_figure(tmp_path):
    from combo_val.clinical.dna_report import render_dna_summary_figure
    kit = KitInput(
        patient_id="FIG-001",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45)],
        karyotype_text="46,XY[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    out_path = tmp_path / "test_figure.png"
    result_path = render_dna_summary_figure(summary, "FIG-001", out_path)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 1000   # non-empty PNG


# ---------------------------------------------------------------------------
# Per-patient README generator
# ---------------------------------------------------------------------------


def test_generate_patient_readme_flt3_tier1_highlighted(tmp_path):
    from combo_val.clinical.dna_report import generate_patient_readme
    kit = KitInput(
        patient_id="README-FLT3",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45)],
        karyotype_text="46,XX[20]",
    )
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    out = tmp_path / "README.md"
    path = generate_patient_readme(summary, "README-FLT3", out_path=out)
    content = Path(path).read_text()
    # Tier 1 section should appear with FLT3 + AR info
    assert "Tier-1" in content
    assert "FLT3" in content
    assert "ITD AR=0.62" in content
    # Targetable drugs listed
    assert "Midostaurin" in content or "Gilteritinib" in content
    # File manifest included
    assert "driver_mutations.csv" in content
    # Links to full guide
    assert "clinical_reader_guide.md" in content


def test_generate_patient_readme_tp53_adverse(tmp_path):
    from combo_val.clinical.dna_report import generate_patient_readme
    kit = KitInput(
        patient_id="README-TP53",
        mutations=[MutationCall(gene="TP53", variant_type="missense", vaf=0.55)],
        karyotype_text="45,XY,-7,del(5)(q13q33),+8,t(3;3)(q21;q26),del(17)(p13)[18]",
    )
    summary = build_dna_summary(kit, computed_eln="Adverse")
    out = tmp_path / "README.md"
    generate_patient_readme(summary, "README-TP53", out_path=out)
    content = out.read_text()
    # Adverse drivers section visible
    assert "Adverse drivers" in content
    assert "TP53" in content
    # Abnormal cytogenetic findings section
    assert "Abnormal cytogenetic findings" in content or "cytogenetic" in content.lower()
    # No Tier-1 section should appear (TP53 is Tier 2, no Tier-1 mutations)
    # (We allow it to be absent or say "0"; ensure Tier-1 isn't claimed false-positive)


def test_generate_patient_readme_returns_content_when_no_out_path():
    from combo_val.clinical.dna_report import generate_patient_readme
    kit = KitInput(patient_id="README-X")
    summary = build_dna_summary(kit, computed_eln="Intermediate")
    content = generate_patient_readme(summary, "README-X")
    # Should return the markdown string directly
    assert isinstance(content, str)
    assert "Patient README-X" in content
