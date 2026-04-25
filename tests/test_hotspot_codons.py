"""Tests for issues #9 + #13 — driver gene hotspot codon classifier."""

from __future__ import annotations

import pytest

from combo_val.clinical.hotspot_codons import (
    HotspotResult,
    classify_hotspot,
    hotspot_summary_for_report,
)
from combo_val.clinical.kit_schema import MutationCall


# ---------------------------------------------------------------------------
# DNMT3A R882 (issue #9 main case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codon", ["R882H", "R882C", "R882P", "R882Y", "R882S"])
def test_dnmt3a_r882_variants_are_hotspot(codon):
    r = classify_hotspot("DNMT3A", codon)
    assert r.is_hotspot is True
    assert "R882" in r.hotspot_label
    assert "adverse" in r.interpretation.lower()
    assert "MRD" in r.interpretation
    assert r.confidence == "high"


def test_dnmt3a_non_r882_recognized_separately():
    r = classify_hotspot("DNMT3A", "R635W")
    assert r.is_hotspot is True
    assert "non-R882" in r.hotspot_label or "non-R882" in r.interpretation
    assert r.confidence == "medium"  # less defined


def test_dnmt3a_unknown_codon_low_confidence():
    """Generic frameshift / non-codon DNMT3A — flagged but low confidence."""
    r = classify_hotspot("DNMT3A", "fs*32")  # nonsense / frameshift annotation
    assert r.is_hotspot is False
    assert r.confidence == "low"


def test_dnmt3a_no_codon_data():
    r = classify_hotspot("DNMT3A", None)
    assert r.is_hotspot is False
    assert "Codon not extracted" in r.interpretation


# ---------------------------------------------------------------------------
# IDH1 R132 / IDH2 R140-R172 (issue #9 + targets Ivo/Ena)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codon", ["R132H", "R132C", "R132G", "R132L", "R132S"])
def test_idh1_r132_is_hotspot(codon):
    r = classify_hotspot("IDH1", codon)
    assert r.is_hotspot is True
    assert "R132" in r.hotspot_label
    assert "Ivosidenib" in r.interpretation
    assert "AGILE" in r.interpretation


def test_idh1_non_r132_not_hotspot():
    r = classify_hotspot("IDH1", "R110Q")
    assert r.is_hotspot is False
    assert r.confidence == "low"


def test_idh2_r140_is_hotspot():
    r = classify_hotspot("IDH2", "R140Q")
    assert r.is_hotspot is True
    assert "R140" in r.hotspot_label
    assert "Enasidenib" in r.interpretation


def test_idh2_r172_is_hotspot_distinct_from_r140():
    r = classify_hotspot("IDH2", "R172K")
    assert r.is_hotspot is True
    assert "R172" in r.hotspot_label
    # Should note the distinct phenotype
    assert "R140" in r.interpretation or "global hypermethylation" in r.interpretation.lower()


# ---------------------------------------------------------------------------
# FLT3-TKD D835/I836/F691L — issue #13
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codon", ["D835Y", "D835F", "D835V", "D835H", "D835N"])
def test_flt3_d835_hotspot_prefers_gilteritinib(codon):
    r = classify_hotspot("FLT3", codon)
    assert r.is_hotspot is True
    assert "D835" in r.hotspot_label
    assert "Gilteritinib" in r.interpretation
    assert "Type-1" in r.interpretation


def test_flt3_i836_hotspot():
    r = classify_hotspot("FLT3", "I836S")
    assert r.is_hotspot is True
    assert "I836" in r.hotspot_label


def test_flt3_f691l_gatekeeper_resistance():
    """F691L is a gatekeeper mutation — Gilt + Quiz both fail; Crenolanib/SCT."""
    r = classify_hotspot("FLT3", "F691L")
    assert r.is_hotspot is True
    assert "gatekeeper" in r.hotspot_label.lower()
    assert "Crenolanib" in r.interpretation or "SCT" in r.interpretation
    assert "resistance" in r.interpretation.lower()


# ---------------------------------------------------------------------------
# NPM1 exon 12 frameshift (most common AML mutation)
# ---------------------------------------------------------------------------


def test_npm1_w288_or_tctg_recognized():
    r = classify_hotspot("NPM1", "W288Cfs*12")
    assert r.is_hotspot is True
    assert "exon 12" in r.hotspot_label.lower()


def test_npm1_tctg_synonym_recognized():
    r = classify_hotspot("NPM1", "c.860_863dupTCTG")
    assert r.is_hotspot is True


# ---------------------------------------------------------------------------
# KIT D816V — adverse modifier in CBF-AML
# ---------------------------------------------------------------------------


def test_kit_d816v_adverse_modifier():
    r = classify_hotspot("KIT", "D816V")
    assert r.is_hotspot is True
    assert "CBF-AML" in r.interpretation
    assert "imatinib" in r.interpretation.lower() or "dasatinib" in r.interpretation.lower()


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_hotspot_summary_collects_multiple_drivers():
    """Patient with 4 driver mutations → returns 4 hotspot summaries."""
    mutations = [
        MutationCall(gene="DNMT3A", protein_codon="R882H"),
        MutationCall(gene="NPM1", protein_codon="W288Cfs*12"),
        MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6),
        MutationCall(gene="IDH1", protein_codon="R132H"),
    ]
    summary = hotspot_summary_for_report(mutations)
    # FLT3 ITD has no protein_codon → low confidence → still listed (but
    # filtered out for low-confidence non-hotspots).
    seen_genes = {s["gene"] for s in summary}
    assert "DNMT3A" in seen_genes
    assert "NPM1" in seen_genes
    assert "IDH1" in seen_genes


def test_hotspot_summary_skips_genes_not_in_table():
    """Non-AML gene → no entry in summary."""
    mutations = [MutationCall(gene="BRCA1", protein_codon="P85L")]
    summary = hotspot_summary_for_report(mutations)
    assert summary == []  # not in our hotspot table → skipped at low confidence


def test_protein_codon_field_is_optional():
    """MutationCall must default protein_codon to None for backwards compat."""
    m = MutationCall(gene="DNMT3A")
    assert m.protein_codon is None
