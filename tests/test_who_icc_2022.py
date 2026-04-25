"""Tests for WHO 2022 + ICC 2022 classification (issue #12)."""

from __future__ import annotations

import pytest

from combo_val.clinical.kit_schema import MutationCall
from combo_val.clinical.who_icc_2022 import classify_who_icc_2022


def test_apl_distinct_entity():
    r = classify_who_icc_2022(
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        mutations=[],
        fusions=["PML-RARA"],
    )
    assert "promyelocytic" in r.who_2022.lower() or "PML" in r.who_2022
    assert "PML" in r.icc_2022 or "APL" in r.icc_2022
    assert r.concordant is True


def test_t_8_21_is_defining():
    r = classify_who_icc_2022(
        karyotype_text="46,XY,t(8;21)(q22;q22)[20]",
        mutations=[],
    )
    assert "RUNX1" in r.who_2022
    assert "RUNX1" in r.icc_2022
    assert r.concordant is True


def test_inv_16_is_defining():
    r = classify_who_icc_2022(
        karyotype_text="46,XX,inv(16)(p13.1q22)[20]",
        mutations=[],
    )
    assert "CBFB" in r.who_2022 or "CBFB" in r.icc_2022


def test_npm1_with_high_blasts_concordant():
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="NPM1")],
        blast_pct=45,
    )
    assert "NPM1" in r.who_2022
    assert "NPM1" in r.icc_2022
    assert r.concordant is True


def test_npm1_with_low_blasts_who_icc_diverge():
    """NPM1mut with BM blast < 10% — WHO calls it AML, ICC calls it MDS."""
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="NPM1")],
        blast_pct=8,
    )
    assert "NPM1" in r.who_2022
    assert "MDS" in r.icc_2022 or "blast" in r.icc_2022.lower()
    assert r.concordant is False


def test_cebpa_bzip_is_defining_in_both():
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="CEBPA", is_bzip=True)],
    )
    assert "CEBPA" in r.who_2022
    assert "CEBPA" in r.icc_2022
    assert r.concordant is True


def test_tp53_multi_hit_icc_dedicated_entity():
    """TP53 multi-hit — ICC has dedicated entity, WHO folds into AML-MR."""
    r = classify_who_icc_2022(
        karyotype_text="46,XY,del(17)(p13p13)[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.4)],
    )
    assert "TP53" in r.icc_2022
    assert "myelodysplasia" in r.who_2022.lower() or "AML-MR" in r.who_2022
    assert r.concordant is False


def test_mds_related_gene_concordant():
    """SF3B1 alone → AML-MR in both."""
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="SF3B1", vaf=0.3)],
    )
    assert "myelodysplasia" in r.who_2022.lower() or "MDS" in r.who_2022
    assert "myelodysplasia" in r.icc_2022.lower() or "MDS" in r.icc_2022


def test_complex_karyotype_blast_threshold_diverge():
    """Complex karyotype, blast 15% — WHO accepts as AML-MR; ICC requires
    ≥ 20% blast for cytogenetic-only AML-MR."""
    r = classify_who_icc_2022(
        karyotype_text="47,XY,+8,del(5)(q13q33),del(7)(q22q36),"
                       "t(3;3)(q21;q26)[10]",
        mutations=[],
        blast_pct=15,
    )
    assert "myelodysplasia" in r.who_2022.lower() or "AML-MR" in r.who_2022
    # ICC: blast 15% < 20% → should NOT be AML, should be MDS-AML
    assert r.concordant is False
    assert "MDS" in r.icc_2022 or "10-19" in r.icc_2022


def test_default_aml_nos():
    """Patient with no defining abnormality → AML, NOS."""
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="DNMT3A")],
    )
    assert "NOS" in r.who_2022 or "not otherwise" in r.who_2022.lower()
    assert r.concordant is True


def test_classification_has_rationale():
    r = classify_who_icc_2022(
        karyotype_text="46,XY,t(8;21)(q22;q22)[20]",
        mutations=[],
    )
    assert r.rationale
    assert isinstance(r.rationale, list)
    assert len(r.rationale) >= 1


def test_kmt2a_rearrangement():
    r = classify_who_icc_2022(
        karyotype_text="46,XX[20]",
        mutations=[],
        fusions=["KMT2A-MLLT3"],
    )
    assert "KMT2A" in r.who_2022
    assert "KMT2A" in r.icc_2022
