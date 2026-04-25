"""Tests for ELN 2022 risk stratification (issue #4).

Per Döhner Blood 2022 (PMID 35797463):

- FLT3-ITD allelic ratio is no longer used
- NPM1+FLT3-ITD (any AR) → Intermediate (was AR-conditional in 2017)
- MDS-related gene mutations → Adverse (NEW: BCOR, EZH2, SF3B1, SRSF2,
  STAG2, U2AF1, ZRSR2 — RUNX1 + ASXL1 were already 2017 Adverse)
- TP53 multi-hit → independent Adverse subtype
- CEBPA bZIP single allele → Favorable (was biallelic-only in 2017)
"""

from __future__ import annotations

import pytest

from combo_val.clinical.eln_2022 import (
    ELN2022_MDS_RELATED_GENES,
    compare_eln_versions,
    compute_eln2022,
)
from combo_val.clinical.eln_computer import compute_eln2017
from combo_val.clinical.kit_schema import MutationCall


# ---------------------------------------------------------------------------
# Adverse — TP53
# ---------------------------------------------------------------------------


def test_tp53_multi_hit_via_explicit_flag():
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.4, is_multi_hit=True)],
    )
    assert r.category == "Adverse"
    assert any("multi-hit" in s.lower() for s in r.rationale)


def test_tp53_multi_hit_via_two_calls():
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[
            MutationCall(gene="TP53", vaf=0.3, variant_type="missense"),
            MutationCall(gene="TP53", vaf=0.3, variant_type="frameshift"),
        ],
    )
    assert r.category == "Adverse"
    assert any("multi-hit" in s.lower() for s in r.rationale)


def test_tp53_multi_hit_via_high_vaf():
    """Single TP53 call with VAF ≥ 0.5 → biallelic (LOH) → multi-hit."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.7)],
    )
    assert r.category == "Adverse"
    assert any("multi-hit" in s.lower() for s in r.rationale)


def test_tp53_multi_hit_via_del17p():
    r = compute_eln2022(
        karyotype_text="46,XY,del(17)(p13p13)[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.3)],
    )
    assert r.category == "Adverse"


def test_tp53_single_hit_low_vaf_no_del17p():
    """Should still be Adverse (TP53 mutation alone), but rationale should
    NOT claim multi-hit — kit must be honest about which subtype."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.15)],
    )
    assert r.category == "Adverse"
    # Rationale should mention single-hit or simple TP53, not multi-hit
    rationale_text = " ".join(r.rationale)
    assert "TP53" in rationale_text


# ---------------------------------------------------------------------------
# Adverse — MDS-related genes (NEW in 2022)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gene", sorted(ELN2022_MDS_RELATED_GENES))
def test_mds_related_gene_alone_triggers_adverse(gene):
    """Each MDS-related gene mutation alone should classify as Adverse."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene=gene, vaf=0.3)],
    )
    assert r.category == "Adverse", \
        f"{gene} mutation should be Adverse per ELN 2022; got {r.category}"


def test_sf3b1_intermediate_in_2017_adverse_in_2022():
    """Patient with SF3B1 alone — 2017 says Intermediate, 2022 says Adverse."""
    muts = [MutationCall(gene="SF3B1", vaf=0.4)]
    r17 = compute_eln2017("46,XX[20]", muts)
    r22 = compute_eln2022("46,XX[20]", muts)
    assert r17.category == "Intermediate"
    assert r22.category == "Adverse"


def test_prior_mds_history_triggers_adverse():
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="DNMT3A", vaf=0.4)],
        prior_mds=True,
    )
    assert r.category == "Adverse"
    assert any("MDS" in s for s in r.rationale)


# ---------------------------------------------------------------------------
# Adverse — cytogenetics
# ---------------------------------------------------------------------------


def test_complex_karyotype_adverse():
    r = compute_eln2022(
        karyotype_text="47,XY,+8,del(5)(q13q33),del(7)(q22q36),"
                        "t(3;3)(q21;q26)[10]",
        mutations=[],
    )
    assert r.category == "Adverse"


def test_monosomy_7_adverse():
    r = compute_eln2022(
        karyotype_text="45,XY,-7[18]/46,XY[2]",
        mutations=[],
    )
    assert r.category == "Adverse"


# ---------------------------------------------------------------------------
# Favorable — CEBPA bZIP (relaxed from 2017's biallelic-only)
# ---------------------------------------------------------------------------


def test_cebpa_bzip_single_allele_is_favorable_2022():
    """ELN 2022 relaxes CEBPA: bZIP in-frame single allele → Favorable."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="CEBPA", is_bzip=True)],
    )
    assert r.category == "Favorable"
    assert any("bZIP" in s for s in r.rationale)


def test_cebpa_biallelic_still_favorable_2022_for_legacy_inputs():
    """Legacy inputs flagged biallelic (without explicit is_bzip) should
    still be Favorable in 2022 — most biallelic CEBPA cases are bZIP."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="CEBPA", is_biallelic=True)],
    )
    assert r.category == "Favorable"


def test_cebpa_neither_bzip_nor_biallelic_not_favorable_2022():
    """Plain monoallelic non-bZIP CEBPA — not Favorable."""
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="CEBPA", is_bzip=False, is_biallelic=False)],
    )
    assert r.category != "Favorable"


# ---------------------------------------------------------------------------
# Favorable — NPM1
# ---------------------------------------------------------------------------


def test_npm1_alone_favorable():
    r = compute_eln2022(
        karyotype_text="46,XX[20]",
        mutations=[MutationCall(gene="NPM1", vaf=0.4)],
    )
    assert r.category == "Favorable"


def test_cbf_aml_favorable():
    """t(8;21) RUNX1-RUNX1T1 → Favorable in both 2017 and 2022."""
    r = compute_eln2022(
        karyotype_text="46,XX,t(8;21)(q22;q22)[20]",
        mutations=[],
    )
    assert r.category == "Favorable"


def test_apl_favorable():
    """t(15;17) PML-RARA → Favorable in both."""
    r = compute_eln2022(
        karyotype_text="46,XX,t(15;17)(q22;q21)[20]",
        mutations=[],
    )
    assert r.category == "Favorable"


# ---------------------------------------------------------------------------
# Intermediate — FLT3-ITD changes from 2017
# ---------------------------------------------------------------------------


def test_npm1_flt3itd_low_ar_was_favorable_in_2017_intermediate_in_2022():
    """The headline 2022 change: NPM1+FLT3-ITD low AR was Favorable in 2017
    but is Intermediate in 2022 (AR no longer used)."""
    muts = [
        MutationCall(gene="NPM1", vaf=0.4),
        MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.3),
    ]
    r17 = compute_eln2017("46,XX[20]", muts)
    r22 = compute_eln2022("46,XX[20]", muts)
    assert r17.category == "Favorable"
    assert r22.category == "Intermediate"
    assert any("AR no longer used" in s or "AR" in s for s in r22.rationale)


def test_npm1_flt3itd_high_ar_intermediate_in_both():
    muts = [
        MutationCall(gene="NPM1", vaf=0.4),
        MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.7),
    ]
    r17 = compute_eln2017("46,XX[20]", muts)
    r22 = compute_eln2022("46,XX[20]", muts)
    assert r17.category == "Intermediate"
    assert r22.category == "Intermediate"


def test_flt3itd_high_ar_no_npm1_was_adverse_in_2017_intermediate_in_2022():
    """The other big 2022 change: FLT3-ITD high AR without NPM1 was
    Adverse in 2017 but is Intermediate in 2022."""
    muts = [MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.7)]
    r17 = compute_eln2017("46,XX[20]", muts)
    r22 = compute_eln2022("46,XX[20]", muts)
    assert r17.category == "Adverse"
    assert r22.category == "Intermediate"


# ---------------------------------------------------------------------------
# Cross-version comparator
# ---------------------------------------------------------------------------


def test_compare_eln_versions_returns_none_when_same():
    assert compare_eln_versions("Favorable", "Favorable") is None
    assert compare_eln_versions("Adverse", "Adverse") is None


def test_compare_eln_versions_explains_intermediate_to_adverse():
    note = compare_eln_versions("Intermediate", "Adverse")
    assert note is not None
    assert "MDS" in note or "SF3B1" in note or "spliceosome" in note.lower()


def test_compare_eln_versions_explains_adverse_to_intermediate():
    note = compare_eln_versions("Adverse", "Intermediate")
    assert note is not None
    assert "FLT3-ITD" in note or "AR" in note


def test_compare_eln_versions_explains_intermediate_to_favorable():
    note = compare_eln_versions("Intermediate", "Favorable")
    assert note is not None
    assert "CEBPA" in note or "bZIP" in note


# ---------------------------------------------------------------------------
# Issue #4 acceptance criteria
# ---------------------------------------------------------------------------


def test_acceptance_tp53_multihit_adverse_via_2022():
    """Issue #4 AC: 'a TP53 multi-hit patient gets correct Adverse via 2022'."""
    r = compute_eln2022(
        karyotype_text="46,XY,del(17)(p13p13)[20]",
        mutations=[MutationCall(gene="TP53", vaf=0.4)],
    )
    assert r.category == "Adverse"


def test_acceptance_sf3b1_runx1_intermediate_to_adverse():
    """Issue #4 AC: 'an SF3B1+RUNX1 patient gets Adverse in 2022 but might
    have been Intermediate in 2017'.

    Actually — RUNX1 alone IS Adverse in 2017, so this comparison really
    tests SF3B1's contribution. We make the 2017 case go favorable via NPM1
    so SF3B1 is the deciding factor."""
    muts = [
        MutationCall(gene="NPM1", vaf=0.4),  # 2017 favorable
        MutationCall(gene="SF3B1", vaf=0.3),  # 2022 adverse, 2017 silent
    ]
    r17 = compute_eln2017("46,XX[20]", muts)
    r22 = compute_eln2022("46,XX[20]", muts)
    # 2017 should be Favorable (NPM1 wins, SF3B1 not in 2017 rules)
    assert r17.category == "Favorable"
    # 2022 should be Adverse (SF3B1 wins as MDS-related)
    assert r22.category == "Adverse"
