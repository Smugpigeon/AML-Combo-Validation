"""Tests for the clinical kit pipeline: karyotype parser, ELN computer, feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.clinical.eln_computer import compute_eln2017, ELN_ordinal_from_string
from combo_val.clinical.karyotype_parser import parse_karyotype
from combo_val.clinical.kit_schema import KitInput, MutationCall


# ---------------------------------------------------------------------------
# Karyotype parser
# ---------------------------------------------------------------------------


def test_karyo_normal():
    k = parse_karyotype("46,XX[20]")
    assert k.normal == 1
    assert k.complex == 0


def test_karyo_t_8_21():
    k = parse_karyotype("46,XX,t(8;21)(q22;q22)[20]")
    assert k.t_8_21 == 1
    assert k.complex == 0


def test_karyo_inv_16():
    k = parse_karyotype("46,XY,inv(16)(p13.1q22)[18]/46,XY[2]")
    assert k.inv_16 == 1


def test_karyo_t_15_17_apl():
    k = parse_karyotype("46,XX,t(15;17)(q22;q21)[20]")
    assert k.t_15_17 == 1


def test_karyo_complex():
    k = parse_karyotype("47,XY,+8,del(5)(q13q33),del(7)(q22q36),t(3;3)(q21;q26)[10]")
    assert k.complex == 1


def test_karyo_monosomy_7():
    k = parse_karyotype("45,XY,-7[18]/46,XY[2]")
    assert k.monosomy_5_or_7 == 1


def test_karyo_del_17p():
    k = parse_karyotype("46,XY,del(17)(p13p13)[20]")
    assert k.del_17p == 1


def test_karyo_empty_or_unknown():
    assert parse_karyotype(None).complex == 0
    assert parse_karyotype("").complex == 0
    assert parse_karyotype("NK").complex == 0


# ---------------------------------------------------------------------------
# ELN 2017 computer
# ---------------------------------------------------------------------------


def test_eln_favorable_cbf_t821():
    r = compute_eln2017("46,XX,t(8;21)(q22;q22)[20]", [], fusions=[])
    assert r.category == "Favorable"
    assert r.ordinal == 0.0


def test_eln_favorable_apl_t1517():
    r = compute_eln2017(
        "46,XX,t(15;17)(q22;q21)[20]", [], fusions=["PML-RARA"],
    )
    assert r.category == "Favorable"


def test_eln_favorable_npm1_no_flt3():
    r = compute_eln2017(
        "46,XX[20]",
        [MutationCall(gene="NPM1", variant_type="missense", vaf=0.4)],
    )
    assert r.category == "Favorable"


def test_eln_favorable_cebpa_biallelic():
    r = compute_eln2017(
        "46,XX[20]",
        [MutationCall(gene="CEBPA", is_biallelic=True)],
    )
    assert r.category == "Favorable"


def test_eln_adverse_complex_karyotype():
    r = compute_eln2017(
        "47,XY,+8,del(5)(q13q33),del(7)(q22q36),t(3;3)(q21;q26)[10]", []
    )
    assert r.category == "Adverse"


def test_eln_adverse_tp53():
    r = compute_eln2017(
        "46,XY[20]",
        [MutationCall(gene="TP53", variant_type="missense", vaf=0.5)],
    )
    assert r.category == "Adverse"


def test_eln_adverse_runx1():
    r = compute_eln2017(
        "46,XY[20]",
        [MutationCall(gene="RUNX1", variant_type="frameshift", vaf=0.4)],
    )
    assert r.category == "Adverse"


def test_eln_intermediate_flt3_high_ar_with_npm1():
    r = compute_eln2017(
        "46,XY[20]",
        [
            MutationCall(gene="NPM1"),
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6),
        ],
    )
    assert r.category == "Intermediate"


def test_eln_favorable_flt3_low_ar_with_npm1():
    r = compute_eln2017(
        "46,XY[20]",
        [
            MutationCall(gene="NPM1"),
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.3),
        ],
    )
    assert r.category == "Favorable"


def test_eln_adverse_flt3_high_ar_no_npm1():
    r = compute_eln2017(
        "46,XY[20]",
        [MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6)],
    )
    assert r.category == "Adverse"


def test_eln_ordinal_string_map():
    assert ELN_ordinal_from_string("Favorable") == 0.0
    assert ELN_ordinal_from_string("Intermediate") == 1.0
    assert ELN_ordinal_from_string("Adverse") == 2.0
    assert ELN_ordinal_from_string("Unknown") == 1.0  # default → Intermediate


# ---------------------------------------------------------------------------
# Feature builder (requires the persisted preprocessor bundle)
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessor_path(tmp_path):
    """If the real preprocessor isn't at the default location, skip tests that need it."""
    from pathlib import Path
    p = Path("data/canonical/beataml_rna_preprocessor.joblib")
    if not p.exists():
        pytest.skip(f"preprocessor not available at {p} — run beataml_etl first")
    return p


def test_feature_builder_end_to_end(preprocessor_path):
    from combo_val.clinical.feature_builder import build_patient_features_from_raw
    import joblib
    bundle = joblib.load(preprocessor_path)
    # Build a synthetic FLT3-mut NPM1-mut new patient
    rna = pd.Series(
        {g: 10.0 for g in bundle["kept_genes"][:1000]}  # 20% gene coverage
    )
    kit = KitInput(
        patient_id="test_flt3_npm1",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.55),
            MutationCall(gene="NPM1"),
            MutationCall(gene="DNMT3A"),
        ],
        karyotype_text="46,XY[20]",
        fusions=[],
        wbc=85.0, platelet=40.0, hemoglobin=9.2,
        ldh=650.0, alt=35.0, ast=40.0, albumin=3.8,
        blast_pct_bm=72.0,
        age=55, sex="male",
        is_relapse=False, prior_mds=False, prior_chemo=False,
        is_initial_diagnosis=True,
    )
    feat, diag = build_patient_features_from_raw(rna, kit, preprocessor_path)
    assert feat.shape == (len(bundle["feature_columns"]),)
    assert np.isfinite(feat).all()

    # FLT3 ITD + NPM1 + high AR → ELN = Intermediate
    assert diag["eln_predicted"] == "Intermediate"

    # Coverage diagnostic is reasonable
    assert diag["rna_gene_coverage_pct"] == 20.0


def test_feature_builder_handles_missing_labs(preprocessor_path):
    """Missing labs should fall back to BeatAML medians, not crash."""
    from combo_val.clinical.feature_builder import build_patient_features_from_raw
    import joblib
    bundle = joblib.load(preprocessor_path)
    rna = pd.Series({g: 5.0 for g in bundle["kept_genes"][:500]})
    kit = KitInput(
        patient_id="sparse_patient",
        mutations=[MutationCall(gene="IDH2")],
        karyotype_text="46,XY[20]",
        age=68,
        # No labs at all
    )
    feat, diag = build_patient_features_from_raw(rna, kit, preprocessor_path)
    assert feat.shape == (len(bundle["feature_columns"]),)
    assert np.isfinite(feat).all()
    # Most clinical lab fields should have been imputed
    assert "clin_wbc_log" in diag["imputed_fields"]
    assert "clin_ldh_log" in diag["imputed_fields"]
