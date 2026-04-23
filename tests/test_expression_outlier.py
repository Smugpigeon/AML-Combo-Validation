"""Tests for RNA-Seq expression-outlier analysis (Route B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combo_val.clinical.expression_outlier import (
    EXPRESSION_HINTS,
    TIER_GROUPS_FOR_EXPR,
    _classify_z,
    _infer_scale_flavor,
    _load_ref_stats,
    _transform_to_reference_scale,
    build_rnaseq_outlier_markdown,
    compute_expression_outliers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_stats():
    """Load the canonical BeatAML reference stats."""
    return _load_ref_stats()


@pytest.fixture
def baseline_expr(ref_stats):
    """A patient expression Series exactly at BeatAML mean for every gene."""
    return pd.Series({g: s["mean"] for g, s in ref_stats["genes"].items()})


# ---------------------------------------------------------------------------
# Reference stats file integrity
# ---------------------------------------------------------------------------


def test_ref_stats_file_has_all_25_core_drivers(ref_stats):
    core_25 = {"FLT3", "IDH1", "IDH2", "KMT2A", "NPM1", "TP53", "RUNX1",
                "ASXL1", "CEBPA", "DNMT3A", "TET2", "KIT", "NRAS", "KRAS",
                "PTPN11", "WT1", "BCOR", "STAG2", "PHF6", "SRSF2", "SF3B1",
                "U2AF1", "EZH2", "MECOM", "CBFB"}
    assert core_25 <= set(ref_stats["genes"].keys())


def test_ref_stats_file_has_key_hint_genes(ref_stats):
    for g in ("HOXA9", "MEIS1", "BCL2", "MCL1", "BAALC", "MN1"):
        assert g in ref_stats["genes"], f"missing hint gene {g}"


def test_ref_stats_ref_cohort_name(ref_stats):
    assert "BeatAML" in ref_stats["reference_cohort"]
    assert ref_stats["n_samples"] >= 600


def test_ref_stats_mean_and_std_reasonable(ref_stats):
    """Means should be log2-CPM-ish; std bound is generous because several AML
    marker genes (MECOM, HOXA9, WT1, BAALC) have truly bimodal expression."""
    for g, s in ref_stats["genes"].items():
        # MECOM can have negative log-CPM-ish mean due to low baseline expression
        assert -10 <= s["mean"] <= 15, f"{g} mean out of range: {s['mean']}"
        assert 0 < s["std"] <= 5, f"{g} std out of range: {s['std']}"


def test_ref_stats_missing_file_raises(tmp_path):
    """Explicit path override that doesn't exist must raise."""
    with pytest.raises(FileNotFoundError):
        _load_ref_stats(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


def test_scale_flavor_raw_counts():
    rng = np.random.default_rng(0)
    counts = rng.lognormal(mean=5.0, sigma=1.5, size=200)  # median ~150
    assert _infer_scale_flavor(counts) == "raw_counts"


def test_scale_flavor_log_cpm():
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=4.0, scale=1.5, size=200)
    assert _infer_scale_flavor(vals) == "log_cpm"


def test_scale_flavor_ambiguous_edge():
    # median ~20, max < 200 — ambiguous
    vals = np.linspace(15, 150, 30)
    assert _infer_scale_flavor(vals) == "ambiguous"


def test_auto_log2_when_raw_counts_given():
    raw = pd.Series({"FLT3": 1200.0, "NPM1": 900.0, "TP53": 500.0,
                      "BCL2": 800.0, "HOXA9": 700.0})
    aligned, note = _transform_to_reference_scale(
        raw, ["FLT3", "NPM1", "TP53", "BCL2", "HOXA9"],
    )
    assert "auto-log2" in note
    assert (aligned < 20).all()  # post-log2, all values should be < 20


def test_log_scale_passed_through():
    log_like = pd.Series({"FLT3": 8.2, "NPM1": 8.6, "TP53": 5.8,
                           "BCL2": 7.4, "HOXA9": 6.0})
    aligned, note = _transform_to_reference_scale(
        log_like, ["FLT3", "NPM1", "TP53", "BCL2", "HOXA9"],
    )
    assert "as-is" in note or "log" in note
    np.testing.assert_allclose(aligned.values, log_like.values)


# ---------------------------------------------------------------------------
# Z-score classification
# ---------------------------------------------------------------------------


def test_classify_z_extreme_high():
    assert _classify_z(3.1) == "↑↑↑"
    assert _classify_z(1.8) == "↑↑"
    assert _classify_z(1.0) == "↑"


def test_classify_z_extreme_low():
    assert _classify_z(-3.1) == "↓↓↓"
    assert _classify_z(-1.8) == "↓↓"
    assert _classify_z(-1.0) == "↓"


def test_classify_z_normal_zone():
    assert _classify_z(0.0) == "·"
    assert _classify_z(0.3) == "·"
    assert _classify_z(-0.5) == "·"


def test_classify_z_none_returns_nan_marker():
    assert _classify_z(None) == "n/a"
    assert _classify_z(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# Core outlier computation
# ---------------------------------------------------------------------------


def test_no_rna_input_returns_all_unavailable_rows():
    rows, meta = compute_expression_outliers(None, mutated_genes={"FLT3"})
    assert len(rows) > 30  # all panel + hint genes
    assert all(not r.available for r in rows)
    assert meta["n_genes_available"] == 0


def test_baseline_expression_yields_zero_z(baseline_expr):
    rows, meta = compute_expression_outliers(baseline_expr, mutated_genes=set())
    available = [r for r in rows if r.available]
    assert len(available) >= 25
    # All should be very close to z=0
    for r in available:
        assert abs(r.z_score) < 0.1, f"{r.gene} z={r.z_score}"
    assert meta["n_outliers_high"] == 0
    assert meta["n_outliers_low"] == 0


def test_elevated_flt3_with_mutation_flags_double_evidence(ref_stats, baseline_expr):
    """FLT3 mutated + expression +2σ → 'double evidence' note."""
    expr = baseline_expr.copy()
    expr["FLT3"] = ref_stats["genes"]["FLT3"]["mean"] + 2.0 * ref_stats["genes"]["FLT3"]["std"]
    rows, meta = compute_expression_outliers(expr, mutated_genes={"FLT3"})
    flt3 = next(r for r in rows if r.gene == "FLT3")
    assert flt3.z_score is not None and flt3.z_score >= 1.5
    assert "双证据" in flt3.note


def test_cryptic_mecom_wildtype_flags_inv3_warning(ref_stats, baseline_expr):
    """MECOM wild-type but z > 2.5 → cryptic event warning with FISH recommendation."""
    expr = baseline_expr.copy()
    expr["MECOM"] = ref_stats["genes"]["MECOM"]["mean"] + 3.0 * ref_stats["genes"]["MECOM"]["std"]
    rows, _ = compute_expression_outliers(expr, mutated_genes=set())
    mecom = next(r for r in rows if r.gene == "MECOM")
    assert mecom.z_score >= 2.5
    assert "野生型但表达异常高" in mecom.note or "inv(3)" in mecom.note or "EVI1" in mecom.note


def test_tp53_low_flags_biallelic_possibility(ref_stats, baseline_expr):
    """TP53 z < -1.5 → hint about possible bi-allelic loss."""
    expr = baseline_expr.copy()
    expr["TP53"] = ref_stats["genes"]["TP53"]["mean"] - 2.5 * ref_stats["genes"]["TP53"]["std"]
    rows, _ = compute_expression_outliers(expr, mutated_genes=set())
    tp53 = next(r for r in rows if r.gene == "TP53")
    assert tp53.z_score <= -1.5
    assert "bi-allelic" in tp53.note or "del(17p)" in tp53.note


def test_hint_only_gene_with_high_bcl2_shows_venetoclax_note(ref_stats, baseline_expr):
    expr = baseline_expr.copy()
    expr["BCL2"] = ref_stats["genes"]["BCL2"]["mean"] + 2.0 * ref_stats["genes"]["BCL2"]["std"]
    rows, _ = compute_expression_outliers(expr, mutated_genes=set())
    bcl2 = next(r for r in rows if r.gene == "BCL2")
    assert bcl2.dna_status == "hint-only"
    assert bcl2.z_score >= 1.5
    assert "Venetoclax" in bcl2.note


def test_flat_rows_have_no_hint_text_to_avoid_noise(baseline_expr):
    """Genes near z=0 should NOT surface their 'what-if' hint text."""
    rows, _ = compute_expression_outliers(baseline_expr, mutated_genes=set())
    for r in rows:
        if r.z_score is not None and abs(r.z_score) < 0.5:
            assert r.note == "", f"{r.gene} has stray note despite z={r.z_score}: {r.note}"


def test_missing_gene_marked_na_not_crashed(baseline_expr):
    """Genes absent from input should render as 'n/a' rows, not crash."""
    sparse = baseline_expr.drop(["FLT3", "MECOM"], errors="ignore")
    rows, _ = compute_expression_outliers(sparse, mutated_genes=set())
    flt3 = next(r for r in rows if r.gene == "FLT3")
    assert flt3.z_score is None
    assert flt3.direction == "n/a"
    assert not flt3.available


def test_raw_count_input_is_auto_transformed(ref_stats):
    """A raw-count-style input should auto-transform and still compute reasonable z."""
    # Construct raw-count-style input by un-log-transforming BeatAML means
    raw = pd.Series({g: 2.0 ** s["mean"] - 1.0
                      for g, s in ref_stats["genes"].items()})
    rows, meta = compute_expression_outliers(raw, mutated_genes=set())
    available = [r for r in rows if r.available]
    assert len(available) >= 25
    # Should land approximately at z=0
    abs_zs = [abs(r.z_score) for r in available]
    assert np.median(abs_zs) < 1.0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_markdown_has_table_headers(baseline_expr):
    md = build_rnaseq_outlier_markdown(baseline_expr, mutated_genes=set())
    assert "| Tier 组 |" in md
    assert "基因" in md and "DNA 状态" in md and "z-score" in md


def test_markdown_includes_legend(baseline_expr):
    md = build_rnaseq_outlier_markdown(baseline_expr, mutated_genes=set())
    assert "↑↑↑" in md and "↓↓↓" in md
    assert "双证据支持" in md or "野生型但表达异常高" in md


def test_markdown_empty_input_shows_graceful_message():
    md = build_rnaseq_outlier_markdown(None, mutated_genes={"FLT3"})
    assert "RNA-Seq" in md and ("未提供" in md or "为空" in md)


def test_markdown_bolds_outlier_genes(ref_stats, baseline_expr):
    expr = baseline_expr.copy()
    expr["MECOM"] = ref_stats["genes"]["MECOM"]["mean"] + 3.0 * ref_stats["genes"]["MECOM"]["std"]
    md = build_rnaseq_outlier_markdown(expr, mutated_genes=set())
    assert "**MECOM**" in md


# ---------------------------------------------------------------------------
# Integration with kit_predict
# ---------------------------------------------------------------------------


def test_kit_output_populates_rna_outlier_when_provided(baseline_expr, ref_stats):
    from combo_val.clinical.kit_predict import predict_for_patient
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    import joblib

    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    rng = np.random.default_rng(0)
    rna = pd.Series(rng.lognormal(4.0, 1.2, size=len(bundle["kept_genes"])),
                     index=bundle["kept_genes"])

    expr = baseline_expr.copy()
    expr["FLT3"] = ref_stats["genes"]["FLT3"]["mean"] + 2.0 * ref_stats["genes"]["FLT3"]["std"]
    kit = KitInput(
        patient_id="TEST", age=50, sex="male",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6,
                                 vaf=0.4)],
        rna_expression_full=expr,
    )
    out = predict_for_patient(rna, kit)
    assert hasattr(out, "rna_outlier")
    assert out.rna_outlier and "rows" in out.rna_outlier
    rows = out.rna_outlier["rows"]
    assert len(rows) >= 30
    flt3 = next(r for r in rows if r["gene"] == "FLT3")
    assert flt3["z_score"] >= 1.5


def test_kit_output_empty_rna_outlier_when_not_provided():
    from combo_val.clinical.kit_predict import predict_for_patient
    from combo_val.clinical.kit_schema import KitInput, MutationCall
    import joblib

    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    rng = np.random.default_rng(0)
    rna = pd.Series(rng.lognormal(4.0, 1.2, size=len(bundle["kept_genes"])),
                     index=bundle["kept_genes"])

    kit = KitInput(
        patient_id="TEST", age=50, sex="male",
        mutations=[MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.6,
                                 vaf=0.4)],
        # rna_expression_full NOT provided
    )
    out = predict_for_patient(rna, kit)
    assert hasattr(out, "rna_outlier")
    # All rows should be flagged unavailable
    rows = out.rna_outlier.get("rows", [])
    assert all(not r.get("available", False) for r in rows)
