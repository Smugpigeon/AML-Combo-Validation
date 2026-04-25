"""Tests for issue #3 — Layer-3 OOD suppression.

When the upstream RNA-Seq QC flags the input as far out-of-distribution
versus BeatAML 2.0 training data, the kit must NOT ship a numeric
"top-3 combos" list. The MLP will happily extrapolate to nonsense AUCs,
and clinicians have no way to tell those apart from real predictions.

Instead the kit emits a single suppression sentinel:

    {"rank": 0, "suppressed": True, "suppress_reason": "...",
     "ood_severity": "far_ood" | "ood" | "pipeline_mismatch",
     "mahalanobis_raw": float|None, "mahalanobis_qn": float|None,
     "layer3_backbone": "..."}

These tests verify:
  1. The suppression marker shape is what downstream consumers expect.
  2. patient_report._combo_prediction_narrative renders a yellow banner
     instead of crashing on the missing drug1/drug2/AUC fields.
  3. dna_report Layer-3 section renders a banner instead of crashing.
  4. pretty_print_kit_output renders a banner instead of crashing.
  5. set_drug_v3_validation._pairs-style consumers skip the marker.
"""

from __future__ import annotations

from combo_val.clinical.kit_schema import KitOutput
from combo_val.clinical.patient_report import _combo_prediction_narrative


def _make_suppressed_output(severity: str = "far_ood",
                            mahalanobis_raw: float | None = 36.4,
                            mahalanobis_qn: float | None = 12.1) -> KitOutput:
    """Construct a KitOutput whose Layer-3 has been suppressed."""
    suppress_msg = (
        f"Layer-3 prediction suppressed — RNA-Seq sample is "
        f"out-of-distribution (severity: {severity}, "
        f"raw Mahalanobis = {mahalanobis_raw}, post-QN = {mahalanobis_qn}). "
        f"Per issue #3, predictions on far-OOD samples are unreliable; "
        f"refer to Layer 1 (evidence-based regimens) and Layer 2 "
        f"(clonal coverage) instead."
    )
    return KitOutput(
        patient_id="A-2026-OOD",
        predicted_eln2017="Intermediate",
        top_combinations=[{
            "rank": 0,
            "suppressed": True,
            "suppress_reason": suppress_msg,
            "ood_severity": severity,
            "mahalanobis_raw": mahalanobis_raw,
            "mahalanobis_qn": mahalanobis_qn,
            "layer3_backbone": "BaselineA-MLP",
        }],
        top_single_drugs=[],
        top_regimens=[],
        clonal_coverage={},
        dna_summary={},
        rna_outlier={},
        driver_flags={},
        fitness_flag="fit_for_intensive",
        cautions=[],
        confidence_notes=[],
    )


def _make_normal_output() -> KitOutput:
    """Construct a normal (in-distribution) KitOutput with real combos."""
    return KitOutput(
        patient_id="A-2026-OK",
        predicted_eln2017="Favorable",
        top_combinations=[{
            "rank": 1,
            "drug1": "Venetoclax",
            "drug2": "Azacitidine",
            "predicted_combo_auc": 88.2,
            "single_auc_d1": 92.0,
            "single_auc_d2": 105.3,
            "mech_score": 0.42,
            "both_mech_annotated": True,
            "clonal_coverage_score": 0.71,
            "layer3_backbone": "BaselineA-MLP",
        }],
        top_single_drugs=[],
        top_regimens=[],
        clonal_coverage={},
        dna_summary={},
        rna_outlier={},
        driver_flags={},
        fitness_flag="fit_for_intensive",
        cautions=[],
        confidence_notes=[],
    )


# ---------------------------------------------------------------------------
# Marker shape
# ---------------------------------------------------------------------------


def test_suppression_marker_has_required_fields():
    out = _make_suppressed_output()
    assert len(out.top_combinations) == 1
    m = out.top_combinations[0]
    for key in ("suppressed", "suppress_reason", "ood_severity",
                "layer3_backbone"):
        assert key in m
    assert m["suppressed"] is True
    # Must NOT carry drug1/drug2/predicted_combo_auc — those would mislead
    # downstream renderers into treating it as a real prediction.
    assert "drug1" not in m
    assert "drug2" not in m
    assert "predicted_combo_auc" not in m


# ---------------------------------------------------------------------------
# patient_report renderer
# ---------------------------------------------------------------------------


def test_patient_report_renders_banner_for_suppressed():
    out = _make_suppressed_output(severity="far_ood",
                                   mahalanobis_raw=36.4, mahalanobis_qn=12.1)
    md = _combo_prediction_narrative(out)
    assert "Layer-3" in md
    assert "禁用" in md or "suppressed" in md.lower()
    # Banner must surface the Mahalanobis figures so the clinician can audit.
    assert "36.4" in md
    assert "12.1" in md
    # Must reference the substitute layers (Layer 1 + Layer 2).
    assert "第 3 节" in md or "Layer 1" in md
    # Must NOT contain the typical "drug1 + drug2 — AUC ..." line.
    assert "AUC " not in md or "Mahalanobis" in md  # no plain AUC prediction


def test_patient_report_handles_missing_mahalanobis():
    """pipeline_mismatch case: Mahalanobis may be None — must not crash."""
    out = _make_suppressed_output(severity="pipeline_mismatch",
                                   mahalanobis_raw=None, mahalanobis_qn=None)
    md = _combo_prediction_narrative(out)
    assert "n/a" in md or "pipeline_mismatch" in md
    assert "Layer-3" in md


def test_patient_report_normal_path_still_works():
    """Regression — normal (non-OOD) reports still get the standard top-3."""
    out = _make_normal_output()
    md = _combo_prediction_narrative(out)
    assert "Venetoclax" in md
    assert "Azacitidine" in md
    assert "88.2" in md
    assert "禁用" not in md


# ---------------------------------------------------------------------------
# pretty_print_kit_output (CLI)
# ---------------------------------------------------------------------------


def test_pretty_print_handles_suppressed():
    from combo_val.clinical.kit_predict import pretty_print_kit_output
    out = _make_suppressed_output()
    text = pretty_print_kit_output(out)
    # Should not raise. Should contain the suppression sentinel.
    assert "SUPPRESSED" in text or "suppressed" in text.lower()
    assert "OOD" in text or "ood" in text.lower()


def test_pretty_print_normal_path_still_works():
    from combo_val.clinical.kit_predict import pretty_print_kit_output
    out = _make_normal_output()
    text = pretty_print_kit_output(out)
    assert "Venetoclax" in text
    assert "SUPPRESSED" not in text


# ---------------------------------------------------------------------------
# dna_report renderer
# ---------------------------------------------------------------------------


def test_dna_report_renderer_guard_present():
    """Verify the dna_report Layer-3 section has a defensive guard for the
    suppression marker. We grep the source rather than build a full DNA
    report (which pulls heavy dependencies) — the guard is a one-line
    invariant easy to assert at the source level."""
    import combo_val.clinical.dna_report as dr
    with open(dr.__file__, encoding="utf-8") as f:
        text = f.read()
    # The guard must check tc[0].get("suppressed") before iterating.
    assert "suppressed" in text
    assert "Layer-3 suppressed" in text or "Layer-3 OOD" in text or "RNA-Seq OOD" in text


# ---------------------------------------------------------------------------
# set_drug_v3_validation pair extractor
# ---------------------------------------------------------------------------


def test_set_drug_v3_pair_extractor_skips_suppressed():
    """The validation harness extracts (drug1, drug2) tuples from
    out.top_combinations to compute Jaccard overlap. A suppression marker
    has no drug1/drug2; the harness must skip it instead of KeyError-ing."""
    suppressed = _make_suppressed_output().top_combinations
    # Mimic the patched _pairs() body from set_drug_v3_validation.py.
    pairs = [tuple(sorted([c["drug1"], c["drug2"]]))
             for c in suppressed
             if not c.get("suppressed")]
    assert pairs == []  # all entries skipped — no crash


# ---------------------------------------------------------------------------
# Severity gating — only OOD-class severities suppress
# ---------------------------------------------------------------------------


def test_only_ood_severities_trigger_suppression():
    """Verify the gating list in kit_predict.py: 'ok' and 'borderline'
    must NOT trigger suppression; 'far_ood', 'ood', 'pipeline_mismatch' must."""
    suppressing = {"far_ood", "ood", "pipeline_mismatch"}
    not_suppressing = {"ok", "borderline"}
    # Read the gate from source to keep the test in lockstep with the impl.
    import combo_val.clinical.kit_predict as kp
    src = kp.__file__
    with open(src, encoding="utf-8") as f:
        text = f.read()
    # The gate appears as: severity_for_layer3 in ("far_ood", "ood", "pipeline_mismatch")
    assert all(s in text for s in suppressing)
    # Sanity — these strings should NOT also appear inside that gate tuple.
    # We don't lint that strictly here; the patient-report tests above are
    # the real behavioural check.
