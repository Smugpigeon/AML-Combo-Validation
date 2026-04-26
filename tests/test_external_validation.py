"""Tests for the external-validation framework.

Per concern #2 (post-deployment clinical reviewer): the framework must
work on a synthetic cohort end-to-end so that adopting institutions can
trust the pipeline before pointing it at real patient data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combo_val.validation.external_validation import (
    ExternalValidationConfig,
    REQUIRED_COLUMNS,
    compute_brier_score,
    compute_ece,
    compute_eln_concordance,
    compute_regimen_top1_match,
    run_external_validation,
)


def _make_synthetic_cohort(n: int = 100, seed: int = 0,
                           agreement: float = 0.7) -> pd.DataFrame:
    """Synthetic cohort: ELN classes with controllable agreement,
    plus CR outcomes consistent with ELN priors."""
    rng = np.random.default_rng(seed)
    classes = ["Favorable", "Intermediate", "Adverse"]
    true_eln = rng.choice(classes, size=n, p=[0.3, 0.5, 0.2])
    pred_eln = []
    for tc in true_eln:
        if rng.random() < agreement:
            pred_eln.append(tc)
        else:
            others = [c for c in classes if c != tc]
            pred_eln.append(rng.choice(others))

    eln_cr_rate = {"Favorable": 0.80, "Intermediate": 0.55, "Adverse": 0.30}
    true_cr = np.array([
        1 if rng.random() < eln_cr_rate[tc] else 0 for tc in true_eln
    ])

    return pd.DataFrame({
        "patient_id": [f"SYN-{i:04d}" for i in range(n)],
        "predicted_eln_2022": pred_eln,
        "true_eln_class": true_eln,
        "predicted_top1_id": rng.choice(
            ["7_3_midostaurin", "ven_aza", "atra_ato_low", "ivo_mono"], size=n,
        ),
        "true_regimen_id": rng.choice(
            ["7_3_midostaurin", "ven_aza", "atra_ato_low", "ivo_mono"], size=n,
        ),
        "true_cr": true_cr,
        "true_os_months": rng.exponential(scale=18, size=n),
        "true_os_event": rng.binomial(1, 0.5, size=n),
        "age": rng.integers(20, 85, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def test_brier_score_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_brier_score(y_true, y_prob) == 0.0


def test_brier_score_random_balanced():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.5, size=10000)
    y_prob = np.full(10000, 0.5)
    score = compute_brier_score(y_true, y_prob)
    # Random guess on balanced data: Brier ≈ 0.25
    assert 0.24 < score < 0.26


def test_ece_perfect_calibration():
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    assert compute_ece(y_true, y_prob) == 0.0


def test_eln_concordance_perfect():
    pred = pd.Series(["Favorable", "Intermediate", "Adverse"] * 10)
    true = pred.copy()
    m = compute_eln_concordance(pred, true)
    assert m["accuracy"] == 1.0
    assert m["cohens_kappa"] == 1.0
    assert "almost perfect" in m["interpretation_kappa"]


def test_eln_concordance_random():
    rng = np.random.default_rng(0)
    classes = ["Favorable", "Intermediate", "Adverse"]
    pred = pd.Series(rng.choice(classes, size=300))
    true = pd.Series(rng.choice(classes, size=300))
    m = compute_eln_concordance(pred, true)
    # Random agreement on 3 classes with marginals ~ 0.33 each → kappa near 0
    assert -0.15 < m["cohens_kappa"] < 0.15


def test_regimen_top1_match_returns_rate():
    # Position-wise: mid×5 vs (mid×4, ven, ven×4, atra×2)
    # → matches: idx 0-3 (mid=mid), idx 5-7 (ven=ven) = 7 matches; idx 4,8,9 mismatch
    pred = pd.Series(["7_3_midostaurin"] * 5 + ["ven_aza"] * 5)
    true = pd.Series(["7_3_midostaurin"] * 4 + ["ven_aza"] * 4 + ["atra_ato_low"] * 2)
    m = compute_regimen_top1_match(pred, true)
    assert m["n_with_truth"] == 10
    assert m["match_rate"] == 0.7


def test_regimen_top1_match_handles_missing_truth():
    pred = pd.Series(["7_3_midostaurin"] * 3)
    true = pd.Series([np.nan, np.nan, np.nan])
    m = compute_regimen_top1_match(pred, true)
    assert m["match_rate"] is None


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_run_external_validation_end_to_end(tmp_path):
    """Synthesize a 100-patient cohort with ELN agreement ~0.85, write CSV,
    run the full pipeline, verify outputs."""
    df = _make_synthetic_cohort(n=200, seed=42, agreement=0.85)
    cohort_csv = tmp_path / "synthetic_cohort.csv"
    df.to_csv(cohort_csv, index=False)

    cfg = ExternalValidationConfig(
        cohort_csv=cohort_csv,
        out_dir=tmp_path / "results",
        bootstrap_n=200,  # smaller for test speed
        institution_name="Synthetic Test Cohort",
        cohort_description="Smoke-test cohort with ~85% ELN agreement",
    )
    metrics = run_external_validation(cfg)

    # Output files exist
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "results" / "calibration_plot.png").exists()
    assert (tmp_path / "results" / "eln_confusion.png").exists()
    assert (tmp_path / "results" / "REPORT.html").exists()
    assert (tmp_path / "results" / "regimen_match_summary.md").exists()

    # Metric sanity
    assert metrics["n_patients"] == 200
    # 85% agreement → kappa should be substantial / almost perfect
    assert metrics["eln_concordance"]["cohens_kappa"] > 0.6
    assert metrics["eln_concordance"]["accuracy"] > 0.75

    # Calibration computed
    cal = metrics["calibration_via_eln_prior"]
    assert "brier_score" in cal
    assert 0.0 < cal["brier_score"] < 0.4

    # HTML report is self-contained (base64 PNGs)
    html = (tmp_path / "results" / "REPORT.html").read_text(encoding="utf-8")
    assert "data:image/png;base64" in html
    assert "Synthetic Test Cohort" in html


def test_missing_required_columns_raises(tmp_path):
    """Bad CSV → clear error, not a cryptic stack trace."""
    df = pd.DataFrame({"patient_id": ["x"], "true_cr": [1]})
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)

    cfg = ExternalValidationConfig(cohort_csv=csv, out_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="missing required columns"):
        run_external_validation(cfg)


def test_pass_criteria_documented_in_summary(tmp_path):
    """The summary markdown must spell out the pass criteria so reviewers
    don't have to dig in the source for the threshold."""
    df = _make_synthetic_cohort(n=100, seed=0, agreement=0.9)
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    cfg = ExternalValidationConfig(
        cohort_csv=csv, out_dir=tmp_path / "out",
        bootstrap_n=100, institution_name="X",
    )
    run_external_validation(cfg)

    md = (tmp_path / "out" / "regimen_match_summary.md").read_text(encoding="utf-8")
    assert "Pass criteria" in md
    assert "kappa" in md.lower()
    assert "Brier" in md
    assert "0.60" in md or "0.6" in md
    assert "0.20" in md or "0.2" in md


def test_required_columns_constant_includes_minimum_fields():
    assert "patient_id" in REQUIRED_COLUMNS
    assert "predicted_eln_2022" in REQUIRED_COLUMNS
    assert "true_eln_class" in REQUIRED_COLUMNS
    assert "true_cr" in REQUIRED_COLUMNS
