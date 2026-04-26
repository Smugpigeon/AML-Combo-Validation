"""Tests for prospective-validation infrastructure (v0.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from combo_val.prospective.locked_prediction import (
    ChainVerificationResult,
    LockedPrediction,
    lock_prediction,
    verify_chain,
)
from combo_val.prospective.outcome_capture import (
    Day30Outcome,
    Month12Outcome,
    Month6Outcome,
    assemble_cohort_for_analysis,
    read_outcome,
    write_outcome,
)
from combo_val.prospective.registry import (
    Site,
    add_site,
    assert_site_active,
    deactivate_site,
    get_site,
    list_active_sites,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_add_get_site(tmp_path):
    p = tmp_path / "site_registry.json"
    s = Site(
        site_id="TEST_IRB_2026_001",
        institution="Test Hospital",
        country="CN",
        principal_investigator="Dr. Test",
        irb_protocol="2026-01",
        irb_approval_date="2026-04-01",
        target_n=50,
        enrolled_at="2026-04-01",
    )
    add_site(s, path=p)
    assert get_site("TEST_IRB_2026_001", path=p).institution == "Test Hospital"
    actives = list_active_sites(path=p)
    assert len(actives) == 1


def test_registry_deactivate_preserves_history(tmp_path):
    p = tmp_path / "reg.json"
    s = Site(
        site_id="X", institution="X", country="CN",
        principal_investigator="X", irb_protocol="X",
        irb_approval_date="2026-01-01", target_n=10, enrolled_at="2026-01-01",
    )
    add_site(s, path=p)
    deactivate_site("X", reason="study closed", path=p)
    site = get_site("X", path=p)
    assert site.active is False
    assert "DEACTIVATED" in site.notes
    assert list_active_sites(path=p) == []


def test_registry_assert_active_rejects_unknown(tmp_path):
    p = tmp_path / "r.json"
    with pytest.raises(ValueError, match="not in prospective-validation registry"):
        assert_site_active("UNKNOWN_SITE", path=p)


def test_registry_assert_active_rejects_deactivated(tmp_path):
    p = tmp_path / "r.json"
    add_site(Site(site_id="A", institution="A", country="CN",
                   principal_investigator="A", irb_protocol="A",
                   irb_approval_date="2026-01-01", target_n=1,
                   enrolled_at="2026-01-01"), path=p)
    deactivate_site("A", reason="test", path=p)
    with pytest.raises(ValueError, match="deactivated"):
        assert_site_active("A", path=p)


# ---------------------------------------------------------------------------
# Locked predictions (immutable hash chain)
# ---------------------------------------------------------------------------


def _fake_kit_input() -> dict:
    return {
        "patient_id": "TEST-001",
        "mutations": [{"gene": "FLT3", "is_ITD": True, "allelic_ratio": 0.62}],
        "karyotype_text": "46,XX[20]",
        "wbc": 95.0,
        "age": 45,
    }


def _fake_kit_output() -> dict:
    return {
        "predicted_eln2017": "Intermediate",
        "eln_2022": {"category": "Intermediate", "rationale": ["NPM1+FLT3-ITD"]},
        "top_combinations": [
            {"rank": 1, "drug1": "Venetoclax", "drug2": "Azacitidine",
             "predicted_combo_auc": 88.2, "mech_score": 0.42,
             "clonal_coverage_score": 0.71},
        ],
        "top_regimens": [
            {"regimen_id": "7_3_midostaurin", "name": "7+3 + Midostaurin"},
            {"regimen_id": "7_3_quizartinib", "name": "7+3 + Quizartinib"},
        ],
    }


def test_lock_prediction_writes_record_and_chain(tmp_path):
    rec = lock_prediction(
        runs_root=tmp_path,
        site_id="TEST_SITE",
        patient_id="P-001",
        enrolled_at="2026-04-26T10:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "no-such-ckpt.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    # Record file exists
    site_dir = tmp_path / "prospective" / "TEST_SITE" / "P-001"
    locks = list(site_dir.glob("*_locked_prediction.json"))
    assert len(locks) == 1
    inputs = list(site_dir.glob("*_input.json"))
    assert len(inputs) == 1
    # Chain index has one entry
    idx = tmp_path / "prospective" / "TEST_SITE" / "_chain.jsonl"
    assert idx.exists()
    with open(idx) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1
    # Record has self_hash + prev_hash
    assert rec.self_hash != ""
    assert rec.prev_hash == ""  # first in chain
    assert rec.predicted_top1_regimen_id == "7_3_midostaurin"
    # checkpoint missing → "missing"
    assert rec.model_checkpoint_sha256 == "missing"


def test_lock_prediction_chains_correctly(tmp_path):
    """Two predictions for the same site → second's prev_hash == first's self_hash."""
    rec1 = lock_prediction(
        runs_root=tmp_path, site_id="S1", patient_id="P1",
        enrolled_at="2026-04-26T10:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "x.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    rec2 = lock_prediction(
        runs_root=tmp_path, site_id="S1", patient_id="P2",
        enrolled_at="2026-04-26T11:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "x.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    assert rec2.prev_hash == rec1.self_hash
    assert rec1.self_hash != rec2.self_hash


def test_verify_chain_passes_on_intact_chain(tmp_path):
    for i in range(3):
        lock_prediction(
            runs_root=tmp_path, site_id="S", patient_id=f"P{i}",
            enrolled_at="2026-04-26T10:00:00+00:00",
            kit_version="v0.5.0",
            model_checkpoint_path=tmp_path / "x.pt",
            backbone="mlp",
            kit_input_dict=_fake_kit_input(),
            kit_output_dict=_fake_kit_output(),
        )
    result = verify_chain(tmp_path, "S")
    assert isinstance(result, ChainVerificationResult)
    assert result.n_records == 3
    assert result.chain_intact is True
    assert result.failures == []


def test_verify_chain_detects_tampering(tmp_path):
    """Modify a locked record after-the-fact → chain verification fails."""
    rec = lock_prediction(
        runs_root=tmp_path, site_id="S", patient_id="P0",
        enrolled_at="2026-04-26T10:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "x.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    # Tamper: change the predicted top-1 in the saved file
    site_dir = tmp_path / "prospective" / "S" / "P0"
    rec_path = list(site_dir.glob("*_locked_prediction.json"))[0]
    with open(rec_path) as f:
        d = json.load(f)
    d["predicted_top1_regimen_id"] = "ven_aza"  # forged tampering
    with open(rec_path, "w") as f:
        json.dump(d, f)
    result = verify_chain(tmp_path, "S")
    assert result.chain_intact is False
    assert any("self_hash mismatch" in m for m in result.failures)


def test_verify_chain_empty_site(tmp_path):
    result = verify_chain(tmp_path, "EMPTY_SITE")
    assert result.n_records == 0
    assert result.chain_intact is True


# ---------------------------------------------------------------------------
# Outcome capture
# ---------------------------------------------------------------------------


def test_write_and_read_day30_outcome(tmp_path):
    outc = Day30Outcome(
        prediction_id="abc", site_id="S1", patient_id="P1",
        assessment_date="2026-05-26", days_from_diagnosis=30,
        actual_regimen_id="7_3_midostaurin",
        cr_status="CR",
        bone_marrow_blast_pct=2.0,
    )
    p = write_outcome(tmp_path, "day30", outc)
    assert p.exists()
    loaded = read_outcome(tmp_path, "S1", "P1", "day30")
    assert loaded["cr_status"] == "CR"
    assert loaded["bone_marrow_blast_pct"] == 2.0


def test_write_outcome_unknown_timepoint_raises(tmp_path):
    outc = Day30Outcome(
        prediction_id="x", site_id="S", patient_id="P",
        assessment_date="2026-05-26", days_from_diagnosis=30,
        actual_regimen_id="7_3",
    )
    with pytest.raises(ValueError, match="Unknown timepoint"):
        write_outcome(tmp_path, "month42", outc)


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def test_assemble_cohort_joins_locked_with_outcome(tmp_path):
    # 1) Lock a prediction
    lock_prediction(
        runs_root=tmp_path, site_id="S1", patient_id="P1",
        enrolled_at="2026-04-26T10:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "x.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    # 2) Write the 12-month outcome
    outc = Month12Outcome(
        prediction_id="any", site_id="S1", patient_id="P1",
        assessment_date="2027-04-26", days_from_diagnosis=365,
        alive=True,
        cr_status_at_12mo="CR",
        mrd_status_at_12mo="MRD_neg",
    )
    write_outcome(tmp_path, "month12", outc)

    # 3) Assemble cohort
    rows = assemble_cohort_for_analysis(tmp_path, require_timepoint="month12")
    assert len(rows) == 1
    r = rows[0]
    assert r["patient_id"] == "P1"
    assert r["predicted_top1_id"] == "7_3_midostaurin"
    assert r["true_cr"] == 1  # CR → 1
    assert r["true_os_event"] == 0  # alive → 0
    assert r["kit_commit_hash"]  # set
    assert r["backbone"] == "mlp"


def test_assemble_cohort_skips_patients_without_outcome(tmp_path):
    lock_prediction(
        runs_root=tmp_path, site_id="S1", patient_id="P1",
        enrolled_at="2026-04-26T10:00:00+00:00",
        kit_version="v0.5.0",
        model_checkpoint_path=tmp_path / "x.pt",
        backbone="mlp",
        kit_input_dict=_fake_kit_input(),
        kit_output_dict=_fake_kit_output(),
    )
    # No outcome filed yet
    rows = assemble_cohort_for_analysis(tmp_path, require_timepoint="month12")
    assert rows == []
