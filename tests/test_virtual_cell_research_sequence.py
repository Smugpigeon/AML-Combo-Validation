from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.virtual_cell.cell_identity import (
    build_retrospective_identity_gate,
    require_retrospective_identity_gate_summary,
)
from combo_val.virtual_cell.challenge_firewall import assert_label_free
from combo_val.virtual_cell.retrospective_validation import (
    CombinationTrainingPolicy,
    audit_combination_training_readiness,
    combination_unlock_decision,
    evaluate_monotherapy_predictions,
    extract_monotherapy_edges,
    model_drug_name,
    summarize_monotherapy_edges,
)
from combo_val.virtual_cell.stage_review import build_stage_review


def _identity_cells() -> pd.DataFrame:
    malignant = ["malignant"] * 80
    healthy = ["healthy"] * 20
    return pd.DataFrame(
        {
            "sample_id": ["p1"] * 100,
            "virtual_state_id": ["VS01"] * 100,
            "known_normal_reference": [False] * 80 + [True] * 20,
            "sctype_malignant_healthy": malignant + healthy,
            "copyKat_output": malignant + healthy,
            "SCEVAN_output": malignant + healthy,
        }
    )


def test_rna_ensemble_can_unlock_only_retrospective_validation() -> None:
    _, patients, states, summary = build_retrospective_identity_gate(
        _identity_cells(),
        pd.DataFrame({"patient_id": ["p1"], "scrna_blast_pct": [80.0]}),
    )
    assert patients.loc[0, "retrospective_identity_gate_pass"]
    assert states.loc[0, "retrospective_state_identity_pass"]
    assert summary["retrospective_drug_validation_stage_unlocked"]
    assert not summary["strict_orthogonal_identity_gate_pass"]
    assert not summary["clinical_identity_stage_unlocked"]
    require_retrospective_identity_gate_summary(summary, patient_ids=["p1"])


def test_rna_ensemble_disagreement_keeps_retrospective_gate_locked() -> None:
    cells = _identity_cells()
    cells["copyKat_output"] = np.where(
        cells["sctype_malignant_healthy"].eq("malignant"), "healthy", "malignant"
    )
    cells["SCEVAN_output"] = "Unknown"
    _, patients, _, summary = build_retrospective_identity_gate(
        cells,
        pd.DataFrame({"patient_id": ["p1"], "scrna_blast_pct": [80.0]}),
    )
    assert not patients.loc[0, "retrospective_identity_gate_pass"]
    with pytest.raises(RuntimeError, match="RNA identity ensemble"):
        require_retrospective_identity_gate_summary(summary, patient_ids=["p1"])


def test_zero_dose_edges_are_real_monotherapy_only() -> None:
    combo = pd.DataFrame(
        {
            "Patient": [5, 5, 5, 5, 5],
            "Drug1": ["Trametinib"] * 5,
            "Drug2": ["Dasatinib"] * 5,
            "Dose1": [10, 10, 0, 0, 10],
            "Dose2": [0, 0, 20, 0, 20],
            "DoseUnit": ["nM"] * 5,
            "Response": [40, 40, 60, 0, 85],
        }
    )
    edges = extract_monotherapy_edges(combo)
    assert len(edges) == 2
    assert edges["repeated_edge_count"].max() == 2
    assert edges["repeated_edge_range"].max() == 0
    assert not (edges["observed_inhibition"] == 85).any()
    assert model_drug_name("Trametinib") == "Trametinib (GSK1120212)"


def test_monotherapy_summary_requires_a_dose_curve() -> None:
    edges = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "drug_id": ["Trametinib", "Trametinib"],
            "model_drug_id": ["Trametinib (GSK1120212)"] * 2,
            "dose": [1.0, 100.0],
            "dose_unit": ["nM", "nM"],
            "observed_inhibition": [10.0, 70.0],
            "repeated_edge_range": [0.0, 0.0],
        }
    )
    summary = summarize_monotherapy_edges(edges)
    assert len(summary) == 1
    assert 0 < summary.loc[0, "observed_dss_like_0_1um"] < 100
    assert summary.loc[0, "n_positive_doses_le_1um"] == 2


def test_patient_specific_monotherapy_gate_beats_drug_mean_baseline() -> None:
    outcomes_by_patient = {
        "p1": [1.0, 2.0, 3.0, 4.0],
        "p2": [2.0, 4.0, 1.0, 3.0],
        "p3": [4.0, 1.0, 3.0, 2.0],
    }
    prediction_rows = []
    outcome_rows = []
    for patient_id, outcomes in outcomes_by_patient.items():
        for index, observed in enumerate(outcomes):
            row_id = f"{patient_id}_{index}"
            prediction_rows.append(
                {
                    "challenge_row_id": row_id,
                    "patient_id": patient_id,
                    "drug_id": f"drug{index}",
                    "predicted_sensitivity_score": observed,
                    "drug_mean_sensitivity_score": float(index + 1),
                }
            )
            outcome_rows.append(
                {
                    "challenge_row_id": row_id,
                    "observed_dss_like_0_1um": observed,
                    "maximum_repeated_edge_range": 0.0,
                }
            )
    metrics, summary = evaluate_monotherapy_predictions(
        pd.DataFrame(prediction_rows), pd.DataFrame(outcome_rows)
    )
    assert len(metrics) == 3
    assert summary["viability_direction_gate_pass"]
    assert summary["median_patient_specific_spearman"] == pytest.approx(1.0)
    decision = combination_unlock_decision(
        {"retrospective_drug_validation_stage_unlocked": True}, summary
    )
    assert decision["combination_benchmark_unlocked"]
    assert not decision["combination_training_unlocked"]
    assert not decision["patient_specific_combination_prediction_unlocked"]
    assert not decision["clinical_combination_use_unlocked"]


def test_combination_training_requires_data_readiness_and_validation() -> None:
    rows = []
    for patient in range(1, 5):
        for pair_index, (drug1, drug2) in enumerate(
            [("A", "B"), ("A", "C")], start=1
        ):
            rows.append(
                {
                    "Patient": patient,
                    "Drug1": drug1,
                    "Drug2": drug2,
                    "Dose1": 10.0,
                    "Dose2": 20.0,
                    "Response": float(5 * patient + pair_index),
                }
            )
    readiness = audit_combination_training_readiness(
        pd.DataFrame(rows),
        policy=CombinationTrainingPolicy(
            minimum_patients=4,
            minimum_unique_unordered_pairs=2,
            minimum_combination_wells=8,
            minimum_median_patients_per_pair=4,
            maximum_single_patient_well_fraction=0.25,
            minimum_response_standard_deviation=1.0,
        ),
    )
    assert readiness["combination_training_data_ready"]
    decision = combination_unlock_decision(
        {"retrospective_drug_validation_stage_unlocked": True},
        {"viability_direction_gate_pass": True},
        readiness,
        {"patient_specific_combination_validation_pass": True},
    )
    assert decision["combination_training_unlocked"]
    assert decision["patient_specific_combination_prediction_unlocked"]
    assert not decision["clinical_combination_use_unlocked"]


def test_firewall_rejects_derived_observed_outcomes() -> None:
    with pytest.raises(ValueError, match="observed-outcome"):
        assert_label_free(pd.DataFrame({"observed_dss_like_0_1um": [20.0]}))


def test_adversarial_review_does_not_turn_rna_ensemble_into_clinical_proof() -> None:
    review = build_stage_review(
        "identity",
        {
            "retrospective_drug_validation_stage_unlocked": True,
            "n_cells": 100,
            "patients": ["p1"],
        },
    )
    assert review["current_stronger_side"] == "conditional_support"
    assert "retrospective" in review["neutral_verdict"]
    assert "clinical" in review["neutral_verdict"]
