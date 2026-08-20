"""Tests for the defensive AML Virtual Cell v1.6 perturbation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.virtual_cell.challenge_firewall import (
    assert_label_free,
    find_outcome_columns,
    split_challenge_table,
)
from combo_val.virtual_cell.perturbation_v16 import (
    assess_prediction_support,
    audit_embedding_discriminability,
    grouped_pair_split,
    require_valid_synergy_model,
    state_aware_aggregate,
)


def test_firewall_rejects_observed_response_but_allows_prediction() -> None:
    assert_label_free(pd.DataFrame({"predicted_auc": [1.0]}), allow_predicted=True)
    with pytest.raises(ValueError, match="Response"):
        assert_label_free(
            pd.DataFrame({"drug": ["A"], "Response": [0.8]}),
            allow_predicted=False,
        )


def test_patient_label_is_identity_not_outcome() -> None:
    detected = find_outcome_columns(
        ["patient_label", "prediction_label", "cell_inhibition_pct"],
        allow_predicted=False,
    )
    assert "patient_label" not in detected
    assert set(detected) == {"prediction_label", "cell_inhibition_pct"}


def test_challenge_split_removes_labels_and_aligns_row_ids() -> None:
    source = pd.DataFrame(
        {
            "Patient": ["Patient5", "Patient5"],
            "Drug1": ["A", "A"],
            "Drug2": ["B", "B"],
            "Dose1": [1.0, 2.0],
            "Dose2": [1.0, 2.0],
            "Response": [0.2, 0.8],
        }
    )
    split = split_challenge_table(
        source,
        identity_columns=["Patient", "Drug1", "Drug2", "Dose1", "Dose2"],
        outcome_columns=["Response"],
        prefix="combo",
    )
    assert "Response" not in split.public.columns
    assert list(split.sealed.columns) == ["challenge_row_id", "Response"]
    assert split.public["challenge_row_id"].is_unique
    assert split.public["challenge_row_id"].equals(split.sealed["challenge_row_id"])


def test_cross_platform_directional_support_is_not_full_support() -> None:
    support = assess_prediction_support(
        qc_severity="pipeline_mismatch",
        gene_coverage_fraction=0.667,
        molecular_similarity=1.0,
        model_in_vocabulary=True,
    )
    assert support.level == "directional"
    assert any("gene coverage" in reason for reason in support.reasons)


@pytest.mark.parametrize(
    ("qc", "coverage", "similarity"),
    [
        ("ood", 0.95, 1.0),
        ("ok", 0.50, 1.0),
        ("ok", 0.95, 0.20),
    ],
)
def test_support_gate_rejects_unsafe_inputs(
    qc: str,
    coverage: float,
    similarity: float,
) -> None:
    support = assess_prediction_support(
        qc_severity=qc,
        gene_coverage_fraction=coverage,
        molecular_similarity=similarity,
        model_in_vocabulary=False,
    )
    assert support.level == "rejected"


def test_state_aware_aggregate_uses_malignant_weighted_state_mass() -> None:
    states = pd.DataFrame(
        {
            "patient_id": ["patient5", "patient5"],
            "drug_id": ["DrugA", "DrugA"],
            "predicted_auc": [10.0, 20.0],
            "state_fraction": [0.6, 0.4],
            "malignant_probability": [0.5, 1.0],
        }
    )
    result = state_aware_aggregate(states)
    expected = (10.0 * 0.3 + 20.0 * 0.4) / 0.7
    assert result.loc[0, "predicted_auc"] == pytest.approx(expected)
    assert result.loc[0, "malignant_weight_mass"] == pytest.approx(0.7)
    assert result.loc[0, "between_state_sd"] > 0


def test_grouped_split_has_zero_unordered_pair_overlap() -> None:
    frame = pd.DataFrame(
        {
            "drug_a": ["A", "B", "A", "C", "D", "E", "D", "F"],
            "drug_b": ["B", "A", "C", "A", "E", "D", "F", "D"],
            "cell": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        }
    )
    train, validation, audit = grouped_pair_split(
        frame,
        drug_a_column="drug_a",
        drug_b_column="drug_b",
        validation_fraction=0.25,
        random_state=7,
    )
    train_pairs = {
        tuple(sorted(pair))
        for pair in frame.iloc[train][["drug_a", "drug_b"]].itertuples(
            index=False,
            name=None,
        )
    }
    validation_pairs = {
        tuple(sorted(pair))
        for pair in frame.iloc[validation][["drug_a", "drug_b"]].itertuples(
            index=False,
            name=None,
        )
    }
    assert train_pairs.isdisjoint(validation_pairs)
    assert audit.pair_overlap_count == 0


def test_embedding_collapse_disables_synergy_head() -> None:
    embeddings = np.zeros((11, 8), dtype=float)
    embeddings[-1, 0] = 1.0
    predictions = np.full(55, 2.030191)
    audit = audit_embedding_discriminability(
        embeddings,
        predictions=predictions,
    )
    assert not audit.enabled
    assert audit.unique_embedding_fraction == pytest.approx(2 / 11)
    with pytest.raises(RuntimeError, match="disabled"):
        require_valid_synergy_model(audit)


def test_discriminative_embeddings_pass_gate() -> None:
    embeddings = np.eye(12, dtype=float)
    predictions = np.linspace(-4.0, 4.0, 66)
    audit = audit_embedding_discriminability(
        embeddings,
        predictions=predictions,
    )
    assert audit.enabled
    require_valid_synergy_model(audit)
