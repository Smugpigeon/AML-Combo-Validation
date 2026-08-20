from __future__ import annotations

import pandas as pd
import pytest

from combo_val.virtual_cell.cell_identity import (
    build_identity_gate,
    require_identity_gate_summary,
)


def base_cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["p1"] * 6,
            "virtual_state_id": ["VS01"] * 4 + ["VS02"] * 2,
            "consensus_lineage": ["stem_progenitor"] * 4 + ["T_cell"] * 2,
            "lineage_confidence": [0.4] * 4 + [0.9] * 2,
            "confident_normal_lineage_anchor": [False] * 4 + [True] * 2,
            "transcriptomic_suspicion_index": [0.8] * 4 + [0.1] * 2,
            "state_assignment_confidence": [0.5] * 6,
            "blast_prior_probability": [0.8] * 4 + [0.1] * 2,
        }
    )


def patient_summary(clinical_blast: float = 60.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1"],
            "scrna_blast_pct": [60.0],
            "clinical_blast_pct": [clinical_blast],
            "potential_driver_mutations": ["FLT3"],
            "chromosomal_abnormalities": ["Normal karyotype"],
        }
    )


def test_expression_and_blast_prior_do_not_prove_cell_identity() -> None:
    cells, patients, states, _, summary = build_identity_gate(
        base_cells(), patient_summary()
    )
    assert not patients.loc[0, "identity_gate_pass"]
    assert patients.loc[0, "validated_identity_fraction"] == 0
    assert set(cells["blast_prior_counted_as_independent_evidence"]) == {False}
    assert not states["state_identity_gate_pass"].all()
    assert not summary["drug_perturbation_stage_unlocked"]


def test_cell_level_genomic_anchors_can_pass_identity_gate() -> None:
    cells = base_cells()
    cells["cnv_malignant_call"] = [True] * 4 + [False] * 2
    cells["flow_malignant_call"] = [True] * 4 + [False] * 2
    cells["aml_reference_malignant_call"] = [True] * 4 + [False] * 2
    _, patients, states, _, summary = build_identity_gate(cells, patient_summary())
    assert patients.loc[0, "identity_gate_pass"]
    assert patients.loc[0, "validated_identity_fraction"] == 1.0
    assert states["state_identity_gate_pass"].all()
    assert summary["drug_perturbation_stage_unlocked"]


def test_conflicting_independent_calls_block_gate() -> None:
    cells = base_cells()
    cells["cnv_malignant_call"] = True
    cells["flow_malignant_call"] = False
    cells["aml_reference_malignant_call"] = False
    audited, patients, _, _, _ = build_identity_gate(cells, patient_summary())
    assert set(audited["cell_identity_status"]) == {"conflicting_evidence"}
    assert patients.loc[0, "identity_gate_status"] == "blocked"


def test_large_blast_gap_requires_explicit_reconciliation() -> None:
    cells = base_cells()
    cells["cnv_malignant_call"] = [True] * 4 + [False] * 2
    cells["flow_malignant_call"] = [True] * 4 + [False] * 2
    cells["aml_reference_malignant_call"] = [True] * 4 + [False] * 2
    _, blocked, _, _, _ = build_identity_gate(cells, patient_summary(clinical_blast=20))
    assert not blocked.loc[0, "identity_gate_pass"]
    manifest = pd.DataFrame(
        {"patient_id": ["p1"], "blast_discrepancy_reconciled": [True]}
    )
    _, passed, _, _, _ = build_identity_gate(
        cells,
        patient_summary(clinical_blast=20),
        evidence_manifest=manifest,
    )
    assert passed.loc[0, "identity_gate_pass"]


def test_bulk_mutation_context_is_not_counted_as_cell_identity() -> None:
    _, patients, _, _, _ = build_identity_gate(base_cells(), patient_summary())
    assert patients.loc[0, "bulk_mutations_context_only"] == "FLT3"
    assert patients.loc[0, "validated_identity_fraction"] == 0


def test_perturbation_stage_refuses_failed_identity_summary() -> None:
    with pytest.raises(RuntimeError, match="identity is not proven"):
        require_identity_gate_summary(
            {
                "drug_perturbation_stage_unlocked": False,
                "inputs": {"patients": ["p1"]},
            },
            patient_ids=["p1"],
        )


def test_perturbation_stage_requires_same_audited_patients() -> None:
    summary = {
        "drug_perturbation_stage_unlocked": True,
        "inputs": {"patients": ["p1"]},
    }
    require_identity_gate_summary(summary, patient_ids=["p1"])
    with pytest.raises(RuntimeError, match="absent from identity audit"):
        require_identity_gate_summary(summary, patient_ids=["p2"])
