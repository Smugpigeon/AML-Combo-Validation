from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from combo_val.virtual_cell.single_cell_v15 import (
    StateAtlasConfig,
    add_reference_shift_metrics,
    calibrate_probability_to_prevalence,
    classify_marker_lineage,
    combine_lineage_annotations,
    fit_gene_set_reference,
    fit_state_atlas,
    map_celltypist_label,
    normalize_log1p_to_10k,
    reference_shift_contributions,
    score_gene_sets,
    summarize_patient_states,
    transform_gene_set_reference,
    transform_state_atlas,
)


def test_normalize_log1p_to_10k_preserves_sparse_shape_and_target() -> None:
    raw = sparse.csr_matrix([[2.0, 0.0, 3.0], [0.0, 4.0, 1.0]])
    source_scaled = raw.multiply(np.array([[1000.0], [2000.0]]))
    source_log = source_scaled.copy()
    source_log.data = np.log1p(source_log.data)
    normalized = normalize_log1p_to_10k(source_log)
    recovered = normalized.copy()
    recovered.data = np.expm1(recovered.data)
    totals = np.asarray(recovered.sum(axis=1)).ravel()
    assert normalized.shape == raw.shape
    assert sparse.isspmatrix_csr(normalized)
    assert np.allclose(totals, 10000.0, rtol=1e-5)


def test_gene_set_scores_report_coverage_and_expected_direction() -> None:
    matrix = sparse.csr_matrix(
        [
            [6.0, 5.0, 0.0, 0.0],
            [5.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 5.0],
            [0.0, 0.0, 5.0, 4.0],
        ]
    )
    scores, coverage = score_gene_sets(
        matrix,
        ["CD3D", "TRAC", "LYZ", "LST1"],
        {"T": ("CD3D", "TRAC"), "myeloid": ("LYZ", "LST1", "MPO")},
    )
    assert scores.loc[0, "T"] > scores.loc[0, "myeloid"]
    assert scores.loc[2, "myeloid"] > scores.loc[2, "T"]
    myeloid = coverage.set_index("module").loc["myeloid"]
    assert myeloid["available_genes"] == 2
    assert myeloid["coverage_fraction"] == 2 / 3


def test_frozen_gene_reference_reproduces_training_scores() -> None:
    matrix = sparse.csr_matrix(
        [[5.0, 4.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 4.0]]
    )
    fitted, _, reference = fit_gene_set_reference(
        matrix,
        ["A", "B", "C"],
        {"first": ("A", "B"), "second": ("B", "C")},
    )
    transformed, coverage = transform_gene_set_reference(
        matrix, ["A", "B", "C"], reference
    )
    assert np.allclose(fitted, transformed, equal_nan=True)
    assert coverage["score_available"].all()


def test_lineage_consensus_exposes_disagreement() -> None:
    scores = pd.DataFrame(
        {
            "T_cell": [4.0, 0.0],
            "monocyte_macrophage": [0.0, 4.0],
        }
    )
    marker = classify_marker_lineage(scores, minimum_confidence=0.2)
    consensus = combine_lineage_annotations(
        marker,
        ["Classical monocytes", "Classical monocytes"],
        [0.95, 0.95],
    )
    assert consensus.loc[0, "consensus_lineage"] == "ambiguous"
    assert consensus.loc[0, "lineage_evidence_status"] == "discordant"
    assert consensus.loc[1, "consensus_lineage"] == "monocyte_macrophage"
    assert consensus.loc[1, "lineage_evidence_status"] == "concordant"


def test_celltypist_mapping_covers_common_broad_lineages() -> None:
    assert map_celltypist_label("Tem/Temra cytotoxic T cells") == "T_cell"
    assert map_celltypist_label("CD16+ NK cells") == "NK_cell"
    assert map_celltypist_label("Classical monocytes") == "monocyte_macrophage"
    assert map_celltypist_label("Neutrophil-myeloid progenitor") == "granulocytic"


def test_prevalence_calibration_matches_requested_mean_and_preserves_rank() -> None:
    probability = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    calibrated = calibrate_probability_to_prevalence(probability, 0.35)
    assert np.isclose(calibrated.mean(), 0.35, atol=1e-8)
    assert np.array_equal(np.argsort(probability), np.argsort(calibrated))


def test_state_atlas_is_deterministic_and_assigns_all_cells() -> None:
    rng = np.random.default_rng(5)
    matrix = np.vstack(
        [
            rng.normal(-2, 0.2, size=(30, 8)),
            rng.normal(0, 0.2, size=(30, 8)),
            rng.normal(2, 0.2, size=(30, 8)),
        ]
    ).astype(np.float32)
    config = StateAtlasConfig(
        n_components=5,
        n_states=3,
        random_state=17,
        batch_size=32,
        stability_repeats=2,
    )
    first = fit_state_atlas(matrix, config)
    second = fit_state_atlas(matrix, config)
    assert np.array_equal(first.state_ids, second.state_ids)
    assert len(first.state_ids) == 90
    assert set(first.state_ids) == {"VS01", "VS02", "VS03"}
    assert np.all((first.assignment_confidence >= 0) & (first.assignment_confidence <= 1))
    assert np.all(
        (first.initialization_stability >= 0) & (first.initialization_stability <= 1)
    )
    assert len(first.initialization_ami) == 2
    coordinates, labels, confidence, distance = transform_state_atlas(
        matrix,
        first.svd,
        first.scaler,
        first.kmeans,
        first.stable_label_mapping,
    )
    assert coordinates.shape == first.coordinates.shape
    assert np.array_equal(labels, first.state_ids)
    assert np.allclose(confidence, first.assignment_confidence)
    assert np.allclose(distance, first.nearest_distance)


def test_patient_summary_and_reference_shift_contract() -> None:
    annotations = pd.DataFrame(
        {
            "sample_id": ["p1", "p1", "p2", "p2"],
            "consensus_lineage": ["T_cell", "stem_progenitor", "B_cell", "B_cell"],
            "lineage_confidence": [0.9, 0.6, 0.8, 0.7],
            "virtual_state_id": ["VS01", "VS02", "VS01", "VS01"],
            "state_assignment_confidence": [0.8, 0.7, 0.9, 0.8],
            "transcriptomic_suspicion_index": [0.1, 0.9, 0.2, 0.3],
            "confident_normal_lineage_anchor": [True, False, True, True],
            "qc_pass": [True, True, True, False],
            "program::stem": [-1.0, 2.0, -0.5, 0.0],
        }
    )
    summary, programs, vectors = summarize_patient_states(
        annotations, ["program::stem"]
    )
    shifted = add_reference_shift_metrics(vectors)
    assert set(summary["patient_id"]) == {"p1", "p2"}
    assert len(programs) == 2
    assert "state_fraction::VS02" in vectors
    assert np.isfinite(shifted["reference_shift_distance"]).all()


def test_reference_shift_contributions_are_ranked() -> None:
    from combo_val.virtual_cell.single_cell_v15 import fit_reference_shift_model

    vectors = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "feature_a": [0.0, 0.0, 9.0],
            "feature_b": [1.0, 2.0, 1.0],
        }
    )
    shifted, model = fit_reference_shift_model(vectors)
    drivers = reference_shift_contributions(shifted, model, top_n=1)
    assert drivers.loc[drivers["patient_id"] == "p3", "feature"].iloc[0] == "feature_a"
    assert (drivers["rank"] == 1).all()
