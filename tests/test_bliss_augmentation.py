"""Tests for Bliss-IDA augmentation (Path B v3 foundation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combo_val.combo.bliss_augmentation import (
    bliss_combo_auc,
    bliss_combo_auc_vectorized,
    BlissAugmentedPairDataset,
    generate_all_patient_bliss_triples,
    AUC_MAX,
)


# ---------------------------------------------------------------------------
# Bliss formula correctness
# ---------------------------------------------------------------------------


def test_bliss_single_drug_unchanged():
    """Single-drug 'combo' = the drug itself."""
    assert bliss_combo_auc([150.0]) == 150.0
    assert bliss_combo_auc([0.0]) == 0.0
    assert bliss_combo_auc([300.0]) == 300.0


def test_bliss_perfect_killer_dominates():
    """If ANY drug has AUC = 0 (perfect killer), combo = 0."""
    assert bliss_combo_auc([0.0, 300.0]) == 0.0
    assert bliss_combo_auc([300.0, 0.0]) == 0.0
    assert bliss_combo_auc([200.0, 0.0, 150.0]) == 0.0


def test_bliss_resistant_stays_resistant():
    """All drugs with AUC = 300 (fully resistant) → combo = 300."""
    assert bliss_combo_auc([300.0, 300.0]) == 300.0
    assert bliss_combo_auc([300.0, 300.0, 300.0]) == 300.0


def test_bliss_two_half_killers():
    """Both drugs at half-kill (AUC=150) → combo kills 3/4 → AUC=75."""
    result = bliss_combo_auc([150.0, 150.0])
    assert result == pytest.approx(75.0, abs=1e-6)


def test_bliss_three_drug_formula():
    """AUC_combo = prod(AUC_i) / 300^(N-1) for N=3."""
    # 3 drugs each at AUC=150
    result = bliss_combo_auc([150.0, 150.0, 150.0])
    expected = (150.0 * 150.0 * 150.0) / (300.0 ** 2)
    assert result == pytest.approx(expected, abs=1e-6)
    # Each alone kills 50%, combined should kill 1 - 0.5^3 = 87.5%
    # → AUC = 300 * 0.125 = 37.5
    assert result == pytest.approx(37.5, abs=1e-6)


def test_bliss_monotone_adding_more_drugs_decreases_auc():
    """Adding a drug with AUC < 300 should decrease combo AUC (more killing)."""
    base = bliss_combo_auc([200.0, 200.0])
    plus_one = bliss_combo_auc([200.0, 200.0, 200.0])
    plus_two = bliss_combo_auc([200.0, 200.0, 200.0, 200.0])
    assert plus_one < base
    assert plus_two < plus_one


def test_bliss_commutative():
    """Order of drugs shouldn't matter."""
    a = bliss_combo_auc([100.0, 200.0, 150.0])
    b = bliss_combo_auc([200.0, 150.0, 100.0])
    c = bliss_combo_auc([150.0, 200.0, 100.0])
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(c, abs=1e-9)


def test_bliss_vectorized_matches_scalar():
    """Vectorized version must match scalar version."""
    matrix = np.array([
        [150.0, 150.0],
        [100.0, 200.0],
        [50.0, 50.0],
        [300.0, 0.0],
    ])
    vec_result = bliss_combo_auc_vectorized(matrix, axis=-1)
    scalar_result = np.array([bliss_combo_auc(row) for row in matrix])
    np.testing.assert_allclose(vec_result, scalar_result, atol=1e-9)


def test_bliss_clips_out_of_range():
    """Values outside [0, 300] should be clipped."""
    # AUC = -50 should be treated as 0 (fully killed)
    assert bliss_combo_auc([-50.0, 300.0]) == 0.0
    # AUC = 400 should be treated as 300 (fully resistant)
    assert bliss_combo_auc([400.0, 300.0]) == 300.0


def test_bliss_empty_raises():
    with pytest.raises(ValueError):
        bliss_combo_auc([])


# ---------------------------------------------------------------------------
# BlissAugmentedPairDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_long_df():
    """3 patients × 4 drugs each."""
    rows = []
    for pid in [1001, 1002, 1003]:
        for drug, auc in [("DrugA", 100.0), ("DrugB", 150.0),
                          ("DrugC", 250.0), ("DrugD", 50.0)]:
            rows.append({"patient_id": pid, "drug_id": drug, "auc": auc})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_patient_features():
    return pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3], "f2": [1.0, 2.0, 3.0]},
        index=[1001, 1002, 1003],
    )


@pytest.fixture
def tiny_drug_to_int():
    return {"DrugA": 0, "DrugB": 1, "DrugC": 2, "DrugD": 3}


def test_augmented_dataset_produces_pairs(
    tiny_long_df, tiny_patient_features, tiny_drug_to_int,
):
    ds = BlissAugmentedPairDataset(
        tiny_long_df, tiny_patient_features, tiny_drug_to_int,
        n_pairs_per_patient=3,
        random_state=0,
    )
    # 3 patients × 3 pairs = up to 9 pair samples
    assert 3 <= len(ds) <= 9


def test_augmented_dataset_bliss_labels_match_formula(
    tiny_long_df, tiny_patient_features, tiny_drug_to_int,
):
    ds = BlissAugmentedPairDataset(
        tiny_long_df, tiny_patient_features, tiny_drug_to_int,
        n_pairs_per_patient=6,
        random_state=0,
    )
    # Each sample's label must match Bliss formula applied to the auc_d1/auc_d2
    for row in ds.rows:
        expected = (row["auc_d1"] * row["auc_d2"]) / AUC_MAX
        assert row["bliss_auc"] == pytest.approx(expected, abs=1e-6)


def test_augmented_dataset_getitem_shape(
    tiny_long_df, tiny_patient_features, tiny_drug_to_int,
):
    ds = BlissAugmentedPairDataset(
        tiny_long_df, tiny_patient_features, tiny_drug_to_int,
        n_pairs_per_patient=5,
        random_state=0,
    )
    item = ds[0]
    assert item["drug_ids"].shape == (2,)
    assert item["drug_mask"].shape == (2,)
    assert item["patient_features"].shape == (2,)
    assert item["auc"].shape == ()
    # drug_ids must be shifted by +1 (reserving 0 for padding)
    assert (item["drug_ids"] >= 1).all()


def test_triple_generator(tiny_long_df, tiny_patient_features, tiny_drug_to_int):
    triples = generate_all_patient_bliss_triples(
        tiny_long_df, tiny_patient_features, tiny_drug_to_int,
        n_triples_per_patient=3,
    )
    # Each patient has 4 drugs → C(4,3) = 4 possible triples
    # With 3 sampled per patient × 3 patients = up to 9
    assert 3 <= len(triples) <= 12
    for t in triples:
        expected = (t["auc_d1"] * t["auc_d2"] * t["auc_d3"]) / (AUC_MAX ** 2)
        assert t["bliss_auc"] == pytest.approx(expected, abs=1e-6)
