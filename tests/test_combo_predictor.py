"""Tests for Week 3 combo predictor (SynergyMLP + end-to-end pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from combo_val.combo.combo_predictor import (
    ComboConfig,
    SynergyMLP,
    load_combo_training_data,
    train_combo_predictor,
)


# ---------------------------------------------------------------------------
# SynergyMLP unit tests
# ---------------------------------------------------------------------------


def test_synergy_mlp_symmetric():
    """f(d1, d2) must equal f(d2, d1) by construction (symmetric pooling)."""
    torch.manual_seed(42)
    m = SynergyMLP(n_drugs=10, drug_emb_dim=8, hidden_dim=16, dropout=0.0)
    m.eval()
    d1 = torch.tensor([0, 1, 2, 5])
    d2 = torch.tensor([5, 2, 1, 0])
    with torch.no_grad():
        out12 = m(d1, d2)
        out21 = m(d2, d1)
    assert torch.allclose(out12, out21, atol=1e-6)


def test_synergy_mlp_output_shape():
    m = SynergyMLP(n_drugs=10, drug_emb_dim=8, hidden_dim=16, dropout=0.0)
    m.eval()
    d1 = torch.tensor([0, 1, 2])
    d2 = torch.tensor([3, 4, 5])
    with torch.no_grad():
        out = m(d1, d2)
    assert out.shape == (3,)


def test_synergy_mlp_gradient_flow():
    m = SynergyMLP(n_drugs=10, drug_emb_dim=8, hidden_dim=16, dropout=0.0)
    m.train()
    d1 = torch.tensor([0, 1])
    d2 = torch.tensor([2, 3])
    y = torch.tensor([0.5, -0.5])
    out = m(d1, d2)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    # All embedding rows actually used should have gradients
    grad = m.drug_emb.weight.grad
    assert grad is not None
    assert grad[0].abs().sum() > 0
    assert grad[1].abs().sum() > 0
    assert grad[2].abs().sum() > 0
    assert grad[3].abs().sum() > 0


# ---------------------------------------------------------------------------
# load_combo_training_data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pairs_file(tmp_path):
    """Tiny synthetic pair table matching the strict-pairs schema."""
    rows = [
        {"drug1_id": "A", "drug2_id": "B", "synergy_loewe": -10.0,
         "synergy_bliss": -5.0, "synergy_zip": -7.0, "synergy_hsa": -8.0,
         "cell_line_id": "HL-60", "source_study": "ALMANAC"},
        {"drug1_id": "A", "drug2_id": "C", "synergy_loewe": 3.0,
         "synergy_bliss": 1.0, "synergy_zip": 2.0, "synergy_hsa": 4.0,
         "cell_line_id": "HL-60", "source_study": "ALMANAC"},
        {"drug1_id": "B", "drug2_id": "C", "synergy_loewe": -20.0,
         "synergy_bliss": -12.0, "synergy_zip": -15.0, "synergy_hsa": -18.0,
         "cell_line_id": "HL-60", "source_study": "ALMANAC"},
        {"drug1_id": "A", "drug2_id": "D", "synergy_loewe": -5.0,
         "synergy_bliss": -2.0, "synergy_zip": -3.0, "synergy_hsa": -4.0,
         "cell_line_id": "HL-60", "source_study": "ALMANAC"},
    ]
    # Duplicate rows so 5-fold KFold has at least a few samples per fold
    rows_full = []
    for r in rows:
        for _ in range(10):
            rows_full.append(r)
    f = tmp_path / "pairs.csv"
    pd.DataFrame(rows_full).to_csv(f, index=False)
    return f


def test_load_combo_training_data(synthetic_pairs_file):
    df, drugs, mapping = load_combo_training_data(synthetic_pairs_file, "synergy_loewe")
    assert len(df) == 40
    assert set(drugs) == {"A", "B", "C", "D"}
    assert all(mapping[d] < 4 for d in drugs)
    assert "d1_int" in df.columns
    assert "d2_int" in df.columns


# ---------------------------------------------------------------------------
# End-to-end training (slow — small model, few epochs)
# ---------------------------------------------------------------------------


def test_end_to_end_training(synthetic_pairs_file, tmp_path):
    """Full pipeline on synthetic data. Ensures no runtime errors + manifest shape."""
    # Synthetic BeatAML predictions: 3 patients × 4 drugs.
    beat_pred = pd.DataFrame(
        {
            "A": [100.0, 150.0, 200.0],
            "B": [110.0, 160.0, 210.0],
            "C": [120.0, 170.0, 220.0],
            "D": [130.0, 180.0, 230.0],
        },
        index=["P1", "P2", "P3"],
    )
    beat_pred.index.name = "patient_id"
    beat_path = tmp_path / "baseline_predictions.csv"
    beat_pred.to_csv(beat_path)

    out = tmp_path / "combo_out"
    cfg = ComboConfig(
        drug_emb_dim=4, hidden_dim=8, dropout=0.0,
        max_epochs=20, patience=5, n_folds=5, batch_size=8,
    )
    manifest = train_combo_predictor(
        pairs_path=synthetic_pairs_file,
        baseline_pred_path=beat_path,
        out_dir=out,
        cfg=cfg,
    )

    assert manifest["n_pairs_training"] == 40
    assert manifest["n_drugs_combo_vocab"] == 4
    assert manifest["n_drugs_beataml_vocab"] == 4
    assert len(manifest["fold_val_rmses"]) == 5

    # Output files exist
    assert (out / "cv_predictions.csv").exists()
    assert (out / "per_patient_top5_combos.csv").exists()
    assert (out / "combo_auc_predictions.npz").exists()
    assert (out / "combo_predictor_manifest.json").exists()

    # Per-patient top-5 has 3 patients × 5 ranks (= 15, but only 6 unique pairs so will be less)
    top5 = pd.read_csv(out / "per_patient_top5_combos.csv")
    assert top5["patient_id"].nunique() == 3

    # NPZ tensor shape: (n_patients, n_drugs, n_drugs)
    npz = np.load(out / "combo_auc_predictions.npz", allow_pickle=True)
    assert npz["combo_auc"].shape == (3, 4, 4)
