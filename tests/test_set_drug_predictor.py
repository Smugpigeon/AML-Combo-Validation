"""Unit tests for the Set Transformer combo predictor (Path B).

Tests the pure architecture properties — permutation invariance, padding
invariance, set-size handling, gradient flow — without needing trained
checkpoints or real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from combo_val.combo.set_drug_predictor import (
    ISAB, MAB, PMA, MultiHeadAttention,
    PatientDrugSetDataset, SetDrugPredictor, SetDrugPredictorConfig,
    collate_variable_arity,
)


# ---------------------------------------------------------------------------
# Attention primitive tests
# ---------------------------------------------------------------------------


def test_mha_shape_and_mask():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    Q = torch.randn(2, 3, 16)
    K = torch.randn(2, 5, 16)
    V = torch.randn(2, 5, 16)
    out = mha(Q, K, V)
    assert out.shape == (2, 3, 16)

    # With mask: last 2 keys masked
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    masked_out = mha(Q, K, V, key_mask=mask)
    assert masked_out.shape == (2, 3, 16)
    # Masked output differs from unmasked
    assert not torch.allclose(out, masked_out)


def test_isab_shape():
    isab = ISAB(d_model=32, n_heads=4, n_inducing=8)
    X = torch.randn(4, 10, 32)
    out = isab(X)
    assert out.shape == (4, 10, 32)


def test_pma_pools_to_one():
    pma = PMA(d_model=32, n_heads=4, n_seeds=1)
    X = torch.randn(4, 10, 32)
    out = pma(X)
    assert out.shape == (4, 1, 32)


# ---------------------------------------------------------------------------
# SetDrugPredictor architecture tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    cfg = SetDrugPredictorConfig(
        drug_emb_dim=16, patient_hidden_dim=32, patient_emb_dim=16,
        set_hidden_dim=32, n_attn_heads=2, n_inducing_points=4,
        n_isab_layers=2, head_hidden_dim=32, dropout=0.0,
    )
    torch.manual_seed(42)
    return SetDrugPredictor(n_drugs=20, n_patient_features=10, cfg=cfg), cfg


def test_forward_size_1(tiny_model):
    model, _ = tiny_model
    model.eval()
    drug_ids = torch.tensor([[5]], dtype=torch.long)
    drug_mask = torch.tensor([[1.0]])
    pf = torch.randn(1, 10)
    with torch.no_grad():
        y = model(drug_ids, drug_mask, pf)
    assert y.shape == (1,)
    assert torch.isfinite(y).all()


def test_forward_size_4(tiny_model):
    model, _ = tiny_model
    model.eval()
    drug_ids = torch.tensor([[1, 5, 12, 19]], dtype=torch.long)
    drug_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    pf = torch.randn(1, 10)
    with torch.no_grad():
        y = model(drug_ids, drug_mask, pf)
    assert y.shape == (1,)
    assert torch.isfinite(y).all()


def test_permutation_invariance(tiny_model):
    """f(π(S)) == f(S) for any permutation π."""
    model, _ = tiny_model
    model.eval()
    torch.manual_seed(7)
    pf = torch.randn(1, 10)
    d1 = torch.tensor([[3, 11, 17, 2]], dtype=torch.long)
    m1 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    d2 = torch.tensor([[11, 2, 17, 3]], dtype=torch.long)
    m2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    d3 = torch.tensor([[17, 3, 11, 2]], dtype=torch.long)
    m3 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    with torch.no_grad():
        y1 = model(d1, m1, pf)
        y2 = model(d2, m2, pf)
        y3 = model(d3, m3, pf)
    # All three should be numerically identical (modulo fp noise)
    assert abs(y1.item() - y2.item()) < 1e-4
    assert abs(y1.item() - y3.item()) < 1e-4


def test_padding_invariance(tiny_model):
    """f({a, b}) == f({a, b, PAD, PAD})."""
    model, _ = tiny_model
    model.eval()
    pf = torch.randn(1, 10)
    d_short = torch.tensor([[5, 12]], dtype=torch.long)
    m_short = torch.tensor([[1.0, 1.0]])
    d_padded = torch.tensor([[5, 12, 0, 0, 0]], dtype=torch.long)
    m_padded = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        y_s = model(d_short, m_short, pf)
        y_p = model(d_padded, m_padded, pf)
    assert abs(y_s.item() - y_p.item()) < 1e-4


def test_padding_ids_do_not_influence_output(tiny_model):
    """If drug_mask correctly zeroes padding, the specific drug IDs in padded
    positions should not matter."""
    model, _ = tiny_model
    model.eval()
    pf = torch.randn(1, 10)
    d1 = torch.tensor([[5, 12, 0, 0]], dtype=torch.long)
    m1 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    d2 = torch.tensor([[5, 12, 1, 19]], dtype=torch.long)  # different padding fill
    m2 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        y1 = model(d1, m1, pf)
        y2 = model(d2, m2, pf)
    assert abs(y1.item() - y2.item()) < 1e-4


def test_different_patient_changes_output(tiny_model):
    """Same drug set, different patient → different prediction.
    Tolerance is loose because untrained heads barely use the patient signal;
    we just verify the output isn't *identical* (patient_features is wired in).
    """
    model, _ = tiny_model
    model.eval()
    torch.manual_seed(123)
    d = torch.tensor([[5, 12]], dtype=torch.long)
    m = torch.tensor([[1.0, 1.0]])
    pf1 = torch.randn(1, 10)
    pf2 = torch.randn(1, 10)
    with torch.no_grad():
        y1 = model(d, m, pf1)
        y2 = model(d, m, pf2)
    # Patient signal must be in the computation graph (non-identical output).
    assert y1.item() != y2.item()


def test_gradient_flow(tiny_model):
    model, _ = tiny_model
    model.train()
    # When calling model.forward directly, drug IDs are the embedding indices
    # as-passed (the +1 padding shift is only done by PatientDrugSetDataset
    # and predict_set helper, not inside forward).
    drug_ids = torch.tensor([[5, 12]], dtype=torch.long)
    drug_mask = torch.tensor([[1.0, 1.0]])
    pf = torch.randn(1, 10)
    target = torch.tensor([100.0])
    y = model(drug_ids, drug_mask, pf)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    emb_grad = model.drug_embedding.weight.grad
    assert emb_grad is not None
    # Drugs passed at positions 5, 12 should accumulate gradient
    assert emb_grad[5].abs().sum() > 0
    assert emb_grad[12].abs().sum() > 0
    # Padding index 0 MUST have ZERO gradient (by nn.Embedding(padding_idx=0))
    assert emb_grad[0].abs().sum() == 0


def test_predict_set_helper(tiny_model):
    model, _ = tiny_model
    model.eval()
    drug_sets = [[5], [5, 12], [5, 12, 17]]
    pf = torch.randn(3, 10)
    with torch.no_grad():
        y = model.predict_set(drug_sets, pf, device=torch.device("cpu"))
    assert y.shape == (3,)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Dataset + collator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_long_df():
    return pd.DataFrame({
        "patient_id": [1, 1, 2, 2, 3],
        "drug_id": ["A", "B", "A", "B", "A"],
        "auc": [100.0, 120.0, 80.0, 90.0, 110.0],
    })


@pytest.fixture
def synthetic_features():
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(3, 5)),
        index=[1, 2, 3],
        columns=[f"f{i}" for i in range(5)],
    )
    df.index.name = "patient_id"
    return df


def test_dataset_returns_size_1_sets(synthetic_long_df, synthetic_features):
    ds = PatientDrugSetDataset(
        synthetic_long_df, synthetic_features,
        drug_id_to_int={"A": 0, "B": 1},
    )
    assert len(ds) == 5
    x = ds[0]
    assert x["drug_ids"].shape == (1,)
    assert x["drug_mask"].shape == (1,)
    # Drug "A" is int 0 → shifted to 1 (padding takes 0)
    assert x["drug_ids"].item() == 1


def test_collator_pads_to_max_arity(synthetic_long_df, synthetic_features):
    ds = PatientDrugSetDataset(
        synthetic_long_df, synthetic_features,
        drug_id_to_int={"A": 0, "B": 1},
    )
    # Manually create a mixed-arity batch
    item1 = ds[0]
    item2 = {
        "patient_features": ds[1]["patient_features"],
        "drug_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "drug_mask": torch.tensor([1.0, 1.0, 1.0]),
        "auc": torch.tensor(50.0),
    }
    batch = collate_variable_arity([item1, item2])
    assert batch["drug_ids"].shape == (2, 3)         # padded to max 3
    assert batch["drug_mask"].shape == (2, 3)
    # First sample (size 1) should have mask [1, 0, 0] and ids [id, 0, 0]
    assert batch["drug_mask"][0].tolist() == [1.0, 0.0, 0.0]
    assert batch["drug_ids"][0, 0].item() == 1
    assert batch["drug_ids"][0, 1].item() == 0      # padding
    # Second sample: full mask
    assert batch["drug_mask"][1].tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Quick train-one-batch sanity
# ---------------------------------------------------------------------------


def test_train_one_step_reduces_loss(tiny_model):
    """Overfit on one mini-batch to verify gradients flow end-to-end."""
    model, _ = tiny_model
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    drug_ids = torch.tensor([
        [5, 0], [12, 17], [1, 8]
    ], dtype=torch.long)
    drug_mask = torch.tensor([
        [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]
    ])
    pf = torch.randn(3, 10)
    target = torch.tensor([100.0, 50.0, 200.0])

    initial_loss = None
    for _ in range(30):
        y = model(drug_ids, drug_mask, pf)
        loss = ((y - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, (
        f"loss did not reduce sufficiently: {initial_loss:.2f} → {final_loss:.2f}"
    )
