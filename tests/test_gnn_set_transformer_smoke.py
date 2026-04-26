"""Smoke tests for the v0.6 GNN Set-Transformer combo head."""

from __future__ import annotations

import pytest


def _import_or_skip():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as e:
        pytest.skip(f"torch_geometric missing: {e}")


def test_import_resolves():
    _import_or_skip()
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )
    cfg = GNNSetTransformerConfig()
    assert cfg.drug_emb_dim == 64
    assert cfg.set_hidden == 128


def test_forward_2drug_combo():
    _import_or_skip()
    import torch
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )

    B = 4
    N_max = 3   # supports up to 3-drug combos
    cfg = GNNSetTransformerConfig()
    model = GNNSetTransformerCombo(cfg)

    # 2 drugs in each combo (third slot padded)
    drug_embs = torch.randn(B, N_max, cfg.drug_emb_dim)
    drug_mask = torch.tensor([[1, 1, 0]] * B)  # 2 drugs per combo
    patient_dna_emb = torch.randn(B, cfg.patient_emb_dim)
    rna_pc = torch.randn(B, cfg.rna_pc_dim)
    clinical = torch.randn(B, cfg.clinical_dim)
    cyto = torch.randn(B, cfg.cyto_dim)

    out = model(drug_embs, drug_mask, patient_dna_emb, rna_pc, clinical, cyto)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_forward_3drug_combo():
    _import_or_skip()
    import torch
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )

    B = 2
    cfg = GNNSetTransformerConfig()
    model = GNNSetTransformerCombo(cfg)

    drug_embs = torch.randn(B, 3, cfg.drug_emb_dim)
    drug_mask = torch.ones(B, 3, dtype=torch.long)   # all 3 real
    out = model(
        drug_embs, drug_mask,
        patient_dna_emb=torch.randn(B, cfg.patient_emb_dim),
        rna_pc=torch.randn(B, cfg.rna_pc_dim),
        clinical=torch.randn(B, cfg.clinical_dim),
        cyto=torch.randn(B, cfg.cyto_dim),
    )
    assert out.shape == (B,)


def test_permutation_invariance():
    """Combo (A, B) and (B, A) should produce the same prediction."""
    _import_or_skip()
    import torch
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )

    cfg = GNNSetTransformerConfig()
    model = GNNSetTransformerCombo(cfg).eval()

    drug_a = torch.randn(1, cfg.drug_emb_dim)
    drug_b = torch.randn(1, cfg.drug_emb_dim)

    embs_ab = torch.stack([drug_a, drug_b], dim=1)        # [1, 2, dim]
    embs_ba = torch.stack([drug_b, drug_a], dim=1)
    mask = torch.ones(1, 2, dtype=torch.long)
    pdna = torch.randn(1, cfg.patient_emb_dim)
    rna = torch.randn(1, cfg.rna_pc_dim)
    clin = torch.randn(1, cfg.clinical_dim)
    cyto = torch.randn(1, cfg.cyto_dim)

    with torch.no_grad():
        out_ab = model(embs_ab, mask, pdna, rna, clin, cyto)
        out_ba = model(embs_ba, mask, pdna, rna, clin, cyto)
    # Tolerance: floating-point noise in attention softmax. Permutation
    # invariance is mathematical, not learned, so should be tight.
    assert torch.allclose(out_ab, out_ba, atol=1e-4)


def test_padded_drugs_ignored():
    """Adding a padded drug (mask=0) should NOT change the prediction."""
    _import_or_skip()
    import torch
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )

    cfg = GNNSetTransformerConfig()
    model = GNNSetTransformerCombo(cfg).eval()

    drug_a = torch.randn(1, cfg.drug_emb_dim)
    drug_b = torch.randn(1, cfg.drug_emb_dim)
    drug_pad = torch.randn(1, cfg.drug_emb_dim)  # arbitrary noise — should be ignored

    embs_2 = torch.stack([drug_a, drug_b], dim=1)              # [1, 2, dim]
    embs_2_padded = torch.stack(
        [drug_a, drug_b, drug_pad], dim=1,
    )                                                            # [1, 3, dim]
    mask_2 = torch.ones(1, 2, dtype=torch.long)
    mask_3_padded = torch.tensor([[1, 1, 0]])
    pdna = torch.randn(1, cfg.patient_emb_dim)
    rna = torch.randn(1, cfg.rna_pc_dim)
    clin = torch.randn(1, cfg.clinical_dim)
    cyto = torch.randn(1, cfg.cyto_dim)

    with torch.no_grad():
        out_2 = model(embs_2, mask_2, pdna, rna, clin, cyto)
        out_3p = model(embs_2_padded, mask_3_padded, pdna, rna, clin, cyto)
    assert torch.allclose(out_2, out_3p, atol=1e-4)


def test_differentiable_backward():
    _import_or_skip()
    import torch
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig,
    )

    cfg = GNNSetTransformerConfig()
    model = GNNSetTransformerCombo(cfg)

    out = model(
        torch.randn(2, 2, cfg.drug_emb_dim),
        torch.ones(2, 2, dtype=torch.long),
        torch.randn(2, cfg.patient_emb_dim),
        torch.randn(2, cfg.rna_pc_dim),
        torch.randn(2, cfg.clinical_dim),
        torch.randn(2, cfg.cyto_dim),
    )
    loss = out.mean()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_param_count_reasonable():
    _import_or_skip()
    from combo_val.combo.gnn_set_transformer import (
        GNNSetTransformerCombo, GNNSetTransformerConfig, count_parameters,
    )
    n = count_parameters(GNNSetTransformerCombo(GNNSetTransformerConfig()))
    # Realistic: ~250k-1M params for default config (set_hidden=128, head=256)
    assert 100_000 < n < 5_000_000, f"Unusual param count: {n}"
