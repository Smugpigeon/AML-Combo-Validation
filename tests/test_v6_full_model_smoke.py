"""Smoke tests for the v0.6 full composite model (GIN + GAT + Combo head)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PPI_PATH = REPO_ROOT / "data" / "canonical" / "gene_ppi_subgraph.json"
FEATURES_CSV = REPO_ROOT / "data" / "canonical" / "beataml_patient_features.csv"


def _import_or_skip():
    try:
        import torch_geometric  # noqa: F401
        from rdkit import Chem  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Dependency missing: {e}")
    if not PPI_PATH.exists():
        pytest.skip(f"PPI subgraph not present at {PPI_PATH}")


def _gene_order() -> list[str]:
    with open(FEATURES_CSV) as f:
        header = f.readline().strip().split(",")
    return [h[len("mut_"):] for h in header if h.startswith("mut_")]


def test_v6_full_model_default_config():
    _import_or_skip()
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    cfg = V6FullModelConfig()
    assert cfg.use_gin_drug is True
    assert cfg.use_gat_gene is True
    model = V6FullModel(PPI_PATH, _gene_order(), cfg)
    assert model.cfg.use_gin_drug is True


def test_v6_forward_full_gin_path():
    _import_or_skip()
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    model = V6FullModel(PPI_PATH, _gene_order(), V6FullModelConfig()).eval()
    B = 2
    N_max = 2
    drug_input = [
        ["CCO", "CC(=O)O"],          # 2-drug combo
        ["c1ccccc1", "CN(C)C=O"],
    ]
    drug_mask = torch.ones(B, N_max, dtype=torch.long)
    mut_features = torch.bernoulli(torch.full((B, 25), 0.3))
    rna_pc = torch.randn(B, 50)
    clinical = torch.randn(B, 21)
    cyto = torch.randn(B, 8)

    with torch.no_grad():
        out = model(drug_input, drug_mask, mut_features, rna_pc, clinical, cyto)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_v6_forward_baseline_path_no_gin_no_gat():
    """v0.5-equivalent baseline: nn.Embedding + linear gene proj."""
    _import_or_skip()
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    cfg = V6FullModelConfig(use_gin_drug=False, use_gat_gene=False)
    model = V6FullModel(PPI_PATH, _gene_order(), cfg).eval()

    B = 2
    drug_ids = torch.tensor([[0, 5], [10, 20]], dtype=torch.long)
    drug_mask = torch.ones(B, 2, dtype=torch.long)
    mut_features = torch.bernoulli(torch.full((B, 25), 0.3))
    rna_pc = torch.randn(B, 50)
    clinical = torch.randn(B, 21)
    cyto = torch.randn(B, 8)

    with torch.no_grad():
        out = model(drug_ids, drug_mask, mut_features, rna_pc, clinical, cyto)
    assert out.shape == (B,)


def test_v6_forward_gin_only_no_gat():
    _import_or_skip()
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    cfg = V6FullModelConfig(use_gin_drug=True, use_gat_gene=False)
    model = V6FullModel(PPI_PATH, _gene_order(), cfg).eval()

    B = 1
    drug_input = [["CCO", "c1ccccc1"]]
    drug_mask = torch.ones(B, 2, dtype=torch.long)
    mut_features = torch.zeros(B, 25)
    rna_pc = torch.randn(B, 50)
    clinical = torch.randn(B, 21)
    cyto = torch.randn(B, 8)

    with torch.no_grad():
        out = model(drug_input, drug_mask, mut_features, rna_pc, clinical, cyto)
    assert out.shape == (B,)


def test_v6_forward_gat_only_no_gin():
    _import_or_skip()
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    cfg = V6FullModelConfig(use_gin_drug=False, use_gat_gene=True)
    model = V6FullModel(PPI_PATH, _gene_order(), cfg).eval()

    B = 1
    drug_ids = torch.tensor([[3, 7]], dtype=torch.long)
    drug_mask = torch.ones(B, 2, dtype=torch.long)
    mut_features = torch.bernoulli(torch.full((B, 25), 0.4))

    with torch.no_grad():
        out = model(
            drug_ids, drug_mask, mut_features,
            torch.randn(B, 50), torch.randn(B, 21), torch.randn(B, 8),
        )
    assert out.shape == (B,)


def test_v6_handles_invalid_smiles_gracefully():
    """Invalid SMILES → zero embedding for that drug, model still runs."""
    _import_or_skip()
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig
    model = V6FullModel(PPI_PATH, _gene_order(), V6FullModelConfig()).eval()
    B = 1
    drug_input = [["CCO", "INVALID_SMILES_!!!"]]
    drug_mask = torch.ones(B, 2, dtype=torch.long)

    with torch.no_grad():
        out = model(
            drug_input, drug_mask, torch.zeros(B, 25),
            torch.randn(B, 50), torch.randn(B, 21), torch.randn(B, 8),
        )
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_v6_param_counts_for_ablation_arms():
    """Sanity check param counts for each ablation arm."""
    _import_or_skip()
    from combo_val.encoders.v6_full_model import (
        V6FullModel, V6FullModelConfig, count_parameters,
    )

    arms = {
        "v0.6.0_full": V6FullModelConfig(use_gin_drug=True, use_gat_gene=True),
        "v0.6.A_drug_only": V6FullModelConfig(use_gin_drug=True, use_gat_gene=False),
        "v0.6.B_gene_only": V6FullModelConfig(use_gin_drug=False, use_gat_gene=True),
        "v0.5_baseline": V6FullModelConfig(use_gin_drug=False, use_gat_gene=False),
    }
    for name, cfg in arms.items():
        model = V6FullModel(PPI_PATH, _gene_order(), cfg)
        n = count_parameters(model)
        # All arms should have meaningful capacity
        assert 100_000 < n < 5_000_000, f"{name}: {n} params outside expected range"
