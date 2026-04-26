"""Smoke tests for the v0.6 GAT gene-PPI encoder.

These tests confirm the GAT wires up to the real STRING PPI subgraph
fetched in Phase 0.4 and produces sensible patient-level embeddings.
"""

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
    except ImportError as e:
        pytest.skip(f"torch_geometric missing: {e}")
    if not PPI_PATH.exists():
        pytest.skip(f"PPI subgraph not present at {PPI_PATH} — run Phase 0.4")


def _gene_order() -> list[str]:
    with open(FEATURES_CSV) as f:
        header = f.readline().strip().split(",")
    return [h[len("mut_"):] for h in header if h.startswith("mut_")]


def test_ppi_subgraph_loads():
    _import_or_skip()
    with open(PPI_PATH) as f:
        g = json.load(f)
    assert g["nodes"]
    assert g["edges"]
    assert len(g["nodes"]) == 25


def test_gat_loads_with_real_ppi():
    _import_or_skip()
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    gat = GeneContextGAT(PPI_PATH, _gene_order(), GeneGATConfig())
    assert gat.cfg.n_genes == 25
    # Edge index should have 2*E + 25 self-loops
    n_edges_expected = 2 * 13 + 25  # 13 PPI bidirectional + 25 self-loops
    assert gat.edge_index.shape == (2, n_edges_expected)


def test_gat_forward_single_patient():
    _import_or_skip()
    import torch
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    cfg = GeneGATConfig(out_dim=64)
    gat = GeneContextGAT(PPI_PATH, _gene_order(), cfg)
    # Single patient with FLT3, NPM1, DNMT3A mutated
    mut = torch.zeros(1, 25)
    gene_to_idx = gat.gene_to_idx
    for g in ("FLT3", "NPM1", "DNMT3A"):
        mut[0, gene_to_idx[g]] = 1.0
    out = gat(mut)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_gat_forward_batched():
    _import_or_skip()
    import torch
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    cfg = GeneGATConfig(out_dim=64)
    gat = GeneContextGAT(PPI_PATH, _gene_order(), cfg)
    # Batch of 8 patients with varying mutation patterns
    mut = torch.bernoulli(torch.full((8, 25), 0.3))
    out = gat(mut)
    assert out.shape == (8, 64)
    assert torch.isfinite(out).all()


def test_gat_is_differentiable():
    _import_or_skip()
    import torch
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    gat = GeneContextGAT(PPI_PATH, _gene_order(), GeneGATConfig())
    mut = torch.bernoulli(torch.full((4, 25), 0.3))
    out = gat(mut)
    loss = (out ** 2).mean()
    loss.backward()
    for p in gat.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_gat_different_inputs_produce_different_outputs():
    _import_or_skip()
    import torch
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    gat = GeneContextGAT(PPI_PATH, _gene_order(), GeneGATConfig())
    gat.eval()
    mut_a = torch.zeros(1, 25)
    mut_a[0, 0] = 1.0  # only FLT3 mutated
    mut_b = torch.zeros(1, 25)
    mut_b[0, 5] = 1.0  # only TP53 mutated
    with torch.no_grad():
        emb_a = gat(mut_a)
        emb_b = gat(mut_b)
    # Different mutation patterns should produce distinguishable embeddings
    assert not torch.allclose(emb_a, emb_b, atol=1e-3)


def test_gat_handles_no_mutations():
    """Patient with all-zero mutation panel — model should still work
    (fall back to gene_id_emb baseline + PPI message passing)."""
    _import_or_skip()
    import torch
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
    gat = GeneContextGAT(PPI_PATH, _gene_order(), GeneGATConfig())
    mut = torch.zeros(1, 25)
    out = gat(mut)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_gat_gene_order_mismatch_raises():
    _import_or_skip()
    from combo_val.encoders.gene_ppi_gat import GeneContextGAT
    bad_order = ["FAKE_GENE_1"] * 25
    with pytest.raises(ValueError, match="mismatch"):
        GeneContextGAT(PPI_PATH, bad_order)
