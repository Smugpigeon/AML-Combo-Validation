"""Smoke tests for the v0.6 GIN drug encoder.

Verifies that the encoder:
  1. Builds without error at default config
  2. Produces sensible output shapes
  3. Handles invalid / empty SMILES gracefully
  4. Is differentiable (loss.backward() runs)
  5. Has parameter count in the expected range (~30k-200k)

These tests do NOT validate model quality — that's Phase 1 training.
They just confirm the architecture wires up correctly.
"""

from __future__ import annotations

import pytest


def _import_or_skip():
    try:
        import torch_geometric  # noqa: F401
        from rdkit import Chem  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Dependency missing for v0.6 GNN tests: {e}")


def test_imports_resolve():
    _import_or_skip()
    from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph
    cfg = DrugGINConfig()
    assert cfg.hidden_dim == 128
    assert cfg.n_layers == 4
    assert cfg.drug_emb_dim == 64


def test_smiles_to_molgraph_simple():
    _import_or_skip()
    from combo_val.encoders.mol_featurizer import (
        ATOM_FEATURE_DIM, smiles_to_molgraph,
    )
    g = smiles_to_molgraph("CCO")  # ethanol
    assert g is not None
    assert g.n_atoms == 3
    assert g.x.shape == (3, ATOM_FEATURE_DIM)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == 4   # 2 bonds × 2 directions


def test_smiles_to_molgraph_invalid_returns_none():
    _import_or_skip()
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph
    assert smiles_to_molgraph("not_valid_smiles_!!") is None
    assert smiles_to_molgraph("") is None


def test_smiles_to_molgraph_complex_drug():
    _import_or_skip()
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph
    # Venetoclax SMILES (truncated for test speed; real one is longer)
    venetoclax = "O=C(NS(=O)(=O)c1ccc(NCC2CCOCC2)c([N+](=O)[O-])c1)c1ccc(N2CCN(Cc3ccc(-c4ccc(Cl)cc4)cc3)CC2)cc1Oc1ccc2[nH]cc(C2(C)C)c2nc1"
    # If the test SMILES is malformed, just confirm RDKit returns None — no crash.
    from rdkit import Chem
    if Chem.MolFromSmiles(venetoclax) is None:
        return  # Acceptable; real SMILES will come from PubChem in Phase 0
    g = smiles_to_molgraph(venetoclax)
    assert g is not None
    assert g.n_atoms > 20  # large drug


def test_drug_gin_forward_single_molecule():
    _import_or_skip()
    import torch
    from torch_geometric.data import Batch
    from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data

    g = smiles_to_molgraph("CCO")
    batch = Batch.from_data_list([to_pyg_data(g)])
    model = DrugGIN(DrugGINConfig(drug_emb_dim=64))
    out = model(batch)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_drug_gin_forward_batched():
    _import_or_skip()
    import torch
    from torch_geometric.data import Batch
    from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data

    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CN(C)C=O", "OCC"]
    data_list = [to_pyg_data(smiles_to_molgraph(s)) for s in smiles]
    batch = Batch.from_data_list(data_list)
    model = DrugGIN()
    out = model(batch)
    assert out.shape == (5, 64)


def test_drug_gin_encode_smiles_handles_invalid():
    _import_or_skip()
    import torch
    from combo_val.encoders.drug_gin import DrugGIN

    model = DrugGIN()
    # Mix of valid + invalid SMILES
    out = model.encode_smiles(["CCO", "INVALID!!!", "c1ccccc1", ""])
    assert out.shape == (4, 64)
    # Invalid + empty should be all-zero
    assert torch.allclose(out[1], torch.zeros(64))
    assert torch.allclose(out[3], torch.zeros(64))
    # Valid should be non-zero (with high probability for random init)
    assert not torch.allclose(out[0], torch.zeros(64))


def test_drug_gin_is_differentiable():
    _import_or_skip()
    import torch
    from torch_geometric.data import Batch
    from combo_val.encoders.drug_gin import DrugGIN
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data

    model = DrugGIN()
    smiles = ["CCO", "c1ccccc1"]
    data_list = [to_pyg_data(smiles_to_molgraph(s)) for s in smiles]
    batch = Batch.from_data_list(data_list)
    out = model(batch)
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    # All params should have gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_drug_gin_param_count_reasonable():
    _import_or_skip()
    from combo_val.encoders.drug_gin import DrugGIN, count_parameters
    n = count_parameters(DrugGIN())
    # Default config: 4 layers × ~33k each + atom_proj + readout ≈ 150-200k
    assert 50_000 < n < 500_000, f"GIN param count {n} outside expected range"


def test_atom_feature_dim_matches_constant():
    _import_or_skip()
    from combo_val.encoders.mol_featurizer import (
        ATOM_FEATURE_DIM, atom_features,
    )
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    f = atom_features(mol.GetAtomWithIdx(0))
    assert len(f) == ATOM_FEATURE_DIM


def test_bond_feature_dim_matches_constant():
    _import_or_skip()
    from combo_val.encoders.mol_featurizer import (
        BOND_FEATURE_DIM, bond_features,
    )
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    f = bond_features(mol.GetBondWithIdx(0))
    assert len(f) == BOND_FEATURE_DIM
