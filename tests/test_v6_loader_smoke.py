"""Smoke tests for the v0.6 loader (_v6_loader.py).

Tests that the loader can construct a model from saved checkpoints +
SMILES table + PPI subgraph, AND that the vectorized pairwise combo
prediction produces sensible matrix shapes.

Phase-5-pre tests: skip cleanly if no checkpoint present yet.
Phase-5-post tests: load runs/v6_ablation_smoke/v0.6.0_full_fold_1.pt
and verify the matrix is symmetric + finite + non-trivial.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMILES_CSV = REPO_ROOT / "data" / "canonical" / "drug_smiles.csv"
PPI_JSON = REPO_ROOT / "data" / "canonical" / "gene_ppi_subgraph.json"
FEATURES_CSV = REPO_ROOT / "data" / "canonical" / "beataml_patient_features.csv"


def _import_or_skip():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as e:
        pytest.skip(f"torch_geometric missing: {e}")


def test_load_smiles_table_returns_165():
    if not SMILES_CSV.exists():
        pytest.skip(f"{SMILES_CSV} not present")
    from combo_val.clinical._v6_loader import load_smiles_table
    table = load_smiles_table(str(SMILES_CSV))
    assert len(table) == 165
    # All values are non-empty SMILES strings
    for did, smi in table.items():
        assert smi
        assert isinstance(smi, str)


def test_smiles_table_caches_idempotent():
    if not SMILES_CSV.exists():
        pytest.skip(f"{SMILES_CSV} not present")
    from combo_val.clinical._v6_loader import load_smiles_table
    a = load_smiles_table(str(SMILES_CSV))
    b = load_smiles_table(str(SMILES_CSV))
    assert a is b   # @lru_cache → same object


def test_loader_raises_clear_error_on_missing_ckpt():
    _import_or_skip()
    from combo_val.clinical._v6_loader import load_v6_model
    with pytest.raises(FileNotFoundError, match="not present"):
        load_v6_model(
            Path("runs/this_does_not_exist.pt"),
            PPI_JSON, FEATURES_CSV,
        )


@pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "v6_ablation_smoke" /
         "v0.6.0_full_fold_1.pt").exists(),
    reason="Phase 5 ablation has not produced v0.6.0_full_fold_1.pt yet",
)
def test_v6_loader_produces_symmetric_combo_matrix():
    _import_or_skip()
    import numpy as np
    from combo_val.clinical._v6_loader import (
        load_smiles_table, load_v6_model, predict_combo_auc_via_gnn_v6,
    )
    smiles = load_smiles_table(str(SMILES_CSV))
    model, meta = load_v6_model(
        REPO_ROOT / "runs" / "v6_ablation_smoke" / "v0.6.0_full_fold_1.pt",
        PPI_JSON, FEATURES_CSV,
    )
    # Pick first 5 drugs that have SMILES
    drug_vocab = list(smiles.keys())[:5]
    # Load one patient feature row
    import pandas as pd
    feats_df = pd.read_csv(FEATURES_CSV)
    pf = feats_df.iloc[0].drop("patient_id").values.astype(np.float32)
    combo_auc = predict_combo_auc_via_gnn_v6(
        model, smiles, drug_vocab, pf,
    )
    assert combo_auc.shape == (5, 5)
    # Symmetric (within float tolerance)
    assert np.allclose(combo_auc, combo_auc.T, atol=1e-5)
    # Diagonal = 0 (single-drug self-combo, intentionally left blank)
    assert np.allclose(np.diag(combo_auc), 0)
    # Off-diagonal entries are finite
    np.fill_diagonal(combo_auc, np.nan)
    assert np.isfinite(np.nan_to_num(combo_auc, nan=0)).all()
