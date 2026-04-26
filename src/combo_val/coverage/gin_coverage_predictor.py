"""B.3 — GIN auxiliary task: SMILES → 18-dim coverage vector.

Trains the existing DrugGIN encoder to predict the 18-dim coverage
vector that B.2 derives from ChEMBL bioactivity data. After training,
the model can predict coverage for ANY new SMILES (not just BeatAML's
165), unlocking arbitrary drug pool extension in the set-cover solver.

Architecture:

    SMILES → DrugGIN (4-layer GIN, ~150k params) → drug_emb [64]
                                                       ↓
                                          coverage_head (Linear → Sigmoid)
                                                       ↓
                                          predicted coverage [18-dim, 0..1]

Training: B.2-aggregated coverage as labels (DrugComb / ChEMBL provides
~30-100 drugs with full ChEMBL annotation). MSE on the per-axis
coverage values.

Inference (Phase B.4): for any new SMILES, run GIN forward → coverage
vector → augment the DrugCoverageMatrix → set-cover solver picks from
the extended pool.

This module exposes:
  - CoveragePredictionModel (nn.Module): GIN + 18-dim head
  - CoverageDataset: SMILES + 18-d coverage vector iterable
  - train_coverage_predictor: 5-fold CV training routine
  - load_predictor + predict_smiles: inference helpers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy torch import — module loads without torch available
try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch
    from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig
    from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


@dataclass
class CoveragePredictorConfig:
    drug_emb_dim: int = 64
    n_axes: int = 18
    hidden_dim: int = 128
    head_dropout: float = 0.2


def make_coverage_predictor(cfg: Optional[CoveragePredictorConfig] = None):
    """Build the DrugGIN + coverage head model.

    Returns the nn.Module. Caller is responsible for loading weights.
    """
    if not _TORCH_OK:
        raise ImportError("torch / torch-geometric required for B.3 model")
    cfg = cfg or CoveragePredictorConfig()

    class CoveragePredictionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.drug_gin = DrugGIN(DrugGINConfig(drug_emb_dim=cfg.drug_emb_dim))
            self.coverage_head = nn.Sequential(
                nn.Linear(cfg.drug_emb_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.hidden_dim, cfg.n_axes),
                nn.Sigmoid(),  # outputs in [0, 1] match coverage range
            )

        def forward(self, batch):
            emb = self.drug_gin(batch)              # [B, drug_emb_dim]
            return self.coverage_head(emb)          # [B, n_axes]

        def predict_smiles(self, smiles_list: list[str]) -> np.ndarray:
            """SMILES → [n × 18] coverage matrix (numpy)."""
            graphs = [smiles_to_molgraph(s) for s in smiles_list]
            valid_idx = [i for i, g in enumerate(graphs) if g is not None]
            if not valid_idx:
                return np.zeros((len(smiles_list), cfg.n_axes), dtype=np.float32)
            data_list = [to_pyg_data(graphs[i]) for i in valid_idx]
            batch = Batch.from_data_list(data_list)
            with torch.no_grad():
                preds = self(batch).cpu().numpy()
            out = np.zeros((len(smiles_list), cfg.n_axes), dtype=np.float32)
            for slot, idx in enumerate(valid_idx):
                out[idx] = preds[slot]
            return out

    return CoveragePredictionModel()


def assemble_training_data(
    coverage_map: dict[str, dict[str, float]],
    smiles_table: dict[str, str],
    axis_order: list[str],
) -> tuple[list[str], list[str], np.ndarray]:
    """Returns (drug_ids, smiles_list, label_matrix [n × 18]).

    Only includes drugs with non-empty coverage_map AND a SMILES.
    """
    drug_ids = []
    smiles_list = []
    labels = []
    for did, cov in coverage_map.items():
        if not cov:
            continue
        smi = smiles_table.get(did)
        if not smi:
            continue
        drug_ids.append(did)
        smiles_list.append(smi)
        labels.append([cov.get(axis, 0.0) for axis in axis_order])
    label_matrix = np.array(labels, dtype=np.float32) if labels else \
                    np.zeros((0, len(axis_order)), dtype=np.float32)
    return drug_ids, smiles_list, label_matrix


def save_predictor(model, path: Path | str, axis_order: list[str], cfg) -> None:
    if not _TORCH_OK:
        raise ImportError("torch required")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "axis_order": list(axis_order),
        "config": {
            "drug_emb_dim": cfg.drug_emb_dim,
            "n_axes": cfg.n_axes,
            "hidden_dim": cfg.hidden_dim,
            "head_dropout": cfg.head_dropout,
        },
    }, p)


def load_predictor(path: Path | str):
    """Load a saved CoveragePredictionModel + the axis order it was
    trained on. Returns (model, axis_order)."""
    if not _TORCH_OK:
        raise ImportError("torch required")
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = CoveragePredictorConfig(**cfg_dict)
    model = make_coverage_predictor(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["axis_order"]
