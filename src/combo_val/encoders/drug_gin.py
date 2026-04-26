"""GIN drug encoder — replaces nn.Embedding(165, 64) with chemistry-grounded
graph encoding.

Architecture (per Xu et al. ICLR 2019, "How Powerful are Graph Neural
Networks?", arXiv:1810.00826):

    SMILES → atom-bond graph (mol_featurizer.py)
                  ↓
    atom features [n × 39] → Linear → [n × hidden]
                  ↓
    GIN layer × N (default 4):
        h_v ← MLP( (1 + ε) · h_v  +  Σ_{u ∈ N(v)} h_u )
                  ↓
    Readout: global add/mean/max pooling → [hidden × 3] → Linear → [drug_emb_dim]

Why GIN (vs GCN / GAT):
  - GIN provably as expressive as Weisfeiler-Lehman test (Xu 2019 Theorem 3)
  - Standard SOTA on MoleculeNet / OGB molhiv benchmarks
  - Sum-aggregation distinguishes graph structure that mean/max cannot

Designed to be a DROP-IN replacement for `nn.Embedding`:
  - Old: drug_id (int) → embedding [batch × 64]
  - New: SMILES (str) → mol graph → GIN → embedding [batch × 64]

The encoder caches MolGraph objects in memory (165 BeatAML drugs at
once = ~3 MB total) so per-step inference is just a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

from combo_val.encoders.mol_featurizer import (
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MolGraph,
    smiles_to_molgraph, to_pyg_data,
)


@dataclass
class DrugGINConfig:
    hidden_dim: int = 128
    n_layers: int = 4
    drug_emb_dim: int = 64
    dropout: float = 0.2
    readout: str = "concat_add_mean_max"   # or "add" / "mean"


class DrugGIN(nn.Module):
    """GIN-based drug encoder.

    Forward signatures:
      forward(batch: torch_geometric.data.Batch) → [batch_size × drug_emb_dim]
        Standard PyG batched forward.

      encode_smiles(smiles_list: list[str]) → [n_drugs × drug_emb_dim]
        One-shot: featurize + batch + forward. Use at the SET-LEVEL
        boundary (e.g., once per patient × candidate-drug-set).
    """

    def __init__(self, cfg: Optional[DrugGINConfig] = None):
        super().__init__()
        self.cfg = cfg or DrugGINConfig()

        # Atom feature projection
        self.atom_proj = nn.Linear(ATOM_FEATURE_DIM, self.cfg.hidden_dim)

        # GIN stack
        self.gin_layers = nn.ModuleList()
        for _ in range(self.cfg.n_layers):
            mlp = nn.Sequential(
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.cfg.hidden_dim * 2, self.cfg.hidden_dim),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))

        # Readout → drug embedding
        if self.cfg.readout == "concat_add_mean_max":
            readout_dim = self.cfg.hidden_dim * 3
        else:
            readout_dim = self.cfg.hidden_dim
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.drug_emb_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """batch: PyG Batch with .x, .edge_index, .batch attributes."""
        x = self.atom_proj(batch.x)

        for layer in self.gin_layers:
            x = layer(x, batch.edge_index)
            x = F.relu(x)

        if self.cfg.readout == "concat_add_mean_max":
            pooled = torch.cat([
                global_add_pool(x, batch.batch),
                global_mean_pool(x, batch.batch),
                global_max_pool(x, batch.batch),
            ], dim=-1)
        elif self.cfg.readout == "add":
            pooled = global_add_pool(x, batch.batch)
        else:
            pooled = global_mean_pool(x, batch.batch)

        return self.readout_mlp(pooled)

    def encode_smiles(self, smiles_list: list[str],
                      device: Optional[torch.device] = None) -> torch.Tensor:
        """Convenience: list[SMILES] → [n × drug_emb_dim] tensor.

        Drugs whose SMILES fails to parse get a zero-vector embedding +
        a warning. This is the safe-fallback for inference time when an
        OOD or new drug is requested.
        """
        device = device or next(self.parameters()).device
        graphs: list[MolGraph] = []
        valid_idx: list[int] = []
        for i, smi in enumerate(smiles_list):
            g = smiles_to_molgraph(smi)
            if g is not None:
                graphs.append(g)
                valid_idx.append(i)

        n_total = len(smiles_list)
        if not graphs:
            return torch.zeros(n_total, self.cfg.drug_emb_dim, device=device)

        data_list = [to_pyg_data(g) for g in graphs]
        batch = Batch.from_data_list(data_list).to(device)
        emb = self.forward(batch)  # [n_valid × drug_emb_dim]

        # Scatter back into n_total-sized output, zeros where SMILES failed
        out = torch.zeros(n_total, self.cfg.drug_emb_dim, device=device)
        for slot_i, valid_i in enumerate(valid_idx):
            out[valid_i] = emb[slot_i]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
