"""GAT (Graph Attention Network) over the STRING PPI subgraph.

Architecture (per Veličković et al. ICLR 2018, "Graph Attention Networks",
arXiv:1710.10903):

    25 driver genes ─── STRING PPI edges (13 high-confidence physical interactions)
                  ↓
    per-patient mutation binary [25] → node features [25 × 1]
                  ↓
    GAT layer × N (default 3, n_heads=4):
        e_{ij} = LeakyReLU( a^T [W h_i || W h_j] )
        α_{ij} = softmax_j(e_{ij})
        h_i' = σ( Σ_{j ∈ N(i)} α_{ij} W h_j )
                  ↓
    Per-gene node embeddings [25 × hidden]
                  ↓
    Readout: weighted by patient's mutation status →  patient_dna_emb [emb_dim]

Why GAT (vs GIN / GraphSAGE):
  - Attention coefficient α_{ij} is interpretable per-patient: which gene
    pathway (e.g., RAS: NRAS-KRAS-PTPN11) contributed most to the
    embedding for THIS patient. Useful for clinician audit.
  - Multi-head attention captures multiple interaction modes (kinase
    cascade vs scaffold vs co-complex)
  - Edges in our subgraph are sparse (density 0.043) — attention can
    learn to ignore weak connections without hurting strong ones.

Designed to be COMPOSED with the GIN drug encoder + RNA + clinical
features in the v0.6 main model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


@dataclass
class GeneGATConfig:
    n_genes: int = 25
    in_dim: int = 1                  # mutation binary
    hidden_dim: int = 32
    n_heads: int = 4
    n_layers: int = 3
    out_dim: int = 64                # per-patient gene-context embedding
    dropout: float = 0.2
    edge_score_threshold: int = 700  # match STRING fetch threshold


def _load_ppi_edge_index(ppi_json_path: Path,
                          gene_order: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    """Load STRING PPI subgraph → (edge_index [2 × E], gene_to_idx).

    Edges are made bidirectional (undirected protein interactions). Self-loops
    are ADDED for each node so any gene with no PPI partners still receives
    its own message (self-attention).
    """
    with open(ppi_json_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes = g["nodes"]
    if set(nodes) != set(gene_order):
        missing = set(gene_order) - set(nodes)
        extra = set(nodes) - set(gene_order)
        raise ValueError(
            f"Gene order vs PPI nodes mismatch: missing={missing}, extra={extra}"
        )

    gene_to_idx = {g_: i for i, g_ in enumerate(gene_order)}

    src, dst = [], []
    for e in g["edges"]:
        i = gene_to_idx[e["a"]]
        j = gene_to_idx[e["b"]]
        # Bidirectional
        src += [i, j]
        dst += [j, i]
    # Self-loops
    for i in range(len(gene_order)):
        src.append(i)
        dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index, gene_to_idx


class GeneContextGAT(nn.Module):
    """GAT over the patient mutation panel + STRING PPI graph.

    Forward signatures:
      forward(mut_features [B × n_genes]) → patient_dna_emb [B × out_dim]

    Shape rules:
      - n_genes is FIXED at 25 to match the BeatAML mut_* feature set
      - edge_index is shared across the whole batch (same PPI structure)
        but expanded to [2, E × B] inside via PyG batching for efficiency
    """

    def __init__(self, ppi_json_path: Path, gene_order: list[str],
                 cfg: Optional[GeneGATConfig] = None):
        super().__init__()
        self.cfg = cfg or GeneGATConfig()
        if len(gene_order) != self.cfg.n_genes:
            raise ValueError(
                f"gene_order has {len(gene_order)} entries, expected {self.cfg.n_genes}"
            )

        edge_index, gene_to_idx = _load_ppi_edge_index(ppi_json_path, gene_order)
        # Register as buffer so it moves with .to(device) but isn't trainable
        self.register_buffer("edge_index", edge_index)
        self.gene_order = list(gene_order)
        self.gene_to_idx = gene_to_idx

        # Gene identity embedding (one per gene; learned)
        self.gene_id_emb = nn.Embedding(self.cfg.n_genes, self.cfg.hidden_dim)

        # Project (mutation binary, gene_id_emb) → hidden
        self.input_proj = nn.Linear(
            self.cfg.in_dim + self.cfg.hidden_dim, self.cfg.hidden_dim,
        )

        # GAT stack
        self.gat_layers = nn.ModuleList()
        for layer_i in range(self.cfg.n_layers):
            in_d = self.cfg.hidden_dim if layer_i == 0 else self.cfg.hidden_dim
            self.gat_layers.append(
                GATConv(in_channels=in_d,
                         out_channels=self.cfg.hidden_dim // self.cfg.n_heads,
                         heads=self.cfg.n_heads,
                         dropout=self.cfg.dropout,
                         add_self_loops=False)  # we added self-loops manually
            )

        # Patient-level readout: weighted sum of node embeddings, with
        # mutation status as the weighting (mutated genes contribute more).
        # Mathematically: patient_emb = Σ_g (1 + α · mut[g]) · h_g  →  Linear → out_dim
        # We use this gating so absent-mutation genes still contribute baseline
        # signal (catch off-panel context), but mutated genes dominate.
        self.mutation_gate = nn.Parameter(torch.tensor(2.0))   # learnable α

        self.readout = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim),
        )

    def forward(self, mut_features: torch.Tensor) -> torch.Tensor:
        """
        mut_features: [B, n_genes] mutation binary (or float) — 1 if patient
                      has a mutation in that gene, 0 otherwise.

        Returns: [B, out_dim] patient gene-context embedding
        """
        B, n_genes = mut_features.shape
        assert n_genes == self.cfg.n_genes, \
            f"Expected {self.cfg.n_genes} genes, got {n_genes}"

        device = mut_features.device

        # Build per-batch node features: concat (mut_bit, gene_id_emb)
        # x: [B, n_genes, hidden_dim + 1]
        gene_ids = torch.arange(n_genes, device=device)
        gene_id_emb = self.gene_id_emb(gene_ids)          # [n_genes, hidden_dim]
        gene_id_emb_b = gene_id_emb.unsqueeze(0).expand(B, -1, -1)
        mut_b = mut_features.unsqueeze(-1)                # [B, n_genes, 1]
        x = torch.cat([mut_b, gene_id_emb_b], dim=-1)     # [B, n_genes, 1+hidden]
        x = self.input_proj(x)                            # [B, n_genes, hidden]

        # PyG message passing expects flat [B*n_genes, hidden] + edge_index
        # offset per batch element. Do batched expansion.
        x_flat = x.reshape(B * n_genes, -1)               # [B*n_genes, hidden]
        # Edge index expansion: shift by k * n_genes for batch element k
        edge_index = self.edge_index                       # [2, E]
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=device) * n_genes).view(B, 1, 1)
        edge_index_b = edge_index.unsqueeze(0) + offsets  # [B, 2, E]
        edge_index_b = edge_index_b.permute(1, 0, 2).reshape(2, B * E)

        for layer in self.gat_layers:
            x_flat = layer(x_flat, edge_index_b)
            x_flat = F.elu(x_flat)

        # Reshape back to per-batch
        h = x_flat.view(B, n_genes, -1)                   # [B, n_genes, hidden]

        # Patient-level readout: gated weighted sum
        gate = 1.0 + self.mutation_gate * mut_features    # [B, n_genes]
        weighted = h * gate.unsqueeze(-1)                 # [B, n_genes, hidden]
        pooled = weighted.sum(dim=1) / (gate.sum(dim=1, keepdim=True) + 1e-6)

        return self.readout(pooled)
