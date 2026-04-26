"""Phase 3 — Set-Transformer combo head over GIN drug node embeddings.

Bridges Phase 1 (DrugGIN per-drug embedding) + Phase 2 (GeneContextGAT
patient embedding) to combo-level prediction:

    For an N-drug combo {d_1, ..., d_N} and a patient p:

      drug_embs = [DrugGIN(SMILES_i) for i in 1..N]    # N × drug_emb_dim
            ↓
      Set Transformer (ISAB → PMA)
            ↓
      combo_set_emb [hidden]
            ↓
      patient_emb = concat(GeneContextGAT(mut), RNA-PC, clinical)
            ↓
      MLP → predicted combo AUC

Why Set Transformer (not MLP / sum / mean):
  - Permutation-invariant by construction (drug ordering doesn't matter)
  - Learns inter-drug interactions via attention (e.g., FLT3i + BCL2i
    interact differently than FLT3i + chemotherapy)
  - Handles arbitrary N (2, 3, 4+ drugs) without architecture change

This head is independent of how drug embeddings are produced — it
accepts ANY (B × N × drug_emb_dim) tensor + per-drug mask. So the same
head works with the v0.5 nn.Embedding baseline (for ablation) and the
v0.6 DrugGIN.

References:
  Lee et al. ICML 2019 — Set Transformer (arXiv:1810.00825)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNSetTransformerConfig:
    drug_emb_dim: int = 64
    patient_emb_dim: int = 64        # from GeneContextGAT
    rna_pc_dim: int = 50             # from BeatAML preprocessor
    clinical_dim: int = 21           # clin_* features
    cyto_dim: int = 8                # fusion + karyo

    set_hidden: int = 128            # internal Set-Transformer hidden
    n_heads: int = 4
    n_inducing: int = 16             # M in ISAB

    head_hidden: int = 256
    dropout: float = 0.2


class MAB(nn.Module):
    """Multi-head Attention Block — MAB(X, Y) = LayerNorm(X + MHA(X,Y,Y)) + FFN.
    Per Set Transformer (Lee 2019)."""
    def __init__(self, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        assert d_out % n_heads == 0
        self.fc_q = nn.Linear(d_in, d_out)
        self.fc_k = nn.Linear(d_in, d_out)
        self.fc_v = nn.Linear(d_in, d_out)
        self.attn = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.ffn = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.ReLU(),
            nn.Linear(d_out * 2, d_out),
        )

    def forward(self, X, Y, key_mask=None):
        Q = self.fc_q(X)
        K = self.fc_k(Y)
        V = self.fc_v(Y)
        attn_out, _ = self.attn(Q, K, V, key_padding_mask=key_mask)
        h = self.ln1(Q + attn_out)
        h = self.ln2(h + self.ffn(h))
        return h


class ISAB(nn.Module):
    """Induced Set Attention Block — O(N · M) instead of O(N²)."""
    def __init__(self, d_in: int, d_out: int, n_heads: int, n_inducing: int):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_out))
        nn.init.xavier_uniform_(self.I)
        self.mab1 = MAB(d_out, d_out, n_heads)
        self.mab2 = MAB(d_in, d_out, n_heads)
        # Project I from d_out to act as MAB query in mab1
        self.in_proj = nn.Linear(d_in, d_out)

    def forward(self, X, key_mask=None):
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        # mab1: inducing points attend to X (key_mask masks padded drugs)
        H = self.mab1(I, self.in_proj(X), key_mask=key_mask)
        # mab2: X attends back to inducing-point summary
        Y = self.mab2(X, H)  # (no mask needed — H is full inducing set)
        return Y


class PMA(nn.Module):
    """Pooling by Multihead Attention — k=1 seed → set-level vector."""
    def __init__(self, d: int, n_heads: int, n_seeds: int = 1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, n_seeds, d))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d, d, n_heads)

    def forward(self, X, key_mask=None):
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_mask=key_mask).squeeze(1)  # [B, d]


class GNNSetTransformerCombo(nn.Module):
    """Composite: drug-set GNN encoder → Set Transformer → combo AUC head.

    Forward signature:
      forward(
        drug_embs:     [B, N_max, drug_emb_dim],   # 0-padded
        drug_mask:     [B, N_max],                  # 1=real, 0=pad
        patient_dna_emb: [B, patient_emb_dim],     # from GeneContextGAT
        rna_pc:        [B, rna_pc_dim],
        clinical:      [B, clinical_dim],
        cyto:          [B, cyto_dim],
      ) → predicted_combo_auc [B]
    """

    def __init__(self, cfg: Optional[GNNSetTransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or GNNSetTransformerConfig()

        # Set-Transformer drug-set encoder (2 ISAB → PMA)
        self.drug_proj = nn.Linear(self.cfg.drug_emb_dim, self.cfg.set_hidden)
        self.isab1 = ISAB(self.cfg.set_hidden, self.cfg.set_hidden,
                           self.cfg.n_heads, self.cfg.n_inducing)
        self.isab2 = ISAB(self.cfg.set_hidden, self.cfg.set_hidden,
                           self.cfg.n_heads, self.cfg.n_inducing)
        self.pma = PMA(self.cfg.set_hidden, self.cfg.n_heads, n_seeds=1)

        # Combo response head
        cat_dim = (
            self.cfg.set_hidden
            + self.cfg.patient_emb_dim
            + self.cfg.rna_pc_dim
            + self.cfg.clinical_dim
            + self.cfg.cyto_dim
        )
        self.head = nn.Sequential(
            nn.Linear(cat_dim, self.cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden // 4),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden // 4, 1),
        )

    def forward(self, drug_embs: torch.Tensor, drug_mask: torch.Tensor,
                patient_dna_emb: torch.Tensor, rna_pc: torch.Tensor,
                clinical: torch.Tensor, cyto: torch.Tensor) -> torch.Tensor:
        # PyTorch MultiheadAttention key_padding_mask: True == ignore
        key_mask = (drug_mask == 0)   # invert: 0 = pad → True (ignore)

        x = self.drug_proj(drug_embs)              # [B, N, set_hidden]
        x = self.isab1(x, key_mask=key_mask)
        x = self.isab2(x, key_mask=key_mask)
        combo_set_emb = self.pma(x, key_mask=key_mask)   # [B, set_hidden]

        cat = torch.cat([
            combo_set_emb, patient_dna_emb, rna_pc, clinical, cyto,
        ], dim=-1)
        return self.head(cat).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
