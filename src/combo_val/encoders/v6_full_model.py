"""v0.6 end-to-end composite: GIN drug + GAT gene + Set-Transformer combo head.

This is the FULL v0.6 model. Composing Phase 1 (DrugGIN), Phase 2
(GeneContextGAT), and Phase 3 (GNNSetTransformerCombo) into a single
nn.Module that the Phase 5 training/ablation script trains end-to-end.

Architecture:

    SMILES list (per combo, N drugs)
         ↓
    DrugGIN per drug → drug_embs [B, N_max, drug_emb_dim]
         ↓
    drug_mask [B, N_max]
         ↓                                      patient mutations [B, 25]
         ↓                                                ↓
         ↓                                  GeneContextGAT → patient_dna_emb [B, 64]
         ↓                                                ↓
         └──────────────┬───────────────────────────────┘
                        ▼
                 GNNSetTransformerCombo
                        ↓
                 predicted combo AUC [B]

For ABLATION purposes, the model has switches:
  - use_gin_drug: True → DrugGIN over SMILES; False → nn.Embedding over drug_id
  - use_gat_gene: True → GeneContextGAT over PPI; False → raw mut binary
  - use_drugcomb_pretrain: just a flag; pretrain happens in a separate script

Ablation matrix (Phase 5):
  v0.6.0  full         : GIN ✓ GAT ✓ pretrain ✓
  v0.6.A  drug-only    : GIN ✓ GAT ✗ pretrain ✓
  v0.6.B  gene-only    : GIN ✗ GAT ✓ pretrain ✗ (no SMILES needed)
  v0.6.C  no-pretrain  : GIN ✓ GAT ✓ pretrain ✗
  v0.5    baseline     : GIN ✗ GAT ✗ pretrain ✗  (i.e., MLP + nn.Embedding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig
from combo_val.encoders.gene_ppi_gat import GeneContextGAT, GeneGATConfig
from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data
from combo_val.combo.gnn_set_transformer import (
    GNNSetTransformerCombo, GNNSetTransformerConfig,
)


@dataclass
class V6FullModelConfig:
    # Component switches (for ablation)
    use_gin_drug: bool = True
    use_gat_gene: bool = True

    # Sub-component configs
    drug_gin: DrugGINConfig = field(default_factory=DrugGINConfig)
    gene_gat: GeneGATConfig = field(default_factory=GeneGATConfig)
    combo_head: GNNSetTransformerConfig = field(default_factory=GNNSetTransformerConfig)

    # For nn.Embedding fallback (when use_gin_drug=False)
    n_drugs_total: int = 165


class V6FullModel(nn.Module):
    """v0.6 composite model — GIN + GAT + Set-Transformer combo head.

    Forward signature:
      forward(
        drug_batch:    PyG Batch  OR  drug_ids: Long[B, N_max]
                                       depending on use_gin_drug
        drug_mask:     [B, N_max]   (1 = real, 0 = pad)
        mut_features:  [B, 25]      (mutation binary per patient)
        rna_pc:        [B, 50]
        clinical:      [B, 21]
        cyto:          [B, 8]
      ) → predicted combo AUC [B]
    """

    def __init__(self, ppi_json_path: Path, gene_order: list[str],
                 cfg: Optional[V6FullModelConfig] = None):
        super().__init__()
        self.cfg = cfg or V6FullModelConfig()

        # --- Drug encoder ---
        if self.cfg.use_gin_drug:
            self.drug_encoder = DrugGIN(self.cfg.drug_gin)
        else:
            self.drug_encoder = nn.Embedding(
                self.cfg.n_drugs_total, self.cfg.drug_gin.drug_emb_dim,
            )

        # --- Gene encoder ---
        if self.cfg.use_gat_gene:
            self.gene_encoder = GeneContextGAT(
                ppi_json_path, gene_order, self.cfg.gene_gat,
            )
            patient_dna_dim = self.cfg.gene_gat.out_dim
        else:
            # Identity / linear projection from raw 25-dim binary
            self.gene_encoder = nn.Linear(25, self.cfg.gene_gat.out_dim)
            patient_dna_dim = self.cfg.gene_gat.out_dim

        # --- Combo head ---
        # Patch combo head config so its patient_emb_dim matches gene_encoder out
        self.cfg.combo_head.patient_emb_dim = patient_dna_dim
        self.combo_head = GNNSetTransformerCombo(self.cfg.combo_head)

    def encode_drugs(self, drug_input, n_max: int) -> torch.Tensor:
        """Returns [B, N_max, drug_emb_dim].

        - If use_gin_drug=True: drug_input is a list[list[str|None]] of
          SMILES per combo, padded with None.
        - If use_gin_drug=False: drug_input is Long[B, N_max] of drug indices,
          0 (or any) for padded slots.
        """
        if self.cfg.use_gin_drug:
            # drug_input: list of [smiles_1, smiles_2, None, ...] per combo
            # Flatten + featurize + batch + scatter back
            assert isinstance(drug_input, list)
            B = len(drug_input)
            assert all(len(combo) == n_max for combo in drug_input)

            valid_smiles = []
            valid_pos = []  # (b, n) per valid drug
            for b, combo in enumerate(drug_input):
                for n, smi in enumerate(combo):
                    if smi:
                        g = smiles_to_molgraph(smi)
                        if g is not None:
                            valid_smiles.append(to_pyg_data(g))
                            valid_pos.append((b, n))

            if not valid_smiles:
                return torch.zeros(B, n_max, self.cfg.drug_gin.drug_emb_dim,
                                    device=next(self.parameters()).device)

            batch = Batch.from_data_list(valid_smiles).to(
                next(self.parameters()).device
            )
            flat_emb = self.drug_encoder(batch)  # [n_valid, drug_emb_dim]

            out = torch.zeros(B, n_max, self.cfg.drug_gin.drug_emb_dim,
                               device=flat_emb.device)
            for slot_i, (b, n) in enumerate(valid_pos):
                out[b, n] = flat_emb[slot_i]
            return out
        else:
            assert isinstance(drug_input, torch.Tensor)
            return self.drug_encoder(drug_input)   # [B, N_max, drug_emb_dim]

    def encode_genes(self, mut_features: torch.Tensor) -> torch.Tensor:
        """[B, 25] → [B, patient_dna_dim]"""
        if self.cfg.use_gat_gene:
            return self.gene_encoder(mut_features)
        else:
            return self.gene_encoder(mut_features)  # nn.Linear path

    def forward(self, drug_input, drug_mask: torch.Tensor,
                mut_features: torch.Tensor, rna_pc: torch.Tensor,
                clinical: torch.Tensor, cyto: torch.Tensor) -> torch.Tensor:
        n_max = drug_mask.shape[1]
        drug_embs = self.encode_drugs(drug_input, n_max)
        patient_dna_emb = self.encode_genes(mut_features)
        return self.combo_head(
            drug_embs, drug_mask, patient_dna_emb, rna_pc, clinical, cyto,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
