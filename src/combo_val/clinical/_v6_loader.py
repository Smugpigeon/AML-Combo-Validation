"""v0.6 GNN backbone loader — bridges checkpoint to V6FullModel.

Used by kit_predict.predict_for_patient when backbone == "gnn-v6".
Until Phase 5 passes the GO threshold, this loader is unreachable
(early-rejection gate in kit_predict raises NotImplementedError before
reaching dispatch). Once GO, the gate is removed and dispatch routes
to predict_combo_auc_via_gnn_v6() in this module.

Inputs at inference:
  - rna_counts (Series) → standard feature builder (same as MLP path)
  - kit (KitInput) → mutations, karyotype, lab values
  - drug_smiles_table (loaded once at module init) → SMILES per drug_id
  - PPI subgraph (loaded once at module init) → STRING 25-gene network

Outputs:
  - combo_auc matrix [n_drugs, n_drugs] — same shape as MLP / ST paths
    so downstream pair-ranking, OOD suppression, etc. all work unchanged
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np


@lru_cache(maxsize=4)
def load_smiles_table(csv_path: str) -> dict[str, str]:
    """Cached lookup. Returns {beataml_drug_id: SMILES}, isomeric preferred."""
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            smi = r.get("isomeric_smiles") or r.get("canonical_smiles") or ""
            if smi:
                out[r["beataml_drug_id"]] = smi
    return out


@lru_cache(maxsize=4)
def _gene_order_from_features(features_csv: str) -> tuple[str, ...]:
    """Read mut_<GENE> columns from BeatAML features CSV (cached)."""
    with open(features_csv) as f:
        header = f.readline().strip().split(",")
    return tuple(h[len("mut_"):] for h in header if h.startswith("mut_"))


def load_v6_model(checkpoint_path: Path, ppi_json: Path,
                   features_csv: Path, device: str = "cpu"):
    """Load V6FullModel from a Phase 5 / Phase 1 checkpoint.

    Returns (model, config_dict). Raises FileNotFoundError if ckpt absent.
    """
    import torch
    from combo_val.encoders.v6_full_model import V6FullModel, V6FullModelConfig

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"v6 checkpoint not present at {checkpoint_path}. "
            f"Run scripts/v6_ablation.py or scripts/train_drug_gin.py first."
        )

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    gene_order = list(_gene_order_from_features(str(features_csv)))

    # Use the saved ablation arm's config if present; otherwise default to full
    arm_name = ckpt.get("arm", "v0.6.0_full")
    cfg = V6FullModelConfig(
        use_gin_drug=("baseline" not in arm_name and "gene_only" not in arm_name),
        use_gat_gene=("baseline" not in arm_name and "drug_only" not in arm_name),
    )
    model = V6FullModel(ppi_json, gene_order, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {"arm": arm_name, "epoch": ckpt.get("best", {}).get("epoch")}


def predict_combo_auc_via_gnn_v6(
    model,
    smiles_table: dict[str, str],
    drug_vocab_filt: list[str],
    patient_features_104: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the [n_drugs × n_drugs] combo AUC matrix using v0.6 model.

    Approach (matching the MLP path semantics):
      1. Get drug_emb [n_drugs × drug_emb_dim] via DrugGIN per drug SMILES
      2. Get patient_dna_emb [1 × patient_emb_dim] via GeneContextGAT
      3. For every (i, j) pair, run combo head → combo_auc[i, j]
         Vectorized: build a single batch of all upper-triangular pairs.

    Returns symmetric (n_drugs, n_drugs) numpy matrix; diagonal = single-drug
    self-combo (degenerate, can be ignored by caller).
    """
    import torch

    n_d = len(drug_vocab_filt)
    if n_d == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # 1) Per-drug embeddings (encode the entire vocab once)
    smiles_list = [smiles_table.get(d, "") for d in drug_vocab_filt]
    if model.cfg.use_gin_drug:
        # encode_drugs takes [B=1, N_max=n_d] worth of SMILES — the model
        # internally handles missing SMILES by zeroing those slots.
        drug_input = [smiles_list]
        drug_embs_one = model.encode_drugs(drug_input, n_max=n_d)  # [1, n_d, dim]
        per_drug_emb = drug_embs_one.squeeze(0)                    # [n_d, dim]
    else:
        # Baseline path uses int drug indices instead of SMILES
        drug_ids = torch.arange(n_d, device=device).unsqueeze(0)
        per_drug_emb = model.drug_encoder(drug_ids).squeeze(0)

    # 2) Patient gene-context embedding (only mutation features — 25 dims)
    feat = torch.from_numpy(patient_features_104.astype(np.float32)).to(device)
    # mut_* columns are 25 genes — find them in the 104-dim feature vector.
    # By convention rna_pc01-50 are first 50, then mut_* (25), then clin_*
    # (21), then fusion_/karyo_ (8). Use the slicing the v6 ablation script
    # used: mut_cols come right after rna_pc.
    mut = feat[50:75].unsqueeze(0)        # [1, 25]
    rna_pc = feat[:50].unsqueeze(0)       # [1, 50]
    clin = feat[75:96].unsqueeze(0)       # [1, 21]
    cyto = feat[96:104].unsqueeze(0)      # [1, 8]
    patient_dna_emb = model.encode_genes(mut)  # [1, gene_emb_dim]

    # 3) Pairwise combo head — build all upper-triangular pairs
    pairs_i, pairs_j = np.triu_indices(n_d, k=1)
    n_pairs = len(pairs_i)

    # Build pair drug_embs: [n_pairs, 2, drug_emb_dim]
    pair_drug_embs = torch.stack([
        per_drug_emb[pairs_i], per_drug_emb[pairs_j],
    ], dim=1)
    pair_mask = torch.ones(n_pairs, 2, dtype=torch.long, device=device)

    # Broadcast patient + RNA + clin + cyto across all pairs
    patient_dna_b = patient_dna_emb.expand(n_pairs, -1)
    rna_b = rna_pc.expand(n_pairs, -1)
    clin_b = clin.expand(n_pairs, -1)
    cyto_b = cyto.expand(n_pairs, -1)

    with torch.no_grad():
        combo_auc_pairs = model.combo_head(
            pair_drug_embs, pair_mask,
            patient_dna_b, rna_b, clin_b, cyto_b,
        )                                   # [n_pairs]
    combo_auc_pairs_np = combo_auc_pairs.cpu().numpy()

    # Pack back into symmetric matrix
    out = np.zeros((n_d, n_d), dtype=np.float32)
    out[pairs_i, pairs_j] = combo_auc_pairs_np
    out[pairs_j, pairs_i] = combo_auc_pairs_np
    # Diagonal = single-drug self-combo; leave at 0 (caller should ignore)
    return out
