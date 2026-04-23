"""Inference wrapper for Route 4 synergy head.

Loads the trained head checkpoint and exposes:
  - predict_synergy(drug_name_pairs) → np.ndarray of synergy scores
  - predict_combo_auc(patient_single_aucs, pair) → combined AUC using
    additive baseline + learned synergy

For API consistency with the existing kit_predict pipeline.
"""

from __future__ import annotations

import itertools
from dataclasses import fields as _fields
from pathlib import Path

import numpy as np
import torch

from combo_val.combo.synergy_head import SynergyHead, SynergyHeadConfig


class SynergyInference:
    """Wrapper around the trained SynergyHead for kit-side inference."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=self.device)

        cfg_dict = ckpt["cfg"]
        kwargs = {f.name: cfg_dict[f.name] for f in _fields(SynergyHeadConfig)
                  if f.name in cfg_dict}
        cfg = SynergyHeadConfig(**kwargs)

        self.head = SynergyHead(
            ckpt["n_drugs_vocab"], ckpt["drug_emb_dim"], cfg,
        ).to(self.device)
        self.head.load_state_dict(ckpt["synergy_head_state_dict"])
        self.head.eval()

        self.drug_vocab: list[str] = ckpt["drug_vocab"]
        self.drug_to_int: dict[str, int] = ckpt["drug_to_int"]
        self.synergy_col: str = ckpt["synergy_col"]

    @torch.no_grad()
    def predict_synergy_matrix(self, drug_indices: list[int]) -> np.ndarray:
        """Returns symmetric (n, n) matrix of predicted synergy_loewe for all pairs.

        Symmetric by averaging forward(d1, d2) and forward(d2, d1).
        Diagonal is unused (single-drug, not synergistic).
        """
        n = len(drug_indices)
        shifted = np.array([i + 1 for i in drug_indices], dtype=np.int64)
        d1_grid = np.repeat(shifted, n)
        d2_grid = np.tile(shifted, n)
        d1 = torch.tensor(d1_grid, dtype=torch.long, device=self.device)
        d2 = torch.tensor(d2_grid, dtype=torch.long, device=self.device)
        pred_ab = self.head(d1, d2).cpu().numpy()
        pred_ba = self.head(d2, d1).cpu().numpy()
        sym = 0.5 * (pred_ab + pred_ba)
        return sym.reshape(n, n)

    @torch.no_grad()
    def predict_synergy(self, drug1_names: list[str], drug2_names: list[str]) -> np.ndarray:
        """Batch synergy prediction by drug names."""
        d1 = torch.tensor(
            [self.drug_to_int[d] + 1 for d in drug1_names],
            dtype=torch.long, device=self.device,
        )
        d2 = torch.tensor(
            [self.drug_to_int[d] + 1 for d in drug2_names],
            dtype=torch.long, device=self.device,
        )
        pred_ab = self.head(d1, d2).cpu().numpy()
        pred_ba = self.head(d2, d1).cpu().numpy()
        return 0.5 * (pred_ab + pred_ba)


def load_synergy_head(
    checkpoint_path: Path = Path("runs/set_drug_predictor_v3_route4/synergy_head.pt"),
    device: str = "cpu",
) -> SynergyInference | None:
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    return SynergyInference(cp, device=device)
