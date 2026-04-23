"""Inference wrapper for Path B Set Transformer — drop-in Layer-3 replacement.

Mirrors the API shape of Baseline A's SingleDrugMLP so kit_predict can swap
between backbones without structural changes:

    predictor = load_set_drug_predictor("runs/set_drug_predictor_v2_stable/final_model.pt")
    singles    = predictor.predict_singles(patient_features_104d)
        # → np.ndarray (n_drugs,) = AUC for each drug as a 1-element set

    pairs      = predictor.predict_pairs(patient_features_104d, drug_indices)
        # → np.ndarray (n_drugs, n_drugs) = AUC for each 2-element set
        # Symmetric by construction (permutation-invariant architecture).

    triples    = predictor.predict_triples(
                    patient_features_104d, drug_indices, top_k=20)
        # → np.ndarray (n_triples, 4): drug1_idx, drug2_idx, drug3_idx, predicted_auc
        # Enumerates C(n, 3) triplets if n ≤ 20; otherwise heuristic sampling.

The big win over the MLP: `predict_pairs` returns learned combo AUC, not the
0.5·(AUC_d1 + AUC_d2) + mech_prior approximation. The Set Transformer was
trained on 55K single-drug samples and extrapolates N=2 via permutation
invariance; zero-shot on triplets.
"""

from __future__ import annotations

import itertools
from dataclasses import fields as _fields
from pathlib import Path

import numpy as np
import torch

from combo_val.combo.set_drug_predictor import (
    SetDrugPredictor,
    SetDrugPredictorConfig,
)


class SetDrugInference:
    """Batch-oriented inference over learned combo AUC."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=self.device)

        cfg_dict = ckpt["cfg"]
        kwargs = {f.name: cfg_dict[f.name] for f in _fields(SetDrugPredictorConfig)
                  if f.name in cfg_dict}
        cfg = SetDrugPredictorConfig(**kwargs)

        self.model = SetDrugPredictor(
            n_drugs=ckpt["n_drugs"],
            n_patient_features=ckpt["n_patient_features"],
            cfg=cfg,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.drug_vocab: list[str] = ckpt["drug_vocab"]
        self.drug_to_int: dict[str, int] = ckpt["drug_to_int"]
        self.feature_cols: list[str] = ckpt["feature_cols"]
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    def _standardize(self, features: np.ndarray) -> torch.Tensor:
        """Apply the saved StandardScaler to a raw feature vector."""
        safe_scale = np.where(self.scaler_scale > 0, self.scaler_scale, 1.0)
        std = (features - self.scaler_mean) / safe_scale
        return torch.tensor(std, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_singles(self, features: np.ndarray) -> np.ndarray:
        """Predict AUC for each drug as a singleton set. Returns (n_drugs,)."""
        pf = self._standardize(features).unsqueeze(0).expand(len(self.drug_vocab), -1)
        singleton_sets = [[i] for i in range(len(self.drug_vocab))]
        preds = self.model.predict_set(singleton_sets, pf, self.device)
        return preds.cpu().numpy()

    @torch.no_grad()
    def predict_pairs(
        self, features: np.ndarray, drug_indices: list[int] | None = None,
    ) -> np.ndarray:
        """Predict AUC for every 2-drug set over the given drug subset.

        drug_indices : list of 0-based indices into self.drug_vocab. Default = all.
        Returns    : (n, n) symmetric matrix. Diagonal = singleton-as-pair (same drug twice)
                     which is meaningless; callers should mask it before ranking.
        """
        if drug_indices is None:
            drug_indices = list(range(len(self.drug_vocab)))
        n = len(drug_indices)
        pair_list = [[drug_indices[i], drug_indices[j]]
                     for i in range(n) for j in range(n)]
        pf = self._standardize(features).unsqueeze(0).expand(len(pair_list), -1)
        preds = self.model.predict_set(pair_list, pf, self.device).cpu().numpy()
        return preds.reshape(n, n)

    @torch.no_grad()
    def predict_triples(
        self, features: np.ndarray, drug_indices: list[int] | None = None,
        top_k: int = 20,
    ) -> list[tuple[int, int, int, float]]:
        """Enumerate C(n, 3) triplets and return top-k lowest-AUC ones.

        Returns list of (drug_idx_0, drug_idx_1, drug_idx_2, predicted_auc).
        """
        if drug_indices is None:
            drug_indices = list(range(len(self.drug_vocab)))
        triples = list(itertools.combinations(drug_indices, 3))
        if not triples:
            return []
        triple_lists = [list(t) for t in triples]
        pf = self._standardize(features).unsqueeze(0).expand(len(triples), -1)
        preds = self.model.predict_set(triple_lists, pf, self.device).cpu().numpy()

        order = np.argsort(preds)
        top = order[:top_k]
        return [(*triples[i], float(preds[i])) for i in top]

    def predict_set_for_patient(
        self, features: np.ndarray, drug_names: list[str],
    ) -> float:
        """Predict AUC for an arbitrary-arity set specified by drug names."""
        missing = [d for d in drug_names if d not in self.drug_to_int]
        if missing:
            raise KeyError(f"Drug(s) not in predictor vocab: {missing}")
        drug_ids = [[self.drug_to_int[d] for d in drug_names]]
        pf = self._standardize(features).unsqueeze(0)
        pred = self.model.predict_set(drug_ids, pf, self.device)
        return float(pred.item())


def load_set_drug_predictor(
    checkpoint_path: Path = Path("runs/set_drug_predictor_v2_stable/final_model.pt"),
    device: str = "cpu",
) -> SetDrugInference | None:
    """Load the Path B v2 checkpoint. Returns None if the file doesn't exist
    (letting callers gracefully fall back to Baseline A MLP)."""
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    return SetDrugInference(cp, device=device)
