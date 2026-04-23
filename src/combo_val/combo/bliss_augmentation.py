"""Bliss-independence synthetic combo labels for Set Transformer augmentation.

Theoretical foundation (Palmer & Sorger, Cancer Discovery 2022):
  The Bliss independence model assumes drugs act on non-overlapping target
  populations. For a single cell, the probability of surviving the combination
  equals the product of single-drug survival probabilities:

      P(survive | d1, ..., dN) = ∏ P(survive | d_i)

  This gives us a theoretically-grounded SYNTHETIC label for any pair/triplet,
  derived from single-drug measurements we already have.

In BeatAML's AUC space (0 = fully killed, 300 = fully resistant):
  resp_i       = (300 − AUC_i) / 300                     # killing fraction
  resp_combo   = 1 − ∏(1 − resp_i)                        # Bliss
  AUC_combo    = 300 × (1 − resp_combo) = ∏ AUC_i / 300^(N−1)

So for any patient p where we've observed AUC for k drugs individually, we
can generate up to C(k, 2) synthetic pair labels without ANY wet-lab combo
screening. This is what lets Path B learn from "pair-like" supervision in
the absence of real combo data.

CAVEAT: Bliss is a theoretical baseline. Real combos can synergize (AUC <
Bliss) or antagonize (AUC > Bliss). Training on Bliss labels teaches ST the
ADDITIVE+INDEPENDENT baseline; it will not discover synergy that deviates
from this. For true synergy discovery, you need real combo measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# AUC range in BeatAML; sometimes small outliers (~350) exist; clip to
# physical range to keep Bliss math well-defined.
AUC_MIN = 0.0
AUC_MAX = 300.0


def bliss_combo_auc(auc_list: list[float] | np.ndarray) -> float:
    """Compute Bliss-independence AUC for an arbitrary-arity combo.

    AUC_combo = ∏ AUC_i / 300^(N-1)

    Parameters
    ----------
    auc_list : iterable of per-drug AUCs, each in [0, 300].

    Returns
    -------
    Predicted combo AUC under Bliss independence assumption.
    """
    auc_arr = np.asarray(auc_list, dtype=np.float64)
    auc_arr = np.clip(auc_arr, AUC_MIN, AUC_MAX)
    n = len(auc_arr)
    if n == 0:
        raise ValueError("Empty auc_list")
    if n == 1:
        return float(auc_arr[0])
    return float(np.prod(auc_arr) / (AUC_MAX ** (n - 1)))


def bliss_combo_auc_vectorized(auc_matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """Vectorized Bliss along a given axis.

    auc_matrix : array with one axis being drug-dimension.
    axis       : which axis to contract.
    """
    clipped = np.clip(auc_matrix, AUC_MIN, AUC_MAX)
    n = clipped.shape[axis]
    return np.prod(clipped, axis=axis) / (AUC_MAX ** (n - 1))


class BlissAugmentedPairDataset(Dataset):
    """Generates synthetic pair (patient, d1, d2, Bliss-AUC) samples.

    For each patient in the provided single-drug long_df, sample up to
    n_pairs_per_patient random pairs of drugs the patient has been measured on,
    and compute their Bliss-IDA combo AUC from the observed singles.

    Intended to be mixed with the single-drug Dataset in a CONCATENATED
    DataLoader so each training batch contains both singletons (real labels)
    and pairs (synthetic Bliss labels).
    """

    def __init__(
        self,
        long_df: pd.DataFrame,                    # must have: patient_id, drug_id, auc
        patient_features: pd.DataFrame,           # indexed by patient_id
        drug_id_to_int: dict[str, int],
        n_pairs_per_patient: int = 50,
        random_state: int = 42,
    ):
        self.patient_features = patient_features
        self.drug_id_to_int = drug_id_to_int
        self.rng = np.random.default_rng(random_state)

        # Build patient → {drug_id: auc} map
        long_df = long_df.dropna(subset=["auc"]).copy()
        long_df["patient_id"] = long_df["patient_id"].astype(int)

        # Pre-compute pair samples
        self.rows: list[dict] = []
        for pid, grp in long_df.groupby("patient_id"):
            if pid not in patient_features.index:
                continue
            drugs_in_this_patient = grp["drug_id"].tolist()
            aucs_in_this_patient = grp["auc"].tolist()
            k = len(drugs_in_this_patient)
            if k < 2:
                continue

            # Sample unique pairs (i, j) with i < j
            n_samples = min(n_pairs_per_patient, k * (k - 1) // 2)
            pair_indices_set: set[tuple[int, int]] = set()
            attempts = 0
            while len(pair_indices_set) < n_samples and attempts < n_samples * 10:
                i, j = self.rng.integers(0, k, size=2)
                if i == j:
                    attempts += 1
                    continue
                if i > j:
                    i, j = j, i
                pair_indices_set.add((int(i), int(j)))
                attempts += 1

            for (i, j) in pair_indices_set:
                d1 = drugs_in_this_patient[i]
                d2 = drugs_in_this_patient[j]
                auc_d1 = aucs_in_this_patient[i]
                auc_d2 = aucs_in_this_patient[j]
                auc_bliss = bliss_combo_auc([auc_d1, auc_d2])
                self.rows.append({
                    "patient_id": int(pid),
                    "drug_int_1": self.drug_id_to_int[d1],
                    "drug_int_2": self.drug_id_to_int[d2],
                    "bliss_auc": auc_bliss,
                    "auc_d1": auc_d1,
                    "auc_d2": auc_d2,
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        pid = row["patient_id"]
        pfeat = self.patient_features.loc[pid].to_numpy(dtype=np.float32)
        # +1 shift: reserve 0 for padding
        d1 = row["drug_int_1"] + 1
        d2 = row["drug_int_2"] + 1
        return {
            "patient_features": torch.from_numpy(pfeat),
            "drug_ids": torch.tensor([d1, d2], dtype=torch.long),
            "drug_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "auc": torch.tensor(float(row["bliss_auc"]), dtype=torch.float32),
        }


def generate_all_patient_bliss_triples(
    long_df: pd.DataFrame,
    patient_features: pd.DataFrame,
    drug_id_to_int: dict[str, int],
    n_triples_per_patient: int = 20,
    random_state: int = 43,
) -> list[dict]:
    """Generate synthetic triple labels analogous to pairs.

    Used for held-out TRIPLE-validation (not training) — we want to see if
    a model trained on pair-Bliss can extrapolate to triple-Bliss correctly
    even though it never saw a triple during training.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []
    long_df = long_df.dropna(subset=["auc"]).copy()
    long_df["patient_id"] = long_df["patient_id"].astype(int)
    for pid, grp in long_df.groupby("patient_id"):
        if pid not in patient_features.index:
            continue
        drugs = grp["drug_id"].tolist()
        aucs = grp["auc"].tolist()
        k = len(drugs)
        if k < 3:
            continue
        seen: set[tuple[int, int, int]] = set()
        attempts = 0
        while len(seen) < n_triples_per_patient and attempts < n_triples_per_patient * 10:
            idxs = tuple(sorted(rng.choice(k, size=3, replace=False)))
            seen.add(idxs)
            attempts += 1
        for (i, j, m) in seen:
            auc_bliss = bliss_combo_auc([aucs[i], aucs[j], aucs[m]])
            rows.append({
                "patient_id": int(pid),
                "drug_int_1": drug_id_to_int[drugs[i]],
                "drug_int_2": drug_id_to_int[drugs[j]],
                "drug_int_3": drug_id_to_int[drugs[m]],
                "bliss_auc": auc_bliss,
                "auc_d1": aucs[i],
                "auc_d2": aucs[j],
                "auc_d3": aucs[m],
            })
    return rows
