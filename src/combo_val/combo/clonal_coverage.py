"""Path A — Clonal-Coverage × Independent Drug Action (IDA) combo scorer.

Theoretical premise (Palmer & Sorger, Cancer Discovery 2022):
  Most clinical combination-therapy benefit arises from INDEPENDENT drug action
  over patient (or sub-population) heterogeneity, not molecular synergy. A
  triplet like Azacitidine + Venetoclax + Gilteritinib works for FLT3-mut AML
  because each drug targets a different sub-clone (FLT3-ITD blasts →
  Gilteritinib; BCL2-dependent cells → Venetoclax; differentiation-blocked
  cells → Azacitidine). At the individual-patient level, a combination is
  "good" if the patient's clonal structure is well-covered.

This module operationalizes that idea:

  1. Each patient is decomposed into a set of present "clonal archetypes"
     based on their mutation profile (and, optionally, RNA-derived states).

  2. Each drug's mechanism vector maps onto one or more clonal archetypes
     — a drug "covers" clone c if it hits the axes defining c.

  3. For a combination G of arbitrary arity N, coverage is computed via a
     Bliss-independence aggregation across drugs:

         covers(G, c) = 1 − ∏_{d ∈ G} (1 − covers(d, c))

     That is: the clone is covered if AT LEAST ONE drug in the combo hits it,
     with partial hits combining via the IDA probability formula.

  4. Patient-level combo score is the weighted average of clone coverages:

         score(p, G) = Σ_{c ∈ C(p)} w_c · covers(G, c)  /  Σ_{c ∈ C(p)} w_c

     where C(p) is the set of clones present in patient p with weights w_c.

Key distinction from the original `mechanism_prior.py`:
  - Old: flat axis × drug × patient max-aggregation over 2-drug pairs only.
  - New: explicit clonal structure, Bliss-IDA aggregation, arity-agnostic
    (any N-drug combo scored via the same formula).

Validation experiments live in `combo_val.validation.path_a_validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from combo_val.combo.mechanism_prior import (
    BEATAML_TO_MECH_ID,
    load_drug_mechanism_matrix,
)


# ---------------------------------------------------------------------------
# Clonal archetypes — biology-curated AML sub-populations
# ---------------------------------------------------------------------------

# Each archetype has:
#   presence_markers: list of mut_* feature columns; clone is PRESENT if any
#                     marker is 1 (or if "DEFAULT" is in the list → always present).
#   weight:           importance in the coverage score. Mutation-driven clones
#                     get weight 1.0; universal background clones get 0.5 or
#                     lower (they're easy to cover).
#   covered_by_axes:  mechanism axes (tgt_*, cs_*) that, when hit by a drug,
#                     count as covering this clone. Max-aggregated across axes
#                     for a single drug.
#
# References for the biology:
#   - FLT3-ITD / NPM1 / IDH / TP53 / RAS co-mutation patterns: Papaemmanuil 2016
#   - MENIN-HOX axis for NPM1-mut and KMT2A-r: Krivtsov 2019, Issa 2023
#   - BCL2 dependency universal in AML blasts: Pan 2014, DiNardo 2019
#   - CD34+ LSC compartment: Ng 2016 LSC17
CLONE_ARCHETYPES: dict[str, dict] = {
    # ---- Mutation-driven clones (weight 1.0) ----
    "FLT3_clone": {
        "description": "FLT3-ITD / FLT3-TKD kinase-driven blast population",
        "presence_markers": ["mut_FLT3"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_FLT3"],
    },
    "IDH1_clone": {
        "description": "IDH1-R132 neomorphic; 2-HG-driven differentiation block",
        "presence_markers": ["mut_IDH1"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_IDH1", "cs_differentiation_induction"],
    },
    "IDH2_clone": {
        "description": "IDH2-R140/R172 neomorphic; 2-HG-driven differentiation block",
        "presence_markers": ["mut_IDH2"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_IDH2", "cs_differentiation_induction"],
    },
    "MENIN_HOX_clone": {
        "description": "NPM1-mut or KMT2A-rearranged; HOX-dependent self-renewal",
        "presence_markers": ["mut_NPM1", "mut_KMT2A"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_MENIN_HOX", "cs_differentiation_induction"],
    },
    "TP53_clone": {
        "description": ("TP53-mut AML is genomic-instability-tolerant and "
                        "resistant to DNA-damaging cytotoxic therapy. ONLY direct "
                        "TP53-pathway agents (e.g. eprenetapopt / APR-246) cover "
                        "this clone. Current BeatAML drug panel has no such agents "
                        "→ this clone stays uncovered for TP53-mut patients, which "
                        "correctly flags them as hard-to-treat with existing combos."),
        "presence_markers": ["mut_TP53"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_TP53_PATHWAY"],
    },
    "RAS_MAPK_clone": {
        "description": "NRAS/KRAS/PTPN11-mut; RAS-MAPK pathway-active sub-clone",
        "presence_markers": ["mut_NRAS", "mut_KRAS", "mut_PTPN11"],
        "weight": 1.0,
        "covered_by_axes": ["tgt_RAS_MAPK"],
    },
    # ---- Background clones (always present, weighted lower) ----
    "BCL2_dependent_clone": {
        "description": "AML-general BCL2-dependent apoptotic vulnerability",
        "presence_markers": ["DEFAULT"],
        "weight": 0.5,
        "covered_by_axes": ["tgt_BCL2", "cs_apoptosis_priming"],
    },
    "proliferative_clone": {
        "description": "Cycling blast population requiring cytotoxic S-phase/M-phase kill",
        "presence_markers": ["DEFAULT"],
        "weight": 0.5,
        "covered_by_axes": [
            "tgt_DNA_SYNTHESIS", "tgt_TOPO_II",
            "cs_DNA_damage", "cs_cell_cycle_block",
        ],
    },
    "LSC_compartment": {
        "description": "Leukemic stem cell (CD34+ HOXA9/MEIS1-high) reservoir",
        "presence_markers": ["DEFAULT"],
        "weight": 0.3,
        "covered_by_axes": ["cs_stem_cell_targeting", "tgt_MENIN_HOX"],
    },
}


# ---------------------------------------------------------------------------
# Patient clonal decomposition
# ---------------------------------------------------------------------------


def build_patient_clone_matrix(patient_features: pd.DataFrame,
                                archetypes: dict = CLONE_ARCHETYPES,
                                ) -> pd.DataFrame:
    """Decompose each patient into the set of present clonal archetypes.

    Returns a DataFrame indexed by patient_id with columns = clone names and
    values ∈ {0, weight} indicating clone presence × weight.
    """
    clone_cols = list(archetypes.keys())
    out = pd.DataFrame(0.0, index=patient_features.index, columns=clone_cols)

    for clone_name, spec in archetypes.items():
        if "DEFAULT" in spec["presence_markers"]:
            out[clone_name] = spec["weight"]
        else:
            markers = [m for m in spec["presence_markers"]
                       if m in patient_features.columns]
            if not markers:
                continue
            present = (patient_features[markers].sum(axis=1) > 0).astype(float)
            out[clone_name] = present * spec["weight"]

    out.index.name = patient_features.index.name or "patient_id"
    return out


# ---------------------------------------------------------------------------
# Drug → clone coverage matrix
# ---------------------------------------------------------------------------


def build_drug_clone_coverage(beataml_drug_ids: list[str],
                               archetypes: dict = CLONE_ARCHETYPES,
                               drug_matrix: pd.DataFrame | None = None,
                               ) -> pd.DataFrame:
    """For each BeatAML drug, compute how well it covers each clonal archetype.

    covers(drug, clone) = max over axes in clone.covered_by_axes of drug[axis]

    Unannotated drugs (not in BEATAML_TO_MECH_ID) → all zeros.

    Returns (n_drugs, n_clones) DataFrame indexed by BeatAML drug name.
    """
    m = drug_matrix if drug_matrix is not None else load_drug_mechanism_matrix()
    clone_names = list(archetypes.keys())
    out = pd.DataFrame(0.0, index=beataml_drug_ids, columns=clone_names)

    for beat_id in beataml_drug_ids:
        mech_id = BEATAML_TO_MECH_ID.get(beat_id)
        if mech_id is None or mech_id not in m.index:
            continue
        drug_row = m.loc[mech_id]
        for clone_name, spec in archetypes.items():
            axes = [a for a in spec["covered_by_axes"] if a in drug_row.index]
            if not axes:
                continue
            out.loc[beat_id, clone_name] = float(drug_row[axes].max())

    return out


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def combo_clone_coverage_bliss(drug_clone_cov_subset: np.ndarray) -> np.ndarray:
    """Bliss-independence aggregation of drug-clone coverage across a combo.

    Input:  (n_drugs_in_combo, n_clones) coverage matrix for the drugs in the combo.
    Output: (n_clones,) — per-clone combined coverage ∈ [0, 1].

    covers(combo, c) = 1 − ∏_d (1 − covers(d, c))
    """
    assert drug_clone_cov_subset.ndim == 2
    return 1.0 - np.prod(1.0 - np.clip(drug_clone_cov_subset, 0.0, 1.0), axis=0)


def score_combo_for_patient(
    patient_clones: np.ndarray,     # (n_clones,) weighted presence
    drug_clone_cov: np.ndarray,     # (n_drugs_in_combo, n_clones)
) -> tuple[float, np.ndarray]:
    """Weighted-average coverage score for one patient × one combo.

    Returns (normalized_score, per_clone_coverage). normalized_score ∈ [0, 1].
    """
    present_mask = patient_clones > 0
    if not present_mask.any():
        return 0.0, np.zeros_like(patient_clones)
    per_clone_cov = combo_clone_coverage_bliss(drug_clone_cov)
    weights = patient_clones * present_mask
    total_w = weights.sum()
    if total_w <= 0:
        return 0.0, per_clone_cov
    score = float((weights * per_clone_cov).sum() / total_w)
    return score, per_clone_cov


def score_all_combos_for_patient(
    patient_clones: np.ndarray,
    drug_clone_cov_matrix: np.ndarray,  # (n_drugs, n_clones)
    combo_arity: int,
    drug_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Enumerate all C(|drugs|, arity) combos and score each.

    Returns a DataFrame: row per combo with columns combo_indices, score,
    per_clone coverages.
    """
    n_drugs = drug_clone_cov_matrix.shape[0]
    idx = drug_indices if drug_indices is not None else list(range(n_drugs))
    n_clones = drug_clone_cov_matrix.shape[1]
    rows = []
    for combo in combinations(idx, combo_arity):
        subset = drug_clone_cov_matrix[list(combo)]
        score, per_clone = score_combo_for_patient(patient_clones, subset)
        rows.append({
            "combo_indices": combo,
            "score": score,
            **{f"clone_{i}_cov": per_clone[i] for i in range(n_clones)},
        })
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch scoring — patient × combo score matrix
# ---------------------------------------------------------------------------


def score_named_regimens(
    patient_clones: pd.DataFrame,           # (n_patients, n_clones)
    drug_clone_cov: pd.DataFrame,            # (n_drugs, n_clones)
    regimens: dict[str, list[str]],          # regimen_name → [drug_name, ...]
) -> pd.DataFrame:
    """Score a fixed dict of named regimens for every patient.

    Returns (n_patients, n_regimens) DataFrame. Unknown drugs in a regimen
    contribute zero coverage (but still count as members of the combo —
    effectively reducing coverage per clone for that patient).
    """
    out = pd.DataFrame(0.0, index=patient_clones.index, columns=list(regimens.keys()))
    clones_arr = patient_clones.to_numpy(dtype=np.float64)

    for name, drug_list in regimens.items():
        # Build drug × clone matrix for this regimen
        known = [d for d in drug_list if d in drug_clone_cov.index]
        if not known:
            out[name] = 0.0
            continue
        cov_sub = drug_clone_cov.loc[known].to_numpy(dtype=np.float64)  # (k, n_clones)
        per_clone_cov = 1.0 - np.prod(1.0 - np.clip(cov_sub, 0.0, 1.0), axis=0)
        present = (clones_arr > 0).astype(np.float64)
        weights = clones_arr * present
        total_w = weights.sum(axis=1)
        num = (weights * per_clone_cov[None, :]).sum(axis=1)
        out[name] = np.where(total_w > 0, num / total_w, 0.0)

    return out


def rank_top_combos_per_patient(
    patient_clones: pd.DataFrame,
    drug_clone_cov: pd.DataFrame,
    arity: int,
    top_k: int = 10,
    drug_subset: list[str] | None = None,
) -> pd.DataFrame:
    """For each patient, return the top-k combos of given arity by score.

    Returns long-form DataFrame: patient_id, rank, drug1, drug2, ..., score.
    """
    drugs = drug_subset if drug_subset is not None else drug_clone_cov.index.tolist()
    cov_mat = drug_clone_cov.loc[drugs].to_numpy(dtype=np.float64)
    clones_arr = patient_clones.to_numpy(dtype=np.float64)

    n_drugs = len(drugs)
    all_combos = list(combinations(range(n_drugs), arity))
    # Vectorized scoring: for each combo, compute per-clone coverage across
    # all patients simultaneously.
    n_patients = clones_arr.shape[0]
    n_combos = len(all_combos)
    scores = np.zeros((n_patients, n_combos), dtype=np.float32)

    present = (clones_arr > 0).astype(np.float64)
    weights = clones_arr * present
    total_w = weights.sum(axis=1)

    for ci, combo in enumerate(all_combos):
        cov_sub = cov_mat[list(combo)]                                 # (arity, n_clones)
        per_clone_cov = 1.0 - np.prod(1.0 - np.clip(cov_sub, 0.0, 1.0), axis=0)
        num = (weights * per_clone_cov[None, :]).sum(axis=1)
        scores[:, ci] = np.where(total_w > 0, num / total_w, 0.0)

    rows = []
    for pi, pid in enumerate(patient_clones.index):
        top_idx = np.argsort(scores[pi])[::-1][:top_k]
        for rank, ci in enumerate(top_idx):
            combo = all_combos[ci]
            row = {"patient_id": pid, "arity": arity, "rank": rank + 1,
                   "score": float(scores[pi, ci])}
            for j, drug_i in enumerate(combo):
                row[f"drug{j+1}"] = drugs[drug_i]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathADiagnostics:
    n_patients: int
    n_drugs: int
    n_clones: int
    clone_prevalence: dict           # clone → fraction of patients with it present
    drugs_annotated: int
    drugs_unannotated: int


def diagnose(patient_clones: pd.DataFrame,
              drug_clone_cov: pd.DataFrame) -> PathADiagnostics:
    prevalence = {
        c: float(((patient_clones[c] > 0).sum()) / max(1, len(patient_clones)))
        for c in patient_clones.columns
    }
    drugs_annotated = int((drug_clone_cov.sum(axis=1) > 0).sum())
    return PathADiagnostics(
        n_patients=int(len(patient_clones)),
        n_drugs=int(len(drug_clone_cov)),
        n_clones=int(len(patient_clones.columns)),
        clone_prevalence=prevalence,
        drugs_annotated=drugs_annotated,
        drugs_unannotated=int(len(drug_clone_cov) - drugs_annotated),
    )
