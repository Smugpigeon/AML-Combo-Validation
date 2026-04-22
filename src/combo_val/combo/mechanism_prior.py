"""Mechanism-aware combo-prior scoring (simplified, no HyGReM prototypes).

Week 3 addition: a knowledge-driven prior that rewards drug pairs jointly
addressing a patient's curated driver deficit. The prior is ADDITIVE to the
residual-learned synergy term, so a pair can win by strong pair-chemistry
(ALMANAC residual), strong driver-matching (this module), or both.

Scoring formula:

    mech_bonus(patient, d1, d2)
        = target_coverage(patient, {d1, d2})
        + cell_state_coverage(patient, {d1, d2})
        - toxicity_stacking_penalty({d1, d2})

Where:
  target_coverage = sum_{axis ∈ mutation-derived targets}
                        patient_deficit[axis] * max(d1[axis], d2[axis])
  cell_state_coverage (for this simplified version) = 0 — we don't have
  the HyGReM prototypes to derive cell-state deficits; only mutation-based
  target deficits are used.
  toxicity_stacking_penalty = sum over high-stacking toxicity axes
                              (myelosuppression, hepatotoxicity, etc.)

Design notes:
  - The "max" aggregation rewards COMBINATIONS that cover DIFFERENT mechanisms
    — e.g. FLT3-mut patient gets max coverage from venetoclax+quizartinib
    (BCL2 + FLT3) than from quizartinib+midostaurin (two FLT3 inhibitors).
  - Only drugs present in drug_mechanism_v1.csv get a non-zero mech bonus.
    The 20-drug vocab covers AML-approved + registrational-stage drugs.
  - Patient target-deficit maps are explicit: mut_FLT3=1 → tgt_FLT3 = 1, etc.
  - The prior is deliberately SPARSE — most (patient, pair) combos will have
    mech_bonus = 0. This is by design: the prior only kicks in when the
    biology matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


_DEFAULT_DRUG_MATRIX_PATH = (
    Path(__file__).resolve().parent.parent / "knowledge" / "drug_mechanism_v1.csv"
)


# Mapping from patient feature column → mechanism target axis.
# Drives `patient_target_deficit` below. Only mutations where a targeted
# therapy exists in our drug vocab are included.
_MUT_TO_TGT: dict[str, str] = {
    "mut_FLT3":   "tgt_FLT3",
    "mut_IDH1":   "tgt_IDH1",
    "mut_IDH2":   "tgt_IDH2",
    "mut_NPM1":   "tgt_MENIN_HOX",       # NPM1-mut confers MENIN-HOX dependency
    "mut_KMT2A":  "tgt_MENIN_HOX",       # KMT2A-r also MENIN-dependent
    # DNA damage / topoisomerase / HMA are universal deficits for intensive
    # or HMA-based regimens; not patient-specific. Left out.
}

# BeatAML drug_id (165 vocab) → mechanism-matrix drug_id (20 vocab).
# Only includes drugs where BeatAML has the drug AND we have mechanism
# annotation. Built manually from inspection.
BEATAML_TO_MECH_ID: dict[str, str] = {
    "Venetoclax": "venetoclax",
    "Azacytidine": "azacitidine_injectable",
    "Midostaurin": "midostaurin",
    "Quizartinib (AC220)": "quizartinib",
    "Gilteritinib": "gilteritinib",
    "Ivosidenib": "ivosidenib",
    "Enasidenib": "enasidenib",
    "Cytarabine": "cytarabine_intensive_context",
    "Daunorubicin": "daunorubicin_intensive_context",
    "Idarubicin": "idarubicin_intensive_context",
    # Olutasidenib, Decitabine, Gemtuzumab-ozogamicin, Glasdegib not in BeatAML.
}


# Toxicity axes where pair-sum exceeds a soft cap → penalty.
# Cap of 1.5 is sum of two "mild" drugs (0.75 ea) or one "heavy" drug (1.5).
_TOX_STACK_CAPS: Mapping[str, float] = {
    "tox_myelosuppression": 1.8,
    "tox_hepatotoxicity": 1.3,
    "tox_cardiotoxicity": 1.3,
    "tox_QT_prolongation": 1.5,
    "tox_GI_toxicity": 1.8,
}


@dataclass(frozen=True)
class MechPriorConfig:
    drug_matrix_path: Path = _DEFAULT_DRUG_MATRIX_PATH
    # Per-axis weights for combining target vs cell-state vs regimen-role
    target_weight: float = 1.0
    cell_state_weight: float = 0.6
    # Toxicity penalty magnitude (penalty_per_unit_overshoot)
    toxicity_penalty_scale: float = 0.5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_drug_mechanism_matrix(path: Path | None = None) -> pd.DataFrame:
    """Load drug × mechanism feature table, indexed by mech drug_id."""
    p = path or _DEFAULT_DRUG_MATRIX_PATH
    df = pd.read_csv(p).set_index("drug_id")
    mech_cols = [
        c for c in df.columns
        if c.startswith(("tgt_", "cs_", "role_", "tox_"))
    ]
    df[mech_cols] = df[mech_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Patient deficit
# ---------------------------------------------------------------------------


def patient_target_deficit(patient_features: pd.DataFrame,
                            mut_to_tgt: Mapping[str, str] | None = None,
                            ) -> pd.DataFrame:
    """Map mutation features → target-axis deficit vectors.

    Input: DataFrame indexed by patient_id with mut_* binary columns.
    Output: DataFrame indexed by patient_id, columns = tgt_* axes,
            values = 1.0 if the corresponding mutation is present.
    """
    mut_to_tgt = mut_to_tgt or _MUT_TO_TGT
    # Collect unique target axes
    tgt_axes = sorted(set(mut_to_tgt.values()))
    deficit = pd.DataFrame(0.0, index=patient_features.index, columns=tgt_axes)
    for mut_col, tgt in mut_to_tgt.items():
        if mut_col in patient_features.columns:
            deficit[tgt] = np.maximum(deficit[tgt].values,
                                       patient_features[mut_col].values.astype(float))
    return deficit


# ---------------------------------------------------------------------------
# Combo scoring
# ---------------------------------------------------------------------------


def _drug_mech_lookup(drug_matrix: pd.DataFrame,
                      beataml_drug_ids: list[str],
                      mech_cols: list[str]) -> pd.DataFrame:
    """Look up mech vectors for each BeatAML drug. Missing ones get zeros."""
    out = pd.DataFrame(0.0, index=beataml_drug_ids, columns=mech_cols)
    for beat_id in beataml_drug_ids:
        mech_id = BEATAML_TO_MECH_ID.get(beat_id)
        if mech_id is not None and mech_id in drug_matrix.index:
            out.loc[beat_id] = drug_matrix.loc[mech_id, mech_cols].values
    return out


def compute_combo_mech_scores(
    patient_features: pd.DataFrame,
    beataml_drug_ids: list[str],
    cfg: MechPriorConfig | None = None,
) -> np.ndarray:
    """Compute mech_bonus(patient, d1, d2) for all (patient, drug_pair) combos.

    Returns
    -------
    ndarray of shape (n_patients, n_drugs, n_drugs), symmetric in last two axes.
    Entry [p, i, j] is the mechanism-prior score for patient p, drugs i, j.
    Higher = more biologically rational pair for that patient.
    """
    cfg = cfg or MechPriorConfig()
    drug_matrix = load_drug_mechanism_matrix(cfg.drug_matrix_path)

    tgt_cols = [c for c in drug_matrix.columns if c.startswith("tgt_")]
    tox_cols = [c for c in drug_matrix.columns if c.startswith("tox_")]

    # Patient × target deficit
    deficit = patient_target_deficit(patient_features)
    # Align patient deficit cols with drug-matrix tgt cols
    deficit = deficit.reindex(columns=tgt_cols, fill_value=0.0)

    # Drug × mech vectors, reindexed to BeatAML drugs
    drug_tgt = _drug_mech_lookup(drug_matrix, beataml_drug_ids, tgt_cols)
    drug_tox = _drug_mech_lookup(drug_matrix, beataml_drug_ids, tox_cols)

    n_p, n_d = len(patient_features), len(beataml_drug_ids)
    # Target coverage: for each pair (i, j), max(drug_tgt[i], drug_tgt[j]) is
    # a vector of length |tgt|. Multiply by deficit (n_p × |tgt|) then sum.
    tgt_arr = drug_tgt.to_numpy(dtype=np.float32)         # (n_d, |tgt|)
    # Pairwise max across drug axis:  (n_d, n_d, |tgt|)
    pair_tgt_max = np.maximum(tgt_arr[:, None, :], tgt_arr[None, :, :])
    # Contract with patient deficit: (n_p, |tgt|) · (n_d, n_d, |tgt|).T
    deficit_arr = deficit.to_numpy(dtype=np.float32)      # (n_p, |tgt|)
    target_cov = np.einsum("pt,ijt->pij", deficit_arr, pair_tgt_max)

    # Toxicity stacking penalty: pair sum over selected axes minus cap
    tox_arr = drug_tox.to_numpy(dtype=np.float32)         # (n_d, |tox|)
    tox_names = list(drug_tox.columns)
    pair_tox_sum = tox_arr[:, None, :] + tox_arr[None, :, :]  # (n_d, n_d, |tox|)
    pair_penalty = np.zeros((n_d, n_d), dtype=np.float32)
    for tox_name, cap in _TOX_STACK_CAPS.items():
        if tox_name in tox_names:
            k = tox_names.index(tox_name)
            overshoot = np.maximum(0.0, pair_tox_sum[:, :, k] - cap)
            pair_penalty += overshoot
    pair_penalty *= cfg.toxicity_penalty_scale

    mech_score = cfg.target_weight * target_cov - pair_penalty[None, :, :]
    return mech_score


# ---------------------------------------------------------------------------
# Simple diagnostics
# ---------------------------------------------------------------------------


def diagnostics(patient_features: pd.DataFrame,
                 beataml_drug_ids: list[str]) -> dict:
    """Return a small dict with QC info about the mechanism-prior input."""
    deficit = patient_target_deficit(patient_features)
    drugs_with_mech = sum(1 for d in beataml_drug_ids if d in BEATAML_TO_MECH_ID)
    return {
        "n_patients": int(len(patient_features)),
        "n_patients_with_any_deficit": int((deficit > 0).any(axis=1).sum()),
        "target_axis_coverage": {
            c: int((deficit[c] > 0).sum()) for c in deficit.columns
        },
        "n_beataml_drugs": int(len(beataml_drug_ids)),
        "n_drugs_with_mech_annotation": int(drugs_with_mech),
    }
