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
# Drives `patient_target_deficit` below. Expanded in v2 after adding mech
# vectors for RAS/MAPK and TP53-pathway drugs.
_MUT_TO_TGT: dict[str, str] = {
    # ---- Primary AML driver → direct target axis ----
    "mut_FLT3":   "tgt_FLT3",
    "mut_IDH1":   "tgt_IDH1",
    "mut_IDH2":   "tgt_IDH2",
    "mut_NPM1":   "tgt_MENIN_HOX",       # NPM1-mut confers MENIN-HOX dependency
    "mut_KMT2A":  "tgt_MENIN_HOX",       # KMT2A-r also MENIN-dependent
    # ---- v2: RAS/MAPK axis (hit by trametinib, selumetinib, sorafenib) ----
    "mut_NRAS":   "tgt_RAS_MAPK",
    "mut_KRAS":   "tgt_RAS_MAPK",
    "mut_PTPN11": "tgt_RAS_MAPK",        # SHP2 → RAS activation
    # ---- v2: TP53 pathway (adverse-risk marker; matches nothing we have;
    #          kept for future TP53-pathway agents) ----
    "mut_TP53":   "tgt_TP53_PATHWAY",
    # ---- v2: Spliceosome (research-stage splicing-modulator drugs,
    #          currently no annotated drug but flag the deficit) ----
    "mut_SRSF2":  "tgt_SPLICEOSOME",
    "mut_SF3B1":  "tgt_SPLICEOSOME",
    "mut_U2AF1":  "tgt_SPLICEOSOME",
    # DNA damage / topoisomerase / HMA are universal deficits for intensive
    # or HMA-based regimens; not patient-specific. Left out.
}

# BeatAML drug_id (165 vocab) → mechanism-matrix drug_id (32 vocab after v2).
# Only includes drugs where BeatAML has the drug AND we have mechanism
# annotation. Built manually from inspection of BeatAML drug name strings.
BEATAML_TO_MECH_ID: dict[str, str] = {
    # ---- v1: AML-approved / registrational (10) ----
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
    # ---- v2 (Problem-3 fix): AML clinical-filter drugs (12 more) ----
    "Trametinib (GSK1120212)": "trametinib",
    "Selumetinib (AZD6244)": "selumetinib",
    "Ruxolitinib (INCB018424)": "ruxolitinib",
    "Sorafenib": "sorafenib",
    "Dasatinib": "dasatinib",
    "Imatinib": "imatinib",
    "Nilotinib": "nilotinib",
    "Ponatinib": "ponatinib",
    "Crenolanib": "crenolanib",
    "Crizotinib (PF-2341066)": "crizotinib",
    "Alisertib (MLN8237)": "alisertib",
    "Pacritinib": "pacritinib",
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

    # Tiebreaker: prefer pairs with BOTH drugs annotated AND hitting DIFFERENT
    # target axes. For a FLT3-only-mut patient, (Quizartinib + Venetoclax)
    # and (Quizartinib + Enasidenib) both cover FLT3, but the former uses
    # complementary axes (FLT3 + BCL2 apoptosis-priming) while the latter
    # wastes Enasidenib's IDH2 axis on a non-deficit. The diversity bonus
    # breaks the tie toward the complementary pair.
    annotated_mask = np.array(
        [beat_id in BEATAML_TO_MECH_ID for beat_id in beataml_drug_ids],
        dtype=np.float32,
    )  # (n_d,)
    pair_annot_count = annotated_mask[:, None] + annotated_mask[None, :]  # (n_d, n_d)

    # Axis-diversity: number of target axes that get a contribution ≥ 0.5
    # from at least one drug in the pair. More axes engaged = more mechanism-
    # diverse combo. Computed on the drug matrix (patient-independent).
    axis_active = (tgt_arr >= 0.5).astype(np.float32)   # (n_d, |tgt|)
    pair_axis_any = np.logical_or(
        axis_active[:, None, :], axis_active[None, :, :],
    ).sum(axis=2).astype(np.float32)   # (n_d, n_d)

    # Redundancy: axes where BOTH drugs are active (≥ 0.5). Penalized to
    # discourage double-target-hit combos like dual-MEKi or dual-FLT3i that
    # have no clinical rationale and get picked up by argsort ties.
    pair_both_active = np.logical_and(
        axis_active[:, None, :], axis_active[None, :, :],
    ).sum(axis=2).astype(np.float32)  # (n_d, n_d)

    tiebreaker = (
        0.01 * pair_annot_count            # both drugs curated
        + 0.01 * pair_axis_any             # mechanistic diversity
        - 0.02 * pair_both_active          # redundancy penalty
    )
    mech_score = cfg.target_weight * target_cov - pair_penalty[None, :, :] + tiebreaker[None, :, :]
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
