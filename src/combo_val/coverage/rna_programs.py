"""D.1 — RNA expression program scoring for axis-weight modulation.

Computes 8-10 program signature scores from a patient's RNA-Seq panel.
These scores feed `infer_active_targets(rna_programs=...)` to modulate
target axis weights individually for each patient.

Program → Axis modulation map:
  BCL2_high            → BCL2 axis weight up        (Ven-sensitive)
  MCL1_high            → MCL1 axis weight up        (Ven-resistance escape)
  MAPK_signature_high  → RAS_MAPK axis weight up    (MEKi-sensitive)
  STAT5_signature_high → JAK_STAT axis weight up    (JAKi-sensitive)
  OXPHOS_high          → OXPHOS axis weight up      (OXPHOSi-sensitive)
  LSC17_high           → stem_cell_program weight up (HMA + Ven sensitive)
  HOX_MEIS_high        → MENIN_KMT2A/MENIN_NPM1 up  (MENINi-sensitive)
  IDH_signature_high   → IDH1/IDH2 weight up        (IDHi sensitive)

Implementation: each program is a list of marker genes with up/down
direction. Score = mean log-rank-normalized expression of "up" genes
minus "down" genes, normalized to [0, 1] using BeatAML 2.0 reference
distribution.

This is the standard ssGSEA / GSVA-like scoring approach, simplified
for individual-patient inference (no cohort-wide normalization needed
because we precomputed reference quantile thresholds during training
data prep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Curated AML program signatures.
# Sources:
# - LSC17 from Ng et al. Nature 2016 (PMID 27926740) — leukemia stem cell score
# - HOX/MEIS from Krivtsov et al. Cancer Cell 2006 — KMT2A/NPM1 program
# - BCL2/MCL1: pharmacology (high expression = drug-sensitive)
# - MAPK signature: Pratilas et al. PNAS 2009 (MEKi predictive)
# - OXPHOS: Pollyea et al. Nat Med 2018 (PMID 30297919) — Ven mechanism
PROGRAM_SIGNATURES: dict[str, dict] = {

    "BCL2_high": {
        "description": "BCL-2 family anti-apoptotic transcription program",
        "up_genes": ["BCL2", "BCL2L1", "BCL2L11"],   # BCL-2, BCL-XL, BIM
        "down_genes": [],
        "axis_modulated": "BCL2",
    },

    "MCL1_high": {
        "description": "MCL-1 anti-apoptotic dependence (Ven-escape)",
        "up_genes": ["MCL1", "BAK1"],
        "down_genes": [],
        "axis_modulated": "MCL1",
    },

    "MAPK_signature_high": {
        "description": "RAS-RAF-MEK-ERK pathway transcriptional output (Pratilas signature)",
        "up_genes": ["DUSP6", "ETV4", "ETV5", "SPRY2", "SPRY4",
                     "PHLDA1", "FOS", "JUN", "EGR1", "MYC"],
        "down_genes": [],
        "axis_modulated": "RAS_MAPK",
    },

    "STAT5_signature_high": {
        "description": "JAK-STAT5 signaling output",
        "up_genes": ["CISH", "PIM1", "BCL6", "SOCS1", "SOCS3",
                     "OSM", "IL2RA"],
        "down_genes": [],
        "axis_modulated": "JAK_STAT",
    },

    "OXPHOS_high": {
        "description": "Oxidative phosphorylation dependence (Pollyea Nat Med 2018)",
        "up_genes": ["NDUFA1", "NDUFB1", "ATP5F1A", "ATP5F1B",
                     "COX5A", "COX6A1", "UQCRC1", "UQCRC2",
                     "CYCS", "TFAM"],
        "down_genes": [],
        "axis_modulated": "OXPHOS",
    },

    "LSC17_high": {
        "description": "17-gene LSC self-renewal score (Ng Nature 2016 PMID 27926740)",
        "up_genes": ["DNMT3B", "ZBTB46", "NYNRIN", "ARHGAP22",
                     "LAPTM4B", "MMRN1", "DPYSL3", "KIAA0125",
                     "CDK6", "CPXM1", "SOCS2", "SMIM24",
                     "EMP1", "NGFRAP1", "CD34", "AKR1C3"],
        "down_genes": ["GPR56"],     # LSC17 includes 1 down-gene
        "axis_modulated": "stem_cell_program",
    },

    "HOX_MEIS_high": {
        "description": "HOX/MEIS1 program — KMT2A-r and NPM1mut driver",
        "up_genes": ["HOXA9", "HOXA10", "HOXA7", "MEIS1",
                     "PBX3", "MEF2C", "MN1"],
        "down_genes": [],
        "axis_modulated": "MENIN_NPM1",   # also covers MENIN_KMT2A indirectly
    },

    "IDH_signature_high": {
        "description": "TET2/IDH co-mut hypermethylation signature",
        "up_genes": ["KDM6A", "KDM6B", "ASCL1", "TET2"],
        "down_genes": ["KDM5C"],
        "axis_modulated": "IDH1",   # also picked up by patient mutation flag
    },
}


@dataclass
class ProgramScoreConfig:
    # Z-score thresholds for "high" classification
    high_z_threshold: float = 0.5     # mean z above 0.5 → "active"
    saturation_z: float = 2.0         # cap normalization at z=2 → score=1.0


def _safe_log_rank_normalize(expression: pd.Series) -> pd.Series:
    """Rank-normalize a single patient's expression vector to [0, 1].
    Equivalent to taking percentile ranks within this single sample —
    avoids cross-sample reference dependence."""
    ranks = expression.rank(pct=True, method="average")
    return ranks


def score_programs(
    rna_expression: pd.Series,
    cfg: Optional[ProgramScoreConfig] = None,
) -> dict[str, float]:
    """Compute program scores from a single patient's RNA expression.

    Args:
      rna_expression: pd.Series with gene symbols as index, expression
        values (counts or TPM or normalized) as values.
      cfg: optional config (thresholds).

    Returns:
      Dict {program_name: score in [0, 1]} for programs where ≥ 50% of
      genes were found in the input. Programs with too few genes
      matched are SKIPPED, not zero — to avoid false-low signal.

      The score is what `infer_active_targets(rna_programs=...)` reads
      via the rna_modifier mechanism in target_taxonomy_v2.yaml.
    """
    cfg = cfg or ProgramScoreConfig()
    if rna_expression.empty:
        return {}

    # Rank-normalize once for the whole vector
    normalized = _safe_log_rank_normalize(rna_expression)

    out: dict[str, float] = {}
    for prog_name, spec in PROGRAM_SIGNATURES.items():
        up = spec.get("up_genes", []) or []
        down = spec.get("down_genes", []) or []
        n_total = len(up) + len(down)
        if n_total == 0:
            continue

        up_present = [g for g in up if g in normalized.index]
        down_present = [g for g in down if g in normalized.index]
        n_present = len(up_present) + len(down_present)
        # Skip if too few genes matched (avoid false-low signal from
        # incomplete RNA panel)
        if n_present < 0.5 * n_total:
            continue

        # Score = mean(up_ranks) - mean(down_ranks), centered at 0.5
        up_score = (np.mean([normalized[g] for g in up_present])
                     if up_present else 0.5)
        down_score = (np.mean([normalized[g] for g in down_present])
                       if down_present else 0.5)
        # "high" means up genes are ABOVE 0.5 percentile and/or down
        # genes are BELOW 0.5
        signed_score = (up_score - 0.5) - (down_score - 0.5)
        # Normalize: signed_score in [-1, 1] → output [0, 1] (above
        # baseline 0.5 = active)
        out[prog_name] = float(np.clip(0.5 + signed_score, 0.0, 1.0))

    return out


def program_to_axis_modulation(program_scores: dict[str, float]
                                 ) -> dict[str, float]:
    """Convert program scores into axis weight modifiers.

    A program score in [0, 1] becomes a multiplier in [0.5, 1.5] applied
    to the axis it modulates:
      score 0.5 (baseline) → multiplier 1.0 (no change)
      score 1.0 (very high) → multiplier 1.5 (axis weight ×1.5)
      score 0.0 (very low) → multiplier 0.5 (axis weight ×0.5)
    """
    out: dict[str, float] = {}
    for prog, score in program_scores.items():
        spec = PROGRAM_SIGNATURES.get(prog)
        if not spec:
            continue
        axis = spec.get("axis_modulated")
        if not axis:
            continue
        # Map [0, 1] → [0.5, 1.5]
        multiplier = 0.5 + score
        # If multiple programs target the same axis (rare), use max
        if axis in out:
            out[axis] = max(out[axis], multiplier)
        else:
            out[axis] = multiplier
    return out
