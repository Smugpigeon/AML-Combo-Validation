"""Infer active vulnerability targets for a given patient.

Input: patient features (mutation flags, RNA expression, clinical)
Output: dict {target_id: weight} where weight ∈ [0, 1] reflects how
         strongly that target is "active" / clinically relevant for THIS
         patient.

Logic per target:
  1. If `patient_evidence.mut` lists genes and any are mutated → activate
  2. If `patient_evidence.fusion` lists fusions and any present → activate
  3. If `patient_evidence.flag` lists clin flags (e.g. clin_flt3_itd) → activate
  4. If `patient_evidence.tp53_multihit: true` AND TP53 multihit detected → activate
  5. If `patient_evidence.always_active_baseline` → activate at that level
  6. If `patient_evidence.rna_modifier` and RNA program score is high → boost

Final weight = max(target.default_weight × evidence_strength,
                    always_active_baseline)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from combo_val.coverage.taxonomy import Taxonomy, TargetSpec


def infer_active_targets(
    patient_features: pd.Series | dict,
    taxonomy: Taxonomy,
    rna_programs: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Returns {target_id: weight} for the targets active in this patient.

    Args:
      patient_features: Series or dict with mut_<GENE>, fusion_<X>,
        clin_<x>, tp53_multihit etc. fields. Compatible with the
        BeatAML patient_features schema.
      taxonomy: loaded Taxonomy
      rna_programs: optional {program_name: score} dict for RNA
        modifiers (e.g. {"BCL2_high": 1.5, "MAPK_signature_high": 0.3})
    """
    rna_programs = rna_programs or {}
    if isinstance(patient_features, pd.Series):
        pf = patient_features.to_dict()
    else:
        pf = dict(patient_features)

    out: dict[str, float] = {}
    for tgt in taxonomy.targets:
        evidence = tgt.patient_evidence or {}
        activated = False
        evidence_strength = 0.0

        # 1. Mutation evidence
        for gene in evidence.get("mut", []) or []:
            key = f"mut_{gene}"
            if pf.get(key, 0) and float(pf[key]) > 0.5:
                activated = True
                evidence_strength = max(evidence_strength, 1.0)

        # 2. Fusion evidence
        for fus in evidence.get("fusion", []) or []:
            key = f"fusion_{fus}"
            if pf.get(key, 0) and float(pf[key]) > 0.5:
                activated = True
                evidence_strength = max(evidence_strength, 1.0)

        # 3. Clin flag evidence
        for flag in evidence.get("flag", []) or []:
            if pf.get(flag, 0) and float(pf[flag]) > 0.5:
                activated = True
                evidence_strength = max(evidence_strength, 1.0)

        # 4. TP53 multi-hit special case
        if evidence.get("tp53_multihit") is True:
            if pf.get("tp53_multihit", False) is True:
                activated = True
                evidence_strength = max(evidence_strength, 1.0)

        # 5. Always-active baseline
        baseline = float(evidence.get("always_active_baseline", 0.0))
        if baseline > 0:
            activated = True
            evidence_strength = max(evidence_strength, baseline)

        # 6. RNA modifier boost
        rna_mod_name = evidence.get("rna_modifier")
        if rna_mod_name and rna_mod_name in rna_programs:
            rna_boost = float(rna_programs[rna_mod_name])
            evidence_strength = max(evidence_strength, min(1.0, rna_boost))
            activated = True

        if activated:
            # Final weight = default_weight × evidence_strength, capped at 1.0
            out[tgt.id] = float(min(1.0, tgt.default_weight * evidence_strength))

    # Phase D.1 — Apply RNA-program → axis modulation as a post-pass.
    # If patient's RNA expression shows e.g. BCL2_high, multiply BCL2
    # axis weight by a [0.5, 1.5] factor (clipped to [0, 1] final).
    # This is a separate signal from rna_modifier in patient_evidence
    # (which only boosts ACTIVATION); modulation here SCALES weight.
    try:
        from combo_val.coverage.rna_programs import program_to_axis_modulation
        modulation = program_to_axis_modulation(rna_programs or {})
    except ImportError:
        modulation = {}
    for axis_id, mult in modulation.items():
        if axis_id in out:
            out[axis_id] = float(min(1.0, out[axis_id] * mult))

    return out
