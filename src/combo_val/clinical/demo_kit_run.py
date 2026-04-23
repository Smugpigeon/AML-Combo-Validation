"""Demo: run the full clinical kit pipeline on two synthetic new patients.

Two synthetic patients representing two opposite ends of the driver spectrum:

  Patient 1: 45-year-old, NPM1-mut + FLT3-ITD (high AR), classic
             "young, targetable driver" profile.  Expected: kit flags FLT3-ITD,
             recommends FLT3i + BCL2i or FLT3i + MEKi combos.

  Patient 2: 72-year-old, TP53-mut, complex karyotype. Classic
             "elderly, adverse-risk" profile.  Expected: kit flags TP53 +
             cautions against intensive induction, recommends low-intensity
             combos.

Usage:
  PYTHONPATH=src python -m combo_val.clinical.demo_kit_run

Output is printed to stdout; nothing is persisted.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from combo_val.clinical.kit_predict import (
    predict_for_patient,
    pretty_print_kit_output,
)
from combo_val.clinical.kit_schema import KitInput, MutationCall


def _synthetic_rna_counts(kept_genes: list[str], profile: str) -> pd.Series:
    """Generate a plausible RNA count profile. Uses random counts but
    deterministic seed per profile so runs are reproducible."""
    rng = np.random.default_rng({"young_flt3": 17, "elderly_tp53": 42}[profile])
    counts = rng.lognormal(mean=4.0, sigma=1.2, size=len(kept_genes))
    return pd.Series(counts, index=kept_genes)


def demo():
    warnings.filterwarnings("ignore")
    bundle = joblib.load("data/canonical/beataml_rna_preprocessor.joblib")
    kept = bundle["kept_genes"]

    # --- Patient 1: NPM1 + FLT3-ITD high AR ---
    rna1 = _synthetic_rna_counts(kept, "young_flt3")
    kit1 = KitInput(
        patient_id="SYNTHETIC-001",
        mutations=[
            MutationCall(gene="FLT3", is_ITD=True, allelic_ratio=0.62, vaf=0.45),
            MutationCall(gene="NPM1", variant_type="missense", vaf=0.42),
            MutationCall(gene="DNMT3A", variant_type="frameshift", vaf=0.48),
        ],
        karyotype_text="46,XX[20]",
        fusions=[],
        wbc=95.0, platelet=32.0, hemoglobin=8.5,
        ldh=1240.0, alt=28.0, ast=35.0, albumin=3.5,
        creatinine=0.9,
        blast_pct_bm=78.0, blast_pct_pb=65.0,
        age=45, sex="female",
        is_relapse=False, prior_mds=False, prior_chemo=False,
        is_initial_diagnosis=True,
    )

    out1 = predict_for_patient(rna1, kit1)
    print(pretty_print_kit_output(out1))
    print()

    # --- Patient 2: TP53 + complex karyotype, elderly ---
    rna2 = _synthetic_rna_counts(kept, "elderly_tp53")
    kit2 = KitInput(
        patient_id="SYNTHETIC-002",
        mutations=[
            MutationCall(gene="TP53", variant_type="missense", vaf=0.55),
            MutationCall(gene="ASXL1", variant_type="nonsense", vaf=0.42),
        ],
        karyotype_text="45,XY,-7,del(5)(q13q33),+8,t(3;3)(q21;q26),del(17)(p13)[18]/46,XY[2]",
        fusions=[],
        wbc=12.0, platelet=25.0, hemoglobin=7.8,
        ldh=850.0, alt=22.0, ast=30.0, albumin=2.9,
        creatinine=1.2,
        blast_pct_bm=55.0, blast_pct_pb=30.0,
        age=72, sex="male",
        is_relapse=False, prior_mds=True, prior_chemo=False,
        is_initial_diagnosis=True,
    )

    out2 = predict_for_patient(rna2, kit2)
    print(pretty_print_kit_output(out2))


if __name__ == "__main__":
    demo()
