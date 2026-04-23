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


# Per-profile perturbations applied on BeatAML's log2-CPM-scale reference means
# for the 25-gene driver panel + expression-hint genes. Values are Δ from the
# BeatAML population mean (so a +2.0 entry makes the gene ~+2 SD above typical).
_PROFILE_EXPR_DELTAS: dict[str, dict[str, float]] = {
    "young_flt3": {
        # FLT3-ITD + NPM1-mut signature: high FLT3, high HOXA9/MEIS1 co-program
        "FLT3": +1.8, "NPM1": +0.3, "DNMT3A": +0.2,
        "HOXA9": +1.6, "MEIS1": +1.2,
        "BCL2": +1.5,  # justifies Ven in triplet
        "MECOM": -0.2,
        "TP53": +0.1,  # not deleted
    },
    "elderly_tp53": {
        # TP53 + complex-karyo + del(5q)/del(7q) pattern: low TP53, high MECOM,
        # high MCL1 (Ven-resistance hint), high BAALC
        "TP53": -2.2, "MECOM": +2.2, "MCL1": +1.5,
        "BAALC": +1.8, "MN1": +1.1,
        "FLT3": -0.1, "NPM1": -0.2,
        "HOXA9": +0.3,
    },
    "apl": {
        # PML-RARA (APL): high BCL2 (differentiation block), high CD33
        # (Mylotarg rationale), normal-ish HOX, low BAALC
        "BCL2": +2.0, "CD33": +1.8, "IL3RA": +0.6,
        "HOXA9": -0.3, "MEIS1": -0.2,
        "BAALC": -0.8, "MN1": -0.4,
        "FLT3": +0.2, "TP53": +0.2,
    },
}


# Per-profile perturbations to a small handful of non-curated genes — these
# are classic AML "extra" outliers that let Section C of the RNA report show
# content for demo patients. In real patients with full RNA-Seq, these
# extras come naturally from the full transcriptome.
_PROFILE_EXTRA_DELTAS: dict[str, dict[str, float]] = {
    "young_flt3": {
        "GATA2":   +3.5,    # high GATA2: HSC/LSC program, classic in FLT3-ITD
        "PROM1":   +3.2,    # CD133: LSC marker, high in FLT3-ITD
        "MPO":     -3.1,    # mature myeloid marker, low in primitive AML
        "CD34":    +3.3,    # blast marker
    },
    "elderly_tp53": {
        "ERG":     +3.6,    # ERG overexpression — classic TP53/complex karyo
        "S100A8":  +3.3,    # inflammatory signature, often high in MDS→AML
        "DNMT1":   +3.1,    # HMA target, expression varies
        "NFKB1":   -3.2,    # reduced NF-kB activity in TP53-null
    },
    "apl": {
        "PRAM1":   +4.0,    # PRAM/PML-related, very high in APL
        "CTSG":    +3.8,    # Cathepsin G, promyelocytic marker
        "ELANE":   +3.7,    # elastase, promyelocytic marker
        "FUT3":    +3.2,    # fucosyltransferase, APL-specific
    },
}


def _synthetic_rna_expression_full(profile: str,
                                     ref_stats_path: Path | str | None = None,
                                     full_ref_stats_path: Path | str | None = None,
                                     ) -> pd.Series:
    """Generate a full-transcriptome-style Series that covers:
      (a) the 25-gene core panel + 8 expression-hint genes on BeatAML
          log-CPM scale, with profile-specific perturbations,
      (b) a handful of profile-specific "extra" outliers (GATA2, PROM1,
          ERG, PRAM1, etc.) so the report's Section C can demonstrate
          the full-transcriptome scan feature,
      (c) all remaining BeatAML genes at their population mean + N(0, 0.25)
          noise so the full transcriptome is available for scanning.

    Baseline = BeatAML population mean for each gene. Noise = N(0, 0.25).
    """
    import json
    from pathlib import Path as _P
    if ref_stats_path is None:
        ref_stats_path = _P("data/canonical/driver_gene_ref_stats.json")
        if not _P(ref_stats_path).exists():
            ref_stats_path = (_P("/Users/ericktom/AML-combo-validation") /
                              "data/canonical/driver_gene_ref_stats.json")
    stats = json.loads(_P(ref_stats_path).read_text())
    gene_stats = stats["genes"]
    deltas = _PROFILE_EXPR_DELTAS.get(profile, {})
    extras = _PROFILE_EXTRA_DELTAS.get(profile, {})

    rng = np.random.default_rng({"young_flt3": 17, "elderly_tp53": 42,
                                  "apl": 31}.get(profile, 0))

    expr = {}
    # 25+8 core genes with targeted perturbations
    for g, s in gene_stats.items():
        base = s["mean"] + s["std"] * float(deltas.get(g, 0.0))
        expr[g] = base + rng.normal(0, 0.25)

    # Optional: load the full-transcriptome ref file to populate remaining
    # genes. If missing, we just ship the 25+8 + extras — the kit's
    # transcriptome scan will still find the extras.
    if full_ref_stats_path is None:
        full_ref_stats_path = _P(
            "data/canonical/full_transcriptome_ref_stats.npz",
        )
        if not _P(full_ref_stats_path).exists():
            full_ref_stats_path = (
                _P("/Users/ericktom/AML-combo-validation") /
                "data/canonical/full_transcriptome_ref_stats.npz"
            )
    if _P(full_ref_stats_path).exists():
        data = np.load(full_ref_stats_path, allow_pickle=False)
        all_genes = data["genes"]
        all_means = data["means"]
        all_stds = data["stds"]
        for gene, mean, std in zip(all_genes, all_means, all_stds):
            if gene in expr:  # already covered by curated path
                continue
            if gene in extras:
                # extras get a big delta to become visible outliers
                expr[gene] = (float(mean) + float(std) * extras[gene]
                               + rng.normal(0, 0.25))
            else:
                # Background: population mean + small noise
                expr[gene] = float(mean) + rng.normal(0, 0.25)
    else:
        # No full-transcriptome ref — just add the extras directly
        for gene, delta in extras.items():
            expr[gene] = 6.0 + delta * 1.0 + rng.normal(0, 0.25)

    return pd.Series(expr)


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
