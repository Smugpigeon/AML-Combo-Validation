"""Build a 104-dim BeatAML-schema patient feature vector for a NEW patient.

Takes:
  - RNA counts as gene_symbol → count (pandas Series)
  - Mutation panel (list of MutationCall from kit_schema)
  - Clinical data as KitInput

Outputs:
  - numpy array shape (n_features,) matching the feature_columns saved in
    the preprocessor bundle. Feature order EXACTLY matches BeatAML training,
    so the saved Baseline A MLP can consume it directly.

Imputation policy:
  - Missing RNA genes → 0 count (unexpressed).
  - Missing mutations → 0 (no call).
  - Missing clinical → saved BeatAML medians (loaded from the bundle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from combo_val.clinical.eln_computer import ELN_ordinal_from_string, compute_eln2017
from combo_val.clinical.karyotype_parser import parse_karyotype
from combo_val.clinical.kit_schema import KitInput, MutationCall
from combo_val.clinical.rna_qc import project_and_qc, RNAQCReport


def _mut_flags(mutations: list[MutationCall], gene_list: list[str]) -> dict[str, int]:
    out = {f"mut_{g}": 0 for g in gene_list}
    for m in mutations:
        key = f"mut_{m.gene.upper()}"
        if key in out:
            out[key] = 1
    return out


def _clinical_to_dict(kit: KitInput) -> dict[str, float]:
    """Translate KitInput → the 29 clinical feature columns matching training."""
    out: dict[str, float] = {}

    # Core
    out["clin_age"] = float(kit.age) if kit.age is not None else np.nan
    # ELN — compute if not provided
    if kit.eln2017:
        out["clin_eln_ordinal"] = ELN_ordinal_from_string(kit.eln2017)
    else:
        eln = compute_eln2017(kit.karyotype_text, kit.mutations, kit.fusions)
        out["clin_eln_ordinal"] = float(eln.ordinal)

    # Blast % — prefer BM, fall back to PB
    if kit.blast_pct_bm is not None:
        out["clin_blast_pct"] = float(kit.blast_pct_bm)
    elif kit.blast_pct_pb is not None:
        out["clin_blast_pct"] = float(kit.blast_pct_pb)
    else:
        out["clin_blast_pct"] = np.nan

    prior_mds = bool(kit.prior_mds) if kit.prior_mds is not None else False
    out["clin_secondary_aml"] = int(prior_mds)
    fit = (kit.age is not None and kit.age <= 65)
    out["clin_fit_for_intensive"] = int(fit) if kit.age is not None else np.nan

    # Demographics
    out["clin_sex_male"] = int((kit.sex or "").lower() in {"male", "m"})
    out["clin_is_relapse"] = int(bool(kit.is_relapse)) if kit.is_relapse is not None else 0

    # Labs (log where applicable)
    def _log(x):
        if x is None:
            return np.nan
        return float(np.log1p(max(0.0, x)))

    out["clin_wbc_log"] = _log(kit.wbc)
    out["clin_platelet_log"] = _log(kit.platelet)
    out["clin_hemoglobin"] = float(kit.hemoglobin) if kit.hemoglobin is not None else np.nan
    out["clin_ldh_log"] = _log(kit.ldh)
    out["clin_alt_log"] = _log(kit.alt)
    out["clin_ast_log"] = _log(kit.ast)
    out["clin_albumin"] = float(kit.albumin) if kit.albumin is not None else np.nan

    # FLT3 detail
    flt3_itd = any(m.gene.upper() == "FLT3" and m.is_ITD for m in kit.mutations)
    flt3_tkd = any(m.gene.upper() == "FLT3" and m.is_TKD for m in kit.mutations)
    flt3_ar = 0.0
    for m in kit.mutations:
        if m.gene.upper() == "FLT3" and m.is_ITD and m.allelic_ratio is not None:
            flt3_ar = float(m.allelic_ratio)
            break
    out["clin_flt3_itd"] = int(flt3_itd)
    out["clin_flt3_tkd"] = int(flt3_tkd)
    out["clin_flt3_allelic_ratio"] = flt3_ar

    # CEBPA biallelic
    out["clin_cebpa_biallelic"] = int(any(
        m.gene.upper() == "CEBPA" and m.is_biallelic for m in kit.mutations
    ))

    # Fusions
    fus_u = [f.upper() for f in (kit.fusions or [])]
    out["fusion_PML_RARA"] = int(any("PML-RARA" in f for f in fus_u))
    out["fusion_KMT2A_r"] = int(any(
        "KMT2A" in f or "MLLT" in f for f in fus_u
    ))
    out["fusion_CBFB_MYH11"] = int(any("CBFB-MYH11" in f for f in fus_u))
    out["fusion_RUNX1_RUNX1T1"] = int(any("RUNX1-RUNX1T1" in f for f in fus_u))
    any_known = (
        out["fusion_PML_RARA"] + out["fusion_KMT2A_r"]
        + out["fusion_CBFB_MYH11"] + out["fusion_RUNX1_RUNX1T1"]
    ) > 0
    out["fusion_other"] = int(bool(fus_u) and not any_known)

    # Karyotype flags
    k = parse_karyotype(kit.karyotype_text)
    kflags = k.as_dict()
    out["karyo_complex"] = kflags["karyo_complex"]
    out["karyo_monosomy_5_or_7"] = kflags["karyo_monosomy_5_or_7"]
    out["karyo_del_17p"] = kflags["karyo_del_17p"]

    # Disease state
    out["clin_prior_mds"] = int(bool(kit.prior_mds)) if kit.prior_mds is not None else 0
    out["clin_prior_chemo"] = int(bool(kit.prior_chemo)) if kit.prior_chemo is not None else 0
    out["clin_is_initial_diagnosis"] = (
        int(bool(kit.is_initial_diagnosis)) if kit.is_initial_diagnosis is not None else 1
    )
    return out


def build_patient_features_from_raw(
    rna_counts: pd.Series,
    kit: KitInput,
    preprocessor_path: Path = Path("data/canonical/beataml_rna_preprocessor.joblib"),
    apply_quantile_normalization: bool = True,
    strict_ood: bool = False,
) -> tuple[np.ndarray, dict]:
    """Build a feature vector matching the trained BeatAML schema.

    Parameters
    ----------
    rna_counts : gene_symbol → raw count (or log-space expression). Values must
        be non-negative. Gene symbols should match BeatAML's HUGO names.
    kit : structured clinical intake.
    preprocessor_path : path to the bundle produced by beataml_etl.
    apply_quantile_normalization : default True. Recommended ON for any input
        whose RNA-Seq pipeline may differ from BeatAML's (STAR + featureCounts
        on GENCODE v37). Turn OFF only if you've confirmed the input pipeline
        matches BeatAML's byte-for-byte.
    strict_ood : default False. When True, raises RuntimeError if the sample
        is flagged as OOD (Mahalanobis > 99%ile of training). When False,
        emits the warning in diag["qc"] but still returns a feature vector.

    Returns
    -------
    features : np.ndarray of shape (n_features_trained,), typically (104,).
    diag     : dict with QC report, coverage stats, imputation record.
    """
    bundle = joblib.load(preprocessor_path)
    feature_cols: list[str] = bundle["feature_columns"]
    medians: dict[str, float] = bundle["clinical_medians"]
    mutation_genes: list[str] = bundle["mutation_genes"]

    # --- RNA PCA + QC (quantile norm + Mahalanobis) ---
    rna_pcs, qc_report = project_and_qc(
        rna_counts, bundle,
        apply_quantile_normalization=apply_quantile_normalization,
    )
    rna_dict = {f"rna_pc{i+1:02d}": float(pc) for i, pc in enumerate(rna_pcs)}

    if strict_ood and qc_report.ood_severity in {"ood", "far_ood"}:
        raise RuntimeError(
            f"OOD input rejected (severity={qc_report.ood_severity}, "
            f"Mahalanobis={qc_report.mahalanobis_distance} > "
            f"threshold {qc_report.ood_threshold_99pct}). "
            f"Warnings: {qc_report.warning_messages}"
        )

    # --- Mutation binaries ---
    mut_dict = _mut_flags(kit.mutations, mutation_genes)

    # --- Clinical (may contain NaN for missing fields) ---
    clin_dict = _clinical_to_dict(kit)

    # --- Assemble in the saved order, imputing missing with medians ---
    all_dict = {**rna_dict, **mut_dict, **clin_dict}
    out = np.empty(len(feature_cols), dtype=np.float32)
    imputed_fields: list[str] = []
    for i, col in enumerate(feature_cols):
        v = all_dict.get(col, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = float(medians.get(col, 0.0))
            imputed_fields.append(col)
        else:
            out[i] = float(v)

    # Per issue #4 — compute BOTH ELN 2017 (matches BeatAML training labels)
    # AND ELN 2022 (current Döhner Blood 2022 standard). The kit feeds 2017
    # to the model (label-distribution-matched) but the report shows 2022 as
    # primary clinical guidance.
    from combo_val.clinical.eln_2022 import compute_eln2022
    eln2017_full = compute_eln2017(
        kit.karyotype_text, kit.mutations, kit.fusions
    )
    eln2022_full = compute_eln2022(
        kit.karyotype_text, kit.mutations, kit.fusions,
        prior_mds=kit.prior_mds,
    )
    diag = {
        "n_features": len(feature_cols),
        "rna_genes_in_panel": int(len(bundle["kept_genes"])),
        "rna_genes_present_in_input": qc_report.n_genes_matched_to_panel,
        "rna_gene_coverage_pct": qc_report.gene_coverage_pct,
        "n_imputed_fields": len(imputed_fields),
        "imputed_fields": imputed_fields,
        "eln_predicted": kit.eln2017 or eln2017_full.category,
        "eln2017": {
            "category": eln2017_full.category,
            "rationale": list(eln2017_full.rationale),
        },
        "eln2022": {
            "category": eln2022_full.category,
            "rationale": list(eln2022_full.rationale),
        },
        "qc": {
            "ood_severity": qc_report.ood_severity,
            "mahalanobis_distance": qc_report.mahalanobis_distance,
            "ood_threshold_99pct": qc_report.ood_threshold_99pct,
            "pre_qn_mean_log2": qc_report.pre_qn_mean_log2,
            "post_qn_mean_log2": qc_report.post_qn_mean_log2,
            "warning_messages": qc_report.warning_messages,
            "quantile_normalization_applied": apply_quantile_normalization,
        },
    }
    return out, diag
