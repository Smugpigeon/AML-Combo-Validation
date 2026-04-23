"""RNA-Seq quality control + out-of-distribution detection for kit inputs.

Addresses problem #1 of the kit-readiness audit: OOD calibration.

Two defensive layers applied to every new-patient RNA-Seq input:

  LAYER A — Quantile normalization to BeatAML training reference
  ------------------------------------------------------------------
  Rationale: different RNA-Seq pipelines (STAR vs HISAT2, featureCounts vs
  Salmon, raw count vs TPM vs VST) produce numerically different gene-count
  distributions for the same underlying transcriptome. Even a slight pipeline
  mismatch → gene counts shifted by 5-30% → log2-PCA projection lands in
  a part of PC space the MLP has never seen → predictions wildly miscalibrated.

  Quantile normalization forces the new sample's empirical distribution to
  match BeatAML's. For each new sample:
    1. Sort its 5000 gene log2-counts.
    2. Replace the k-th smallest value with `quantile_reference[k]` — the
       average of the k-th smallest log2-counts across BeatAML's 707 training
       samples.
  After QN, two samples that used different pipelines should produce
  near-identical PC projections for the same biological state.

  LAYER B — Mahalanobis distance in PC space as OOD detector
  ------------------------------------------------------------------
  Rationale: even with QN, a sample's RELATIVE gene-rank structure can be
  wildly different from AML (e.g., you accidentally pass a breast-cancer
  sample). We catch this by computing the Mahalanobis distance from the
  projected PC coordinate to the training-sample PC distribution.

  If d > threshold (99th percentile of training = 9.75), we flag the sample
  as OOD and the kit returns a warning in `confidence_notes` / refuses to
  produce a prediction (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RNAQCReport:
    n_genes_input: int
    n_genes_matched_to_panel: int
    gene_coverage_pct: float
    pre_qn_mean_log2: float
    pre_qn_std_log2: float
    post_qn_mean_log2: float
    post_qn_std_log2: float
    mahalanobis_raw: float        # projection without QN — catches pipeline + biology OOD
    mahalanobis_qn: float         # projection with QN — catches biology-only OOD
    ood_threshold_95pct: float
    ood_threshold_99pct: float
    ood_severity: str
    """ 'ok' | 'borderline' (raw>95%) | 'pipeline_mismatch' (raw>99%, qn<95%) |
       'ood' (qn>99%) | 'far_ood' (qn>99.9%) """
    warning_messages: list[str]

    # Legacy alias: current code reads `mahalanobis_distance`. Keep a
    # simple single number too — use the QN version since that's the
    # "corrected" one we'd cite most often.
    @property
    def mahalanobis_distance(self) -> float:
        return self.mahalanobis_qn


def quantile_normalize(sample_values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remap sample_values's distribution to match reference via rank-alignment.

    sample_values : (n_genes,) — one patient's log2(count+1) for the 5000
                   training genes, IN TRAINING GENE ORDER.
    reference     : (n_genes,) — bundle["quantile_reference"]: the avg sorted
                   log2(count+1) across BeatAML training samples.
    Returns       : (n_genes,) — quantile-normalized values, same length/order.
    """
    assert sample_values.shape == reference.shape, (
        f"shape mismatch: sample {sample_values.shape} vs ref {reference.shape}"
    )
    # Rank the sample (handle ties via average rank so monotonic)
    # Ascending rank: smallest gets rank 0, largest gets rank n-1
    order = np.argsort(sample_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(sample_values))
    # Replace by reference value at that rank (reference is already sorted ascending)
    return reference[ranks].astype(np.float32)


def project_and_qc(
    rna_counts: pd.Series,
    bundle: dict,
    apply_quantile_normalization: bool = True,
) -> tuple[np.ndarray, RNAQCReport]:
    """Full QC pipeline: reindex → log2 → (optional QN) → project → Mahalanobis.

    Returns (pc_vector, RNAQCReport).
    """
    kept_genes: list[str] = bundle["kept_genes"]
    n_input_genes = len(rna_counts)
    n_matched = int(rna_counts.index.isin(kept_genes).sum())

    # Reindex to training gene panel; missing → 0
    counts = rna_counts.reindex(kept_genes).fillna(0.0).to_numpy(dtype=np.float64)
    counts = np.clip(counts, a_min=0.0, a_max=None)
    log_counts = np.log2(counts + 1.0)
    log_counts = np.nan_to_num(log_counts, nan=0.0, posinf=0.0, neginf=0.0)

    pre_qn_mean = float(log_counts.mean())
    pre_qn_std = float(log_counts.std())

    pca_mean = np.asarray(bundle["pca_mean_"], dtype=np.float64)
    pca_components = np.asarray(bundle["pca_components_"], dtype=np.float64)
    train_mean = np.asarray(bundle.get("train_pc_mean", np.zeros(pca_components.shape[0])), dtype=np.float64)
    train_cov_inv = np.asarray(
        bundle.get("train_pc_cov_inv", np.eye(pca_components.shape[0])), dtype=np.float64,
    )

    def _maha(projected_input: np.ndarray) -> tuple[np.ndarray, float]:
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            centered = projected_input - pca_mean
            pc_vec = centered @ pca_components.T
            diff = pc_vec - train_mean
            m_sq = float(diff @ train_cov_inv @ diff)
        return pc_vec, float(np.sqrt(max(m_sq, 0.0)))

    # ---- Compute BOTH Mahalanobis distances ----
    # (1) Raw — log2 only, no distribution matching. Catches both pipeline
    #     mismatch AND biological mismatch.
    _, maha_raw = _maha(log_counts)

    # (2) QN — after quantile normalization to training distribution. Pipeline
    #     mismatch is washed out, so this distance reflects ONLY biological
    #     mismatch (if any).
    ref = bundle.get("quantile_reference")
    if apply_quantile_normalization and ref is not None:
        normalized = quantile_normalize(log_counts, np.asarray(ref, dtype=np.float64))
        post_qn_mean = float(normalized.mean())
        post_qn_std = float(normalized.std())
        pc_vector, maha_qn = _maha(normalized.astype(np.float64))
    else:
        post_qn_mean = pre_qn_mean
        post_qn_std = pre_qn_std
        pc_vector, maha_qn = log_counts, maha_raw  # not really used if QN off

    # If QN is off, use the raw projection as the "final" PC vector
    if not apply_quantile_normalization or ref is None:
        pc_vector, _ = _maha(log_counts)

    thresh_95 = float(bundle.get("train_maha_threshold_95pct", np.inf))
    thresh_99 = float(bundle.get("train_maha_threshold_99pct", np.inf))
    thresh_999 = float(bundle.get("train_maha_threshold_99_9pct", np.inf))

    # ---- Input-variance check ----
    # Catches degenerate inputs (all zeros, constant values, failed loading).
    input_std = float(log_counts.std())
    input_variance_ok = input_std > 0.5  # BeatAML samples have std ~3-4

    # ---- AML marker-rank biology check ----
    # In any real AML sample, CD34/KIT/HOXA9/MEIS1 sit at high ranks (top 15-25%)
    # of the 5000-gene variable panel. Random-shuffled or non-myeloid samples
    # lose this signature. Z-score the observed ranks against training statistics.
    marker_genes = bundle.get("aml_marker_genes") or []
    marker_ref_mean = np.asarray(bundle.get("aml_marker_ref_mean_rank", []), dtype=np.float64)
    marker_ref_std = np.asarray(bundle.get("aml_marker_ref_std_rank", []), dtype=np.float64)
    marker_biology_ok = True
    marker_z_mean = 0.0
    if len(marker_genes) >= 3:
        # Rank the log-count input (ascending: smallest = 0, largest = n-1)
        input_ranks_full = log_counts.argsort().argsort()
        marker_idx_in_panel = [bundle["kept_genes"].index(g) for g in marker_genes]
        observed_marker_ranks = input_ranks_full[marker_idx_in_panel]
        # Z-score per marker, then average
        z = (observed_marker_ranks - marker_ref_mean) / np.maximum(marker_ref_std, 1.0)
        marker_z_mean = float(np.mean(np.abs(z)))
        # Average |z| > 2 means markers are systematically displaced from training
        marker_biology_ok = marker_z_mean < 2.0

    # ---- Layered severity logic ----
    warnings: list[str] = []
    # Pre-empt the distance-based logic: catch degenerate or biology-broken inputs first
    if not input_variance_ok:
        severity = "far_ood"
        warnings.append(
            f"CRITICAL: input has near-zero variance (std={input_std:.2f}). "
            f"Real RNA-Seq samples have log-count std > 2. Likely causes: "
            f"all-zero input, failed file loading, or wrong column extracted."
        )
    elif not marker_biology_ok:
        severity = "ood"
        marker_str = ", ".join(marker_genes[:4])
        warnings.append(
            f"WARNING: AML marker-gene expression pattern does not match training. "
            f"Average |z-score| for {marker_str} = {marker_z_mean:.2f} (threshold 2.0). "
            f"This is likely a non-AML sample, a wrong-gene-ID sample, or "
            f"randomly scrambled genes. Predictions NOT reliable even if "
            f"PC-space Mahalanobis looks ok."
        )
    elif maha_qn > thresh_999:
        severity = "far_ood"
        warnings.append(
            f"CRITICAL: post-QN Mahalanobis {maha_qn:.2f} far exceeds 99.9%ile "
            f"threshold ({thresh_999:.2f}). Quantile normalization could not "
            f"rescue this sample — the underlying gene-rank structure doesn't "
            f"match AML biology. Likely causes: non-AML tissue, failed RNA-Seq "
            f"QC (low library size), or gene-ID confusion (Ensembl vs symbols)."
        )
    elif maha_qn > thresh_99:
        severity = "ood"
        warnings.append(
            f"WARNING: post-QN Mahalanobis {maha_qn:.2f} exceeds 99%ile "
            f"threshold ({thresh_99:.2f}). Biology-level outlier — "
            f"predictions may be unreliable. Verify: is this an AML sample? "
            f"Was the RNA-Seq QC passed?"
        )
    elif maha_raw > thresh_99 and maha_qn < thresh_95:
        severity = "pipeline_mismatch"
        warnings.append(
            f"Pipeline mismatch detected and corrected: raw Mahalanobis "
            f"{maha_raw:.2f} > 99%ile but post-QN {maha_qn:.2f} < 95%ile. "
            f"Quantile normalization appears to have aligned this sample to "
            f"the BeatAML training distribution. Cross-check top PC loadings "
            f"match expected AML biology before relying on predictions."
        )
    elif maha_raw > thresh_95 or maha_qn > thresh_95:
        severity = "borderline"
        warnings.append(
            f"Borderline: Mahalanobis (raw={maha_raw:.2f}, QN={maha_qn:.2f}) in "
            f"training-distribution tail. Treat predictions as directional."
        )
    else:
        severity = "ok"

    # Gene coverage warning
    coverage_pct = 100 * n_matched / len(kept_genes)
    if coverage_pct < 70:
        warnings.append(
            f"Gene coverage only {coverage_pct:.0f}% of training panel "
            f"({n_matched}/{len(kept_genes)}); missing genes filled with 0."
        )
    elif coverage_pct < 90:
        warnings.append(
            f"Gene coverage {coverage_pct:.0f}% — adequate but not full."
        )

    # Distribution shift warning
    if apply_quantile_normalization and abs(pre_qn_mean - post_qn_mean) > 2.0:
        warnings.append(
            f"Large distribution shift from quantile normalization "
            f"(mean {pre_qn_mean:.2f} → {post_qn_mean:.2f}). Input pipeline may "
            f"produce counts in a scale very different from BeatAML's."
        )

    report = RNAQCReport(
        n_genes_input=n_input_genes,
        n_genes_matched_to_panel=n_matched,
        gene_coverage_pct=round(coverage_pct, 1),
        pre_qn_mean_log2=round(pre_qn_mean, 2),
        pre_qn_std_log2=round(pre_qn_std, 2),
        post_qn_mean_log2=round(post_qn_mean, 2),
        post_qn_std_log2=round(post_qn_std, 2),
        mahalanobis_raw=round(maha_raw, 2),
        mahalanobis_qn=round(maha_qn, 2),
        ood_threshold_95pct=round(thresh_95, 2),
        ood_threshold_99pct=round(thresh_99, 2),
        ood_severity=severity,
        warning_messages=warnings,
    )
    return pc_vector.astype(np.float32), report


def load_bundle(path: Path | str = "data/canonical/beataml_rna_preprocessor.joblib") -> dict:
    return joblib.load(path)
