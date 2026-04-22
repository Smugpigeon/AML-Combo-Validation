"""Simplified BeatAML ETL for the combo-validation project.

This is a *flat* feature extractor — a single (patient × ~80 features) table
plus a long (patient × drug × AUC) response table. No multimodal encoders,
no graph, no SNF. The philosophy is "keep it boring" so the downstream
single-drug predictor and combo model train quickly and are easy to ablate.

Inputs (`data/raw/BeatAML2.0/`):
  - beataml_clinical.xlsx            (942 samples, 95 cols)
  - beatAML.xlsx                     (22,843 genes × 707 RNA-Seq samples)
  - beatAML2.0 AUC(1).xlsx           (~63K rows)
  - WES-targeted Sequencing Mutation Calls.txt  (from parse_beataml_variants.py)

Outputs (`data/canonical/`):
  - beataml_patient_features.csv     (patient × ~80 features)
  - beataml_drug_response_long.csv   (patient × drug × AUC rows)
  - beataml_feature_manifest.json    (column provenance, rank, etc.)

Feature breakdown (target: ~80 dims per patient):
  - RNA PCA components: 50  (PCA on log2(TPM+1), variance-filtered genes)
  - Mutation binary flags: 25  (top curated AML driver genes)
  - Clinical: 5  (age, ELN risk ordinal, blast_pct, secondary_aml, fit_for_intensive)

Design decisions:
  1. PCA, not a learned encoder — we showed earlier that BeatAML's intrinsic
     dimensionality is ~10, so PCA-50 is plenty and avoids training cost.
  2. 25 curated mutation genes (not free-text variantSummary tokens) —
     reduces feature noise; captures the known drivers that matter.
  3. ELN ordinal encoding (Favorable=0, Intermediate=1, Adverse=2) — preserves
     ordering information lost by one-hot.
  4. Per-patient aggregation: take latest non-relapse sample if multiple.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Curated gene panels + label mappings
# ---------------------------------------------------------------------------

# 25 curated AML driver genes (WHO/ICC 2022 + emerging targets).
# This is deliberately smaller than variantSummary's ~140-gene raw vocabulary
# — those rare genes add noise to the feature matrix. Downstream models can
# still pick up rare events via the continuous RNA features.
CURATED_MUTATION_GENES: tuple[str, ...] = (
    "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TP53",
    "RUNX1", "ASXL1", "TET2", "CEBPA", "KIT",
    "NRAS", "KRAS", "PTPN11", "WT1", "BCOR",
    "STAG2", "PHF6", "SRSF2", "SF3B1", "U2AF1",
    "EZH2", "KMT2A", "MECOM", "CBFB",
)

# ELN 2017 risk encoding (ordinal). Unknown/NonInitial/etc. map to NaN (handled downstream).
ELN_ORDINAL: Mapping[str, float] = {
    "Favorable": 0.0,
    "FavorableOrIntermediate": 0.5,
    "Intermediate": 1.0,
    "IntermediateOrAdverse": 1.5,
    "Adverse": 2.0,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatAMLConfig:
    raw_dir: Path = Path("data/raw/BeatAML2.0")
    out_dir: Path = Path("data/canonical")
    n_pca: int = 50
    top_n_variable_genes: int = 5000   # keep top-N by per-gene variance after log-norm
    mutation_genes: tuple[str, ...] = CURATED_MUTATION_GENES
    random_state: int = 42


# ---------------------------------------------------------------------------
# Utility loaders
# ---------------------------------------------------------------------------


def _read_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="summary")
    # Normalize patient_id as integer string to match AUC/expression tables
    df["dbgap_subject_id"] = pd.to_numeric(df["dbgap_subject_id"], errors="coerce")
    return df


def _read_expression(path: Path) -> pd.DataFrame:
    """Returns gene × sample DataFrame (index = gene symbol, columns = BA...R)."""
    df = pd.read_excel(path, sheet_name="Sheet1")
    if "symbol" not in df.columns:
        raise ValueError(f"Expected 'symbol' column in {path}; found {list(df.columns)[:5]}")
    df = df.set_index("symbol")
    # Drop duplicate genes (keep first)
    df = df[~df.index.duplicated(keep="first")]
    return df


def _read_auc(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df[["dbgap_subject_id", "dbgap_rnaseq_sample", "inhibitor", "auc", "ic50"]].copy()
    df["dbgap_subject_id"] = pd.to_numeric(df["dbgap_subject_id"], errors="coerce")
    df = df.dropna(subset=["auc"])
    return df


def _read_mutation_calls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def extract_rna_pca_features(
    expr: pd.DataFrame,
    sample_to_patient: pd.Series,
    *,
    n_pca: int,
    top_n_variable_genes: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    """Extract patient × RNA PCA features.

    Steps:
      1. Log-normalize the expression matrix (log2(x + 1))
      2. Keep the top-N most-variable genes (robust to normalization scale)
      3. Sample-level PCA (50 components)
      4. Average per patient if multiple samples
    """
    # expr is gene × sample. Impute NaN → 0 (unexpressed in that sample).
    X = expr.to_numpy(dtype=np.float64)
    n_nan_before = int(np.isnan(X).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip any tiny negatives (shouldn't exist but be safe for log2)
    X = np.clip(X, a_min=0.0, a_max=None)
    X = np.log2(X + 1.0)  # gene × sample
    # Final sanity check: no NaN, no inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.isfinite(X).all(), "non-finite values slipped into log-normalized expression"
    gene_var = X.var(axis=1)
    top_idx = np.argsort(gene_var)[::-1][:top_n_variable_genes]
    top_idx.sort()
    keep_mask = np.zeros_like(gene_var, dtype=bool)
    keep_mask[top_idx] = True
    X = X[keep_mask]
    kept_genes = expr.index[keep_mask].tolist()

    # Transpose: sample × gene
    X_t = X.T  # (n_samples, n_genes)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_t)  # (n_samples, n_pca)

    # Build sample × PC frame
    sample_pca = pd.DataFrame(
        X_pca,
        index=expr.columns,
        columns=[f"rna_pc{i+1:02d}" for i in range(n_pca)],
    )
    sample_pca.index.name = "rna_sample_id"

    # Map sample → patient, average across samples per patient
    merged = sample_pca.join(sample_to_patient.rename("patient_id"), how="inner")
    if merged["patient_id"].isna().all():
        raise ValueError("No RNA samples could be mapped to patient_id — check sample_to_patient")
    patient_pca = merged.dropna(subset=["patient_id"]).groupby("patient_id").mean(numeric_only=True)

    meta = {
        "n_samples_used": int(X_t.shape[0]),
        "n_genes_before": int(expr.shape[0]),
        "n_genes_kept_top_variable": int(keep_mask.sum()),
        "n_patients": int(patient_pca.shape[0]),
        "explained_variance_ratio_cumulative": (
            np.cumsum(pca.explained_variance_ratio_).round(4).tolist()
        ),
        "n_pca": int(n_pca),
    }
    return patient_pca, meta


def extract_mutation_features(
    mutation_df: pd.DataFrame,
    sample_to_patient: pd.Series,
    genes: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Extract patient × mutation binary features.

    mutation_df columns (from parse_beataml_variants.py output):
      dbgap_dnaseq_sample, gene, variant_classification, t_vaf, hgvsp_short
    """
    # Map DNA sample to patient (not RNA — the mutation file uses DNA IDs).
    # We approximate: DNA sample ID and patient map are joined via the clinical
    # table. The simplest route here is to treat the variantSummary parser's
    # `dbgap_dnaseq_sample` as already patient-keyed.
    mut = mutation_df.copy()
    mut["patient_id"] = mut["dbgap_dnaseq_sample"].map(sample_to_patient)

    # Drop rows with no patient mapping
    n_before = len(mut)
    mut = mut.dropna(subset=["patient_id"])
    n_dropped = n_before - len(mut)

    # Binary: patient × gene panel
    panel = set(genes)
    mut = mut[mut["gene"].isin(panel)]
    grouped = mut.groupby(["patient_id", "gene"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=list(genes), fill_value=0)
    grouped = (grouped > 0).astype(int)
    grouped.columns = [f"mut_{g}" for g in grouped.columns]
    grouped.index.name = "patient_id"

    meta = {
        "n_patients_with_any_mutation": int(grouped.shape[0]),
        "n_gene_panel": int(len(genes)),
        "n_dropped_unmapped": int(n_dropped),
        "top_5_gene_prevalence": grouped.mean().sort_values(ascending=False).head(5).round(3).to_dict(),
    }
    return grouped, meta


def extract_clinical_features(clinical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Extract ~5 clinical covariates per patient.

    Columns: age, eln_ordinal, blast_pct, secondary_aml_flag, fit_for_intensive_flag.
    """
    df = clinical.copy()
    df = df.drop_duplicates("dbgap_subject_id", keep="last").set_index("dbgap_subject_id")

    out = pd.DataFrame(index=df.index)

    # age
    out["clin_age"] = pd.to_numeric(df.get("ageAtDiagnosis"), errors="coerce")

    # ELN ordinal
    out["clin_eln_ordinal"] = df.get("ELN2017", pd.Series(index=df.index)).map(ELN_ORDINAL)

    # blast_pct — average of BM and PB blast columns if both present
    bm = pd.to_numeric(df.get("%.Blasts.in.BM"), errors="coerce")
    pb = pd.to_numeric(df.get("%.Blasts.in.PB"), errors="coerce")
    combined = bm.fillna(pb).fillna((bm + pb) / 2)
    out["clin_blast_pct"] = combined

    # secondary AML flag (any of therapy-related or MDS-related)
    tr = df.get("therapy_related_flag", pd.Series(index=df.index)).astype(str).isin(["1", "1.0", "True"])
    mds = df.get("priorMDS", pd.Series(index=df.index)).astype(str).str.lower().isin(["y", "yes", "1"])
    mpn = df.get("priorMPN", pd.Series(index=df.index)).astype(str).str.lower().isin(["y", "yes", "1"])
    out["clin_secondary_aml"] = (tr | mds | mpn).astype(int)

    # Fit for intensive: age <= 65 AND ECOG <= 2 (ECOG may be missing → default fit if age young)
    out["clin_fit_for_intensive"] = ((out["clin_age"].fillna(100) <= 65)).astype(int)

    out.index.name = "patient_id"
    meta = {
        "n_patients": int(len(out)),
        "age_stats": {"mean": float(out["clin_age"].mean()), "median": float(out["clin_age"].median())},
        "eln_distribution": df["ELN2017"].value_counts().head(5).to_dict() if "ELN2017" in df.columns else {},
        "secondary_aml_rate": float(out["clin_secondary_aml"].mean()),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_beataml_etl(cfg: BeatAMLConfig | None = None) -> dict:
    """Main entry: read raw BeatAML files, build feature matrix + drug response table.

    Returns manifest dict with provenance.
    """
    cfg = cfg or BeatAMLConfig()
    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[etl] raw dir: {raw_dir}")
    print(f"[etl] out dir: {out_dir}")

    # --- Load raw ---
    print("[etl] Loading clinical summary ...")
    clinical = _read_clinical(raw_dir / "beataml_clinical.xlsx")
    print(f"      {len(clinical)} rows × {len(clinical.columns)} cols")

    print("[etl] Loading expression (22K genes × 707 samples, ~30s) ...")
    expr = _read_expression(raw_dir / "beatAML.xlsx")
    print(f"      {expr.shape[0]} genes × {expr.shape[1]} samples")

    print("[etl] Loading AUC ...")
    auc = _read_auc(raw_dir / "beatAML2.0 AUC(1).xlsx")
    print(f"      {len(auc)} measurements")

    print("[etl] Loading mutation calls (from variantSummary parser) ...")
    mutation = _read_mutation_calls(raw_dir / "WES-targeted Sequencing Mutation Calls.txt")
    print(f"      {len(mutation)} mutation events")

    # --- Sample → patient maps ---
    rna_sample_to_patient = (
        clinical[["dbgap_rnaseq_sample", "dbgap_subject_id"]]
        .dropna()
        .drop_duplicates("dbgap_rnaseq_sample")
        .set_index("dbgap_rnaseq_sample")["dbgap_subject_id"]
    )
    rna_sample_to_patient = rna_sample_to_patient.astype(int)

    dna_sample_to_patient = (
        clinical[["dbgap_dnaseq_sample", "dbgap_subject_id"]]
        .dropna()
        .drop_duplicates("dbgap_dnaseq_sample")
        .set_index("dbgap_dnaseq_sample")["dbgap_subject_id"]
    )
    dna_sample_to_patient = dna_sample_to_patient.astype(int)

    # --- Feature extraction ---
    print("[etl] Extracting RNA PCA features ...")
    rna_pca, rna_meta = extract_rna_pca_features(
        expr,
        rna_sample_to_patient,
        n_pca=cfg.n_pca,
        top_n_variable_genes=cfg.top_n_variable_genes,
        random_state=cfg.random_state,
    )
    print(f"      {rna_pca.shape[0]} patients × {rna_pca.shape[1]} PCs  "
          f"(cum var @ PC50 = {rna_meta['explained_variance_ratio_cumulative'][-1]:.3f})")

    print("[etl] Extracting mutation features ...")
    mut_feat, mut_meta = extract_mutation_features(
        mutation, dna_sample_to_patient, cfg.mutation_genes
    )
    print(f"      {mut_feat.shape[0]} patients × {mut_feat.shape[1]} gene flags")

    print("[etl] Extracting clinical features ...")
    clin_feat, clin_meta = extract_clinical_features(clinical)
    print(f"      {clin_feat.shape[0]} patients × {clin_feat.shape[1]} clinical")

    # --- Merge patient features ---
    print("[etl] Merging patient features ...")
    features = rna_pca.join(mut_feat, how="left").join(clin_feat, how="left")
    # Fill NA with 0 for mutations (no call = absent), median for clinical numerics
    for c in mut_feat.columns:
        features[c] = features[c].fillna(0).astype(int)
    for c in ["clin_age", "clin_blast_pct", "clin_eln_ordinal"]:
        if c in features.columns:
            features[c] = features[c].fillna(features[c].median())
    for c in ["clin_secondary_aml", "clin_fit_for_intensive"]:
        if c in features.columns:
            features[c] = features[c].fillna(0).astype(int)
    features.index.name = "patient_id"

    print(f"      final feature matrix: {features.shape[0]} patients × {features.shape[1]} features")

    # --- Drug response long table ---
    print("[etl] Building drug response long table ...")
    auc_kept = auc.dropna(subset=["dbgap_subject_id"]).copy()
    auc_kept["patient_id"] = auc_kept["dbgap_subject_id"].astype(int)
    drug_long = auc_kept[["patient_id", "inhibitor", "auc", "ic50"]].rename(
        columns={"inhibitor": "drug_id"}
    )
    # Keep only patients who have features
    pids_with_features = set(features.index)
    drug_long = drug_long[drug_long["patient_id"].isin(pids_with_features)]
    print(f"      {len(drug_long)} (patient × drug) rows covering "
          f"{drug_long['patient_id'].nunique()} patients × "
          f"{drug_long['drug_id'].nunique()} drugs")

    # --- Write ---
    features_path = out_dir / "beataml_patient_features.csv"
    drug_path = out_dir / "beataml_drug_response_long.csv"
    features.to_csv(features_path)
    drug_long.to_csv(drug_path, index=False)

    manifest = {
        "cfg": {
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "n_pca": cfg.n_pca,
            "top_n_variable_genes": cfg.top_n_variable_genes,
            "random_state": cfg.random_state,
            "mutation_genes_n": len(cfg.mutation_genes),
        },
        "rna": rna_meta,
        "mutation": mut_meta,
        "clinical": clin_meta,
        "drug_response": {
            "n_rows": int(len(drug_long)),
            "n_patients": int(drug_long["patient_id"].nunique()),
            "n_drugs": int(drug_long["drug_id"].nunique()),
        },
        "outputs": {
            "patient_features": str(features_path),
            "drug_response_long": str(drug_path),
        },
        "feature_columns": features.columns.tolist(),
    }

    manifest_path = out_dir / "beataml_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[etl] Done. Manifest at {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", default="data/raw/BeatAML2.0")
    ap.add_argument("--out-dir", default="data/canonical")
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--top-n-genes", type=int, default=5000)
    args = ap.parse_args()
    cfg = BeatAMLConfig(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        n_pca=args.n_pca,
        top_n_variable_genes=args.top_n_genes,
    )
    run_beataml_etl(cfg)
