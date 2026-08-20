#!/usr/bin/env python3
"""Infer baseline AML Virtual Cell v1.5 states for new single-cell samples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from combo_val.virtual_cell.single_cell_v15 import (  # noqa: E402
    PROGRAM_MARKERS,
    calibrate_probability_to_prevalence,
    classify_marker_lineage,
    combine_lineage_annotations,
    normalize_log1p_to_10k,
    reference_shift_contributions,
    summarize_patient_states,
    transcriptomic_suspicion,
    transform_gene_set_reference,
    transform_reference_shift,
    transform_state_atlas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--patient-manifest", type=Path)
    parser.add_argument("--celltypist-model")
    parser.add_argument("--disable-celltypist", action="store_true")
    parser.add_argument("--write-h5ad", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except (ImportError, ModuleNotFoundError):
        fallback = path.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False)
        return fallback


def run_celltypist(
    matrix: sparse.csr_matrix,
    obs: pd.DataFrame,
    var_names: pd.Index,
    sample_column: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import anndata as ad
    import celltypist

    labels = np.full(len(obs), "unavailable", dtype=object)
    confidence = np.zeros(len(obs), dtype=np.float32)
    sample_values = obs[sample_column].astype(str).to_numpy()
    for sample_id in sorted(pd.unique(sample_values)):
        positions = np.flatnonzero(sample_values == sample_id)
        sample = ad.AnnData(
            X=matrix[positions],
            obs=pd.DataFrame(index=obs.index[positions].astype(str)),
            var=pd.DataFrame(index=var_names.astype(str)),
        )
        use_majority = "cluster" in obs
        over_clustering = None
        if use_majority:
            over_clustering = pd.Series(
                obs.iloc[positions]["cluster"].astype(str).to_numpy(),
                index=sample.obs_names,
            )
        result = celltypist.annotate(
            sample,
            model=model,
            majority_voting=use_majority,
            over_clustering=over_clustering,
            mode="best match",
        )
        label_column = (
            "majority_voting"
            if "majority_voting" in result.predicted_labels
            else result.predicted_labels.columns[0]
        )
        labels[positions] = result.predicted_labels[label_column].astype(str).to_numpy()
        confidence[positions] = result.probability_matrix.max(axis=1).to_numpy(
            dtype=np.float32
        )
    return labels, confidence, {
        "enabled": True,
        "model": model,
        "package_version": importlib.metadata.version("celltypist"),
    }


def load_manifest(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["patient_id"])
    frame = pd.read_csv(path)
    if "patient_id" not in frame:
        raise ValueError("Patient manifest requires patient_id")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    return frame


def add_blast_prior(
    annotations: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = annotations.copy()
    output["blast_prior_probability"] = np.nan
    output["blast_prior_source"] = "not_available"
    audit: dict[str, object] = {}
    if manifest.empty or "scrna_blast_pct" not in manifest:
        return output, audit
    lookup = pd.to_numeric(
        manifest.set_index("patient_id")["scrna_blast_pct"], errors="coerce"
    ).to_dict()
    for patient_id, positions in output.groupby("sample_id").groups.items():
        blast_pct = lookup.get(str(patient_id))
        if blast_pct is None or not np.isfinite(float(blast_pct)):
            continue
        target = float(np.clip(float(blast_pct) / 100.0, 0.0, 1.0))
        calibrated = calibrate_probability_to_prevalence(
            output.loc[positions, "transcriptomic_suspicion_index"], target
        )
        output.loc[positions, "blast_prior_probability"] = calibrated
        output.loc[positions, "blast_prior_source"] = "reported_scRNA_blast_fraction"
        audit[str(patient_id)] = {
            "target_fraction": target,
            "calibrated_mean": float(np.mean(calibrated)),
        }
    return output, audit


def validate_hvg_schema(adata, payload: dict[str, object]) -> str:
    if "X_hvg" not in adata.obsm:
        raise ValueError("Input H5AD requires obsm['X_hvg'] from the frozen preprocessor")
    if "highly_variable" not in adata.var:
        raise ValueError("Input H5AD requires var['highly_variable'] for schema validation")
    input_genes = adata.var_names[adata.var["highly_variable"].astype(bool)].astype(str).tolist()
    expected = list(payload["hvg_genes"])
    if input_genes != expected:
        first_mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(input_genes, expected, strict=False))
                if left != right
            ),
            min(len(input_genes), len(expected)),
        )
        raise ValueError(
            "HVG schema mismatch; run the frozen STATE preprocessing contract. "
            f"input={len(input_genes)} expected={len(expected)} first_mismatch={first_mismatch}"
        )
    if adata.obsm["X_hvg"].shape[1] != len(expected):
        raise ValueError("X_hvg width does not match the frozen gene schema")
    digest = hashlib.sha256("\n".join(input_genes).encode("utf-8")).hexdigest()
    if digest != payload["hvg_schema_sha256"]:
        raise ValueError("HVG schema hash mismatch")
    return digest


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import anndata as ad

    payload = joblib.load(args.model)
    if payload.get("scope") != "observed_baseline_expression_state_atlas":
        raise ValueError("Unsupported virtual-cell model scope")
    adata = ad.read_h5ad(args.adata)
    if args.sample_column not in adata.obs:
        raise ValueError(f"Input obs is missing {args.sample_column}")
    adata.obs[args.sample_column] = adata.obs[args.sample_column].astype(str)
    hvg_hash = validate_hvg_schema(adata, payload)
    matrix_10k = normalize_log1p_to_10k(adata.X)

    lineage_scores, lineage_coverage = transform_gene_set_reference(
        matrix_10k, adata.var_names, payload["lineage_gene_set_reference"]
    )
    program_scores, program_coverage = transform_gene_set_reference(
        matrix_10k, adata.var_names, payload["program_gene_set_reference"]
    )
    lineage_scores.index = adata.obs_names
    program_scores.index = adata.obs_names
    marker_calls = classify_marker_lineage(lineage_scores)

    celltypist_model = args.celltypist_model or payload.get("celltypist_model")
    if args.disable_celltypist or not celltypist_model:
        ct_labels = None
        ct_confidence = None
        celltypist_metadata: dict[str, object] = {
            "enabled": False,
            "reason": "disabled_or_not_frozen",
        }
    else:
        ct_labels, ct_confidence, celltypist_metadata = run_celltypist(
            matrix_10k,
            adata.obs,
            adata.var_names,
            args.sample_column,
            str(celltypist_model),
        )
    lineage = combine_lineage_annotations(marker_calls, ct_labels, ct_confidence)
    suspicion = transcriptomic_suspicion(
        program_scores,
        lineage["consensus_lineage"],
        lineage["lineage_confidence"],
    )
    coordinates, state_ids, state_confidence, nearest_distance = transform_state_atlas(
        adata.obsm["X_hvg"],
        payload["svd"],
        payload["scaler"],
        payload["kmeans"],
        payload["stable_label_mapping"],
    )

    annotations = adata.obs.copy()
    annotations.index = adata.obs_names.astype(str)
    annotations.index.name = "cell_id"
    annotations = annotations.rename(columns={args.sample_column: "sample_id"})
    annotations["sample_id"] = annotations["sample_id"].astype(str)
    annotations["qc_pass"] = True
    if "n_genes" in annotations:
        annotations["qc_pass"] &= pd.to_numeric(
            annotations["n_genes"], errors="coerce"
        ).fillna(0).ge(200)
    if "fraction_mt" in annotations:
        annotations["qc_pass"] &= pd.to_numeric(
            annotations["fraction_mt"], errors="coerce"
        ).fillna(100).le(20)
    for column in lineage:
        annotations[column] = lineage[column].to_numpy()
    for column in program_scores:
        annotations[f"program::{column}"] = program_scores[column].to_numpy(dtype=float)
    for column in suspicion:
        annotations[column] = suspicion[column].to_numpy()
    annotations["virtual_state_id"] = state_ids
    annotations["state_assignment_confidence"] = state_confidence
    state_stability = payload.get("state_initialization_stability_by_id", {})
    annotations["state_initialization_stability"] = [
        float(state_stability.get(state_id, float("nan"))) for state_id in state_ids
    ]
    annotations["state_nearest_distance"] = nearest_distance
    for index in range(min(10, coordinates.shape[1])):
        annotations[f"state_component_{index + 1:02d}"] = coordinates[:, index]

    manifest = load_manifest(args.patient_manifest)
    annotations, prevalence_audit = add_blast_prior(annotations, manifest)
    program_columns = [f"program::{name}" for name in PROGRAM_MARKERS]
    summaries, program_summary, vectors = summarize_patient_states(
        annotations, program_columns
    )
    vectors = transform_reference_shift(vectors, payload["reference_shift_model"])
    shift_contributions = reference_shift_contributions(
        vectors, payload["reference_shift_model"]
    )
    summaries = summaries.merge(
        vectors[
            [
                "patient_id",
                "reference_shift_distance",
                "reference_shift_percentile",
                "reference_shift_flag",
            ]
        ],
        on="patient_id",
        how="left",
    )
    if not manifest.empty:
        summaries = summaries.merge(manifest, on="patient_id", how="left")

    files: list[Path] = []
    files.append(
        write_frame(
            annotations.reset_index(), args.out_dir / "cell_state_annotations.parquet"
        )
    )
    for frame, name in (
        (summaries, "patient_state_summary.csv"),
        (program_summary, "patient_program_summary.csv"),
        (vectors, "patient_state_vectors.csv"),
        (shift_contributions, "patient_reference_shift_drivers.csv"),
        (
            pd.concat(
                [
                    lineage_coverage.assign(module_type="lineage"),
                    program_coverage.assign(module_type="program"),
                ],
                ignore_index=True,
            ),
            "marker_coverage_audit.csv",
        ),
    ):
        path = args.out_dir / name
        frame.to_csv(path, index=False)
        files.append(path)
    if args.write_h5ad:
        output_adata = adata.copy()
        output_adata.obs = annotations.copy()
        output_adata.obsm["X_virtual_cell_v15"] = coordinates
        output_adata.uns["virtual_cell_v15_inference"] = {
            "scope": "observed_baseline_state_not_post_treatment",
            "model": str(args.model.resolve()),
            "hvg_schema_sha256": hvg_hash,
        }
        h5ad_path = args.out_dir / "virtual_cell_v15_inference.h5ad"
        output_adata.write_h5ad(h5ad_path, compression="gzip")
        files.append(h5ad_path)

    audit = {
        "scope": "observed_baseline_state_inference",
        "research_use_only": True,
        "input_h5ad": str(args.adata.resolve()),
        "model": str(args.model.resolve()),
        "hvg_schema_sha256": hvg_hash,
        "n_samples": int(annotations["sample_id"].nunique()),
        "n_cells": int(len(annotations)),
        "celltypist": celltypist_metadata,
        "blast_prevalence_prior": prevalence_audit,
        "label_firewall": {
            "drug_response_labels_used": False,
            "post_treatment_cells_used": False,
        },
    }
    audit_path = args.out_dir / "inference_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(audit_path)
    manifest_path = args.out_dir / "ARTIFACT_MANIFEST.csv"
    pd.DataFrame(
        [
            {
                "file": str(path.relative_to(args.out_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ]
    ).to_csv(manifest_path, index=False)
    print(f"samples={annotations['sample_id'].nunique()}")
    print(f"cells={len(annotations)}")
    print(f"output={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
