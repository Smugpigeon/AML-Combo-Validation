#!/usr/bin/env python3
"""Build the AML Virtual Cell v1.5 baseline single-cell state atlas.

The workflow combines a cross-patient expression-state reference, broad immune
lineage annotation, curated resistance-program scores, explicit uncertainty,
and optional blast-prevalence priors. It does not use drug-response validation
labels and does not generate a causal post-treatment transcriptome.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from combo_val.virtual_cell.single_cell_v15 import (  # noqa: E402
    CELL_TYPE_MARKERS,
    PROGRAM_MARKERS,
    StateAtlasConfig,
    calibrate_probability_to_prevalence,
    classify_marker_lineage,
    combine_lineage_annotations,
    fit_gene_set_reference,
    fit_reference_shift_model,
    fit_state_atlas,
    normalize_log1p_to_10k,
    reference_shift_contributions,
    summarize_patient_states,
    summarize_state_prototypes,
    transcriptomic_suspicion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True, help="STATE-preprocessed H5AD")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-samples",
        default="patient5,patient6,patient12",
        help="Comma-separated samples for the focused R/R AML deliverable",
    )
    parser.add_argument("--patient-manifest", type=Path)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--celltypist-model", default="Immune_All_Low.pkl")
    parser.add_argument("--disable-celltypist", action="store_true")
    parser.add_argument("--n-states", type=int, default=12)
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--no-selected-h5ad", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frame(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=index)
            return path
        except (ImportError, ModuleNotFoundError):
            path = path.with_suffix(".csv.gz")
    frame.to_csv(path, index=index)
    return path


def load_manifest(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["patient_id"])
    frame = pd.read_csv(path)
    if "patient_id" not in frame:
        raise ValueError("Patient manifest requires a patient_id column")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    if frame["patient_id"].duplicated().any():
        raise ValueError("Patient manifest has duplicate patient_id values")
    return frame


def run_celltypist(
    matrix_10k: sparse.csr_matrix,
    obs: pd.DataFrame,
    var_names: pd.Index,
    sample_column: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Annotate one patient at a time to cap memory and retain row alignment."""

    import anndata as ad
    import celltypist

    labels = np.full(len(obs), "unavailable", dtype=object)
    confidence = np.zeros(len(obs), dtype=np.float32)
    samples: dict[str, dict[str, object]] = {}
    sample_values = obs[sample_column].astype(str).to_numpy()
    for sample_id in sorted(pd.unique(sample_values)):
        positions = np.flatnonzero(sample_values == sample_id)
        sample = ad.AnnData(
            X=matrix_10k[positions],
            obs=pd.DataFrame(index=obs.index[positions].astype(str)),
            var=pd.DataFrame(index=var_names.astype(str)),
        )
        use_majority_voting = "cluster" in obs.columns
        over_clustering = None
        if use_majority_voting:
            over_clustering = pd.Series(
                obs.iloc[positions]["cluster"].astype(str).to_numpy(),
                index=sample.obs_names,
            )
        prediction = celltypist.annotate(
            sample,
            model=model,
            majority_voting=use_majority_voting,
            over_clustering=over_clustering,
            mode="best match",
        )
        label_column = (
            "majority_voting"
            if "majority_voting" in prediction.predicted_labels
            else prediction.predicted_labels.columns[0]
        )
        predicted = prediction.predicted_labels[label_column].astype(str).to_numpy()
        maximum_probability = (
            prediction.probability_matrix.max(axis=1).to_numpy(dtype=np.float32)
        )
        labels[positions] = predicted
        confidence[positions] = maximum_probability
        samples[sample_id] = {
            "n_cells": int(len(positions)),
            "majority_voting": use_majority_voting,
            "median_max_probability": float(np.median(maximum_probability)),
            "fraction_probability_ge_0_5": float(np.mean(maximum_probability >= 0.5)),
        }
    metadata: dict[str, object] = {
        "enabled": True,
        "model": model,
        "package_version": importlib.metadata.version("celltypist"),
        "per_sample": samples,
    }
    return labels, confidence, metadata


def descriptive_state_markers(
    matrix: sparse.csr_matrix,
    gene_names: pd.Index,
    state_ids: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return descriptive mean-expression contrasts, without pseudo-replicate p-values."""

    overall = np.asarray(matrix.mean(axis=0)).ravel()
    n_cells = len(state_ids)
    genes = np.asarray(gene_names.astype(str), dtype=object)
    excluded = np.asarray(
        [gene.upper().startswith(("MT-", "RPL", "RPS")) for gene in genes], dtype=bool
    )
    rows: list[dict[str, object]] = []
    for state_id in sorted(pd.unique(state_ids)):
        mask = state_ids == state_id
        state_mean = np.asarray(matrix[mask].mean(axis=0)).ravel()
        if int(mask.sum()) == n_cells:
            rest_mean = overall
        else:
            rest_mean = (overall * n_cells - state_mean * int(mask.sum())) / max(
                n_cells - int(mask.sum()), 1
            )
        contrast = state_mean - rest_mean
        contrast[excluded] = -np.inf
        order = np.argpartition(contrast, -top_n)[-top_n:]
        order = order[np.argsort(contrast[order])[::-1]]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "virtual_state_id": state_id,
                    "rank": rank,
                    "gene": genes[index],
                    "mean_log1p_10k_in_state": float(state_mean[index]),
                    "mean_log1p_10k_outside_state": float(rest_mean[index]),
                    "descriptive_mean_difference": float(contrast[index]),
                    "inference_scope": "descriptive_marker_no_cell_level_p_value",
                }
            )
    return pd.DataFrame(rows)


def add_blast_prevalence_priors(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = annotations.copy()
    output["blast_prior_probability"] = np.nan
    output["blast_prior_source"] = "not_available"
    audit: dict[str, object] = {}
    if manifest.empty or "scrna_blast_pct" not in manifest:
        return output, audit
    blast_lookup = pd.to_numeric(
        manifest.set_index("patient_id")["scrna_blast_pct"], errors="coerce"
    ).to_dict()
    for patient_id, positions in output.groupby("sample_id").groups.items():
        blast_pct = blast_lookup.get(str(patient_id))
        if blast_pct is None or not math.isfinite(float(blast_pct)):
            continue
        target = float(np.clip(float(blast_pct) / 100.0, 0.0, 1.0))
        calibrated = calibrate_probability_to_prevalence(
            output.loc[positions, "transcriptomic_suspicion_index"].to_numpy(dtype=float),
            target,
        )
        output.loc[positions, "blast_prior_probability"] = calibrated
        output.loc[positions, "blast_prior_source"] = "reported_scRNA_blast_fraction"
        audit[str(patient_id)] = {
            "reported_scRNA_blast_fraction": target,
            "calibrated_mean_probability": float(np.mean(calibrated)),
            "note": "prevalence-anchored ranking; not independent malignant-cell validation",
        }
    return output, audit


def state_perturbation_inputs(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    selected_samples: set[str],
    program_columns: list[str],
    coordinate_columns: list[str],
) -> pd.DataFrame:
    selected = annotations[annotations["sample_id"].isin(selected_samples)].copy()
    rows: list[dict[str, object]] = []
    manifest_lookup = (
        manifest.set_index("patient_id").to_dict(orient="index") if not manifest.empty else {}
    )
    for (patient_id, state_id), group in selected.groupby(
        ["sample_id", "virtual_state_id"], sort=True
    ):
        patient_size = int((selected["sample_id"] == patient_id).sum())
        lineage = group["consensus_lineage"].value_counts(normalize=True)
        row: dict[str, object] = {
            "patient_id": patient_id,
            "virtual_state_id": state_id,
            "n_cells": int(len(group)),
            "patient_state_fraction": float(len(group) / max(patient_size, 1)),
            "dominant_lineage": str(lineage.index[0]),
            "dominant_lineage_fraction": float(lineage.iloc[0]),
            "mean_state_assignment_confidence": float(
                group["state_assignment_confidence"].mean()
            ),
            "mean_state_initialization_stability": float(
                group["state_initialization_stability"].mean()
            ),
            "mean_transcriptomic_suspicion_index": float(
                group["transcriptomic_suspicion_index"].mean()
            ),
            "perturbation_label": "",
            "drug_id": "",
            "dose_value": "",
            "dose_unit": "",
            "exposure_time_hours": "",
            "output_contract": (
                "delta_expression;viability;malignant_fraction;normal_toxicity;uncertainty"
            ),
            "use_scope": "research_only_requires_ex_vivo_validation",
        }
        for column in program_columns + coordinate_columns:
            row[f"baseline_mean::{column}"] = float(group[column].mean())
        metadata = manifest_lookup.get(str(patient_id), {})
        for field in (
            "disease_stage",
            "fab_type",
            "eln2022_risk",
            "potential_driver_mutations",
            "chromosomal_abnormalities",
        ):
            row[field] = metadata.get(field, "")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["patient_id", "patient_state_fraction"], ascending=[True, False])


def write_patient_cards(
    path: Path,
    selected_samples: set[str],
    summaries: pd.DataFrame,
    programs: pd.DataFrame,
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    manifest_lookup = (
        manifest.set_index("patient_id").to_dict(orient="index") if not manifest.empty else {}
    )
    with path.open("w", encoding="utf-8") as handle:
        for patient_id in sorted(selected_samples):
            summary_rows = summaries[summaries["patient_id"] == patient_id]
            if summary_rows.empty:
                continue
            patient_programs = programs[programs["patient_id"] == patient_id].sort_values(
                "mean", ascending=False
            )
            patient_cells = annotations[annotations["sample_id"] == patient_id]
            state_fraction = patient_cells["virtual_state_id"].value_counts(normalize=True)
            card = {
                "patient_id": patient_id,
                "clinical_context": manifest_lookup.get(patient_id, {}),
                "baseline_state_summary": summary_rows.iloc[0].to_dict(),
                "dominant_programs": patient_programs.head(5).to_dict(orient="records"),
                "dominant_virtual_states": [
                    {"virtual_state_id": state, "fraction": float(fraction)}
                    for state, fraction in state_fraction.head(5).items()
                ],
                "interpretation_boundary": (
                    "Observed baseline expression-state model. Virtual states are not "
                    "genetically proven clones; no post-treatment state is asserted."
                ),
            }
            handle.write(json.dumps(card, ensure_ascii=False, allow_nan=True) + "\n")


def plot_state_map(
    annotations: pd.DataFrame,
    selected_samples: set[str],
    path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if len(annotations) > 25000:
        positions = rng.choice(len(annotations), size=25000, replace=False)
        frame = annotations.iloc[np.sort(positions)].copy()
    else:
        frame = annotations.copy()
    states = sorted(frame["virtual_state_id"].unique())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(states)))
    color_lookup = dict(zip(states, colors, strict=True))
    fig, ax = plt.subplots(figsize=(11, 8))
    for state in states:
        group = frame[frame["virtual_state_id"] == state]
        ax.scatter(
            group["state_component_01"],
            group["state_component_02"],
            s=3,
            alpha=0.35,
            color=color_lookup[state],
            label=state,
            linewidths=0,
        )
    patient_colors = ["#111111", "#b2182b", "#2166ac"]
    patient_markers = ["o", "^", "s"]
    for patient_index, patient_id in enumerate(sorted(selected_samples)):
        focused = annotations[annotations["sample_id"] == patient_id]
        if len(focused) > 150:
            positions = rng.choice(len(focused), size=150, replace=False)
            focused = focused.iloc[np.sort(positions)]
        ax.scatter(
            focused["state_component_01"],
            focused["state_component_02"],
            s=15,
            marker=patient_markers[patient_index % len(patient_markers)],
            facecolors="none",
            edgecolors=patient_colors[patient_index % len(patient_colors)],
            linewidths=0.55,
            alpha=0.75,
            label=patient_id,
        )
    ax.set_title("AML Virtual Cell v1.5: cross-patient baseline state atlas")
    ax.set_xlabel("State component 1")
    ax.set_ylabel("State component 2")
    ax.legend(ncol=3, fontsize=8, frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selected_programs(programs: pd.DataFrame, selected_samples: set[str], path: Path) -> None:
    frame = programs[programs["patient_id"].isin(selected_samples)].copy()
    matrix = frame.pivot(index="patient_id", columns="program", values="mean")
    matrix.columns = [column.removeprefix("program::") for column in matrix.columns]
    matrix = matrix.reindex(sorted(matrix.index))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Selected R/R AML patients: baseline program activity")
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="robust relative activity score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selected_composition(
    annotations: pd.DataFrame, selected_samples: set[str], path: Path
) -> None:
    selected = annotations[annotations["sample_id"].isin(selected_samples)]
    composition = pd.crosstab(
        selected["sample_id"], selected["consensus_lineage"], normalize="index"
    ).reindex(sorted(selected_samples)).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(composition), dtype=float)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(composition.columns), 1)))
    for color, column in zip(colors, composition.columns, strict=False):
        values = composition[column].to_numpy(dtype=float)
        ax.bar(composition.index, values, bottom=bottom, label=column, color=color)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cell fraction")
    ax.set_title("Selected R/R AML patients: consensus broad-lineage composition")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_model_card(path: Path, metrics: dict[str, object]) -> None:
    data = metrics["dataset"]
    annotation = metrics["annotation"]
    text = f"""# AML Virtual Cell v1.5 Model Card

## What changed

This release upgrades the earlier one-vector patient pilot into a hierarchical
baseline state representation:

1. `{data['n_cells']:,}` observed cells from `{data['n_samples']}` public AML samples.
2. Broad lineage calls from curated markers and
   `{annotation['celltypist']['model'] if annotation['celltypist']['enabled'] else 'marker-only fallback'}`.
3. `{data['n_programs']}` resistance/state programs with marker-coverage auditing.
4. `{data['n_virtual_states']}` cross-patient expression-state prototypes.
5. Cell-, state-, and patient-level uncertainty and within-cohort shift metrics.
6. Optional clinical blast-prevalence calibration kept separate from the
   transcriptomic suspicion index.

## What the model means

`virtual_state_id` denotes a reproducible expression-state prototype shared
across patients. It is **not** a genetically proven subclone. Broad lineage
annotations describe transcriptional identity; they do not independently prove
that a myeloid cell is malignant. The `transcriptomic_suspicion_index` is a
research feature assembled from stem/progenitor, proliferation, apoptosis,
DNA-repair, and differentiation programs. It is not a diagnostic probability.

When a reported scRNA blast fraction is available, `blast_prior_probability`
preserves the expression-based ranking while shifting its mean to the reported
prevalence. This is a prior-anchored representation, not an independent
validation of blast identity.

## Perturbation boundary

The package writes state-level perturbation inputs for future STATE/CPA-style
models, but it does not claim a causal post-treatment transcriptome. A valid
transition model still requires matched drug identity, dose, exposure time, and
post-treatment cells. Drug-response and flow-validation labels were not read by
this build and remain available for a later frozen-prediction evaluation.

## Intended use

Research prioritization, patient-state characterization, assay design, and
retrospective validation. Not a prescription engine and not a replacement for
physician or multidisciplinary review.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    selected_samples = {
        value.strip() for value in args.selected_samples.split(",") if value.strip()
    }
    if not selected_samples:
        raise ValueError("At least one selected sample is required")

    import anndata as ad

    print(f"[load] {args.adata}", flush=True)
    adata = ad.read_h5ad(args.adata)
    if args.sample_column not in adata.obs:
        raise ValueError(f"H5AD obs is missing {args.sample_column}")
    if "X_hvg" not in adata.obsm:
        raise ValueError("H5AD is missing STATE-compatible obsm['X_hvg']")
    adata.obs[args.sample_column] = adata.obs[args.sample_column].astype(str)
    available_samples = set(adata.obs[args.sample_column])
    missing_selected = selected_samples - available_samples
    if missing_selected:
        raise ValueError(f"Selected samples are absent from H5AD: {sorted(missing_selected)}")
    manifest = load_manifest(args.patient_manifest)

    print("[normalize] log1p expression -> 10,000-count target", flush=True)
    matrix_10k = normalize_log1p_to_10k(adata.X)
    print("[score] lineage and resistance programs", flush=True)
    lineage_scores, lineage_coverage, lineage_reference = fit_gene_set_reference(
        matrix_10k, adata.var_names, CELL_TYPE_MARKERS
    )
    program_scores, program_coverage, program_reference = fit_gene_set_reference(
        matrix_10k, adata.var_names, PROGRAM_MARKERS
    )
    lineage_scores.index = adata.obs_names
    program_scores.index = adata.obs_names
    marker_calls = classify_marker_lineage(lineage_scores)

    celltypist_metadata: dict[str, object]
    if args.disable_celltypist:
        ct_labels = None
        ct_confidence = None
        celltypist_metadata = {"enabled": False, "reason": "disabled_by_cli"}
    else:
        print(f"[annotate] CellTypist model={args.celltypist_model}", flush=True)
        try:
            ct_labels, ct_confidence, celltypist_metadata = run_celltypist(
                matrix_10k,
                adata.obs,
                adata.var_names,
                args.sample_column,
                args.celltypist_model,
            )
        except (ImportError, ModuleNotFoundError) as error:
            print(f"[annotate] marker-only fallback: {error}", flush=True)
            ct_labels = None
            ct_confidence = None
            celltypist_metadata = {
                "enabled": False,
                "reason": f"dependency_unavailable:{type(error).__name__}",
            }
    lineage = combine_lineage_annotations(marker_calls, ct_labels, ct_confidence)
    suspicion = transcriptomic_suspicion(
        program_scores,
        lineage["consensus_lineage"],
        lineage["lineage_confidence"],
    )

    config = StateAtlasConfig(
        n_components=args.n_components,
        n_states=args.n_states,
        random_state=args.seed,
    )
    print(
        f"[state-atlas] components={config.n_components} states={config.n_states}",
        flush=True,
    )
    state = fit_state_atlas(adata.obsm["X_hvg"], config)

    annotations = adata.obs.copy()
    annotations.index = adata.obs_names.astype(str)
    annotations.index.name = "cell_id"
    annotations = annotations.rename(columns={args.sample_column: "sample_id"})
    if "sample_id" not in annotations:
        annotations["sample_id"] = adata.obs[args.sample_column].astype(str).to_numpy()
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
    for column in lineage.columns:
        annotations[column] = lineage[column].to_numpy()
    for column in program_scores.columns:
        annotations[f"program::{column}"] = program_scores[column].to_numpy(dtype=float)
    for column in suspicion.columns:
        annotations[column] = suspicion[column].to_numpy()
    annotations["virtual_state_id"] = state.state_ids
    annotations["state_assignment_confidence"] = state.assignment_confidence
    annotations["state_initialization_stability"] = state.initialization_stability
    annotations["state_nearest_distance"] = state.nearest_distance
    coordinate_columns: list[str] = []
    for index in range(min(10, state.coordinates.shape[1])):
        column = f"state_component_{index + 1:02d}"
        annotations[column] = state.coordinates[:, index]
        coordinate_columns.append(column)

    annotations, prevalence_audit = add_blast_prevalence_priors(annotations, manifest)
    program_columns = [f"program::{name}" for name in PROGRAM_MARKERS]
    summaries, program_summary, patient_vectors = summarize_patient_states(
        annotations, program_columns
    )
    patient_vectors, reference_shift_model = fit_reference_shift_model(patient_vectors)
    shift_contributions = reference_shift_contributions(
        patient_vectors, reference_shift_model
    )
    summaries = summaries.merge(
        patient_vectors[
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

    prototypes = summarize_state_prototypes(annotations, program_columns)
    state_markers = descriptive_state_markers(
        matrix_10k, adata.var_names, state.state_ids
    )
    perturbation_inputs = state_perturbation_inputs(
        annotations,
        manifest,
        selected_samples,
        program_columns,
        coordinate_columns,
    )

    output_files: list[Path] = []
    output_files.append(
        write_frame(
            annotations.reset_index(), args.out_dir / "cell_state_annotations.parquet"
        )
    )
    selected_annotations = annotations[annotations["sample_id"].isin(selected_samples)]
    output_files.append(
        write_frame(
            selected_annotations.reset_index(),
            args.out_dir / "selected_rr3_cell_state_annotations.parquet",
        )
    )
    for frame, filename in (
        (summaries, "patient_state_summary.csv"),
        (program_summary, "patient_program_summary.csv"),
        (patient_vectors, "patient_state_vectors.csv"),
        (shift_contributions, "patient_reference_shift_drivers.csv"),
        (prototypes, "virtual_state_prototypes.csv"),
        (state_markers, "virtual_state_descriptive_markers.csv"),
        (perturbation_inputs, "selected_rr3_state_perturbation_inputs.csv"),
    ):
        path = args.out_dir / filename
        frame.to_csv(path, index=False)
        output_files.append(path)
    coverage = pd.concat(
        [
            lineage_coverage.assign(module_type="lineage"),
            program_coverage.assign(module_type="program"),
        ],
        ignore_index=True,
    )
    coverage_path = args.out_dir / "marker_coverage_audit.csv"
    coverage.to_csv(coverage_path, index=False)
    output_files.append(coverage_path)

    cards_path = args.out_dir / "selected_rr3_virtual_cell_cards.jsonl"
    write_patient_cards(
        cards_path,
        selected_samples,
        summaries,
        program_summary,
        annotations,
        manifest,
    )
    output_files.append(cards_path)

    hvg_genes = (
        adata.var_names[adata.var["highly_variable"].astype(bool)].astype(str).tolist()
        if "highly_variable" in adata.var
        else []
    )
    if len(hvg_genes) != adata.obsm["X_hvg"].shape[1]:
        raise ValueError(
            "The highly_variable gene list does not align with obsm['X_hvg'] columns"
        )
    hvg_schema_sha256 = hashlib.sha256("\n".join(hvg_genes).encode("utf-8")).hexdigest()
    model_path = args.out_dir / "virtual_state_atlas_model.joblib"
    joblib.dump(
        {
            "scope": "observed_baseline_expression_state_atlas",
            "config": asdict(config),
            "svd": state.svd,
            "scaler": state.scaler,
            "kmeans": state.kmeans,
            "stable_label_mapping": state.stable_label_mapping,
            "state_initialization_stability_by_id": annotations.groupby(
                "virtual_state_id"
            )["state_initialization_stability"].mean().to_dict(),
            "hvg_genes": hvg_genes,
            "hvg_schema_sha256": hvg_schema_sha256,
            "cell_type_markers": CELL_TYPE_MARKERS,
            "program_markers": PROGRAM_MARKERS,
            "lineage_gene_set_reference": lineage_reference,
            "program_gene_set_reference": program_reference,
            "reference_patient_vectors": patient_vectors,
            "reference_shift_model": reference_shift_model,
            "celltypist_model": args.celltypist_model if not args.disable_celltypist else None,
        },
        model_path,
    )
    output_files.append(model_path)

    if not args.no_selected_h5ad:
        selected_mask = annotations["sample_id"].isin(selected_samples).to_numpy()
        selected_adata = adata[selected_mask].copy()
        selected_adata.obs = annotations.loc[selected_adata.obs_names].copy()
        selected_adata.obsm["X_virtual_cell_v15"] = state.coordinates[selected_mask]
        selected_adata.uns["virtual_cell_v15"] = {
            "scope": "observed_baseline_state_not_post_treatment",
            "config": asdict(config),
            "selected_samples": sorted(selected_samples),
        }
        selected_h5ad = args.out_dir / "selected_rr3_virtual_cell_v15.h5ad"
        selected_adata.write_h5ad(selected_h5ad, compression="gzip")
        output_files.append(selected_h5ad)

    state_map = figures_dir / "cross_patient_state_atlas.png"
    program_heatmap = figures_dir / "selected_rr3_program_heatmap.png"
    composition_plot = figures_dir / "selected_rr3_lineage_composition.png"
    plot_state_map(annotations, selected_samples, state_map, args.seed)
    plot_selected_programs(program_summary, selected_samples, program_heatmap)
    plot_selected_composition(annotations, selected_samples, composition_plot)
    output_files.extend([state_map, program_heatmap, composition_plot])

    metrics: dict[str, Any] = {
        "run_name": "AML Virtual Cell v1.5 baseline state atlas",
        "research_use_only": True,
        "dataset": {
            "source_h5ad": str(args.adata.resolve()),
            "n_samples": int(annotations["sample_id"].nunique()),
            "n_cells": int(len(annotations)),
            "n_genes": int(adata.n_vars),
            "n_hvg": int(adata.obsm["X_hvg"].shape[1]),
            "n_programs": len(PROGRAM_MARKERS),
            "n_virtual_states": int(annotations["virtual_state_id"].nunique()),
            "selected_samples": sorted(selected_samples),
            "selected_cells": int(len(selected_annotations)),
        },
        "annotation": {
            "lineage_method": "curated_marker_plus_optional_CellTypist_consensus",
            "celltypist": celltypist_metadata,
            "lineage_concordance_counts": annotations[
                "lineage_evidence_status"
            ].value_counts().to_dict(),
            "mean_lineage_confidence": float(annotations["lineage_confidence"].mean()),
            "marker_coverage_minimum": float(coverage["coverage_fraction"].min()),
            "marker_coverage_median": float(coverage["coverage_fraction"].median()),
        },
        "state_atlas": {
            "config": asdict(config),
            "explained_variance_ratio_sum": float(
                np.sum(state.svd.explained_variance_ratio_)
            ),
            "mean_assignment_confidence": float(state.assignment_confidence.mean()),
            "median_assignment_confidence": float(np.median(state.assignment_confidence)),
            "mean_initialization_stability": float(state.initialization_stability.mean()),
            "median_initialization_stability": float(
                np.median(state.initialization_stability)
            ),
            "initialization_adjusted_mutual_information": list(
                state.initialization_ami
            ),
            "mean_initialization_adjusted_mutual_information": float(
                np.mean(state.initialization_ami)
            )
            if state.initialization_ami
            else float("nan"),
        },
        "blast_prevalence_prior_audit": prevalence_audit,
        "label_firewall": {
            "combo_dose_matrix_used": False,
            "flow_validation_used": False,
            "single_agent_validation_labels_used": False,
        },
        "limitations": [
            "baseline scRNA only; no matched post-treatment cells",
            "expression states are not genetically proven clones",
            "broad lineage annotation is reference/marker based and can misclassify AML blasts",
            "program scores are transcriptomic proxies, not pathway flux or phosphoprotein activity",
            "within-cohort shift percentile is descriptive because the reference has only 12 patients",
            "no clinical treatment recommendation is produced",
        ],
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    output_files.append(metrics_path)
    model_card_path = args.out_dir / "MODEL_CARD.md"
    write_model_card(model_card_path, metrics)
    output_files.append(model_card_path)

    manifest_rows = []
    for path in sorted(output_files):
        manifest_rows.append(
            {
                "file": str(path.relative_to(args.out_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    artifact_manifest = args.out_dir / "ARTIFACT_MANIFEST.csv"
    pd.DataFrame(manifest_rows).to_csv(artifact_manifest, index=False)

    print(f"[done] samples={annotations['sample_id'].nunique()} cells={len(annotations)}")
    print(f"[done] selected_cells={len(selected_annotations)} states={prototypes.shape[0]}")
    print(f"[done] output={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
