#!/usr/bin/env python3
"""Build patient-level BeatAML features from the selected scTherapy samples.

The script aggregates raw single-cell counts into one pseudobulk profile per
patient, adds only metadata explicitly present in the public patient manifest,
and projects the result through the frozen BeatAML preprocessor. It refuses to
run unless the public retrospective RNA-ensemble identity gate has passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.clinical.feature_builder import (  # noqa: E402
    build_patient_features_from_raw,
)
from combo_val.clinical.kit_schema import KitInput, MutationCall  # noqa: E402
from combo_val.virtual_cell.cell_identity import (  # noqa: E402
    require_retrospective_identity_gate_summary,
)


def _split_tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def _mutation_calls(value: object) -> list[MutationCall]:
    calls: list[MutationCall] = []
    for token in _split_tokens(value):
        gene = token.split()[0].split("(")[0].strip().upper()
        if gene:
            # The public manifest does not distinguish FLT3-ITD from FLT3-TKD.
            # Do not infer a subtype that was not reported.
            calls.append(MutationCall(gene=gene))
    return calls


def _kit_from_manifest(row: pd.Series) -> KitInput:
    stage = str(row.get("disease_stage", "")).strip().lower()
    return KitInput(
        patient_id=str(row["patient_id"]),
        mutations=_mutation_calls(row.get("potential_driver_mutations")),
        karyotype_text=(
            str(row["chromosomal_abnormalities"])
            if pd.notna(row.get("chromosomal_abnormalities"))
            else None
        ),
        blast_pct_bm=(
            float(row["clinical_blast_pct"])
            if pd.notna(row.get("clinical_blast_pct"))
            else None
        ),
        is_relapse=stage == "relapse",
        prior_chemo=stage in {"relapse", "refractory"},
        is_initial_diagnosis=stage not in {"relapse", "refractory"},
        intent_comment="public retrospective validation only",
    )


def _pseudobulk_counts(adata: ad.AnnData, patient_id: str) -> pd.Series:
    mask = adata.obs["sample_id"].astype(str).to_numpy() == patient_id
    if not mask.any():
        raise ValueError(f"patient absent from H5AD: {patient_id}")
    summed = np.asarray(adata.X[mask].sum(axis=0)).ravel().astype(float)
    counts = pd.Series(summed, index=adata.var_names.astype(str), dtype=float)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0, sort=False).sum()
    if (counts < 0).any():
        raise ValueError(f"negative values found in raw count matrix for {patient_id}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--patient-manifest", type=Path, required=True)
    parser.add_argument("--identity-gate-summary", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--out-features", type=Path, required=True)
    parser.add_argument("--out-diagnostics", type=Path, required=True)
    parser.add_argument(
        "--patients", default="patient5,patient6,patient12",
        help="Comma-separated patient IDs.",
    )
    args = parser.parse_args()

    patient_ids = [value.strip() for value in args.patients.split(",") if value.strip()]
    identity_summary = json.loads(args.identity_gate_summary.read_text(encoding="utf-8"))
    require_retrospective_identity_gate_summary(
        identity_summary,
        patient_ids=patient_ids,
    )

    manifest = pd.read_csv(args.patient_manifest)
    if manifest["patient_id"].astype(str).duplicated().any():
        raise ValueError("patient manifest contains duplicate patient_id values")
    manifest = manifest.set_index(manifest["patient_id"].astype(str), drop=False)
    missing = sorted(set(patient_ids) - set(manifest.index))
    if missing:
        raise ValueError(f"patients absent from manifest: {missing}")

    bundle = joblib.load(args.preprocessor)
    feature_columns = list(bundle["feature_columns"])
    adata = ad.read_h5ad(args.h5ad)
    if "sample_id" not in adata.obs:
        raise KeyError("H5AD obs is missing sample_id")

    feature_rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "research_use_only": True,
        "aggregation": "sum_raw_counts_per_patient_pseudobulk",
        "identity_gate_summary": str(args.identity_gate_summary),
        "preprocessor": str(args.preprocessor),
        "patients": {},
    }
    for patient_id in patient_ids:
        counts = _pseudobulk_counts(adata, patient_id)
        kit = _kit_from_manifest(manifest.loc[patient_id])
        features, diag = build_patient_features_from_raw(
            counts,
            kit,
            preprocessor_path=args.preprocessor,
            apply_quantile_normalization=True,
            strict_ood=False,
        )
        if len(features) != len(feature_columns):
            raise RuntimeError(
                f"feature width mismatch for {patient_id}: "
                f"{len(features)} != {len(feature_columns)}"
            )
        feature_rows.append(
            {"patient_id": patient_id, **dict(zip(feature_columns, features, strict=True))}
        )
        diagnostics["patients"][patient_id] = {
            **diag,
            "n_cells": int((adata.obs["sample_id"].astype(str) == patient_id).sum()),
            "pseudobulk_library_size": float(counts.sum()),
            "reported_disease_stage": str(manifest.loc[patient_id, "disease_stage"]),
            "reported_driver_mutations": _split_tokens(
                manifest.loc[patient_id, "potential_driver_mutations"]
            ),
            "mutation_subtype_inference": "none",
        }

    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    args.out_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).to_csv(args.out_features, index=False)
    args.out_diagnostics.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
