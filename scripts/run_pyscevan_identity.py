#!/usr/bin/env python3
"""Run a pinned Python SCEVAN reimplementation for identity classification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--identity-input-dir", type=Path, required=True)
    parser.add_argument("--pyscevan-source", type=Path, required=True)
    parser.add_argument("--patients", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--beta-vega", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    patients = [item.strip() for item in args.patients.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pyscevan_source.resolve()))
    from pyscevan import pipeline_cna  # noqa: PLC0415
    from pyscevan.io.anndata_io import adata_to_count_mtx  # noqa: PLC0415

    source_revision = git_revision(args.pyscevan_source)
    full = ad.read_h5ad(args.h5ad, backed="r")
    statuses: dict[str, dict[str, object]] = {}
    for patient in patients:
        print(f"[pyscevan] {patient}", flush=True)
        normal_path = args.identity_input_dir / f"{patient}_normal_cells.txt"
        sample = None
        count_mtx = None
        result = None
        all_calls = None
        try:
            normals = [
                line.strip()
                for line in normal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            sample_mask = full.obs["sample_id"].astype(str).eq(patient).to_numpy()
            sample = full[sample_mask].to_memory()
            raw_barcodes = sample.obs["cell_barcode"].astype(str)
            if raw_barcodes.duplicated().any():
                raise ValueError(f"duplicate raw barcodes for {patient}")
            sample.obs_names = raw_barcodes.to_numpy()
            if len(set(normals).intersection(sample.obs_names)) < 20:
                raise ValueError(f"fewer than 20 normal references found for {patient}")

            count_mtx = adata_to_count_mtx(sample)
            result = pipeline_cna(
                count_mtx,
                norm_cell=normals,
                sample=patient,
                beta_vega=args.beta_vega,
                subclones=False,
                clonal_cn=False,
            )
            all_calls = pd.DataFrame(
                {
                    "sample_id": patient,
                    "cell_barcode": sample.obs_names.astype(str),
                    "scevan_python_class": "filtered",
                }
            ).set_index("cell_barcode", drop=False)
            shared = all_calls.index.intersection(result.class_df.index.astype(str))
            all_calls.loc[shared, "scevan_python_class"] = (
                result.class_df.loc[shared, "class"].astype(str)
            )
            all_calls["SCEVAN_output"] = all_calls["scevan_python_class"].map(
                {"tumor": "malignant", "normal": "healthy"}
            )
            all_calls.reset_index(drop=True).to_csv(
                args.out_dir / f"{patient}_scevan_python_classification.csv",
                index=False,
            )
            counts = all_calls["scevan_python_class"].value_counts().to_dict()
            statuses[patient] = {
                "patient_id": patient,
                "status": "complete",
                "n_cells": int(len(all_calls)),
                "n_analyzed": int(len(result.class_df)),
                "n_malignant": int(counts.get("tumor", 0)),
                "n_healthy": int(counts.get("normal", 0)),
                "n_filtered": int(counts.get("filtered", 0)),
                "n_normal_references": len(normals),
                "beta_vega": args.beta_vega,
                "segmentation_class": True,
                "pyscevan_revision": source_revision,
                "normal_reference_sha256": sha256_file(normal_path),
            }
        except Exception as error:  # preserve failures for the hard gate
            statuses[patient] = {
                "patient_id": patient,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "pyscevan_revision": source_revision,
            }
        (args.out_dir / "run_status.json").write_text(
            json.dumps(statuses, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        del sample, count_mtx, result, all_calls
        gc.collect()
    full.file.close()
    return 2 if any(item["status"] != "complete" for item in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
