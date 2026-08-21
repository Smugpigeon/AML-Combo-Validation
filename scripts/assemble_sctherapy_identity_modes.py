#!/usr/bin/env python3
"""Assemble released-code and true-SCEVAN identity-call modes separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-input-dir", type=Path, required=True)
    parser.add_argument("--legacy-call-dir", type=Path, required=True)
    parser.add_argument("--copykat-dir", type=Path, required=True)
    parser.add_argument("--python-scevan-dir", type=Path, required=True)
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--patients", default="patient5,patient6,patient12")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_call(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "malignant": "malignant",
            "aneuploid": "malignant",
            "healthy": "healthy",
            "normal": "healthy",
            "diploid": "healthy",
        }
    )


def majority_vote(frame: pd.DataFrame) -> pd.Series:
    def vote(row: pd.Series) -> str | pd.NA:
        calls = row.dropna()
        if len(calls) < 2:
            return pd.NA
        counts = calls.value_counts()
        return str(counts.index[0]) if int(counts.iloc[0]) >= 2 else pd.NA

    return frame.apply(vote, axis=1).astype("string")


def load_copykat(args: argparse.Namespace, patient: str) -> pd.DataFrame:
    legacy_path = args.legacy_call_dir / f"{patient}_identity_calls.csv"
    if legacy_path.exists():
        return pd.read_csv(legacy_path, usecols=["cell_barcode", "copyKat_output"])
    component_path = args.copykat_dir / f"{patient}_copykat_calls.csv"
    return pd.read_csv(component_path, usecols=["cell_barcode", "copyKat_output"])


def main() -> int:
    args = parse_args()
    patients = [item.strip() for item in args.patients.split(",") if item.strip()]
    parity = json.loads(args.parity_summary.read_text(encoding="utf-8"))
    if not parity.get("python_reimplementation_parity_gate_pass", False):
        raise RuntimeError("true-SCEVAN mode is blocked because Python-to-R parity failed")

    released_dir = args.out_dir / "released_code_effective"
    corrected_dir = args.out_dir / "true_scevan_corrected"
    released_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, dict[str, object]] = {}

    for patient in patients:
        identity = pd.read_csv(
            args.identity_input_dir / f"{patient}_identity_inputs.csv"
        )
        copykat = load_copykat(args, patient)
        python_scevan = pd.read_csv(
            args.python_scevan_dir / f"{patient}_scevan_python_classification.csv",
            usecols=["cell_barcode", "SCEVAN_output"],
        ).rename(columns={"SCEVAN_output": "true_scevan_output"})
        base = identity.merge(
            copykat,
            on="cell_barcode",
            how="left",
            validate="one_to_one",
        ).merge(
            python_scevan,
            on="cell_barcode",
            how="left",
            validate="one_to_one",
        )
        base["sctype_malignant_healthy"] = normalize_call(
            base["sctype_malignant_healthy"]
        )
        base["copyKat_output"] = normalize_call(base["copyKat_output"])
        base["true_scevan_output"] = normalize_call(base["true_scevan_output"])

        released = base.copy()
        released["SCEVAN_output"] = released["known_normal_reference"].map(
            {True: "healthy", False: "malignant"}
        )
        released["ensemble_output"] = majority_vote(
            released[
                ["sctype_malignant_healthy", "copyKat_output", "SCEVAN_output"]
            ]
        )
        released["scevan_call_source"] = "released_code_normal_reference_rule"

        corrected = base.copy()
        corrected["SCEVAN_output"] = corrected["true_scevan_output"]
        corrected["ensemble_output"] = majority_vote(
            corrected[
                ["sctype_malignant_healthy", "copyKat_output", "SCEVAN_output"]
            ]
        )
        corrected["scevan_call_source"] = "pyscevan_full_segmentation_parity_gated"

        output_columns = [
            "sample_id",
            "cell_barcode",
            "seurat_cluster",
            "sctype_classification",
            "known_normal_reference",
            "sctype_malignant_healthy",
            "copyKat_output",
            "SCEVAN_output",
            "ensemble_output",
            "scevan_call_source",
        ]
        released[output_columns].to_csv(
            released_dir / f"{patient}_identity_calls.csv", index=False
        )
        corrected[output_columns].to_csv(
            corrected_dir / f"{patient}_identity_calls.csv", index=False
        )
        statuses[patient] = {
            "patient_id": patient,
            "status": "complete",
            "n_cells": len(base),
            "copykat_unresolved": int(base["copyKat_output"].isna().sum()),
            "true_scevan_unresolved": int(base["true_scevan_output"].isna().sum()),
        }

    released_status = {
        **statuses,
        "_mode": {
            "name": "released_code_effective",
            "warning": (
                "SCEVAN_output reproduces released run_SCEVAN(all_pred=FALSE) "
                "writeback semantics and is not a SCEVAN tumour classification."
            ),
        },
    }
    corrected_status = {
        **statuses,
        "_mode": {
            "name": "true_scevan_corrected",
            "warning": (
                "SCEVAN tumour calls come from a parity-gated Python "
                "reimplementation; all evidence remains RNA-derived."
            ),
        },
    }
    for directory, status in (
        (released_dir, released_status),
        (corrected_dir, corrected_status),
    ):
        (directory / "run_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
