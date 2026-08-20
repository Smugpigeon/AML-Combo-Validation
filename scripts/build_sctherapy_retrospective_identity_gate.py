#!/usr/bin/env python3
"""Merge official identity calls with virtual states and audit reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.cell_identity import (  # noqa: E402
    build_retrospective_identity_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-annotations", type=Path, required=True)
    parser.add_argument("--identity-call-dir", type=Path, required=True)
    parser.add_argument("--patient-summary", type=Path, required=True)
    parser.add_argument("--patients", default="patient5,patient6,patient12")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    patients = [item.strip() for item in args.patients.split(",") if item.strip()]
    annotations = read_table(args.cell_annotations)
    annotations = annotations.loc[annotations["sample_id"].astype(str).isin(patients)].copy()
    if "cell_id" in annotations:
        annotations["cell_barcode"] = annotations["cell_id"].astype(str).str.split(":").str[-1]
    elif annotations.index.name == "cell_id":
        annotations["cell_barcode"] = annotations.index.astype(str).str.split(":").str[-1]
    else:
        raise KeyError("cell annotations require cell_id as a column or index")

    call_paths = [
        args.identity_call_dir / f"{patient}_identity_calls.csv" for patient in patients
    ]
    missing_paths = [str(path) for path in call_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"identity call files missing: {missing_paths}")
    calls = pd.concat([pd.read_csv(path) for path in call_paths], ignore_index=True)
    if calls.duplicated(["sample_id", "cell_barcode"]).any():
        raise ValueError("official identity calls contain duplicate barcodes")
    merged = annotations.merge(
        calls,
        on=["sample_id", "cell_barcode"],
        how="left",
        validate="one_to_one",
    )
    unmatched = int(merged["ensemble_output"].isna().sum())
    if unmatched == len(merged):
        raise RuntimeError("no official identity calls matched virtual-cell barcodes")

    patient_summary = pd.read_csv(args.patient_summary)
    cells, patient_gate, state_gate, summary = build_retrospective_identity_gate(
        merged,
        patient_summary,
    )
    summary["barcode_unmatched_cells"] = unmatched
    summary["input_sha256"] = {
        "cell_annotations": sha256_file(args.cell_annotations),
        "patient_summary": sha256_file(args.patient_summary),
        "official_identity_calls": {
            path.name: sha256_file(path) for path in call_paths
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(args.out_dir / "retrospective_cell_identity.parquet", index=False)
    patient_gate.to_csv(args.out_dir / "retrospective_patient_identity_gate.csv", index=False)
    state_gate.to_csv(args.out_dir / "retrospective_state_identity_gate.csv", index=False)
    (args.out_dir / "retrospective_identity_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
