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

    call_paths = {
        patient: args.identity_call_dir / f"{patient}_identity_calls.csv"
        for patient in patients
    }
    status_path = args.identity_call_dir / "run_status.json"
    run_status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    call_frames: list[pd.DataFrame] = []
    missing_patients: list[str] = []
    for patient, path in call_paths.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["identity_call_record_present"] = True
            frame["identity_run_status"] = "complete"
            frame["identity_run_error"] = ""
            call_frames.append(frame)
            continue
        missing_patients.append(patient)
        patient_cells = annotations.loc[
            annotations["sample_id"].astype(str).eq(patient),
            ["sample_id", "cell_barcode"],
        ].copy()
        patient_cells["known_normal_reference"] = False
        patient_cells["sctype_malignant_healthy"] = pd.NA
        patient_cells["copyKat_output"] = pd.NA
        patient_cells["SCEVAN_output"] = pd.NA
        patient_cells["ensemble_output"] = pd.NA
        patient_cells["identity_call_record_present"] = False
        patient_cells["identity_run_status"] = str(
            run_status.get(patient, {}).get("status", "missing")
        )
        patient_cells["identity_run_error"] = str(
            run_status.get(patient, {}).get(
                "error", "official identity call file was not produced"
            )
        )
        call_frames.append(patient_cells)
    calls = pd.concat(call_frames, ignore_index=True)
    if calls.duplicated(["sample_id", "cell_barcode"]).any():
        raise ValueError("official identity calls contain duplicate barcodes")
    merged = annotations.merge(
        calls,
        on=["sample_id", "cell_barcode"],
        how="left",
        validate="one_to_one",
    )
    missing_record = ~merged["identity_call_record_present"].fillna(False).astype(bool)
    missing_file = merged["sample_id"].astype(str).isin(missing_patients)
    missing_record_count = int(missing_record.sum())
    barcode_unmatched = int((missing_record & ~missing_file).sum())
    unresolved = int(merged["ensemble_output"].isna().sum())

    patient_summary = pd.read_csv(args.patient_summary)
    cells, patient_gate, state_gate, summary = build_retrospective_identity_gate(
        merged,
        patient_summary,
    )
    patient_gate["identity_run_status"] = patient_gate["patient_id"].map(
        lambda patient: str(run_status.get(str(patient), {}).get("status", "unknown"))
    )
    patient_gate["identity_run_error"] = patient_gate["patient_id"].map(
        lambda patient: str(run_status.get(str(patient), {}).get("error", ""))
    )
    for patient in missing_patients:
        patient_mask = patient_gate["patient_id"].astype(str).eq(patient)
        state_mask = state_gate["patient_id"].astype(str).eq(patient)
        unavailable_patient_fields = [
            "rna_ensemble_call_coverage",
            "rna_ensemble_malignant_fraction",
            "known_normal_reference_cells",
            "major_states_passing",
            "ensemble_vs_scrna_blast_gap_pct",
        ]
        patient_gate.loc[patient_mask, unavailable_patient_fields] = float("nan")
        patient_gate.loc[patient_mask, "retrospective_identity_gate_pass"] = False
        status = str(run_status.get(patient, {}).get("status", "missing"))
        error = str(
            run_status.get(patient, {}).get(
                "error", "official identity call file was not produced"
            )
        )
        patient_gate.loc[patient_mask, "blockers"] = (
            f"official identity workflow {status}: {error}"
        )
        state_gate.loc[
            state_mask,
            [
                "rna_ensemble_call_coverage",
                "dominant_identity_call",
                "dominant_identity_purity",
            ],
        ] = float("nan")
        state_gate.loc[state_mask, "retrospective_state_identity_pass"] = False
    summary["missing_identity_call_record_cells"] = missing_record_count
    summary["barcode_unmatched_cells"] = barcode_unmatched
    summary["algorithm_unresolved_cells"] = unresolved
    summary["missing_identity_call_patients"] = missing_patients
    summary["official_identity_run_status"] = run_status
    if missing_patients:
        summary["all_patients_pass_retrospective_identity_gate"] = False
        summary["retrospective_drug_validation_stage_unlocked"] = False
    summary["input_sha256"] = {
        "cell_annotations": sha256_file(args.cell_annotations),
        "patient_summary": sha256_file(args.patient_summary),
        "official_identity_calls": {
            path.name: sha256_file(path)
            for path in call_paths.values()
            if path.exists()
        },
        "run_status": sha256_file(status_path) if status_path.exists() else None,
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
