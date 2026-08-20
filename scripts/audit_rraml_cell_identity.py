#!/usr/bin/env python3
"""Audit whether AML cell identity is proven before drug perturbation work."""

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

from combo_val.virtual_cell.cell_identity import build_identity_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build patient-, state-, and cell-level AML identity evidence gates. "
            "No drug-response table is read."
        )
    )
    parser.add_argument("--cell-annotations", type=Path, required=True)
    parser.add_argument("--patient-summary", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--patients", nargs="*", default=["patient5", "patient6", "patient12"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--write-cell-table",
        action="store_true",
        help="Also write the cell-level parquet audit; omitted by default to keep artifacts small.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    cells = read_table(args.cell_annotations)
    patients = read_table(args.patient_summary)
    manifest = read_table(args.evidence_manifest) if args.evidence_manifest else None

    requested = {str(patient).strip() for patient in args.patients}
    cells = cells.loc[cells["sample_id"].astype(str).isin(requested)].copy()
    patients = patients.loc[patients["patient_id"].astype(str).isin(requested)].copy()
    if manifest is not None:
        manifest = manifest.loc[manifest["patient_id"].astype(str).isin(requested)].copy()
    observed = set(cells["sample_id"].astype(str))
    missing = sorted(requested - observed)
    if missing:
        raise ValueError(f"requested patients missing from cell annotations: {missing}")

    cell_audit, patient_gate, state_gate, evidence_plan, summary = build_identity_gate(
        cells,
        patients,
        evidence_manifest=manifest,
    )
    summary["inputs"] = {
        "cell_annotations": args.cell_annotations.name,
        "patient_summary": args.patient_summary.name,
        "evidence_manifest": (
            args.evidence_manifest.name if args.evidence_manifest else None
        ),
        "patients": sorted(requested),
        "cell_table_written": bool(args.write_cell_table),
        "paths_redacted_to_basenames": True,
    }
    summary["input_sha256"] = {
        "cell_annotations": sha256_file(args.cell_annotations),
        "patient_summary": sha256_file(args.patient_summary),
        "evidence_manifest": (
            sha256_file(args.evidence_manifest) if args.evidence_manifest else None
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_cell_table:
        cell_audit.to_parquet(args.out_dir / "cell_identity_audit.parquet", index=False)
    patient_gate.to_csv(args.out_dir / "patient_identity_gate.csv", index=False)
    state_gate.to_csv(args.out_dir / "state_identity_gate.csv", index=False)
    evidence_plan.to_csv(args.out_dir / "next_identity_evidence_plan.csv", index=False)
    summary_path = args.out_dir / "identity_gate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths = [
        args.out_dir / "patient_identity_gate.csv",
        args.out_dir / "state_identity_gate.csv",
        args.out_dir / "next_identity_evidence_plan.csv",
        summary_path,
    ]
    if args.write_cell_table:
        artifact_paths.append(args.out_dir / "cell_identity_audit.parquet")
    artifact_manifest = {
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ]
    }
    (args.out_dir / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
