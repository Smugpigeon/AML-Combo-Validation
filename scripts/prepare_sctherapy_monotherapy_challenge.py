#!/usr/bin/env python3
"""Build a physically blinded single-agent challenge from zero-dose edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.challenge_firewall import (  # noqa: E402
    file_digest,
    split_challenge_table,
    write_manifest,
    write_split,
)
from combo_val.virtual_cell.retrospective_validation import (  # noqa: E402
    extract_monotherapy_edges,
    summarize_monotherapy_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo-matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.combo_matrix)
    edges = extract_monotherapy_edges(source)
    summary = summarize_monotherapy_edges(edges)
    summary = summary.loc[summary["model_drug_id"].notna()].copy()
    if summary.empty:
        raise RuntimeError("no scTherapy drugs map to the frozen BeatAML vocabulary")

    split = split_challenge_table(
        summary,
        identity_columns=[
            "patient_id",
            "drug_id",
            "model_drug_id",
            "dose_unit",
        ],
        outcome_columns=[
            "observed_dss_like_0_1um",
            "maximum_repeated_edge_range",
        ],
        prefix="mono",
    )
    public_dir = args.out_dir / "public"
    sealed_dir = args.out_dir / "sealed"
    public_digest, sealed_digest = write_split(
        split,
        public_path=public_dir / "monotherapy_cases.csv",
        sealed_path=sealed_dir / "monotherapy_outcomes.csv",
    )
    metadata = {
        "schema_version": "sctherapy_zero_dose_monotherapy_v1",
        "research_use_only": True,
        "source_rule": "Dose1>0,Dose2=0 or Dose2>0,Dose1=0",
        "primary_outcome": "log-dose integrated inhibition from 0.1 to 1000 nM",
        "n_rows": int(len(split.public)),
        "n_patients": int(split.public["patient_id"].nunique()),
        "n_source_drugs": int(summary["drug_id"].nunique()),
        "n_model_drugs": int(summary["model_drug_id"].nunique()),
        "label_free_prediction_directory": "public",
        "evaluation_only_directory": "sealed",
    }
    write_manifest(
        args.out_dir / "MANIFEST.json",
        files=[public_digest, sealed_digest],
        metadata=metadata,
    )
    write_manifest(
        public_dir / "MANIFEST.json",
        files=[public_digest],
        metadata={"label_free": True, "sealed_files_not_listed": True},
    )
    digest = file_digest(args.out_dir / "MANIFEST.json", relative_to=args.out_dir)
    print(f"rows={len(split.public)}")
    print(f"patients={split.public['patient_id'].nunique()}")
    print(f"model_drugs={split.public['model_drug_id'].nunique()}")
    print(f"manifest_sha256={digest.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
