#!/usr/bin/env python3
"""Create physically separated public and sealed scTherapy challenge tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split scTherapy candidate tables into label-free public inputs "
            "and sealed outcomes. Point prediction code only at OUT_DIR/public."
        )
    )
    parser.add_argument("--combo-matrix", type=Path, required=True)
    parser.add_argument("--flow-validation", type=Path, required=True)
    parser.add_argument("--single-agent-labels", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    by_normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    resolved: list[str] = []
    missing: list[str] = []
    for name in requested:
        column = by_normalized.get(name.lower())
        if column is None:
            missing.append(name)
        else:
            resolved.append(column)
    if missing:
        raise KeyError(f"required columns missing: {missing}")
    return resolved


def main() -> int:
    args = parse_args()
    public_dir = args.out_dir / "public"
    sealed_dir = args.out_dir / "sealed"
    public_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    all_digests = []
    public_digests = []
    metadata: dict[str, object] = {
        "research_use_only": True,
        "prediction_input_directory": "public",
        "evaluation_only_directory": "sealed",
        "single_agent_identity_complete": False,
    }

    combo = pd.read_csv(args.combo_matrix)
    combo_identity = require_columns(
        combo,
        ["Patient", "PairIndex", "Drug1", "Drug2", "Dose1", "Dose2", "DoseUnit"],
    )
    combo_outcomes = require_columns(combo, ["Response"])
    combo_split = split_challenge_table(
        combo,
        identity_columns=combo_identity,
        outcome_columns=combo_outcomes,
        prefix="combo",
    )
    digests = write_split(
        combo_split,
        public_path=public_dir / "combo_candidates.csv",
        sealed_path=sealed_dir / "combo_outcomes.csv",
    )
    public_digests.append(digests[0])
    all_digests.extend(digests)
    metadata["combo_rows"] = len(combo_split.public)

    flow = pd.read_csv(args.flow_validation)
    flow_identity = require_columns(
        flow,
        [
            "patient_number",
            "compartment",
            "drug1",
            "dose1_nm",
            "drug2",
            "dose2_nm",
        ],
    )
    flow_outcomes = require_columns(
        flow,
        ["response1_pct", "response2_pct", "response3_pct"],
    )
    flow_split = split_challenge_table(
        flow,
        identity_columns=flow_identity,
        outcome_columns=flow_outcomes,
        prefix="flow",
    )
    digests = write_split(
        flow_split,
        public_path=public_dir / "flow_cases.csv",
        sealed_path=sealed_dir / "flow_outcomes.csv",
    )
    public_digests.append(digests[0])
    all_digests.extend(digests)
    metadata["flow_rows"] = len(flow_split.public)

    if args.single_agent_labels is not None:
        single = pd.read_csv(args.single_agent_labels)
        single_identity = require_columns(single, ["row_id", "patient_number"])
        single_outcomes = require_columns(
            single,
            ["cell_inhibition_pct", "prediction_label"],
        )
        single_split = split_challenge_table(
            single,
            identity_columns=single_identity,
            outcome_columns=single_outcomes,
            prefix="single",
        )
        digests = write_split(
            single_split,
            public_path=public_dir / "single_agent_cases.csv",
            sealed_path=sealed_dir / "single_agent_outcomes.csv",
        )
        public_digests.append(digests[0])
        all_digests.extend(digests)
        metadata["single_agent_rows"] = len(single_split.public)
        metadata["single_agent_note"] = (
            "The source table has no drug identity; use only for label-distribution "
            "audit, not drug-ranking validation."
        )

    write_manifest(
        args.out_dir / "MANIFEST.json",
        files=all_digests,
        metadata=metadata,
    )
    write_manifest(
        public_dir / "MANIFEST.json",
        files=public_digests,
        metadata={
            "label_free": True,
            "prediction_entrypoint": True,
            "sealed_files_not_listed": True,
        },
    )

    root_manifest = file_digest(args.out_dir / "MANIFEST.json", relative_to=args.out_dir)
    print(f"challenge_dir={args.out_dir}")
    print(f"public_dir={public_dir}")
    print(f"sealed_dir={sealed_dir}")
    print(f"manifest_sha256={root_manifest.sha256}")
    print("Prediction code must receive --public-dir only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
