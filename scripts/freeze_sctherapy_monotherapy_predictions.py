#!/usr/bin/env python3
"""Join label-free cases to frozen BeatAML predictions and hash the result."""

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

from combo_val.virtual_cell.challenge_firewall import (  # noqa: E402
    assert_label_free,
    canonical_csv_bytes,
    verify_public_directory,
)
from combo_val.virtual_cell.retrospective_validation import (  # noqa: E402
    normalize_patient_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--beataml-predictions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    verify_public_directory(args.public_dir)
    cases = pd.read_csv(args.public_dir / "monotherapy_cases.csv")
    assert_label_free(cases, allow_predicted=False, context="monotherapy cases")

    anchors = pd.read_csv(args.beataml_predictions)
    assert_label_free(anchors, allow_predicted=True, context="BeatAML predictions")
    required = {
        "patient_id",
        "drug_id",
        "pred_auc_drug_mean_baseline",
        "pred_auc_blended",
        "ensemble_uncertainty_auc",
        "drug_training_observations",
        "support_flag",
    }
    missing = sorted(required - set(anchors.columns))
    if missing:
        raise KeyError(f"BeatAML prediction columns missing: {missing}")
    anchors = anchors.copy()
    anchors["patient_id"] = anchors["patient_id"].map(normalize_patient_id)
    if anchors.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("BeatAML predictions contain duplicate patient-drug rows")

    selected = anchors.loc[:, list(required)].rename(
        columns={"drug_id": "model_drug_id"}
    )
    merged = cases.merge(
        selected,
        on=["patient_id", "model_drug_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["pred_auc_blended"].isna().any():
        missing_rows = merged.loc[
            merged["pred_auc_blended"].isna(), ["patient_id", "model_drug_id"]
        ]
        raise ValueError(
            "missing frozen predictions for challenge rows: "
            + missing_rows.astype(str).agg("/".join, axis=1).str.cat(sep=", ")
        )

    output = merged.loc[
        :,
        [
            "challenge_row_id",
            "patient_id",
            "drug_id",
            "model_drug_id",
            "pred_auc_blended",
            "pred_auc_drug_mean_baseline",
            "ensemble_uncertainty_auc",
            "drug_training_observations",
            "support_flag",
        ],
    ].copy()
    output["predicted_sensitivity_score"] = -pd.to_numeric(
        output["pred_auc_blended"]
    )
    output["drug_mean_sensitivity_score"] = -pd.to_numeric(
        output["pred_auc_drug_mean_baseline"]
    )
    output["score_direction"] = "higher_is_more_sensitive"
    output["use_scope"] = "research_only_directional_ex_vivo_validation"
    output = output.sort_values("challenge_row_id").reset_index(drop=True)
    assert_label_free(output, allow_predicted=True, context="frozen monotherapy output")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "frozen_monotherapy_predictions.csv"
    output_path.write_bytes(canonical_csv_bytes(output))
    checkpoint_paths = sorted(args.model_dir.glob("model_seed_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no model_seed_*.pt files in {args.model_dir}")
    payload = {
        "schema_version": "frozen_sctherapy_monotherapy_predictions_v1",
        "research_use_only": True,
        "prediction_file": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "n_rows": len(output),
        "n_patients": int(output["patient_id"].nunique()),
        "n_drugs": int(output["model_drug_id"].nunique()),
        "inputs": {
            "public_cases_sha256": sha256_file(
                args.public_dir / "monotherapy_cases.csv"
            ),
            "beataml_predictions_sha256": sha256_file(args.beataml_predictions),
            "model_checkpoints": {
                path.name: sha256_file(path) for path in checkpoint_paths
            },
        },
        "outcome_columns_read": [],
    }
    manifest_path = output_path.with_suffix(".csv.manifest.json")
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
