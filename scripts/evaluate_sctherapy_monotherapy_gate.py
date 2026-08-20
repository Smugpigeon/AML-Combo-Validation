#!/usr/bin/env python3
"""Unblind and evaluate an immutable scTherapy monotherapy prediction file."""

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

from combo_val.virtual_cell.retrospective_validation import (  # noqa: E402
    evaluate_monotherapy_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--sealed-outcomes", type=Path, required=True)
    parser.add_argument("--feature-support-summary", type=Path, required=True)
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
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    observed_hash = sha256_file(args.predictions)
    if observed_hash != manifest.get("sha256"):
        raise RuntimeError("prediction file changed after freezing")

    predictions = pd.read_csv(args.predictions)
    outcomes = pd.read_csv(args.sealed_outcomes)
    feature_support = json.loads(
        args.feature_support_summary.read_text(encoding="utf-8")
    )
    patient_metrics, summary = evaluate_monotherapy_predictions(
        predictions,
        outcomes,
        feature_support_summary=feature_support,
    )
    summary["prediction_sha256"] = observed_hash
    summary["sealed_outcomes_sha256"] = sha256_file(args.sealed_outcomes)
    summary["feature_support_summary_sha256"] = sha256_file(
        args.feature_support_summary
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patient_metrics.to_csv(args.out_dir / "patient_monotherapy_metrics.csv", index=False)
    (args.out_dir / "monotherapy_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
