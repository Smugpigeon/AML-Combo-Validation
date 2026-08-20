#!/usr/bin/env python3
"""Run the hard-gated scTherapy identity-to-monotherapy research sequence.

Official ScType/CopyKAT/SCEVAN calls must be generated first with
``run_sctherapy_identity_ensemble.R``. This orchestrator then enforces:

identity reproduction -> frozen single-drug predictions -> unblinded metrics
-> research-only combination unlock decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.retrospective_validation import (  # noqa: E402
    combination_unlock_decision,
)
from combo_val.virtual_cell.stage_review import (  # noqa: E402
    build_stage_review,
    render_stage_review_markdown,
)


def _run(script: str, *arguments: object) -> None:
    command = [sys.executable, str(SCRIPTS / script), *(str(item) for item in arguments)]
    subprocess.run(command, cwd=ROOT, check=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_review(out_dir: Path, stage: str, summary: dict[str, object]) -> None:
    review = build_stage_review(stage, summary)
    review_dir = out_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / f"{stage}_adversarial_review.json", review)
    (review_dir / f"{stage}_adversarial_review.md").write_text(
        render_stage_review_markdown(review), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-call-dir", type=Path, required=True)
    parser.add_argument("--cell-annotations", type=Path, required=True)
    parser.add_argument("--patient-manifest", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--split-safe-rna-preprocessor", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--combo-matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patients", default="patient5,patient6,patient12")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    identity_dir = args.out_dir / "stage1_identity"
    _run(
        "build_sctherapy_retrospective_identity_gate.py",
        "--cell-annotations",
        args.cell_annotations,
        "--identity-call-dir",
        args.identity_call_dir,
        "--patient-summary",
        args.patient_manifest,
        "--patients",
        args.patients,
        "--out-dir",
        identity_dir,
    )
    identity_path = identity_dir / "retrospective_identity_gate_summary.json"
    identity = _read_json(identity_path)
    _write_review(args.out_dir, "identity", identity)
    readiness_path = args.out_dir / "stage3_combination_readiness.json"
    _run(
        "audit_sctherapy_combination_readiness.py",
        "--combo-matrix",
        args.combo_matrix,
        "--out",
        readiness_path,
    )
    readiness = _read_json(readiness_path)
    if not identity.get("retrospective_drug_validation_stage_unlocked", False):
        monotherapy = {
            "viability_direction_gate_pass": False,
            "blockers": ["not run because the identity reproduction gate failed"],
        }
        decision = combination_unlock_decision(identity, monotherapy, readiness)
        _write_json(args.out_dir / "combination_unlock_decision.json", decision)
        _write_review(args.out_dir, "combination", decision)
        _write_json(
            args.out_dir / "PIPELINE_STATUS.json",
            {"status": "blocked_at_identity", "research_use_only": True},
        )
        return 0

    challenge_dir = args.out_dir / "stage2_monotherapy_challenge"
    _run(
        "prepare_sctherapy_monotherapy_challenge.py",
        "--combo-matrix",
        args.combo_matrix,
        "--out-dir",
        challenge_dir,
    )
    features_dir = args.out_dir / "stage2_patient_features"
    _run(
        "build_sctherapy_beataml_features.py",
        "--h5ad",
        args.h5ad,
        "--patient-manifest",
        args.patient_manifest,
        "--identity-gate-summary",
        identity_path,
        "--preprocessor",
        args.preprocessor,
        "--split-safe-rna-preprocessor",
        args.split_safe_rna_preprocessor,
        "--out-features",
        features_dir / "patient_features.csv",
        "--out-diagnostics",
        features_dir / "patient_feature_diagnostics.json",
        "--patients",
        args.patients,
    )
    inference_dir = args.out_dir / "stage2_frozen_model_inference"
    _run(
        "infer_beataml_virtual_cell_pilot.py",
        "--patient-features",
        features_dir / "patient_features.csv",
        "--model-dir",
        args.model_dir,
        "--out-dir",
        inference_dir,
        "--device",
        "cpu",
    )
    frozen_dir = args.out_dir / "stage2_frozen_predictions"
    _run(
        "freeze_sctherapy_monotherapy_predictions.py",
        "--public-dir",
        challenge_dir / "public",
        "--beataml-predictions",
        inference_dir / "drug_predictions.csv",
        "--model-dir",
        args.model_dir,
        "--out-dir",
        frozen_dir,
    )
    evaluation_dir = args.out_dir / "stage2_unblinded_evaluation"
    frozen_path = frozen_dir / "frozen_monotherapy_predictions.csv"
    _run(
        "evaluate_sctherapy_monotherapy_gate.py",
        "--predictions",
        frozen_path,
        "--prediction-manifest",
        frozen_path.with_suffix(".csv.manifest.json"),
        "--sealed-outcomes",
        challenge_dir / "sealed" / "monotherapy_outcomes.csv",
        "--feature-support-summary",
        inference_dir / "feature_support_summary.json",
        "--out-dir",
        evaluation_dir,
    )
    monotherapy_path = evaluation_dir / "monotherapy_gate_summary.json"
    monotherapy = _read_json(monotherapy_path)
    _write_review(args.out_dir, "monotherapy", monotherapy)
    decision = combination_unlock_decision(identity, monotherapy, readiness)
    _write_json(args.out_dir / "combination_unlock_decision.json", decision)
    _write_review(args.out_dir, "combination", decision)
    _write_json(
        args.out_dir / "PIPELINE_STATUS.json",
        {
            "status": (
                "combination_training_unlocked_pending_validation"
                if decision["combination_training_unlocked"]
                else (
                    "research_combination_benchmark_only"
                    if decision["combination_benchmark_unlocked"]
                    else "blocked_at_monotherapy"
                )
            ),
            "research_use_only": True,
            "clinical_combination_use_unlocked": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
