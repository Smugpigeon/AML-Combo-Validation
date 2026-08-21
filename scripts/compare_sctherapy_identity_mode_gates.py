#!/usr/bin/env python3
"""Require agreement between released-code and corrected SCEVAN identity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-gate-dir", type=Path, required=True)
    parser.add_argument("--corrected-gate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mode(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    patient_path = directory / "retrospective_patient_identity_gate.csv"
    summary_path = directory / "retrospective_identity_gate_summary.json"
    patient_gate = pd.read_csv(patient_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {"patient_id", "retrospective_identity_gate_pass", "blockers"}
    missing = sorted(required - set(patient_gate.columns))
    if missing:
        raise KeyError(f"patient identity gate columns missing: {missing}")
    if patient_gate["patient_id"].astype(str).duplicated().any():
        raise ValueError("patient identity gate contains duplicate patients")
    return patient_gate, summary


def main() -> int:
    args = parse_args()
    released, released_summary = read_mode(args.released_gate_dir)
    corrected, corrected_summary = read_mode(args.corrected_gate_dir)
    comparison = released.merge(
        corrected,
        on="patient_id",
        how="outer",
        suffixes=("_released", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        missing = comparison.loc[
            ~comparison["_merge"].eq("both"), ["patient_id", "_merge"]
        ].to_dict(orient="records")
        raise ValueError(f"identity modes contain different patients: {missing}")

    released_pass = comparison["retrospective_identity_gate_pass_released"].astype(
        bool
    )
    corrected_pass = comparison[
        "retrospective_identity_gate_pass_corrected"
    ].astype(bool)
    comparison["patient_gate_decision_agrees"] = released_pass.eq(corrected_pass)
    comparison["patient_passes_both_modes"] = released_pass & corrected_pass
    comparison = comparison.drop(columns="_merge")

    released_unlocked = bool(
        released_summary.get("retrospective_drug_validation_stage_unlocked", False)
    )
    corrected_unlocked = bool(
        corrected_summary.get("retrospective_drug_validation_stage_unlocked", False)
    )
    decisions_agree = bool(comparison["patient_gate_decision_agrees"].all())
    unlocked = released_unlocked and corrected_unlocked and decisions_agree
    disagreeing = comparison.loc[
        ~comparison["patient_gate_decision_agrees"], "patient_id"
    ].astype(str).tolist()

    blockers: list[str] = []
    if not released_unlocked:
        blockers.append("released-code effective identity mode did not pass")
    if not corrected_unlocked:
        blockers.append("true-SCEVAN corrected identity mode did not pass")
    if disagreeing:
        blockers.append(
            "patient-level gate decisions differ between modes: "
            + ", ".join(disagreeing)
        )

    released_patient_path = (
        args.released_gate_dir / "retrospective_patient_identity_gate.csv"
    )
    corrected_patient_path = (
        args.corrected_gate_dir / "retrospective_patient_identity_gate.csv"
    )
    released_summary_path = (
        args.released_gate_dir / "retrospective_identity_gate_summary.json"
    )
    corrected_summary_path = (
        args.corrected_gate_dir / "retrospective_identity_gate_summary.json"
    )
    decision = {
        "stage": "dual_semantics_identity_sensitivity_gate",
        "research_use_only": True,
        "released_code_mode_unlocked": released_unlocked,
        "true_scevan_corrected_mode_unlocked": corrected_unlocked,
        "all_patient_gate_decisions_agree": decisions_agree,
        "disagreeing_patients": disagreeing,
        "dual_mode_retrospective_drug_validation_unlocked": unlocked,
        "permitted_identity_mode_for_monotherapy": (
            "true_scevan_corrected" if unlocked else None
        ),
        "blockers": blockers,
        "boundary": (
            "This is a sensitivity gate over two identity semantics. Even a pass "
            "does not establish orthogonal or same-cell genomic identity."
        ),
        "input_sha256": {
            "released_patient_gate": sha256_file(released_patient_path),
            "released_summary": sha256_file(released_summary_path),
            "corrected_patient_gate": sha256_file(corrected_patient_path),
            "corrected_summary": sha256_file(corrected_summary_path),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        args.out_dir / "dual_mode_patient_gate_comparison.csv", index=False
    )
    (args.out_dir / "dual_mode_identity_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
