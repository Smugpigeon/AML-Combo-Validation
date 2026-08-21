#!/usr/bin/env python3
"""Audit released scTherapy SCEVAN semantics and Python-to-R call parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

MINIMUM_CALL_AGREEMENT = 0.95
MINIMUM_ADJUSTED_RAND_INDEX = 0.95
MINIMUM_MALIGNANT_JACCARD = 0.90
MINIMUM_COMMON_EVALUABLE_FRACTION = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-call-dir", type=Path, required=True)
    parser.add_argument("--identity-input-dir", type=Path, required=True)
    parser.add_argument("--r-reference-dir", type=Path, required=True)
    parser.add_argument("--python-call-dir", type=Path, required=True)
    parser.add_argument("--reference-patients", default="patient5,patient12")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def malignant_jaccard(left: pd.Series, right: pd.Series) -> float:
    left_positive = left.eq("malignant")
    right_positive = right.eq("malignant")
    union = int((left_positive | right_positive).sum())
    return float((left_positive & right_positive).sum() / union) if union else 1.0


def main() -> int:
    args = parse_args()
    patients = [
        item.strip() for item in args.reference_patients.split(",") if item.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    semantic_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []

    for patient in patients:
        legacy = pd.read_csv(args.legacy_call_dir / f"{patient}_identity_calls.csv")
        identity = pd.read_csv(
            args.identity_input_dir / f"{patient}_identity_inputs.csv"
        )
        semantic = legacy[["cell_barcode", "SCEVAN_output"]].merge(
            identity[["cell_barcode", "known_normal_reference"]],
            on="cell_barcode",
            how="inner",
            validate="one_to_one",
        )
        expected = semantic["known_normal_reference"].map(
            {True: "healthy", False: "malignant"}
        )
        mismatch = int(semantic["SCEVAN_output"].ne(expected).sum())
        semantic_rows.append(
            {
                "patient_id": patient,
                "n_compared": len(semantic),
                "n_mismatch": mismatch,
                "released_output_equals_normal_reference_rule": mismatch == 0,
            }
        )

        r_calls = pd.read_csv(
            args.r_reference_dir / f"{patient}_scevan_r_classification.csv"
        ).rename(columns={"SCEVAN_output": "r_call"})
        py_calls = pd.read_csv(
            args.python_call_dir / f"{patient}_scevan_python_classification.csv"
        ).rename(columns={"SCEVAN_output": "python_call"})
        paired = r_calls[["cell_barcode", "r_call"]].merge(
            py_calls[["cell_barcode", "python_call"]],
            on="cell_barcode",
            how="outer",
            validate="one_to_one",
        )
        evaluable = paired.dropna(subset=["r_call", "python_call"]).copy()
        denominator = min(paired["r_call"].notna().sum(), paired["python_call"].notna().sum())
        common_fraction = len(evaluable) / max(int(denominator), 1)
        agreement = float(evaluable["r_call"].eq(evaluable["python_call"]).mean())
        ari = float(adjusted_rand_score(evaluable["r_call"], evaluable["python_call"]))
        kappa = float(cohen_kappa_score(evaluable["r_call"], evaluable["python_call"]))
        jaccard = malignant_jaccard(evaluable["r_call"], evaluable["python_call"])
        passes = (
            common_fraction >= MINIMUM_COMMON_EVALUABLE_FRACTION
            and agreement >= MINIMUM_CALL_AGREEMENT
            and ari >= MINIMUM_ADJUSTED_RAND_INDEX
            and jaccard >= MINIMUM_MALIGNANT_JACCARD
        )
        parity_rows.append(
            {
                "patient_id": patient,
                "n_common_evaluable": len(evaluable),
                "common_evaluable_fraction": common_fraction,
                "call_agreement": agreement,
                "adjusted_rand_index": ari,
                "cohen_kappa": kappa,
                "malignant_jaccard": jaccard,
                "parity_gate_pass": passes,
            }
        )

    semantic_table = pd.DataFrame(semantic_rows)
    parity_table = pd.DataFrame(parity_rows)
    semantic_table.to_csv(args.out_dir / "released_code_semantics_audit.csv", index=False)
    parity_table.to_csv(args.out_dir / "python_r_scevan_parity.csv", index=False)
    summary = {
        "released_code_semantics_confirmed": bool(
            semantic_table["released_output_equals_normal_reference_rule"].all()
        ),
        "python_reimplementation_parity_gate_pass": bool(
            parity_table["parity_gate_pass"].all()
        ),
        "reference_patients": patients,
        "thresholds_preregistered_before_patient_results": {
            "minimum_call_agreement": MINIMUM_CALL_AGREEMENT,
            "minimum_adjusted_rand_index": MINIMUM_ADJUSTED_RAND_INDEX,
            "minimum_malignant_jaccard": MINIMUM_MALIGNANT_JACCARD,
            "minimum_common_evaluable_fraction": MINIMUM_COMMON_EVALUABLE_FRACTION,
        },
        "interpretation": (
            "The released scTherapy SCEVAN_output is audited separately from "
            "true SCEVAN tumour classification. Python calls may be used for "
            "patient6 only if parity passes on both R-computable patients."
        ),
    }
    (args.out_dir / "scevan_semantics_and_parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(summary[key] for key in (
        "released_code_semantics_confirmed",
        "python_reimplementation_parity_gate_pass",
    )) else 2


if __name__ == "__main__":
    raise SystemExit(main())
