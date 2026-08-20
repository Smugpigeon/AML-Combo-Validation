#!/usr/bin/env python3
"""Freeze outcome-blind, state-aware AML Virtual Cell v1.6 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.cell_identity import require_identity_gate_summary  # noqa: E402
from combo_val.virtual_cell.challenge_firewall import (  # noqa: E402
    assert_label_free,
    verify_public_directory,
)
from combo_val.virtual_cell.perturbation_v16 import (  # noqa: E402
    PredictionSupport,
    assess_prediction_support,
    combine_pair_support,
    freeze_predictions,
    state_aware_aggregate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate precomputed state-level single-drug anchors and freeze "
            "a label-free v1.6 prediction artifact."
        )
    )
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--identity-gate-summary", type=Path, required=True)
    parser.add_argument("--state-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-candidates", type=Path)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def normalize_patient_id(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "")
    if text.startswith("patient"):
        suffix = text[len("patient"):]
        return f"patient{int(suffix)}" if suffix.isdigit() else text
    if text.isdigit():
        return f"patient{int(text)}"
    return text


def support_table(state_predictions: pd.DataFrame) -> pd.DataFrame:
    required = [
        "patient_id",
        "drug_id",
        "qc_severity",
        "gene_coverage_fraction",
        "molecular_similarity",
        "model_in_vocabulary",
    ]
    missing = [column for column in required if column not in state_predictions.columns]
    if missing:
        raise KeyError(f"support columns missing from state predictions: {missing}")

    rows: list[dict[str, object]] = []
    for (patient_id, drug_id), group in state_predictions.groupby(
        ["patient_id", "drug_id"],
        sort=True,
    ):
        consistency_columns = required[2:]
        inconsistent = [
            column
            for column in consistency_columns
            if group[column].dropna().astype(str).nunique() > 1
        ]
        if inconsistent:
            raise ValueError(
                f"inconsistent support metadata for {patient_id}/{drug_id}: "
                f"{inconsistent}"
            )
        first = group.iloc[0]
        support = assess_prediction_support(
            qc_severity=str(first["qc_severity"]),
            gene_coverage_fraction=float(first["gene_coverage_fraction"]),
            molecular_similarity=first["molecular_similarity"],
            model_in_vocabulary=parse_bool(first["model_in_vocabulary"]),
        )
        rows.append(
            {
                "patient_id": normalize_patient_id(patient_id),
                "drug_id": str(drug_id).strip(),
                "prediction_support": support.level,
                "support_reasons": " | ".join(support.reasons),
                "qc_severity": support.qc_severity,
                "gene_coverage_fraction": support.gene_coverage_fraction,
                "molecular_similarity": support.molecular_similarity,
                "model_in_vocabulary": support.model_in_vocabulary,
            }
        )
    return pd.DataFrame(rows)


def support_from_row(row: pd.Series) -> PredictionSupport:
    reasons = tuple(
        item.strip()
        for item in str(row.get("support_reasons", "")).split("|")
        if item.strip()
    )
    similarity = row.get("molecular_similarity")
    return PredictionSupport(
        level=str(row["prediction_support"]),
        reasons=reasons,
        qc_severity=str(row["qc_severity"]),
        gene_coverage_fraction=float(row["gene_coverage_fraction"]),
        molecular_similarity=(
            None if pd.isna(similarity) else float(similarity)
        ),
        model_in_vocabulary=parse_bool(row["model_in_vocabulary"]),
    )


def build_pair_support(
    candidates: pd.DataFrame,
    per_drug_support: pd.DataFrame,
) -> pd.DataFrame:
    rename = {}
    if "Patient" in candidates.columns and "patient_id" not in candidates.columns:
        rename["Patient"] = "patient_id"
    if "Drug1" in candidates.columns and "drug1" not in candidates.columns:
        rename["Drug1"] = "drug1"
    if "Drug2" in candidates.columns and "drug2" not in candidates.columns:
        rename["Drug2"] = "drug2"
    out = candidates.rename(columns=rename).copy()
    required = {"patient_id", "drug1", "drug2"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise KeyError(f"pair candidate columns missing: {missing}")
    assert_label_free(out, allow_predicted=False, context="pair candidates")

    out["patient_id"] = out["patient_id"].map(normalize_patient_id)
    lookup = {
        (str(row.patient_id), str(row.drug_id)): row
        for row in per_drug_support.itertuples(index=False)
    }

    levels: list[str] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        patient_id = str(row.patient_id)
        drug1 = str(row.drug1).strip()
        drug2 = str(row.drug2).strip()
        support_a_row = lookup.get((patient_id, drug1))
        support_b_row = lookup.get((patient_id, drug2))
        if support_a_row is None or support_b_row is None:
            levels.append("rejected")
            missing_drugs = [
                drug
                for drug, support_row in (
                    (drug1, support_a_row),
                    (drug2, support_b_row),
                )
                if support_row is None
            ]
            reasons.append("missing single-drug support: " + ", ".join(missing_drugs))
            continue
        support_a = support_from_row(pd.Series(support_a_row._asdict()))
        support_b = support_from_row(pd.Series(support_b_row._asdict()))
        level, pair_reasons = combine_pair_support(support_a, support_b)
        levels.append(level)
        reasons.append(" | ".join(pair_reasons))

    out["pair_support"] = levels
    out["pair_support_reasons"] = reasons
    out["predicted_synergy"] = np.nan
    out["pair_model_status"] = "disabled_pending_grouped_pair_validation"
    return out


def main() -> int:
    args = parse_args()
    verify_public_directory(args.public_dir)

    states = pd.read_csv(args.state_predictions)
    assert_label_free(states, allow_predicted=True, context="state predictions")
    states["patient_id"] = states["patient_id"].map(normalize_patient_id)
    identity_summary = json.loads(args.identity_gate_summary.read_text(encoding="utf-8"))
    require_identity_gate_summary(
        identity_summary,
        patient_ids=states["patient_id"].dropna().unique(),
    )

    aggregate = state_aware_aggregate(states)
    support = support_table(states)
    predictions = aggregate.merge(
        support,
        on=["patient_id", "drug_id"],
        how="left",
        validate="one_to_one",
    )
    predictions["research_use_only"] = True
    predictions["clinical_recommendation"] = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_manifest = freeze_predictions(
        predictions,
        output_path=args.out_dir / "patient_drug_predictions_v16.csv",
        sort_columns=["patient_id", "patient_drug_rank", "drug_id"],
        metadata={
            "scope": "baseline-state single-drug ex-vivo AUC anchor",
            "lower_predicted_auc_is_more_sensitive": True,
            "synergy_head_enabled": False,
            "post_treatment_transcriptome_generated": False,
            "identity_gate_summary": args.identity_gate_summary.name,
        },
    )

    output = {
        "prediction_manifest": prediction_manifest,
        "n_patients": int(predictions["patient_id"].nunique()),
        "n_drugs": int(predictions["drug_id"].nunique()),
        "support_counts": {
            str(key): int(value)
            for key, value in predictions["prediction_support"].value_counts().items()
        },
        "synergy_head": {
            "enabled": False,
            "reason": (
                "Existing checkpoint failed embedding-discriminability and "
                "grouped-pair validation gates."
            ),
        },
    }

    if args.pair_candidates is not None:
        pairs = build_pair_support(pd.read_csv(args.pair_candidates), support)
        pair_manifest = freeze_predictions(
            pairs,
            output_path=args.out_dir / "pair_support_v16.csv",
            sort_columns=["patient_id", "drug1", "drug2"],
            metadata={
                "synergy_head_enabled": False,
                "pair_scores_are_not_predictions": True,
            },
        )
        output["pair_manifest"] = pair_manifest
        output["pair_support_counts"] = {
            str(key): int(value)
            for key, value in pairs["pair_support"].value_counts().items()
        }

    (args.out_dir / "run_summary_v16.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
