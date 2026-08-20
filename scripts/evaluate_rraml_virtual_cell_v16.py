#!/usr/bin/env python3
"""Evaluate a previously frozen v1.6 artifact against separately sealed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from combo_val.virtual_cell.challenge_firewall import assert_label_free  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen prediction hash, then evaluate against a sealed "
            "outcome table joined only by challenge_row_id."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--sealed-outcomes", type=Path, required=True)
    parser.add_argument("--score-column", default="predicted_score")
    parser.add_argument("--outcome-column", required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--lower-score-is-better", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def safe_correlations(score: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    if len(score) < 3 or np.std(score) == 0 or np.std(outcome) == 0:
        return float("nan"), float("nan")
    pearson = float(pearsonr(score, outcome).statistic)
    spearman = float(spearmanr(score, outcome).statistic)
    return pearson, spearman


def bootstrap_correlations(
    frame: pd.DataFrame,
    *,
    score_column: str,
    outcome_column: str,
    group_column: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    groups = frame[group_column].drop_duplicates().to_numpy()
    estimates = {"pearson": [], "spearman": []}
    by_group = {
        group: part
        for group, part in frame.groupby(group_column, sort=False)
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [by_group[group] for group in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        pearson, spearman = safe_correlations(
            boot[score_column].to_numpy(dtype=float),
            boot[outcome_column].to_numpy(dtype=float),
        )
        if np.isfinite(pearson):
            estimates["pearson"].append(pearson)
        if np.isfinite(spearman):
            estimates["spearman"].append(spearman)
    return {
        metric: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        if values else [float("nan"), float("nan")]
        for metric, values in estimates.items()
    }


def main() -> int:
    args = parse_args()
    payload = args.predictions.read_bytes()
    observed_hash = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    expected_hash = str(manifest.get("sha256", ""))
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"prediction hash mismatch: expected {expected_hash}, got {observed_hash}"
        )

    predictions = pd.read_csv(args.predictions)
    assert_label_free(
        predictions,
        allow_predicted=True,
        context="frozen predictions",
    )
    required_predictions = {"challenge_row_id", args.score_column}
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise KeyError(f"prediction columns missing: {missing}")
    if predictions["challenge_row_id"].duplicated().any():
        raise ValueError("prediction challenge_row_id values must be unique")

    sealed = pd.read_csv(args.sealed_outcomes)
    required_sealed = {"challenge_row_id", args.outcome_column}
    missing = sorted(required_sealed - set(sealed.columns))
    if missing:
        raise KeyError(f"sealed outcome columns missing: {missing}")
    if sealed["challenge_row_id"].duplicated().any():
        raise ValueError("sealed challenge_row_id values must be unique")

    merged = predictions.merge(
        sealed.loc[:, ["challenge_row_id", args.outcome_column]],
        on="challenge_row_id",
        how="inner",
        validate="one_to_one",
    )
    score = pd.to_numeric(merged[args.score_column], errors="coerce")
    outcome = pd.to_numeric(merged[args.outcome_column], errors="coerce")
    valid = score.notna() & outcome.notna()
    merged = merged.loc[valid].copy()
    if len(merged) < 3:
        raise ValueError("fewer than three finite prediction/outcome pairs")

    evaluation_score = merged[args.score_column].to_numpy(dtype=float)
    if args.lower_score_is_better:
        evaluation_score = -evaluation_score
    merged["_evaluation_score"] = evaluation_score
    pearson, spearman = safe_correlations(
        merged["_evaluation_score"].to_numpy(dtype=float),
        merged[args.outcome_column].to_numpy(dtype=float),
    )

    if args.group_column is not None:
        if args.group_column not in predictions.columns:
            raise KeyError(f"group column missing from predictions: {args.group_column}")
        group_column = args.group_column
    else:
        group_column = "challenge_row_id"

    intervals = bootstrap_correlations(
        merged,
        score_column="_evaluation_score",
        outcome_column=args.outcome_column,
        group_column=group_column,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result = {
        "research_use_only": True,
        "prediction_sha256": observed_hash,
        "n_evaluated_rows": int(len(merged)),
        "n_bootstrap": int(args.bootstrap),
        "score_direction_normalized_to_higher_is_better": True,
        "pearson": pearson,
        "pearson_95ci": intervals["pearson"],
        "spearman": spearman,
        "spearman_95ci": intervals["spearman"],
        "note": (
            "Rows are technical dose observations unless group_column identifies "
            "a patient or unordered drug pair. Interpret row-level intervals cautiously."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
