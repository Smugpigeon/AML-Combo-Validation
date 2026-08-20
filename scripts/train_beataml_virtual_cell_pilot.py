#!/usr/bin/env python3
"""Train the first BeatAML virtual patient-state perturbation pilot.

This is a research prototype, not a prescribing system and not a complete
single-cell digital twin. It learns a shared patient encoder and a categorical
single-drug perturbation layer from de-identified BeatAML multi-modal features
and ex-vivo AUC measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from combo_val.virtual_cell.beataml_pilot import (  # noqa: E402
    checkpoint_payload,
    evaluate_predictions,
    fit_blend_alpha,
    load_and_prepare_beataml,
    normalize_identifier,
    predict_all_drugs,
    predict_rows,
    train_one_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO_ROOT / "data/canonical/beataml_patient_features.csv",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=REPO_ROOT / "data/canonical/beataml_drug_response_long.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/virtual_cell_beataml_pilot",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", default="11,29,47")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--first-batch-size", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def ensemble_row_predictions(models, prepared, frame, device):
    standardized = np.stack(
        [predict_rows(model, prepared, frame, device) for model in models], axis=0
    )
    drug_index = frame["drug_index"].to_numpy(dtype=int)
    baseline = prepared.drug_mean[drug_index]
    scale = prepared.drug_scale[drug_index]
    neural_members = baseline[None, :] + standardized * scale[None, :]
    return {
        "standardized_mean": standardized.mean(axis=0),
        "standardized_std": standardized.std(axis=0),
        "baseline": baseline,
        "neural": neural_members.mean(axis=0),
        "neural_std": neural_members.std(axis=0),
    }


def add_predictions(frame, predictions, blend_alpha):
    output = frame.copy()
    output["pred_auc_drug_mean_baseline"] = predictions["baseline"]
    output["pred_auc_neural"] = predictions["neural"]
    output["pred_auc_blended"] = predictions["baseline"] + blend_alpha * (
        predictions["neural"] - predictions["baseline"]
    )
    output["ensemble_uncertainty_auc"] = predictions["neural_std"] * blend_alpha
    return output


def uncertainty_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    uncertainty = frame["ensemble_uncertainty_auc"].to_numpy(dtype=float)
    absolute_error = np.abs(
        frame["pred_auc_blended"].to_numpy(dtype=float)
        - frame["auc"].to_numpy(dtype=float)
    )
    if np.std(uncertainty) < 1e-12:
        correlation = float("nan")
    else:
        correlation = float(spearmanr(uncertainty, absolute_error).statistic)
    return {
        "mean_ensemble_sd_auc": float(np.mean(uncertainty)),
        "median_ensemble_sd_auc": float(np.median(uncertainty)),
        "uncertainty_absolute_error_spearman": correlation,
    }


def paired_patient_bootstrap(
    frame: pd.DataFrame,
    iterations: int = 2000,
    seed: int = 20260819,
) -> dict[str, object]:
    """Estimate patient-paired gains without treating drug rows as independent."""

    rows: list[dict[str, float]] = []
    for _, group in frame.groupby("patient_id"):
        if len(group) < 5:
            continue
        baseline_error = np.abs(
            group["pred_auc_drug_mean_baseline"].to_numpy(dtype=float)
            - group["auc"].to_numpy(dtype=float)
        )
        blended_error = np.abs(
            group["pred_auc_blended"].to_numpy(dtype=float)
            - group["auc"].to_numpy(dtype=float)
        )
        baseline_spearman = float(
            spearmanr(group["auc"], group["pred_auc_drug_mean_baseline"]).statistic
        )
        blended_spearman = float(
            spearmanr(group["auc"], group["pred_auc_blended"]).statistic
        )
        k = min(5, len(group))
        actual_hits = set(group.nsmallest(k, "auc")["drug_id"])
        baseline_hits = set(
            group.nsmallest(k, "pred_auc_drug_mean_baseline")["drug_id"]
        )
        blended_hits = set(group.nsmallest(k, "pred_auc_blended")["drug_id"])
        rows.append(
            {
                "mae_gain": float(baseline_error.mean() - blended_error.mean()),
                "spearman_gain": blended_spearman - baseline_spearman,
                "top5_gain": (len(actual_hits & blended_hits) - len(actual_hits & baseline_hits))
                / k,
            }
        )
    patient_metrics = pd.DataFrame(rows).dropna()
    if patient_metrics.empty:
        raise ValueError("No evaluable patients for paired bootstrap")
    rng = np.random.default_rng(seed)
    values = patient_metrics.to_numpy(dtype=float)
    bootstrap = np.empty((iterations, values.shape[1]), dtype=float)
    for iteration in range(iterations):
        sample = values[rng.integers(0, len(values), size=len(values))]
        bootstrap[iteration, 0] = np.mean(sample[:, 0])
        bootstrap[iteration, 1] = np.median(sample[:, 1])
        bootstrap[iteration, 2] = np.mean(sample[:, 2])
    names = ["mean_patient_mae_gain", "median_patient_spearman_gain", "mean_top5_gain"]
    point = [
        float(patient_metrics["mae_gain"].mean()),
        float(patient_metrics["spearman_gain"].median()),
        float(patient_metrics["top5_gain"].mean()),
    ]
    result: dict[str, object] = {
        "n_patients": int(len(patient_metrics)),
        "iterations": iterations,
        "positive_values_favor_blended": True,
    }
    for index, name in enumerate(names):
        result[name] = {
            "point": point[index],
            "ci95_low": float(np.quantile(bootstrap[:, index], 0.025)),
            "ci95_high": float(np.quantile(bootstrap[:, index], 0.975)),
        }
    return result


def select_first_batch(
    feature_path: Path,
    test_predictions: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    raw_features = pd.read_csv(feature_path)
    raw_features["patient_id"] = raw_features["patient_id"].map(normalize_identifier)
    coverage = (
        test_predictions.groupby("patient_id")
        .agg(
            measured_drugs=("drug_id", "nunique"),
            observed_auc_median=("auc", "median"),
        )
        .reset_index()
    )
    selected_columns = [
        "patient_id",
        "clin_is_relapse",
        "clin_age",
        "clin_eln_ordinal",
        "clin_blast_pct",
        "clin_secondary_aml",
        "clin_fit_for_intensive",
        "mut_FLT3",
        "mut_NPM1",
        "mut_IDH1",
        "mut_IDH2",
        "mut_TP53",
        "mut_RUNX1",
        "mut_ASXL1",
    ]
    selected_columns = [c for c in selected_columns if c in raw_features.columns]
    candidates = coverage.merge(raw_features[selected_columns], on="patient_id", how="left")
    if "clin_is_relapse" not in candidates:
        candidates["clin_is_relapse"] = 0.0
    candidates["adult_high_coverage_eligible"] = (
        candidates["measured_drugs"].ge(50)
        & candidates.get("clin_age", pd.Series(np.nan, index=candidates.index))
        .fillna(18)
        .ge(18)
    )
    candidates["selection_priority"] = np.where(
        candidates["clin_is_relapse"].fillna(0).astype(float).gt(0),
        "held_out_relapse_high_assay_coverage",
        "held_out_high_assay_coverage",
    )
    candidates = candidates[candidates["adult_high_coverage_eligible"]].copy()
    candidates = candidates.sort_values(
        ["clin_is_relapse", "measured_drugs", "patient_id"],
        ascending=[False, False, True],
    )
    return candidates.head(batch_size).reset_index(drop=True)


def build_first_batch_predictions(
    selected: pd.DataFrame,
    prepared,
    all_standardized_mean: np.ndarray,
    all_standardized_std: np.ndarray,
    blend_alpha: float,
) -> pd.DataFrame:
    patient_index = {patient_id: idx for idx, patient_id in enumerate(prepared.patient_ids)}
    observed = (
        prepared.responses.groupby(["patient_id", "drug_id"], as_index=False)["auc"].median()
    )
    observed_lookup = {
        (row.patient_id, row.drug_id): float(row.auc)
        for row in observed.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for patient_id in selected["patient_id"]:
        pidx = patient_index[patient_id]
        neural = prepared.drug_mean + prepared.drug_scale * all_standardized_mean[pidx]
        blended = prepared.drug_mean + blend_alpha * (neural - prepared.drug_mean)
        uncertainty = blend_alpha * prepared.drug_scale * all_standardized_std[pidx]
        order = np.argsort(blended)
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        for didx, drug_id in enumerate(prepared.drug_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "drug_id": drug_id,
                    "observed_auc": observed_lookup.get((patient_id, drug_id), np.nan),
                    "pred_auc_drug_mean_baseline": float(prepared.drug_mean[didx]),
                    "pred_auc_neural": float(neural[didx]),
                    "pred_auc_blended": float(blended[didx]),
                    "ensemble_uncertainty_auc": float(uncertainty[didx]),
                    "predicted_sensitivity_rank": int(rank[didx]),
                    "drug_training_observations": int(prepared.drug_train_count[didx]),
                    "support_flag": (
                        "supported" if prepared.drug_train_count[didx] >= 20 else "low_support"
                    ),
                    "interpretation": "lower predicted AUC indicates greater ex-vivo sensitivity",
                    "use_scope": "research_only_not_a_treatment_recommendation",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["patient_id", "predicted_sensitivity_rank"]
    )


def patient_latent_table(prepared, latent_mean: np.ndarray) -> pd.DataFrame:
    split_lookup = {
        **{p: "train" for p in prepared.split.train},
        **{p: "validation" for p in prepared.split.validation},
        **{p: "test" for p in prepared.split.test},
    }
    data: dict[str, object] = {
        "patient_id": prepared.patient_ids,
        "split": [split_lookup.get(p, "unlabeled_no_auc") for p in prepared.patient_ids],
    }
    for index in range(latent_mean.shape[1]):
        data[f"latent_{index + 1:02d}"] = latent_mean[:, index]
    return pd.DataFrame(data)


def write_model_card(path: Path, metrics: dict[str, object]) -> None:
    baseline = metrics["test_metrics"]["drug_mean_baseline"]
    neural = metrics["test_metrics"]["neural"]
    blended = metrics["test_metrics"]["blended"]
    bootstrap = metrics["paired_patient_bootstrap"]
    text = f"""# BeatAML Virtual Patient-State Pilot

## Intended scope

This pilot encodes de-identified BeatAML patients into a shared 32-dimensional
baseline state and composes that state with a learned single-drug embedding to
predict ex-vivo AUC. Lower AUC indicates greater measured sensitivity.

It is **not** a complete single-cell virtual cell, a post-treatment transcriptome
generator, a combination-synergy model, or a prescribing system. The output is
for research prioritization and prospective validation only.

## Leakage control

The split is patient-wise: all drug rows from one patient stay in exactly one of
train, validation, or test. Feature scaling and per-drug AUC statistics are fit
on training patients only.

## Held-out results

| Predictor | MAE | RMSE | Median within-patient Spearman | Mean top-5 overlap |
|---|---:|---:|---:|---:|
| Drug-mean baseline | {baseline['mae']:.3f} | {baseline['rmse']:.3f} | {baseline['median_patient_spearman']:.3f} | {baseline['mean_top_5_overlap']:.3f} |
| Neural residual | {neural['mae']:.3f} | {neural['rmse']:.3f} | {neural['median_patient_spearman']:.3f} | {neural['mean_top_5_overlap']:.3f} |
| Validation-calibrated blend | {blended['mae']:.3f} | {blended['rmse']:.3f} | {blended['median_patient_spearman']:.3f} | {blended['mean_top_5_overlap']:.3f} |

The blend coefficient is `{metrics['blend_alpha']:.4f}`. A coefficient near zero
means the validation set did not support a large patient-specific neural
residual; a coefficient near one retains the full neural residual.

Patient-paired bootstrap estimates ({bootstrap['iterations']} resamples):

- Mean patient-level MAE gain: {bootstrap['mean_patient_mae_gain']['point']:.3f}
  (95% CI {bootstrap['mean_patient_mae_gain']['ci95_low']:.3f} to
  {bootstrap['mean_patient_mae_gain']['ci95_high']:.3f}).
- Median within-patient Spearman gain:
  {bootstrap['median_patient_spearman_gain']['point']:.3f}
  (95% CI {bootstrap['median_patient_spearman_gain']['ci95_low']:.3f} to
  {bootstrap['median_patient_spearman_gain']['ci95_high']:.3f}).
- Mean top-5 overlap gain: {bootstrap['mean_top5_gain']['point']:.3f}
  (95% CI {bootstrap['mean_top5_gain']['ci95_low']:.3f} to
  {bootstrap['mean_top5_gain']['ci95_high']:.3f}).

## First-batch outputs

- `first_batch_patients.csv`: held-out, de-identified patients selected for high
  assay coverage, with relapsed samples prioritized when available.
- `first_batch_drug_predictions.csv`: all known single drugs ranked per patient,
  including observed AUC where available and ensemble uncertainty.
- `patient_latent_states.csv`: baseline latent vectors for all feature-table
  patients; patients without AUC labels are marked `unlabeled_no_auc`.
- `heldout_test_predictions.csv`: row-level audit table for all test metrics.

## Required next validation

1. External patient-level validation in a cohort not used for fitting.
2. Paired pre/post-treatment RNA or single-cell profiles to learn state
   transitions rather than response scalars alone.
3. Combination-dose matrices for synergy, schedule, and dose optimization.
4. Normal CD34+ and clinical safety data before any clinical routing claim.
5. Prospective organoid/ex-vivo confirmation and clinician review.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    seeds = [int(seed.strip()) for seed in args.ensemble_seeds.split(",") if seed.strip()]
    if args.smoke:
        seeds = seeds[:1]
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    if not seeds:
        raise ValueError("At least one ensemble seed is required")

    print(f"[data] features={args.features}")
    print(f"[data] responses={args.responses}")
    prepared = load_and_prepare_beataml(args.features, args.responses, args.split_seed)
    print(
        "[split] "
        f"train={len(prepared.split.train)} validation={len(prepared.split.validation)} "
        f"test={len(prepared.split.test)} patients"
    )
    print(
        f"[matrix] patients={len(prepared.patient_ids)} features={len(prepared.feature_columns)} "
        f"drugs={len(prepared.drug_ids)} response_rows={len(prepared.responses)}"
    )
    print(f"[device] {device}")

    models = []
    diagnostics = []
    for seed in seeds:
        print(f"[train] seed={seed}", flush=True)
        model, diagnostic = train_one_model(
            prepared=prepared,
            seed=seed,
            device=device,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )
        print(
            f"  best_epoch={diagnostic['best_epoch']} "
            f"validation_loss={diagnostic['best_validation_loss']:.5f}"
        )
        models.append(model)
        diagnostics.append(diagnostic)

    validation_frame = prepared.responses[prepared.responses["split"] == "validation"].copy()
    validation_predictions = ensemble_row_predictions(
        models, prepared, validation_frame, device
    )
    blend_alpha = fit_blend_alpha(
        observed_auc=validation_frame["auc"].to_numpy(dtype=float),
        baseline_auc=validation_predictions["baseline"],
        neural_auc=validation_predictions["neural"],
    )
    print(f"[blend] validation alpha={blend_alpha:.4f}")

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy()
    test_predictions = ensemble_row_predictions(models, prepared, test_frame, device)
    test_output = add_predictions(test_frame, test_predictions, blend_alpha)
    test_metrics = {
        "drug_mean_baseline": evaluate_predictions(
            test_output, "pred_auc_drug_mean_baseline"
        ),
        "neural": evaluate_predictions(test_output, "pred_auc_neural"),
        "blended": evaluate_predictions(test_output, "pred_auc_blended"),
    }
    uncertainty = uncertainty_diagnostics(test_output)
    paired_bootstrap = paired_patient_bootstrap(test_output)

    all_predictions = []
    all_latent = []
    for model in models:
        predicted, latent = predict_all_drugs(
            model, prepared.features_scaled, device=device
        )
        all_predictions.append(predicted)
        all_latent.append(latent)
    all_predictions_array = np.stack(all_predictions, axis=0)
    all_latent_array = np.stack(all_latent, axis=0)
    standardized_mean = all_predictions_array.mean(axis=0)
    standardized_std = all_predictions_array.std(axis=0)
    latent_mean = all_latent_array.mean(axis=0)

    first_batch = select_first_batch(
        args.features, test_output, batch_size=args.first_batch_size
    )
    first_batch_predictions = build_first_batch_predictions(
        selected=first_batch,
        prepared=prepared,
        all_standardized_mean=standardized_mean,
        all_standardized_std=standardized_std,
        blend_alpha=blend_alpha,
    )
    latent_table = patient_latent_table(prepared, latent_mean)

    columns_to_drop = ["patient_index", "drug_index", "target_z"]
    test_output.drop(columns=columns_to_drop).to_csv(
        args.out_dir / "heldout_test_predictions.csv", index=False
    )
    first_batch.to_csv(args.out_dir / "first_batch_patients.csv", index=False)
    first_batch_predictions.to_csv(
        args.out_dir / "first_batch_drug_predictions.csv", index=False
    )
    latent_table.to_csv(args.out_dir / "patient_latent_states.csv", index=False)

    for model, diagnostic in zip(models, diagnostics):
        checkpoint = checkpoint_payload(
            model=model,
            prepared=prepared,
            diagnostics=diagnostic,
            blend_alpha=blend_alpha,
        )
        torch.save(checkpoint, model_dir / f"model_seed_{diagnostic['seed']}.pt")

    model_improves_ranking = (
        test_metrics["blended"]["median_patient_spearman"]
        > test_metrics["drug_mean_baseline"]["median_patient_spearman"]
    )
    model_improves_mae = (
        test_metrics["blended"]["mae"]
        < test_metrics["drug_mean_baseline"]["mae"]
    )
    split_safe_rna_pca = "split_safe" in args.features.name.lower()
    limitations = [
        "bulk multi-modal baseline features, not a single-cell state",
        "single-drug ex-vivo AUC, not a post-treatment transcriptome",
        "no combination synergy, dose schedule, or clinical outcome head",
        "internal BeatAML holdout only; external and prospective validation required",
    ]
    if not split_safe_rna_pca:
        limitations.append(
            "upstream canonical RNA PCA was fitted on the full feature cohort; use the "
            "split-safe feature builder for publication-grade evaluation"
        )
    metrics: dict[str, object] = {
        "run_name": "BeatAML virtual patient-state pilot",
        "scope": "baseline patient state plus single-drug ex-vivo AUC",
        "research_use_only": True,
        "dataset": {
            "feature_table": str(args.features.resolve()),
            "split_safe_rna_pca": split_safe_rna_pca,
            "feature_patients": len(prepared.patient_ids),
            "response_patients": int(prepared.responses["patient_id"].nunique()),
            "response_rows": len(prepared.responses),
            "features": len(prepared.feature_columns),
            "drugs": len(prepared.drug_ids),
        },
        "patient_split": {
            "seed": args.split_seed,
            "train_patients": len(prepared.split.train),
            "validation_patients": len(prepared.split.validation),
            "test_patients": len(prepared.split.test),
            "patient_sets_disjoint": True,
        },
        "ensemble_seeds": seeds,
        "parameter_count_per_model": diagnostics[0]["parameter_count"],
        "training": diagnostics,
        "blend_alpha": blend_alpha,
        "test_metrics": test_metrics,
        "uncertainty": uncertainty,
        "paired_patient_bootstrap": paired_bootstrap,
        "incremental_value_verdict": {
            "blended_improves_median_patient_ranking_vs_drug_mean": model_improves_ranking,
            "blended_improves_mae_vs_drug_mean": model_improves_mae,
            "claim": (
                "patient-specific incremental signal detected on the internal held-out set"
                if model_improves_ranking and model_improves_mae
                else "incremental value is incomplete; do not claim superiority"
            ),
        },
        "limitations": limitations,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_model_card(args.out_dir / "MODEL_CARD.md", metrics)

    print("[test]")
    for name, values in test_metrics.items():
        print(
            f"  {name}: MAE={values['mae']:.3f} RMSE={values['rmse']:.3f} "
            f"median_patient_spearman={values['median_patient_spearman']:.3f} "
            f"top5_overlap={values['mean_top_5_overlap']:.3f}"
        )
    print(f"[output] {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
