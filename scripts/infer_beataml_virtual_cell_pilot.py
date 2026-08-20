#!/usr/bin/env python3
"""Run batch inference with the BeatAML virtual patient-state pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from combo_val.virtual_cell.beataml_pilot import (  # noqa: E402
    load_checkpoint,
    normalize_identifier,
    predict_all_drugs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-features", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "runs/virtual_cell_beataml_pilot/model",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_paths = sorted(args.model_dir.glob("model_seed_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No model_seed_*.pt checkpoints found in {args.model_dir}")
    device = torch.device(args.device)
    models_and_payloads = [load_checkpoint(path, device) for path in checkpoint_paths]
    first_payload = models_and_payloads[0][1]
    feature_columns = list(first_payload["feature_columns"])
    drug_ids = list(first_payload["drug_ids"])
    for _, payload in models_and_payloads[1:]:
        if list(payload["feature_columns"]) != feature_columns:
            raise ValueError("Ensemble checkpoints have inconsistent feature schemas")
        if list(payload["drug_ids"]) != drug_ids:
            raise ValueError("Ensemble checkpoints have inconsistent drug schemas")

    patient_frame = pd.read_csv(args.patient_features)
    if "patient_id" not in patient_frame:
        raise ValueError("Input requires a patient_id column")
    missing = [column for column in feature_columns if column not in patient_frame]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} model features: {missing[:10]}")
    patient_frame = patient_frame.copy()
    patient_frame["patient_id"] = patient_frame["patient_id"].map(normalize_identifier)
    numeric = patient_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Input contains missing/non-numeric model features: {bad[:10]}")

    feature_mean = np.asarray(first_payload["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(first_payload["feature_scale"], dtype=np.float32)
    drug_mean = np.asarray(first_payload["drug_mean"], dtype=np.float32)
    drug_scale = np.asarray(first_payload["drug_scale"], dtype=np.float32)
    drug_train_count = np.asarray(first_payload["drug_train_count"], dtype=int)
    blend_alpha = float(first_payload.get("blend_alpha") or 0.0)
    scaled = ((numeric.to_numpy(dtype=np.float32) - feature_mean) / feature_scale).astype(
        np.float32
    )

    standardized_members = []
    latent_members = []
    for model, _ in models_and_payloads:
        standardized, latent = predict_all_drugs(model, scaled, device=device)
        standardized_members.append(standardized)
        latent_members.append(latent)
    standardized_array = np.stack(standardized_members, axis=0)
    latent_array = np.stack(latent_members, axis=0)
    standardized_mean = standardized_array.mean(axis=0)
    standardized_std = standardized_array.std(axis=0)
    latent_mean = latent_array.mean(axis=0)

    output_rows: list[dict[str, object]] = []
    for patient_index, patient_id in enumerate(patient_frame["patient_id"]):
        neural = drug_mean + drug_scale * standardized_mean[patient_index]
        blended = drug_mean + blend_alpha * (neural - drug_mean)
        uncertainty = blend_alpha * drug_scale * standardized_std[patient_index]
        order = np.argsort(blended)
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for drug_index, drug_id in enumerate(drug_ids):
            output_rows.append(
                {
                    "patient_id": patient_id,
                    "drug_id": drug_id,
                    "pred_auc_drug_mean_baseline": float(drug_mean[drug_index]),
                    "pred_auc_neural": float(neural[drug_index]),
                    "pred_auc_blended": float(blended[drug_index]),
                    "ensemble_uncertainty_auc": float(uncertainty[drug_index]),
                    "predicted_sensitivity_rank": int(ranks[drug_index]),
                    "drug_training_observations": int(drug_train_count[drug_index]),
                    "support_flag": (
                        "supported" if drug_train_count[drug_index] >= 20 else "low_support"
                    ),
                    "use_scope": "research_only_not_a_treatment_recommendation",
                }
            )

    latent_output: dict[str, object] = {"patient_id": patient_frame["patient_id"]}
    for index in range(latent_mean.shape[1]):
        latent_output[f"latent_{index + 1:02d}"] = latent_mean[:, index]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).sort_values(
        ["patient_id", "predicted_sensitivity_rank"]
    ).to_csv(args.out_dir / "drug_predictions.csv", index=False)
    pd.DataFrame(latent_output).to_csv(
        args.out_dir / "patient_latent_states.csv", index=False
    )
    print(f"patients={len(patient_frame)}")
    print(f"drugs={len(drug_ids)}")
    print(f"ensemble_members={len(models_and_payloads)}")
    print(f"blend_alpha={blend_alpha:.4f}")
    print(f"output={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
