"""Secondary analysis: Path B (Set Transformer) vs the existing
factorized combo predictor on 2-drug sets.

The existing pipeline computes:
    combo_auc(p, d1, d2) = 0.5*(AUC_d1(p) + AUC_d2(p))
                         + synergy_loewe(d1, d2)
                         - mech_prior_scale * mech_score(p, d1, d2)

Path B predicts the same quantity zero-shot from single-drug-only training.

If path B is theoretically sound, the two should be:
  - Positively correlated on random (patient, drug-pair) samples
  - Substantially different in ranking (else path B offers nothing new)
  - Both agree on direction for known clinical pairs (Gilt+Ven for FLT3-mut)

Writes `runs/set_drug_predictor_viability/pathB_vs_existing.csv` +
summary statistics in the viability JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from combo_val.combo.set_drug_predictor import SetDrugPredictor, SetDrugPredictorConfig
from combo_val.combo.mechanism_prior import compute_combo_mech_scores


def compare(
    set_model_ckpt: Path = Path("runs/set_drug_predictor/final_model.pt"),
    combo_npz: Path = Path("runs/combo_predictor/combo_auc_predictions.npz"),
    patient_features: Path = Path("data/canonical/beataml_patient_features.csv"),
    n_patient_samples: int = 50,
    n_pairs_per_patient: int = 40,
    out_dir: Path = Path("runs/set_drug_predictor_viability"),
    random_state: int = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    # --- Load path B ---
    ckpt = torch.load(set_model_ckpt, weights_only=False, map_location="cpu")
    model_cfg = SetDrugPredictorConfig(**{
        k: v for k, v in ckpt["cfg"].items()
        if k in SetDrugPredictorConfig.__dataclass_fields__
    })
    model = SetDrugPredictor(
        n_drugs=ckpt["n_drugs"],
        n_patient_features=ckpt["n_patient_features"],
        cfg=model_cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pf_raw = pd.read_csv(patient_features).set_index("patient_id")
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    pf_scaled = (pf_raw.values - mean) / np.where(scale > 0, scale, 1.0)
    pf_scaled_df = pd.DataFrame(pf_scaled, index=pf_raw.index, columns=pf_raw.columns)

    drug_vocab = ckpt["drug_vocab"]
    drug_to_int = ckpt["drug_to_int"]

    # --- Load existing combo_predictor npz ---
    npz = np.load(combo_npz, allow_pickle=True)
    existing_auc = npz["combo_auc"]                  # (n_pat, n_drugs, n_drugs)
    existing_pids = list(npz["patient_ids"])
    existing_drugs = list(npz["drug_names"])
    # Index mapping: existing_drugs → our drug_to_int
    existing_drug_idx = {d: i for i, d in enumerate(existing_drugs)}

    # --- Sample patients + pairs ---
    valid_pids = [p for p in pf_raw.index if int(p) in {int(x) for x in existing_pids}]
    pids = rng.choice(valid_pids,
                      size=min(n_patient_samples, len(valid_pids)),
                      replace=False)

    records = []
    device = torch.device("cpu")
    for pid in pids:
        pf_tensor = torch.tensor(
            pf_scaled_df.loc[pid].values, dtype=torch.float32
        ).unsqueeze(0)
        # Find pid index in existing_pids array
        existing_pi = existing_pids.index(int(pid)) if int(pid) in existing_pids else None
        if existing_pi is None:
            continue
        n_drugs = len(drug_vocab)
        for _ in range(n_pairs_per_patient):
            d1, d2 = rng.choice(n_drugs, size=2, replace=False)
            set_pred = float(model.predict_set(
                [[int(d1), int(d2)]], pf_tensor, device=device,
            ).item())
            # Existing combo_predictor on same pair (indices already aligned:
            # both use sorted BeatAML drug vocab)
            d1_name = drug_vocab[int(d1)]
            d2_name = drug_vocab[int(d2)]
            i1 = existing_drug_idx.get(d1_name)
            i2 = existing_drug_idx.get(d2_name)
            if i1 is None or i2 is None:
                continue
            existing_pred = float(existing_auc[existing_pi, i1, i2])
            records.append({
                "patient_id": int(pid),
                "d1": d1_name, "d2": d2_name,
                "pathB_setxformer_auc": round(set_pred, 2),
                "existing_factorized_auc": round(existing_pred, 2),
                "delta": round(set_pred - existing_pred, 2),
            })
    df = pd.DataFrame(records)
    out_csv = out_dir / "pathB_vs_existing.csv"
    df.to_csv(out_csv, index=False)

    pearson_r = float(pearsonr(df["pathB_setxformer_auc"], df["existing_factorized_auc"]).statistic)
    spearman_r = float(spearmanr(df["pathB_setxformer_auc"], df["existing_factorized_auc"]).correlation)
    mean_abs_delta = float(np.abs(df["delta"]).mean())
    std_delta = float(df["delta"].std())

    summary = {
        "n_pairs_compared": len(df),
        "pearson_r": round(pearson_r, 3),
        "spearman_r": round(spearman_r, 3),
        "mean_absolute_delta_AUC": round(mean_abs_delta, 2),
        "std_delta_AUC": round(std_delta, 2),
        "pathB_mean_auc": round(float(df["pathB_setxformer_auc"].mean()), 2),
        "existing_mean_auc": round(float(df["existing_factorized_auc"].mean()), 2),
    }
    print("[pathB-vs-existing]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  output: {out_csv}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-ckpt", default="runs/set_drug_predictor/final_model.pt")
    ap.add_argument("--combo-npz", default="runs/combo_predictor/combo_auc_predictions.npz")
    ap.add_argument("--out", default="runs/set_drug_predictor_viability")
    args = ap.parse_args()
    compare(
        set_model_ckpt=Path(args.set_ckpt),
        combo_npz=Path(args.combo_npz),
        out_dir=Path(args.out),
    )
