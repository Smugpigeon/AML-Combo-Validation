"""Train Set Transformer v3 — fine-tune from v2 with Bliss-IDA pair augmentation.

Starts from the v2 checkpoint (stability-fixed, 5/5 folds ρ=0.691 ± 0.011)
and adds synthetic pair supervision derived from Bliss independence. Uses
joint loss:

    loss = w_single * MSE(pred_single, auc_true)
         + w_pair   * MSE(pred_pair,   auc_bliss)

The pair labels are THEORETICAL (Bliss formula applied to the patient's
observed single-drug AUCs), not wet-lab measurements. They teach ST that
pairs should follow the additive-independence baseline; real synergy beyond
that still requires pair-level ground truth.

Output: runs/set_drug_predictor_v3_bliss/final_model.pt
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from combo_val.combo.bliss_augmentation import BlissAugmentedPairDataset
from combo_val.combo.set_drug_predictor import (
    PatientDrugSetDataset,
    SetDrugPredictor,
    SetDrugPredictorConfig,
    collate_variable_arity,
)


# Training config — fine-tune: lower LR, fewer epochs, start from v2 weights
V2_CHECKPOINT = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
V3_OUT = Path("runs/set_drug_predictor_v3_bliss")
V3_CHECKPOINT = V3_OUT / "final_model.pt"

FINETUNE_LR = 1e-4           # 20× smaller than v2's 2e-3
FINETUNE_EPOCHS = 8
N_PAIRS_PER_PATIENT = 100    # 487 × 100 = ~49K synthetic pairs
BATCH_SIZE = 512
WEIGHT_SINGLE = 1.0
WEIGHT_PAIR = 1.0            # equal weighting (both MSE in AUC² units)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    V3_OUT.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    print(f"[v3] device={device}")

    # ---- Load v2 checkpoint as starting point ----
    if not V2_CHECKPOINT.exists():
        raise FileNotFoundError(f"v2 checkpoint missing at {V2_CHECKPOINT}")
    ckpt = torch.load(V2_CHECKPOINT, weights_only=False, map_location=device)
    print(f"[v3] loaded v2 checkpoint: {V2_CHECKPOINT}")

    # Re-hydrate config from v2 cfg, override fine-tune params
    from dataclasses import fields as _fields
    cfg_v2 = {f.name: ckpt["cfg"][f.name] for f in _fields(SetDrugPredictorConfig)
              if f.name in ckpt["cfg"]}
    cfg_v2.update({
        "lr": FINETUNE_LR,
        "max_epochs": FINETUNE_EPOCHS,
        "warmup_steps": 0,       # already warmed up in v2
        "n_seeds": 1,            # fine-tune single run
        "min_epochs": FINETUNE_EPOCHS,
        "patience": FINETUNE_EPOCHS,
    })
    cfg = SetDrugPredictorConfig(**cfg_v2)

    # ---- Load data ----
    pf = pd.read_csv("data/canonical/beataml_patient_features.csv").set_index("patient_id")
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    # Use the saved scaler so features match v2's
    feat_scaled = pd.DataFrame(
        (pf.values - scaler.mean_) / np.where(scaler.scale_ > 0, scaler.scale_, 1.0),
        index=pf.index, columns=pf.columns,
    )

    long_df = pd.read_csv("data/canonical/beataml_drug_response_long.csv").dropna(subset=["auc"])
    long_df["patient_id"] = long_df["patient_id"].astype(int)
    long_df = long_df[long_df["patient_id"].isin(feat_scaled.index)].reset_index(drop=True)

    drug_vocab = ckpt["drug_vocab"]
    drug_to_int = ckpt["drug_to_int"]

    print(f"[v3] patients={feat_scaled.shape[0]} drugs={len(drug_vocab)} "
          f"single_samples={len(long_df)}")

    # ---- Build datasets ----
    t0 = time.time()
    single_ds = PatientDrugSetDataset(long_df, feat_scaled, drug_to_int)
    pair_ds = BlissAugmentedPairDataset(
        long_df, feat_scaled, drug_to_int,
        n_pairs_per_patient=N_PAIRS_PER_PATIENT,
        random_state=42,
    )
    print(f"[v3] built datasets in {time.time()-t0:.1f}s: "
          f"{len(single_ds)} singles + {len(pair_ds)} Bliss pairs = {len(single_ds) + len(pair_ds)} total")

    combined = ConcatDataset([single_ds, pair_ds])
    loader = DataLoader(
        combined, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_variable_arity, num_workers=0,
    )

    # ---- Instantiate model + load v2 weights ----
    model = SetDrugPredictor(
        n_drugs=ckpt["n_drugs"],
        n_patient_features=ckpt["n_patient_features"],
        cfg=cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[v3] model loaded from v2 weights, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # ---- Fine-tune ----
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    import torch.nn.functional as F

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_samples = 0
        for batch in loader:
            drug_ids = batch["drug_ids"].to(device)
            drug_mask = batch["drug_mask"].to(device)
            pf_b = batch["patient_features"].to(device)
            auc = batch["auc"].to(device)

            pred = model(drug_ids, drug_mask, pf_b)
            loss = F.mse_loss(pred, auc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item() * auc.shape[0]
            n_samples += auc.shape[0]

        mean_mse = total_loss / n_samples
        print(f"[v3] epoch {epoch + 1}/{FINETUNE_EPOCHS}  "
              f"mean MSE = {mean_mse:.2f}  MAE ≈ {mean_mse**0.5:.2f}  "
              f"[{time.time()-t0:.1f}s]")

    # ---- Save checkpoint ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "drug_vocab": drug_vocab,
        "drug_to_int": drug_to_int,
        "feature_cols": ckpt["feature_cols"],
        "scaler_mean": ckpt["scaler_mean"],
        "scaler_scale": ckpt["scaler_scale"],
        "n_patient_features": ckpt["n_patient_features"],
        "n_drugs": ckpt["n_drugs"],
        "architecture": "SetTransformer-v3-BlissIDA",
        "training_notes": {
            "base": "v2 (5/5 folds ρ=0.691)",
            "augmentation": f"Bliss-IDA synthetic pairs, {N_PAIRS_PER_PATIENT}/patient",
            "finetune_lr": FINETUNE_LR,
            "finetune_epochs": FINETUNE_EPOCHS,
            "n_singles": len(single_ds),
            "n_bliss_pairs": len(pair_ds),
        },
    }, V3_CHECKPOINT)
    print(f"[v3] saved {V3_CHECKPOINT}  size = {V3_CHECKPOINT.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
