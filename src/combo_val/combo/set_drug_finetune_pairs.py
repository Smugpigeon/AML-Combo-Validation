"""Path B v3 — fine-tune Set Transformer v2 on DrugComb ALMANAC-HL-60 pairs.

Route 3 from the kit-roadmap: teach the Set Transformer real pair behavior
using the 186 strict-pair samples we have. No wet-lab cost, no synergy
theory. Pure data-driven correction of v2's pair predictions.

Design (see docs/pathB_v3_pair_finetune.md for full rationale):

  1. HL-60 pseudo-patient feature vector:
     - RNA PCs = 0 (standardized BeatAML training-set mean, 'average AML')
     - Mutation flags = 0 everywhere except mut_NRAS=1 (HL-60 NRAS Q61L known)
     - Clinical = BeatAML medians → 0 after the saved StandardScaler

  2. Target AUC reconstruction (DrugComb gives synergy_loewe, not combo AUC):
     - Freeze v2 at start, compute singleton AUC A_d for each of 19 drugs
     - For each pair (d1, d2) with observed synergy s:
           target_combo_auc = 0.5 * (A_d1 + A_d2) + s
     - Loewe convention: s < 0 → synergistic → target below additive mean

  3. Joint loss (prevent catastrophic forgetting of singletons):
     L = MSE_pair + lambda * MSE_singleton_anchor
     where MSE_singleton_anchor keeps the 19 singleton predictions pinned
     to their frozen v2 values.

  4. Training regime:
     - Low LR (1e-4, 10x less than v2 training)
     - Short duration (max 20 epochs, patience 5)
     - Small batch (16) — full batch fits easily in memory
     - Shared optimizer over ALL model weights (we want pair loss to back-
       propagate through the drug embeddings + ISAB attention)

Known limitations (report these honestly in validation):
  - HL-60 is FLT3-wt → cannot teach FLT3-mut-specific pair rules directly.
  - Only 19 drugs involved, 5 are not in the 20-drug clinical filter
    (Bortezomib, Lapatinib, Vandetanib, etc.) → pair rules transfer only
    through drug embedding similarity, not explicit supervision.
  - HL-60 pseudo-features approximate as 'average AML' — pair rules may
    not transfer to patients with distinct RNA signatures.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from combo_val.combo.set_drug_inference import SetDrugInference
from combo_val.combo.set_drug_predictor import (
    SetDrugPredictor,
    SetDrugPredictorConfig,
)


# HL-60 is documented NRAS Q61L in COSMIC / CCLE / ATCC profiles.
# All other 24 genes in our 25-gene panel are wt in HL-60.
HL60_KNOWN_MUTATIONS = {"NRAS"}


@dataclass(frozen=True)
class FineTuneConfig:
    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 16
    singleton_anchor_weight: float = 1.0
    random_state: int = 42
    # Val split (on 186 pairs)
    val_frac: float = 0.15
    # Source paths
    v2_checkpoint: Path = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
    drugcomb_pairs: Path = Path("data/canonical/drugcomb_aml_pairs.csv")
    features_path: Path = Path("data/canonical/beataml_patient_features.csv")
    out_dir: Path = Path("runs/set_drug_predictor_v3_pair_ft")


def build_hl60_pseudo_features(
    feature_cols: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_medians: dict[str, float] | None = None,
    mutations: set[str] = HL60_KNOWN_MUTATIONS,
) -> np.ndarray:
    """Construct a 104-dim 'pseudo-patient' vector for HL-60.

    Returns a vector ALREADY STANDARDIZED — ready to feed into the model
    (which was trained on standardized features).
    """
    feat_cols_list = list(feature_cols)
    # Raw feature vector (pre-standardization)
    raw = np.zeros(len(feat_cols_list), dtype=np.float32)

    # Fill clinical medians (or zeros if none supplied) for clin_* fields
    if feature_medians is not None:
        for i, col in enumerate(feat_cols_list):
            if col in feature_medians:
                raw[i] = float(feature_medians[col])

    # Override mutation flags
    for i, col in enumerate(feat_cols_list):
        if col.startswith("mut_"):
            gene = col[len("mut_"):]
            raw[i] = 1.0 if gene in mutations else 0.0

    # Standardize
    safe_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    standardized = (raw - scaler_mean) / safe_scale
    return standardized.astype(np.float32)


def build_pair_training_dataset(
    cfg: FineTuneConfig, v2_predictor: SetDrugInference,
    hl60_standardized: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    """Build the 186-pair training data (HL60_features x drug_pair x target_auc).

    Returns:
        (pair_df, singleton_auc_by_drug, hl60_features)
    """
    pairs = pd.read_csv(cfg.drugcomb_pairs)
    pairs = pairs.dropna(subset=["drug1_id", "drug2_id", "synergy_loewe"]).reset_index(drop=True)
    drugs_in_pairs = sorted(set(pairs["drug1_id"]).union(pairs["drug2_id"]))

    # Keep only drugs that exist in the v2 drug vocab
    in_vocab = [d for d in drugs_in_pairs if d in v2_predictor.drug_to_int]
    dropped = [d for d in drugs_in_pairs if d not in v2_predictor.drug_to_int]
    if dropped:
        print(f"[v3] {len(dropped)} drugs in pairs but not in v2 vocab "
              f"(dropped): {dropped}")
    pairs = pairs[
        pairs["drug1_id"].isin(in_vocab) & pairs["drug2_id"].isin(in_vocab)
    ].reset_index(drop=True)

    # Freeze singleton predictions for HL-60 pseudo-patient using v2
    singleton_auc: dict[str, float] = {}
    for d in in_vocab:
        singleton_auc[d] = v2_predictor.predict_set_for_patient(
            hl60_standardized_to_raw(hl60_standardized, v2_predictor),
            [d],
        )
    # NOTE: predict_set_for_patient internally re-standardizes using the
    # saved scaler, so we must pass the RAW feature vector. Convert back.

    # Compute target AUC: 0.5*(A_d1 + A_d2) + synergy_loewe
    pairs["A_d1"] = pairs["drug1_id"].map(singleton_auc)
    pairs["A_d2"] = pairs["drug2_id"].map(singleton_auc)
    pairs["target_combo_auc"] = (
        0.5 * (pairs["A_d1"] + pairs["A_d2"]) + pairs["synergy_loewe"]
    )
    return pairs, singleton_auc, hl60_standardized


def hl60_standardized_to_raw(std_vec: np.ndarray,
                              predictor: SetDrugInference) -> np.ndarray:
    """Invert the standardization so predict_set_for_patient can re-apply it."""
    mean = np.asarray(predictor.scaler_mean, dtype=np.float32)
    scale = np.asarray(predictor.scaler_scale, dtype=np.float32)
    safe_scale = np.where(scale > 0, scale, 1.0)
    return (std_vec * safe_scale + mean).astype(np.float32)


def finetune_on_pairs(cfg: FineTuneConfig | None = None) -> dict:
    cfg = cfg or FineTuneConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    t0 = time.time()

    # --- 1. Load v2 predictor ---
    if not cfg.v2_checkpoint.exists():
        raise FileNotFoundError(
            f"v2 checkpoint not found at {cfg.v2_checkpoint}. "
            "Run scripts/train_set_transformer_v2_stable.py first."
        )
    v2 = SetDrugInference(cfg.v2_checkpoint, device="cpu")
    print(f"[v3] loaded v2 checkpoint ({len(v2.drug_vocab)} drugs, "
          f"{len(v2.feature_cols)} features)")

    # --- 2. HL-60 pseudo-patient features ---
    # Read per-feature medians from BeatAML training set
    pf_train = pd.read_csv(cfg.features_path).set_index("patient_id")
    pf_medians = pf_train.median(numeric_only=True).to_dict()

    hl60_std = build_hl60_pseudo_features(
        feature_cols=v2.feature_cols,
        scaler_mean=v2.scaler_mean,
        scaler_scale=v2.scaler_scale,
        feature_medians=pf_medians,
    )
    hl60_raw = hl60_standardized_to_raw(hl60_std, v2)
    print(f"[v3] HL-60 pseudo-features built: mut_NRAS=1, std vector"
          f" mean={hl60_std.mean():.3f} std={hl60_std.std():.3f}")

    # --- 3. Build pair training dataset ---
    pairs_df, singleton_auc, _ = build_pair_training_dataset(cfg, v2, hl60_std)
    print(f"[v3] {len(pairs_df)} pair samples; "
          f"target AUC mean={pairs_df['target_combo_auc'].mean():.2f}, "
          f"std={pairs_df['target_combo_auc'].std():.2f}")

    # Train/val split (random by pair)
    rng = np.random.default_rng(cfg.random_state)
    n_val = max(1, int(len(pairs_df) * cfg.val_frac))
    shuffled = rng.permutation(len(pairs_df))
    val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]
    train_pairs = pairs_df.iloc[train_idx].reset_index(drop=True)
    val_pairs = pairs_df.iloc[val_idx].reset_index(drop=True)

    # --- 4. Rehydrate the v2 model into a trainable state ---
    device = torch.device("cpu")
    model_cfg = v2.model.cfg if hasattr(v2.model, "cfg") else None
    # v2.model is already constructed and loaded — re-use it
    model = v2.model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Prebuild tensors
    hl60_std_t = torch.tensor(hl60_std, dtype=torch.float32, device=device)

    def _forward_set(drug_ids_list: list[list[int]]) -> torch.Tensor:
        """Forward a batch of drug-id sets given the HL-60 pseudo-patient."""
        B = len(drug_ids_list)
        N_max = max(len(s) for s in drug_ids_list)
        drug_ids = torch.zeros(B, N_max, dtype=torch.long, device=device)
        drug_mask = torch.zeros(B, N_max, dtype=torch.float32, device=device)
        for i, s in enumerate(drug_ids_list):
            drug_ids[i, :len(s)] = torch.tensor(
                [d + 1 for d in s], dtype=torch.long, device=device
            )
            drug_mask[i, :len(s)] = 1.0
        pf = hl60_std_t.unsqueeze(0).expand(B, -1)
        return model(drug_ids, drug_mask, pf)

    # Frozen singleton-anchor targets (from v2's predictions)
    anchor_drugs = list(singleton_auc.keys())
    anchor_ids = [[v2.drug_to_int[d]] for d in anchor_drugs]
    anchor_targets = torch.tensor(
        [singleton_auc[d] for d in anchor_drugs],
        dtype=torch.float32, device=device,
    )

    # --- 5. Fine-tuning loop ---
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        # Shuffle train pairs
        order = rng.permutation(len(train_pairs))
        epoch_pair_loss = 0.0
        epoch_anchor_loss = 0.0
        n_pair_steps = 0
        for start in range(0, len(order), cfg.batch_size):
            batch_idx = order[start:start + cfg.batch_size]
            batch = train_pairs.iloc[batch_idx]
            drug_sets = [
                [v2.drug_to_int[r["drug1_id"]], v2.drug_to_int[r["drug2_id"]]]
                for _, r in batch.iterrows()
            ]
            targets = torch.tensor(
                batch["target_combo_auc"].values, dtype=torch.float32, device=device,
            )
            pred = _forward_set(drug_sets)
            pair_loss = F.mse_loss(pred, targets)

            # Anchor (all 19 singletons every step — tiny batch)
            anchor_pred = _forward_set(anchor_ids)
            anchor_loss = F.mse_loss(anchor_pred, anchor_targets)

            loss = pair_loss + cfg.singleton_anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_pair_loss += pair_loss.item() * len(batch)
            epoch_anchor_loss += anchor_loss.item() * len(batch)
            n_pair_steps += len(batch)

        # Validation
        model.eval()
        with torch.no_grad():
            val_sets = [
                [v2.drug_to_int[r["drug1_id"]], v2.drug_to_int[r["drug2_id"]]]
                for _, r in val_pairs.iterrows()
            ]
            val_targets = torch.tensor(
                val_pairs["target_combo_auc"].values, dtype=torch.float32,
            )
            val_pred = _forward_set(val_sets).cpu()
            val_mse = F.mse_loss(val_pred, val_targets).item()

            anchor_pred_val = _forward_set(anchor_ids).cpu()
            anchor_mae = torch.abs(anchor_pred_val - anchor_targets.cpu()).mean().item()

        entry = {
            "epoch": epoch + 1,
            "train_pair_mse": epoch_pair_loss / max(n_pair_steps, 1),
            "train_anchor_mse": epoch_anchor_loss / max(n_pair_steps, 1),
            "val_pair_mse": val_mse,
            "val_anchor_mae": anchor_mae,
        }
        history.append(entry)
        print(f"[v3] epoch {epoch+1:2d}  "
              f"train pair MSE {entry['train_pair_mse']:7.1f}  "
              f"val pair MSE {val_mse:7.1f}  "
              f"anchor MAE {anchor_mae:5.2f}")

        if val_mse < best_val - 1e-3:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"[v3] early stop at epoch {epoch+1}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    # --- 6. Save v3 checkpoint ---
    ckpt = torch.load(cfg.v2_checkpoint, weights_only=False, map_location="cpu")
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["architecture"] = "SetTransformer-v3-pair-ft"
    ckpt["finetune_cfg"] = asdict(cfg)
    ckpt["finetune_history"] = history
    ckpt["finetune_best_epoch"] = best_epoch
    ckpt["finetune_best_val_mse"] = best_val
    out_ckpt = cfg.out_dir / "final_model.pt"
    torch.save(ckpt, out_ckpt)

    # Also save the target reconstruction table for reproducibility
    pairs_df.to_csv(cfg.out_dir / "pair_training_targets.csv", index=False)
    with open(cfg.out_dir / "hl60_pseudo_features.json", "w") as f:
        json.dump({
            "feature_cols": list(v2.feature_cols),
            "standardized_vector": hl60_std.tolist(),
            "mutation_flags": list(HL60_KNOWN_MUTATIONS),
        }, f, indent=2)

    elapsed = time.time() - t0
    summary = {
        "v2_checkpoint": str(cfg.v2_checkpoint),
        "v3_checkpoint": str(out_ckpt),
        "n_pair_samples": int(len(pairs_df)),
        "n_train": int(len(train_pairs)),
        "n_val": int(len(val_pairs)),
        "n_drugs_in_pairs": int(len(singleton_auc)),
        "best_epoch": int(best_epoch),
        "best_val_pair_mse": round(best_val, 3),
        "final_anchor_mae": round(history[-1]["val_anchor_mae"], 3),
        "elapsed_s": round(elapsed, 1),
        "target_combo_auc_stats": {
            "mean": round(float(pairs_df["target_combo_auc"].mean()), 2),
            "std": round(float(pairs_df["target_combo_auc"].std()), 2),
            "min": round(float(pairs_df["target_combo_auc"].min()), 2),
            "max": round(float(pairs_df["target_combo_auc"].max()), 2),
        },
    }
    with open(cfg.out_dir / "finetune_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[v3] DONE. best val pair MSE {best_val:.2f} @ epoch {best_epoch}  "
          f"anchor MAE {summary['final_anchor_mae']}  "
          f"elapsed {elapsed:.0f}s → {out_ckpt}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v2-checkpoint",
                    default="runs/set_drug_predictor_v2_stable/final_model.pt")
    ap.add_argument("--pairs", default="data/canonical/drugcomb_aml_pairs.csv")
    ap.add_argument("--out", default="runs/set_drug_predictor_v3_pair_ft")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = FineTuneConfig(
        v2_checkpoint=Path(args.v2_checkpoint),
        drugcomb_pairs=Path(args.pairs),
        out_dir=Path(args.out),
        max_epochs=args.max_epochs,
        lr=args.lr,
    )
    finetune_on_pairs(cfg)


if __name__ == "__main__":
    main()
