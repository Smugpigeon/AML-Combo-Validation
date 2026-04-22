"""Baseline A: multi-task MLP for single-drug AUC prediction on BeatAML.

Architecture (compact by design — fits laptop CPU in <5 min):

    patient_features (80-dim)
        │
        ▼
    [Patient MLP]  80 → 128 → patient_emb_dim (64)
        │
        │    [Drug Embedding Table]  165 drugs × drug_emb_dim (64)
        │        │
        ▼        ▼
    ┌──── concat (128) ────┐
    │   [Response Head]     │
    │    128 → 128 → 1      │
    └──────────────────────┘
             │
             ▼
      predicted AUC (raw scale, ~ 0-300)

Loss: MSE on raw AUC. We deliberately stay in raw units so downstream
"best_single = argmin(AUC)" comparisons are done on the same scale as
the DrugComb-derived combo predictor (Week 3).

Evaluation: 5-fold CV by PATIENT (not by measurement) to prevent
leakage — a patient's drug responses are tightly correlated and must
all fall in the same fold.

Primary gate metric: held-out patient-wise mean Spearman ≥ 0.4
  (for each held-out patient, compute Spearman between their predicted
  and observed AUCs across all drugs they have measurements for, then
  average across held-out patients).

Secondary metrics:
  - MAE (raw AUC units)
  - per-drug Spearman (for Week 4 "which drugs are predictable" analysis)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleDrugMLPConfig:
    # architecture
    patient_hidden_dim: int = 128
    patient_emb_dim: int = 64
    drug_emb_dim: int = 64
    head_hidden_dim: int = 128
    dropout: float = 0.2

    # training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 120
    patience: int = 15             # early stopping patience (epochs)
    min_epochs: int = 20
    random_state: int = 42

    # CV
    n_folds: int = 5

    # compute
    device: str = "auto"           # 'auto', 'cpu', 'mps', 'cuda'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatientDrugResponseDataset(Dataset):
    """(patient_features, drug_id_int, auc) triples."""

    def __init__(
        self,
        long_df: pd.DataFrame,
        patient_features: pd.DataFrame,
        drug_id_to_int: dict[str, int],
    ):
        self.long_df = long_df.reset_index(drop=True)
        self.patient_features = patient_features
        self.drug_id_to_int = drug_id_to_int
        self.feat_cols = patient_features.columns.tolist()

    def __len__(self) -> int:
        return len(self.long_df)

    def __getitem__(self, i: int) -> dict:
        row = self.long_df.iloc[i]
        pid = int(row["patient_id"])
        pfeat = self.patient_features.loc[pid].to_numpy(dtype=np.float32)
        drug_int = self.drug_id_to_int[row["drug_id"]]
        auc = float(row["auc"])
        return {
            "patient_features": torch.from_numpy(pfeat),
            "drug_id": torch.tensor(drug_int, dtype=torch.long),
            "auc": torch.tensor(auc, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SingleDrugMLP(nn.Module):
    """Multi-task MLP with patient encoder + drug embedding + response head."""

    def __init__(
        self,
        n_patient_features: int,
        n_drugs: int,
        cfg: SingleDrugMLPConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.patient_mlp = nn.Sequential(
            nn.Linear(n_patient_features, cfg.patient_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.patient_hidden_dim, cfg.patient_emb_dim),
        )

        self.drug_embedding = nn.Embedding(n_drugs, cfg.drug_emb_dim)
        # Small initialization for drug embeddings (reduces dominance of untrained drugs)
        nn.init.normal_(self.drug_embedding.weight, mean=0.0, std=0.1)

        head_in = cfg.patient_emb_dim + cfg.drug_emb_dim
        self.response_head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim // 2, 1),
        )

    def forward(self, patient_features: torch.Tensor, drug_id: torch.Tensor) -> torch.Tensor:
        p = self.patient_mlp(patient_features)
        d = self.drug_embedding(drug_id)
        x = torch.cat([p, d], dim=-1)
        return self.response_head(x).squeeze(-1)

    @torch.no_grad()
    def predict_all_drugs_for_patient(
        self, patient_features: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Batch-predict AUC for all drugs for one patient.

        patient_features: (n_features,) single patient.
        Returns: (n_drugs,) AUC predictions.
        """
        self.eval()
        n_drugs = self.drug_embedding.num_embeddings
        pf = patient_features.to(device).unsqueeze(0).expand(n_drugs, -1)
        dids = torch.arange(n_drugs, device=device)
        return self(pf, dids).cpu()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        pf = batch["patient_features"].to(device)
        did = batch["drug_id"].to(device)
        auc = batch["auc"].to(device)
        pred = model(pf, did)
        loss = F.mse_loss(pred, auc)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * pf.size(0)
        n += pf.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mse, preds, targets, patient_ids, drug_ids)."""
    model.eval()
    preds, targets, pids, dids = [], [], [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        pf = batch["patient_features"].to(device)
        did = batch["drug_id"].to(device)
        auc = batch["auc"].to(device)
        pred = model(pf, did)
        loss = F.mse_loss(pred, auc, reduction="sum")
        total_loss += float(loss.item())
        n += pf.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(auc.cpu().numpy())
        dids.append(did.cpu().numpy())
    return (
        total_loss / max(1, n),
        np.concatenate(preds),
        np.concatenate(targets),
        np.array([]),  # patient_ids filled by caller if needed
        np.concatenate(dids),
    )


def _per_patient_spearman(
    long_df_val: pd.DataFrame, preds: np.ndarray
) -> pd.DataFrame:
    """Per-patient Spearman between predicted and observed AUC across their drugs.

    A patient must have ≥ 5 drug measurements to have a meaningful Spearman.
    Returns DataFrame with columns: patient_id, n_drugs, spearman.
    """
    df = long_df_val.copy().reset_index(drop=True)
    df["pred"] = preds
    rows = []
    for pid, g in df.groupby("patient_id"):
        if len(g) < 5:
            continue
        rho, _ = spearmanr(g["pred"], g["auc"])
        rows.append({"patient_id": pid, "n_drugs": len(g), "spearman": rho})
    return pd.DataFrame(rows)


def _per_drug_spearman(
    long_df_val: pd.DataFrame, preds: np.ndarray
) -> pd.DataFrame:
    """Per-drug Spearman across patients."""
    df = long_df_val.copy().reset_index(drop=True)
    df["pred"] = preds
    rows = []
    for d, g in df.groupby("drug_id"):
        if len(g) < 10:
            continue
        rho, _ = spearmanr(g["pred"], g["auc"])
        rows.append({"drug_id": d, "n_patients": len(g), "spearman": rho})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API: CV training
# ---------------------------------------------------------------------------


def train_with_cv(
    features_df: pd.DataFrame,
    response_long: pd.DataFrame,
    cfg: SingleDrugMLPConfig | None = None,
    out_dir: Path | str | None = None,
) -> dict:
    """Run 5-fold CV by PATIENT. Trains one model per fold; also trains a
    final model on all data for downstream use.

    Returns manifest dict and saves to out_dir:
      - final_model.pt
      - cv_metrics.json           (per-fold + overall)
      - per_patient_spearman.csv  (held-out aggregation across folds)
      - per_drug_spearman.csv
      - predictions_all_patients_all_drugs.csv   (from final_model)
      - scaler.joblib             (StandardScaler on features)
    """
    cfg = cfg or SingleDrugMLPConfig()
    rng = np.random.default_rng(cfg.random_state)
    device = _resolve_device(cfg.device)
    out_dir = Path(out_dir or "runs/baseline_single_drug_mlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Align + build drug vocab ---
    features_df = features_df.copy()
    features_df.index = features_df.index.astype(int)
    response_long = response_long.copy()
    response_long["patient_id"] = response_long["patient_id"].astype(int)

    response_long = response_long[response_long["patient_id"].isin(features_df.index)]
    drug_vocab = sorted(response_long["drug_id"].unique().tolist())
    drug_id_to_int = {d: i for i, d in enumerate(drug_vocab)}

    # --- StandardScaler on patient features ---
    scaler = StandardScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(features_df),
        index=features_df.index,
        columns=features_df.columns,
    )

    # --- K-fold by patient ---
    all_patients = np.array(sorted(features_df.index.tolist()))
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)

    fold_metrics = []
    all_val_preds_rows: list[pd.DataFrame] = []

    print(f"[baseline] device={device}, n_patients={len(all_patients)}, "
          f"n_drugs={len(drug_vocab)}, n_folds={cfg.n_folds}")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_patients)):
        t0 = time.time()
        train_pids = set(all_patients[train_idx])
        val_pids = set(all_patients[val_idx])

        train_long = response_long[response_long["patient_id"].isin(train_pids)]
        val_long = response_long[response_long["patient_id"].isin(val_pids)]

        train_ds = PatientDrugResponseDataset(train_long, feat_scaled, drug_id_to_int)
        val_ds = PatientDrugResponseDataset(val_long, feat_scaled, drug_id_to_int)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        model = SingleDrugMLP(
            n_patient_features=feat_scaled.shape[1],
            n_drugs=len(drug_vocab),
            cfg=cfg,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        best_val_mse = float("inf")
        best_epoch = -1
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            train_loss = _train_one_epoch(model, train_loader, optimizer, device)
            val_mse, val_preds, val_targets, _, _ = _evaluate(model, val_loader, device)
            improved = val_mse < best_val_mse - 1e-4
            if improved:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch >= cfg.min_epochs and patience_counter >= cfg.patience:
                break

        # Load best
        if best_state is not None:
            model.load_state_dict(best_state)
        # Final val eval with best state
        _, val_preds, val_targets, _, _ = _evaluate(model, val_loader, device)
        val_mae = float(np.mean(np.abs(val_preds - val_targets)))

        # Per-patient / per-drug Spearman
        per_patient = _per_patient_spearman(val_long, val_preds)
        per_drug = _per_drug_spearman(val_long, val_preds)
        mean_per_patient_rho = float(per_patient["spearman"].mean()) if len(per_patient) else np.nan
        median_per_drug_rho = float(per_drug["spearman"].median()) if len(per_drug) else np.nan

        elapsed = time.time() - t0
        print(
            f"  Fold {fold_idx + 1}/{cfg.n_folds}  "
            f"best_epoch={best_epoch:3d}  "
            f"val MSE={best_val_mse:7.1f}  "
            f"val MAE={val_mae:6.2f}  "
            f"per-patient ρ={mean_per_patient_rho:.3f}  "
            f"median per-drug ρ={median_per_drug_rho:.3f}  "
            f"[{elapsed:5.1f}s]"
        )

        fold_metrics.append({
            "fold": fold_idx + 1,
            "best_epoch": best_epoch,
            "val_mse": best_val_mse,
            "val_mae": val_mae,
            "mean_per_patient_spearman": mean_per_patient_rho,
            "median_per_drug_spearman": median_per_drug_rho,
            "n_val_patients": len(val_pids),
            "n_val_measurements": len(val_long),
            "elapsed_s": elapsed,
        })

        # collect val predictions for overall metric
        val_df_with_pred = val_long.copy().reset_index(drop=True)
        val_df_with_pred["pred"] = val_preds
        val_df_with_pred["fold"] = fold_idx + 1
        all_val_preds_rows.append(val_df_with_pred)

    cv_df = pd.DataFrame(fold_metrics)
    cv_mean_rho = float(cv_df["mean_per_patient_spearman"].mean())
    cv_mean_mae = float(cv_df["val_mae"].mean())

    overall = {
        "cv_folds": fold_metrics,
        "cv_mean_per_patient_spearman": cv_mean_rho,
        "cv_std_per_patient_spearman": float(cv_df["mean_per_patient_spearman"].std()),
        "cv_mean_mae": cv_mean_mae,
        "gate_pass_spearman_0_4": bool(cv_mean_rho >= 0.4),
    }

    # --- Concatenate all held-out predictions ---
    all_val_preds_df = pd.concat(all_val_preds_rows, ignore_index=True)
    all_val_preds_df.to_csv(out_dir / "cv_held_out_predictions.csv", index=False)

    per_patient_all = _per_patient_spearman(
        all_val_preds_df.rename(columns={"pred": "_pred_placeholder"}),
        all_val_preds_df["pred"].to_numpy(),
    )
    per_patient_all.to_csv(out_dir / "per_patient_spearman.csv", index=False)

    per_drug_all = _per_drug_spearman(
        all_val_preds_df.rename(columns={"pred": "_pred_placeholder"}),
        all_val_preds_df["pred"].to_numpy(),
    )
    per_drug_all.to_csv(out_dir / "per_drug_spearman.csv", index=False)

    # --- Final model: train on all data, save checkpoint + full predictions ---
    print(f"[baseline] Training final model on all {len(all_patients)} patients ...")
    full_ds = PatientDrugResponseDataset(response_long, feat_scaled, drug_id_to_int)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)
    final = SingleDrugMLP(
        n_patient_features=feat_scaled.shape[1], n_drugs=len(drug_vocab), cfg=cfg
    ).to(device)
    final_opt = torch.optim.Adam(final.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    avg_best_epoch = int(cv_df["best_epoch"].median())
    for epoch in range(max(avg_best_epoch + 5, cfg.min_epochs)):
        _train_one_epoch(final, full_loader, final_opt, device)

    # Save final
    ckpt_path = out_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": final.state_dict(),
            "cfg": asdict(cfg),
            "drug_vocab": drug_vocab,
            "feature_cols": feat_scaled.columns.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "n_patient_features": feat_scaled.shape[1],
            "n_drugs": len(drug_vocab),
        },
        ckpt_path,
    )

    # --- Full (patient × drug) prediction matrix ---
    print("[baseline] Generating (patient × drug) predictions matrix ...")
    full_pred = np.zeros((len(all_patients), len(drug_vocab)), dtype=np.float32)
    final.eval()
    with torch.no_grad():
        feat_tensor = torch.from_numpy(feat_scaled.loc[all_patients].to_numpy(dtype=np.float32)).to(device)
        for drug_i in range(len(drug_vocab)):
            did = torch.full((len(all_patients),), drug_i, dtype=torch.long, device=device)
            full_pred[:, drug_i] = final(feat_tensor, did).cpu().numpy()

    pred_df = pd.DataFrame(full_pred, index=all_patients, columns=drug_vocab)
    pred_df.index.name = "patient_id"
    pred_df.to_csv(out_dir / "predictions_all_patients_all_drugs.csv")

    # --- Manifest ---
    manifest = {
        "cfg": asdict(cfg),
        "device": str(device),
        "n_patients": int(len(all_patients)),
        "n_drugs": int(len(drug_vocab)),
        "n_features": int(feat_scaled.shape[1]),
        "overall_cv": overall,
        "output_files": {
            "final_model": str(ckpt_path),
            "per_patient_spearman": str(out_dir / "per_patient_spearman.csv"),
            "per_drug_spearman": str(out_dir / "per_drug_spearman.csv"),
            "cv_held_out_predictions": str(out_dir / "cv_held_out_predictions.csv"),
            "prediction_matrix": str(out_dir / "predictions_all_patients_all_drugs.csv"),
        },
    }
    (out_dir / "cv_metrics.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"[baseline] Done. CV mean per-patient ρ = {cv_mean_rho:.3f}  "
          f"(gate ≥ 0.4: {'PASS' if cv_mean_rho >= 0.4 else 'FAIL'})")
    print(f"[baseline] Artifacts: {out_dir}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", default="data/canonical/beataml_patient_features.csv")
    ap.add_argument("--response", default="data/canonical/beataml_drug_response_long.csv")
    ap.add_argument("--out", default="runs/baseline_single_drug_mlp")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    features = pd.read_csv(args.features, index_col="patient_id")
    response = pd.read_csv(args.response)

    cfg = SingleDrugMLPConfig(
        max_epochs=args.epochs,
        n_folds=args.folds,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    train_with_cv(features, response, cfg=cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
