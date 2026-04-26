#!/usr/bin/env python3
"""Phase 1 — Train DrugGIN on BeatAML single-drug AUC predictions.

This is the BASELINE PARITY check: does a GIN over molecular graphs
predict ex-vivo AUC at LEAST as well as the current
nn.Embedding(165, 64) baseline? If YES, GIN is a drop-in upgrade with
chemistry-grounded inductive bias. If NO, the molecular signal is
overwhelmed by noise at n=55826 / 165 drugs and we need to revisit.

Architecture (Phase 1 only — combo head deferred to Phase 3):

    SMILES → DrugGIN → drug_emb [64]
                              │
    patient_features [104] ─┐ │
                            │ ▼
                            └→ MLP (104 + 64 → 256 → 64 → 1) → predicted AUC

Train objective: MSE on AUC scalar; metric: pooled Spearman + Pearson.

5-fold CV at the patient level (no patient leakage between train/val).

Usage:
  python scripts/train_drug_gin.py \
      --smiles-csv data/canonical/drug_smiles.csv \
      --beataml-features data/canonical/beataml_patient_features.csv \
      --beataml-response data/canonical/beataml_drug_response_long.csv \
      --out-dir runs/drug_gin_phase1 \
      --epochs 30 \
      --batch-size 256

Run on GPU if available; CPU-only completes in 1-2 hours.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy torch + PyG
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.data import Batch
    from scipy.stats import pearsonr, spearmanr
except ImportError as e:
    print(f"ERROR: {e}\n  pip install torch torch-geometric rdkit scipy")
    sys.exit(1)

from combo_val.encoders.drug_gin import DrugGIN, DrugGINConfig, count_parameters
from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data


def load_smiles_table(path: Path) -> dict[str, str]:
    """Returns {beataml_drug_id: SMILES} (isomeric preferred, falls back
    to canonical/connectivity, drops empty)."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            smi = r.get("isomeric_smiles") or r.get("canonical_smiles") or ""
            if smi:
                out[r["beataml_drug_id"]] = smi
    return out


def precompute_drug_graphs(smiles_map: dict) -> dict:
    """Pre-build PyG Data per drug — cached in memory."""
    graphs = {}
    for did, smi in smiles_map.items():
        g = smiles_to_molgraph(smi)
        if g is not None:
            graphs[did] = to_pyg_data(g)
    return graphs


class BeatAMLSingleDrugDataset(Dataset):
    """Patient × drug × AUC long-format dataset for single-drug supervision."""

    def __init__(self, response_df: pd.DataFrame, features_df: pd.DataFrame,
                 drug_graphs: dict, drug_id_to_int: dict):
        # Filter rows to (patient_in_features × drug_in_graphs)
        feat_pids = set(features_df["patient_id"].astype(str))
        valid = (
            response_df["patient_id"].astype(str).isin(feat_pids)
            & response_df["drug_id"].isin(drug_graphs)
        )
        self.rows = response_df[valid].reset_index(drop=True)

        # Cache feature vectors per patient
        feat_cols = [c for c in features_df.columns if c != "patient_id"]
        feat_arr = features_df[feat_cols].values.astype(np.float32)
        self.feat_cols = feat_cols
        self.feat_dim = len(feat_cols)
        feat_pid_index = {str(pid): i for i, pid in
                           enumerate(features_df["patient_id"])}
        self.patient_feats = feat_arr
        self.feat_pid_index = feat_pid_index

        self.drug_graphs = drug_graphs
        self.drug_id_to_int = drug_id_to_int

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        pid = str(row["patient_id"])
        did = row["drug_id"]
        return {
            "patient_features": self.patient_feats[self.feat_pid_index[pid]],
            "drug_graph": self.drug_graphs[did],
            "auc": float(row["auc"]),
            "patient_id": pid,
        }


def collate(batch):
    """Custom collate: patient_features tensor + PyG batched drug graphs."""
    pf = torch.tensor(np.stack([b["patient_features"] for b in batch]))
    drug_batch = Batch.from_data_list([b["drug_graph"] for b in batch])
    auc = torch.tensor([b["auc"] for b in batch], dtype=torch.float32)
    pids = [b["patient_id"] for b in batch]
    return pf, drug_batch, auc, pids


class GINSingleDrugRegressor(nn.Module):
    """DrugGIN drug encoder + patient feature MLP → AUC scalar."""

    def __init__(self, n_patient_features: int, drug_emb_dim: int = 64,
                 hidden: int = 256, drug_gin_cfg: DrugGINConfig | None = None):
        super().__init__()
        self.drug_gin = DrugGIN(drug_gin_cfg or DrugGINConfig(drug_emb_dim=drug_emb_dim))
        self.head = nn.Sequential(
            nn.Linear(n_patient_features + drug_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, patient_features, drug_batch):
        drug_emb = self.drug_gin(drug_batch)
        x = torch.cat([patient_features, drug_emb], dim=-1)
        return self.head(x).squeeze(-1)


def patient_level_kfold_split(patient_ids: list[str], n_folds: int = 5,
                                seed: int = 42) -> list[tuple[set, set]]:
    """Returns list of (train_pids, val_pids) per fold."""
    rng = np.random.default_rng(seed)
    pids = sorted(set(patient_ids))
    rng.shuffle(pids)
    folds_pids = np.array_split(pids, n_folds)
    out = []
    for k in range(n_folds):
        val = set(folds_pids[k])
        train = set().union(*(set(folds_pids[j]) for j in range(n_folds) if j != k))
        out.append((train, val))
    return out


def train_one_fold(model, train_loader, val_loader, device, epochs, lr,
                    log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    history = []
    best_val = {"epoch": -1, "pearson": -float("inf"), "spearman": -float("inf")}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for pf, drug_batch, auc, _ in train_loader:
            pf = pf.to(device)
            drug_batch = drug_batch.to(device)
            auc = auc.to(device)
            pred = model(pf, drug_batch)
            loss = loss_fn(pred, auc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(auc)
            n_seen += len(auc)
        train_loss /= n_seen

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for pf, drug_batch, auc, _ in val_loader:
                pf = pf.to(device)
                drug_batch = drug_batch.to(device)
                pred = model(pf, drug_batch).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(auc.numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        pear, _ = pearsonr(val_preds, val_trues)
        spr, _ = spearmanr(val_preds, val_trues)
        val_mse = float(np.mean((val_preds - val_trues) ** 2))

        elapsed = time.time() - t0
        msg = (f"{log_prefix}epoch {epoch:3d}/{epochs}  "
               f"train_mse={train_loss:.3f}  val_mse={val_mse:.3f}  "
               f"val_pear={pear:.3f}  val_sp={spr:.3f}  ({elapsed:.1f}s)")
        print(msg, flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_mse": val_mse,
            "val_pearson": pear, "val_spearman": spr, "elapsed_sec": elapsed,
        })
        if pear > best_val["pearson"]:
            best_val = {"epoch": epoch, "pearson": pear, "spearman": spr,
                        "val_mse": val_mse}

    return history, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-csv", type=Path, required=True)
    ap.add_argument("--beataml-features", type=Path, required=True)
    ap.add_argument("--beataml-response", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--single-fold-only", action="store_true",
                     help="run only fold-0 (faster smoke test)")
    args = ap.parse_args()

    device = (args.device if args.device != "auto"
               else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] SMILES table: {args.smiles_csv}")
    smiles_map = load_smiles_table(args.smiles_csv)
    print(f"[load]   {len(smiles_map)} drugs with SMILES")

    print(f"[load] Patient features: {args.beataml_features}")
    features_df = pd.read_csv(args.beataml_features)
    print(f"[load]   {len(features_df)} patients × {features_df.shape[1] - 1} features")

    print(f"[load] Drug response long: {args.beataml_response}")
    response_df = pd.read_csv(args.beataml_response)
    print(f"[load]   {len(response_df)} (patient, drug, AUC) records")

    print(f"[precompute] Building molecular graphs for {len(smiles_map)} drugs ...")
    drug_graphs = precompute_drug_graphs(smiles_map)
    print(f"[precompute]   {len(drug_graphs)} drugs successfully featurized")

    drug_id_to_int = {d: i for i, d in enumerate(sorted(drug_graphs))}
    dataset = BeatAMLSingleDrugDataset(response_df, features_df, drug_graphs, drug_id_to_int)
    print(f"[dataset] {len(dataset)} usable rows ({len(response_df) - len(dataset)} dropped: "
           f"missing patient features or drug SMILES)")

    folds = patient_level_kfold_split(
        list(features_df["patient_id"].astype(str)),
        n_folds=args.n_folds, seed=args.seed,
    )
    print(f"[cv] {args.n_folds}-fold patient-level CV", flush=True)

    fold_summary = []
    for k, (train_pids, val_pids) in enumerate(folds):
        if args.single_fold_only and k > 0:
            break
        print(f"\n--- Fold {k+1}/{args.n_folds} ---")
        train_idx = [i for i in range(len(dataset))
                      if dataset.rows.iloc[i]["patient_id"] in train_pids]
        val_idx = [i for i in range(len(dataset))
                    if dataset.rows.iloc[i]["patient_id"] in val_pids]
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        print(f"[fold {k+1}] train={len(train_subset)} val={len(val_subset)}", flush=True)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                    shuffle=True, collate_fn=collate, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate, num_workers=0)

        model = GINSingleDrugRegressor(
            n_patient_features=dataset.feat_dim,
            drug_emb_dim=64, hidden=256,
        ).to(device)
        if k == 0:
            print(f"[model] {count_parameters(model):,} trainable params", flush=True)

        history, best = train_one_fold(
            model, train_loader, val_loader, device, args.epochs, args.lr,
            log_prefix=f"  [f{k+1}] ",
        )
        fold_summary.append({"fold": k + 1, "best_epoch": best["epoch"],
                              "val_pearson": best["pearson"],
                              "val_spearman": best["spearman"],
                              "val_mse": best["val_mse"]})

        # Save checkpoint per fold
        torch.save({
            "model_state": model.state_dict(),
            "history": history,
            "best": best,
            "config": {
                "fold": k + 1,
                "n_patient_features": dataset.feat_dim,
                "drug_emb_dim": 64,
                "hidden": 256,
            },
        }, args.out_dir / f"fold_{k+1}.pt")

    # Pooled summary
    pearsons = [f["val_pearson"] for f in fold_summary]
    spearmans = [f["val_spearman"] for f in fold_summary]
    summary = {
        "fold_summaries": fold_summary,
        "pooled": {
            "mean_pearson": float(np.mean(pearsons)),
            "std_pearson": float(np.std(pearsons)),
            "mean_spearman": float(np.mean(spearmans)),
            "std_spearman": float(np.std(spearmans)),
        },
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== POOLED ===")
    print(f"  Pearson  : {summary['pooled']['mean_pearson']:.3f} ± "
           f"{summary['pooled']['std_pearson']:.3f}")
    print(f"  Spearman : {summary['pooled']['mean_spearman']:.3f} ± "
           f"{summary['pooled']['std_spearman']:.3f}")
    print(f"\nSaved to {args.out_dir}/summary.json + per-fold checkpoints.")


if __name__ == "__main__":
    main()
