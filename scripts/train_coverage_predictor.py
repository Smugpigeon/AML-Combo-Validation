#!/usr/bin/env python3
"""Phase B.3 — Train SMILES → 18-dim coverage predictor.

Inputs:
  - data/canonical/chembl_drug_targets.csv    (B.1 output)
  - data/canonical/drug_smiles.csv            (Phase 0)
  - src/combo_val/knowledge/chembl_target_to_taxonomy.yaml  (B.2 mapping)

Output:
  runs/coverage_predictor/coverage_gin.pt     (CoveragePredictionModel weights)
  runs/coverage_predictor/training_metrics.json

Usage:
  python scripts/train_coverage_predictor.py \
      --epochs 30 --batch-size 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch_geometric.data import Batch
    from scipy.stats import pearsonr
except ImportError as e:
    print(f"ERROR: {e}\n  pip install torch torch-geometric rdkit scipy")
    sys.exit(1)

from combo_val.coverage.chembl_coverage import (
    aggregate_drug_coverage, load_taxonomy_map,
)
from combo_val.coverage.gin_coverage_predictor import (
    CoveragePredictorConfig, assemble_training_data,
    make_coverage_predictor, save_predictor,
)
from combo_val.coverage.taxonomy import load_taxonomy
from combo_val.encoders.mol_featurizer import smiles_to_molgraph, to_pyg_data


def _load_smiles_table(path: Path) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s = r.get("isomeric_smiles") or r.get("canonical_smiles") or ""
            if s:
                out[r["beataml_drug_id"]] = s
    return out


class CoverageDataset(Dataset):
    def __init__(self, smiles_list, label_matrix):
        self.smiles_list = smiles_list
        self.labels = label_matrix
        # Pre-cache mol graphs
        self.graphs = [smiles_to_molgraph(s) for s in smiles_list]
        self.valid_indices = [i for i, g in enumerate(self.graphs) if g is not None]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        return {
            "graph": to_pyg_data(self.graphs[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def collate(batch):
    g = Batch.from_data_list([b["graph"] for b in batch])
    y = torch.stack([b["label"] for b in batch])
    return g, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl-csv", type=Path,
                     default=Path("data/canonical/chembl_drug_targets.csv"))
    ap.add_argument("--smiles-csv", type=Path,
                     default=Path("data/canonical/drug_smiles.csv"))
    ap.add_argument("--out-dir", type=Path,
                     default=Path("runs/coverage_predictor"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[load] taxonomy + chembl coverage...", flush=True)
    tx = load_taxonomy()
    axis_order = tx.target_ids()
    cov_map = aggregate_drug_coverage(args.chembl_csv)
    smiles_table = _load_smiles_table(args.smiles_csv)
    print(f"  18 axes; {len(cov_map)} drugs with ChEMBL data; "
           f"{len(smiles_table)} drugs with SMILES")

    drug_ids, smiles_list, labels = assemble_training_data(
        cov_map, smiles_table, axis_order,
    )
    n = len(drug_ids)
    print(f"[data] {n} training drugs (with both ChEMBL coverage AND SMILES)")
    if n < 10:
        print("ERROR: too few training drugs — Phase B.1 must complete first.")
        sys.exit(2)

    # Train/val split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    print(f"[split] train={len(train_idx)}, val={len(val_idx)}")

    dataset = CoverageDataset(smiles_list, labels)
    train_loader = DataLoader(Subset(dataset, train_idx),
                                batch_size=args.batch_size, shuffle=True,
                                collate_fn=collate)
    val_loader = DataLoader(Subset(dataset, val_idx),
                              batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)

    cfg = CoveragePredictorConfig(n_axes=len(axis_order))
    model = make_coverage_predictor(cfg)
    print(f"[model] {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    # Compute positive-class weight per axis to handle 18-axis sparsity
    # (most drug-axis pairs are 0 because most drugs only hit 1-3 axes).
    train_labels = labels[train_idx]
    pos_frac = train_labels.mean(axis=0).clip(min=0.01, max=0.99)
    pos_weight = torch.tensor(
        ((1 - pos_frac) / pos_frac).clip(max=20.0).astype(np.float32),
    )  # cap to avoid extreme weights for never-positive axes
    print(f"[pos_weight] axis pos-freq:  {pos_frac.round(3).tolist()}")
    print(f"[pos_weight] axis pos_weight: {pos_weight.numpy().round(2).tolist()}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def loss_fn(pred, target):
        """Hybrid MSE + per-axis weighted BCE.

        Pred is in [0, 1] from sigmoid. Target is in [0, 1] (continuous
        coverage). Use:
          - MSE on (pred, target) for overall fit
          - BCEWithLogitsLoss on (logit, target>0.1 binary) with
            pos_weight to avoid collapse on sparse classes
        """
        # MSE for fit on continuous labels
        mse = ((pred - target) ** 2).mean()
        # BCE with pos_weight on each axis to fight class imbalance.
        # Need the pre-sigmoid logit. Recover via inverse:
        #   logit = log(p / (1-p))
        # Clamp p to avoid log(0).
        p_clamped = pred.clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p_clamped / (1 - p_clamped))
        bin_target = (target > 0.1).float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logit, bin_target,
            pos_weight=pos_weight.to(logit.device),
        )
        return mse + 0.5 * bce
    history = []
    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for graph, y in train_loader:
            pred = model(graph)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(y)
            n_seen += len(y)
        train_loss /= n_seen

        # Eval
        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for graph, y in val_loader:
                p = model(graph).cpu().numpy()
                preds.append(p)
                trues.append(y.numpy())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            val_mse = float(np.mean((preds - trues) ** 2))
            # Per-axis Pearson (where there's variance)
            per_axis = []
            for ax in range(preds.shape[1]):
                if trues[:, ax].std() > 1e-6 and preds[:, ax].std() > 1e-6:
                    r, _ = pearsonr(preds[:, ax], trues[:, ax])
                    per_axis.append(float(r))
            mean_pear = float(np.mean(per_axis)) if per_axis else float("nan")

        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d}/{args.epochs}  "
               f"train_mse={train_loss:.4f}  val_mse={val_mse:.4f}  "
               f"mean_axis_pear={mean_pear:.3f}  ({elapsed:.1f}s)", flush=True)
        history.append({
            "epoch": epoch, "train_loss": float(train_loss),
            "val_mse": val_mse, "mean_axis_pearson": mean_pear,
            "elapsed_s": elapsed,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_predictor(model, args.out_dir / "coverage_gin.pt", axis_order, cfg)

    metrics = {
        "n_train_drugs": len(train_idx),
        "n_val_drugs": len(val_idx),
        "axis_order": axis_order,
        "best_val_mse": float(best_val_mse),
        "history": history,
    }
    with open(args.out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] best val MSE: {best_val_mse:.4f}")
    print(f"[saved] {args.out_dir}/coverage_gin.pt")
    print(f"[saved] {args.out_dir}/training_metrics.json")


if __name__ == "__main__":
    main()
