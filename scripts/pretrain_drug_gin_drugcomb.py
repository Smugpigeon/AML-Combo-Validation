#!/usr/bin/env python3
"""Phase 4 — Pretrain DrugGIN on DrugComb cross-cancer pair data.

Goal: give the GIN drug encoder chemistry-grounded weights that
generalize across cell lines, before we fine-tune on the smaller
BeatAML AML-specific cohort. This is a standard transfer-learning move
in drug-combo ML (DeepDDS, DeepSynergy, etc.).

Pretrain objective: predict combo synergy (Loewe / Bliss / HSA / ZIP —
configurable) from (drug_a_smiles, drug_b_smiles, cell_line_id).

Architecture (pretrain-only, head discarded after):

    SMILES_a → DrugGIN → drug_a_emb [64]
    SMILES_b → DrugGIN → drug_b_emb [64]   (shared encoder weights)
                                ↓
    cell_line_id → nn.Embedding → cell_emb [32]
                                ↓
    concat(drug_a, drug_b, cell) → MLP → predicted synergy scalar

After pretraining, only the DrugGIN encoder weights are kept; the cell
embedding + synergy head are discarded. Phase 5 / 1b loads those
weights as the initialization for the v0.6 BeatAML fine-tune.

Usage (after data/canonical/drugcomb_pairs_pretrain.csv is fetched):
  python scripts/pretrain_drug_gin_drugcomb.py \
      --pairs-csv data/canonical/drugcomb_pairs_pretrain.csv \
      --out-dir runs/drug_gin_pretrain_drugcomb \
      --epochs 20 \
      --batch-size 512 \
      --target-col synergy_loewe

Status: NOT YET RUNNABLE. Waits on Phase 0.5 DrugComb data fetch
(currently deferred — proxy can't reach drugcomb.fimm.fi). Once data
is in place, this script is ready.
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


class DrugCombPairDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, target_col: str):
        # Filter rows with valid SMILES + target
        valid_mask = (
            pairs_df["drug_row_smiles"].notna()
            & (pairs_df["drug_row_smiles"] != "")
            & pairs_df["drug_col_smiles"].notna()
            & (pairs_df["drug_col_smiles"] != "")
            & pairs_df[target_col].notna()
        )
        self.rows = pairs_df[valid_mask].reset_index(drop=True)
        self.target_col = target_col

        # Build cell-line vocabulary
        cell_col = (
            "cell_line_name" if "cell_line_name" in self.rows.columns
            else "cell_line"
        )
        self.cell_col = cell_col
        cells = sorted(self.rows[cell_col].astype(str).unique())
        self.cell_to_int = {c: i for i, c in enumerate(cells)}

        # Pre-cache mol graphs (drug SMILES are the bottleneck)
        unique_smiles = set(self.rows["drug_row_smiles"]).union(
            self.rows["drug_col_smiles"],
        )
        print(f"[cache] Featurizing {len(unique_smiles)} unique drugs ...",
               flush=True)
        self.smiles_to_graph = {}
        for s in unique_smiles:
            g = smiles_to_molgraph(s)
            if g is not None:
                self.smiles_to_graph[s] = to_pyg_data(g)

        # Drop rows with un-featurizable SMILES
        good_mask = (
            self.rows["drug_row_smiles"].isin(self.smiles_to_graph)
            & self.rows["drug_col_smiles"].isin(self.smiles_to_graph)
        )
        self.rows = self.rows[good_mask].reset_index(drop=True)
        print(f"[dataset] {len(self.rows)} usable pairs across "
               f"{len(cells)} cell lines", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return {
            "drug_a_graph": self.smiles_to_graph[r["drug_row_smiles"]],
            "drug_b_graph": self.smiles_to_graph[r["drug_col_smiles"]],
            "cell_int": self.cell_to_int[str(r[self.cell_col])],
            "target": float(r[self.target_col]),
        }


def collate_pair(batch):
    drug_a = Batch.from_data_list([b["drug_a_graph"] for b in batch])
    drug_b = Batch.from_data_list([b["drug_b_graph"] for b in batch])
    cell = torch.tensor([b["cell_int"] for b in batch], dtype=torch.long)
    target = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
    return drug_a, drug_b, cell, target


class DrugCombPretrainModel(nn.Module):
    """Shared DrugGIN encoder + cell embedding + MLP synergy head."""
    def __init__(self, n_cells: int, drug_emb_dim: int = 64,
                 cell_emb_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.drug_gin = DrugGIN(DrugGINConfig(drug_emb_dim=drug_emb_dim))
        self.cell_emb = nn.Embedding(n_cells, cell_emb_dim)
        self.synergy_head = nn.Sequential(
            nn.Linear(drug_emb_dim * 2 + cell_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, drug_a_batch, drug_b_batch, cell_int):
        emb_a = self.drug_gin(drug_a_batch)
        emb_b = self.drug_gin(drug_b_batch)
        emb_cell = self.cell_emb(cell_int)
        x = torch.cat([emb_a, emb_b, emb_cell], dim=-1)
        return self.synergy_head(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-col", type=str, default="synergy_loewe")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.pairs_csv.exists():
        print(f"ERROR: pairs CSV not present at {args.pairs_csv}.")
        print(f"  Phase 0.5 must complete first (DrugComb data fetch).")
        print(f"  See scripts/fetch_drugcomb_pairs.py.")
        sys.exit(2)

    device = (args.device if args.device != "auto"
               else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] {args.pairs_csv}")
    pairs_df = pd.read_csv(args.pairs_csv)
    print(f"  {len(pairs_df)} raw pairs")

    if args.target_col not in pairs_df.columns:
        print(f"ERROR: --target-col '{args.target_col}' not in CSV. Available: "
               f"{[c for c in pairs_df.columns if 'syn' in c.lower()]}")
        sys.exit(2)

    dataset = DrugCombPairDataset(pairs_df, args.target_col)
    n_total = len(dataset)
    n_val = max(1, int(args.val_split * n_total))
    n_train = n_total - n_val

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:].tolist()

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pair,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pair,
    )

    model = DrugCombPretrainModel(
        n_cells=len(dataset.cell_to_int),
    ).to(device)
    print(f"[model] {count_parameters(model):,} params")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_pearson = -1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_seen = 0.0, 0
        t0 = time.time()
        for drug_a, drug_b, cell, target in train_loader:
            drug_a = drug_a.to(device)
            drug_b = drug_b.to(device)
            cell = cell.to(device)
            target = target.to(device)
            pred = model(drug_a, drug_b, cell)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(target)
            n_seen += len(target)
        train_loss /= n_seen

        # Eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for drug_a, drug_b, cell, target in val_loader:
                drug_a = drug_a.to(device)
                drug_b = drug_b.to(device)
                cell = cell.to(device)
                pred = model(drug_a, drug_b, cell).cpu().numpy()
                preds.append(pred)
                trues.append(target.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        pear, _ = pearsonr(preds, trues)
        spr, _ = spearmanr(preds, trues)
        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d}/{args.epochs}  "
               f"train_mse={train_loss:.3f}  val_pear={pear:.3f}  "
               f"val_sp={spr:.3f}  ({elapsed:.1f}s)", flush=True)

        if pear > best_pearson:
            best_pearson = pear
            # Save ENCODER ONLY for downstream Phase 1/5 fine-tune
            torch.save({
                "drug_gin_state": model.drug_gin.state_dict(),
                "epoch": epoch,
                "val_pearson": pear,
                "val_spearman": spr,
                "n_train_pairs": n_train,
                "n_cells": len(dataset.cell_to_int),
                "target_col": args.target_col,
            }, args.out_dir / "drug_gin_pretrained.pt")

    print(f"\n[done] Best val Pearson: {best_pearson:.3f}")
    print(f"  Saved encoder weights: {args.out_dir}/drug_gin_pretrained.pt")
    print(f"  Use in Phase 1 / Phase 5 via DrugGIN().load_state_dict(...)")


if __name__ == "__main__":
    main()
