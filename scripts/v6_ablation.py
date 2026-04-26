#!/usr/bin/env python3
"""Phase 5 — v0.6 ablation matrix + HARD GO/NO-GO decision.

Per the Path C plan from the post-v0.5 reviewer dialog:

  Trains 4 model variants on BeatAML 55,826 (patient × drug × AUC) with
  patient-level 5-fold CV, then aggregates internal Pearson r at the
  model level. Threshold-based GO/NO-GO:

    Internal Pearson r >= 0.20  → GO (continue Phase 6 kit integration)
    0.10 <= r < 0.20            → DSMB-style independent review
    r < 0.10                    → NO-GO (delete Layer-3, pivot to Path A)

Ablation arms (V6FullModelConfig switches):

    v0.5_baseline    | nn.Embedding(165, 64) + linear gene proj
    v0.6.B_gene_only | nn.Embedding(165, 64) + GeneContextGAT (PPI)
    v0.6.A_drug_only | DrugGIN(SMILES) + linear gene proj
    v0.6.0_full      | DrugGIN(SMILES) + GeneContextGAT (PPI)

Note: this script trains on SINGLE-DRUG labels (sets of size 1) since
BeatAML has no in-domain combo labels. The Set-Transformer combo head
falls through to the same single-drug-set semantics that an N=1 set
processes correctly. Phase 4 DrugComb pretrain (deferred) would let us
fine-tune the combo head proper on real pair labels.

Usage:
  python scripts/v6_ablation.py \
      --smiles-csv data/canonical/drug_smiles.csv \
      --beataml-features data/canonical/beataml_patient_features.csv \
      --beataml-response data/canonical/beataml_drug_response_long.csv \
      --ppi-json data/canonical/gene_ppi_subgraph.json \
      --out-dir runs/v6_ablation \
      --epochs 20 \
      --n-folds 5

Output:
  runs/v6_ablation/
    {arm}_fold_{k}.pt
    {arm}_summary.json
    ablation_table.md     ← human-readable matrix
    go_no_go_decision.md  ← Phase 6 entry decision
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, Subset
    from scipy.stats import pearsonr, spearmanr
except ImportError as e:
    print(f"ERROR: {e}\n  pip install torch torch-geometric rdkit scipy")
    sys.exit(1)

from combo_val.encoders.v6_full_model import (
    V6FullModel, V6FullModelConfig, count_parameters,
)


# ---------------------------------------------------------------------------
# Ablation arm definitions
# ---------------------------------------------------------------------------


@dataclass
class AblationArm:
    name: str
    description: str
    use_gin_drug: bool
    use_gat_gene: bool


ABLATION_ARMS = [
    AblationArm("v0.5_baseline", "nn.Embedding + linear gene proj (current production)",
                use_gin_drug=False, use_gat_gene=False),
    AblationArm("v0.6.B_gene_only", "nn.Embedding + GeneContextGAT (PPI)",
                use_gin_drug=False, use_gat_gene=True),
    AblationArm("v0.6.A_drug_only", "DrugGIN(SMILES) + linear gene proj",
                use_gin_drug=True, use_gat_gene=False),
    AblationArm("v0.6.0_full", "DrugGIN(SMILES) + GeneContextGAT (PPI) — full v0.6",
                use_gin_drug=True, use_gat_gene=True),
]


# ---------------------------------------------------------------------------
# Dataset (single-drug-set wrapped to N=1 for V6FullModel API)
# ---------------------------------------------------------------------------


class V6SingleDrugDataset(Dataset):
    """Wraps BeatAML single-drug AUC into V6FullModel-compatible batches.

    Each row → (patient_features, single drug SMILES OR drug_id, AUC).
    Drug is treated as a 1-element "set" with mask=[1]; combo head still
    works correctly on this degenerate case (Set Transformer pools 1
    element to itself).
    """

    def __init__(self, response_df: pd.DataFrame, features_df: pd.DataFrame,
                 smiles_map: dict, drug_id_to_int: dict,
                 use_smiles: bool):
        feat_pids = set(features_df["patient_id"].astype(str))
        valid = response_df["patient_id"].astype(str).isin(feat_pids)
        if use_smiles:
            valid = valid & response_df["drug_id"].isin(smiles_map)
        else:
            valid = valid & response_df["drug_id"].isin(drug_id_to_int)
        self.rows = response_df[valid].reset_index(drop=True)

        # mut_* columns
        self.mut_cols = [c for c in features_df.columns if c.startswith("mut_")]
        self.rna_cols = [c for c in features_df.columns if c.startswith("rna_pc")]
        self.clin_cols = [c for c in features_df.columns if c.startswith("clin_")]
        self.cyto_cols = [c for c in features_df.columns
                           if c.startswith(("fusion_", "karyo_"))]

        self.feat_index = {str(pid): i
                            for i, pid in enumerate(features_df["patient_id"])}
        self.mut_arr = features_df[self.mut_cols].values.astype(np.float32)
        self.rna_arr = features_df[self.rna_cols].values.astype(np.float32)
        self.clin_arr = features_df[self.clin_cols].values.astype(np.float32)
        self.cyto_arr = features_df[self.cyto_cols].values.astype(np.float32)

        self.smiles_map = smiles_map
        self.drug_id_to_int = drug_id_to_int
        self.use_smiles = use_smiles

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        pid = str(row["patient_id"])
        idx = self.feat_index[pid]
        return {
            "drug_id": row["drug_id"],
            "drug_smiles": (self.smiles_map.get(row["drug_id"], "")
                             if self.use_smiles else None),
            "drug_int": (self.drug_id_to_int.get(row["drug_id"], 0)
                          if not self.use_smiles else None),
            "mut": self.mut_arr[idx],
            "rna_pc": self.rna_arr[idx],
            "clin": self.clin_arr[idx],
            "cyto": self.cyto_arr[idx],
            "auc": float(row["auc"]),
            "patient_id": pid,
        }


def make_collate(use_smiles: bool):
    def collate(batch):
        B = len(batch)
        if use_smiles:
            drug_input = [[b["drug_smiles"]] for b in batch]   # list[list[str]]
        else:
            drug_input = torch.tensor(
                [[b["drug_int"]] for b in batch], dtype=torch.long,
            )
        drug_mask = torch.ones(B, 1, dtype=torch.long)
        mut = torch.tensor(np.stack([b["mut"] for b in batch]))
        rna = torch.tensor(np.stack([b["rna_pc"] for b in batch]))
        clin = torch.tensor(np.stack([b["clin"] for b in batch]))
        cyto = torch.tensor(np.stack([b["cyto"] for b in batch]))
        auc = torch.tensor([b["auc"] for b in batch], dtype=torch.float32)
        pids = [b["patient_id"] for b in batch]
        return drug_input, drug_mask, mut, rna, clin, cyto, auc, pids
    return collate


# ---------------------------------------------------------------------------
# Train / eval one arm × one fold
# ---------------------------------------------------------------------------


def patient_kfold_split(features_df, n_folds, seed):
    rng = np.random.default_rng(seed)
    pids = sorted(features_df["patient_id"].astype(str).unique())
    rng.shuffle(pids)
    folds = np.array_split(pids, n_folds)
    out = []
    for k in range(n_folds):
        val = set(folds[k])
        train = set().union(*(set(folds[j]) for j in range(n_folds) if j != k))
        out.append((train, val))
    return out


def train_one_fold(model, train_loader, val_loader, device, epochs, lr,
                    log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best = {"epoch": -1, "pearson": -1e9, "spearman": -1e9, "val_mse": 1e9}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_seen = 0.0, 0
        t0 = time.time()
        for drug_input, drug_mask, mut, rna, clin, cyto, auc, _ in train_loader:
            drug_mask = drug_mask.to(device)
            mut = mut.to(device)
            rna = rna.to(device)
            clin = clin.to(device)
            cyto = cyto.to(device)
            auc = auc.to(device)
            if isinstance(drug_input, torch.Tensor):
                drug_input = drug_input.to(device)
            pred = model(drug_input, drug_mask, mut, rna, clin, cyto)
            loss = loss_fn(pred, auc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(auc)
            n_seen += len(auc)
        train_loss /= n_seen

        # Eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for drug_input, drug_mask, mut, rna, clin, cyto, auc, _ in val_loader:
                drug_mask = drug_mask.to(device)
                mut = mut.to(device)
                rna = rna.to(device)
                clin = clin.to(device)
                cyto = cyto.to(device)
                if isinstance(drug_input, torch.Tensor):
                    drug_input = drug_input.to(device)
                pred = model(drug_input, drug_mask, mut, rna, clin, cyto).cpu().numpy()
                preds.append(pred)
                trues.append(auc.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        pear, _ = pearsonr(preds, trues)
        spr, _ = spearmanr(preds, trues)
        val_mse = float(np.mean((preds - trues) ** 2))
        elapsed = time.time() - t0
        msg = (f"{log_prefix}epoch {epoch:3d}/{epochs}  "
               f"train_mse={train_loss:.3f}  val_mse={val_mse:.3f}  "
               f"val_pear={pear:.3f}  val_sp={spr:.3f}  ({elapsed:.1f}s)")
        print(msg, flush=True)
        history.append({"epoch": int(epoch), "train_loss": float(train_loss),
                         "val_mse": float(val_mse), "val_pearson": float(pear),
                         "val_spearman": float(spr),
                         "elapsed_sec": float(elapsed)})
        if pear > best["pearson"]:
            best = {"epoch": int(epoch), "pearson": float(pear),
                     "spearman": float(spr), "val_mse": float(val_mse)}
    return history, best


# ---------------------------------------------------------------------------
# GO / NO-GO decision
# ---------------------------------------------------------------------------


def decide_go_no_go(arm_results: dict) -> dict:
    """Compare v0.6.0_full vs v0.5_baseline; emit decision per Path C."""
    full = arm_results.get("v0.6.0_full", {})
    base = arm_results.get("v0.5_baseline", {})

    full_r = full.get("mean_pearson", -1.0)
    base_r = base.get("mean_pearson", -1.0)

    if full_r >= 0.20:
        decision = "GO"
        rationale = (
            f"v0.6.0_full Pearson r = {full_r:.3f} ≥ 0.20 → "
            f"continue Phase 6 (kit integration)."
        )
    elif full_r >= 0.10:
        decision = "REVIEW"
        rationale = (
            f"v0.6.0_full Pearson r = {full_r:.3f} ∈ [0.10, 0.20) → "
            f"DSMB-style independent review required before Phase 6. "
            f"Compare to baseline {base_r:.3f}."
        )
    else:
        decision = "NO-GO"
        rationale = (
            f"v0.6.0_full Pearson r = {full_r:.3f} < 0.10 → "
            f"DELETE Layer-3 from kit; pivot to Path A "
            f"(clinical decision support without ML). "
            f"Baseline r = {base_r:.3f}."
        )
    return {
        "decision": decision,
        "rationale": rationale,
        "v0.6.0_full_pearson": full_r,
        "v0.5_baseline_pearson": base_r,
        "delta_vs_baseline": full_r - base_r,
    }


def render_table(arm_results: dict) -> str:
    lines = [
        "| Arm | Description | Mean Pearson r | Std | Mean Spearman | Params |",
        "|-----|-------------|---------------:|----:|-------------:|-------:|",
    ]
    for arm in ABLATION_ARMS:
        r = arm_results.get(arm.name)
        if not r:
            lines.append(f"| `{arm.name}` | {arm.description} | — | — | — | — |")
            continue
        lines.append(
            f"| `{arm.name}` | {arm.description} | "
            f"**{r['mean_pearson']:.3f}** | {r['std_pearson']:.3f} | "
            f"{r['mean_spearman']:.3f} | {r.get('n_params', '?'):,} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-csv", type=Path, required=True)
    ap.add_argument("--beataml-features", type=Path, required=True)
    ap.add_argument("--beataml-response", type=Path, required=True)
    ap.add_argument("--ppi-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--single-fold-only", action="store_true",
                     help="run only fold-0 per arm (faster smoke)")
    ap.add_argument("--arms", nargs="+", default=None,
                     help="restrict to specific arm names")
    args = ap.parse_args()

    device = (args.device if args.device != "auto"
               else "cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] features ...")
    features_df = pd.read_csv(args.beataml_features)
    print(f"  {len(features_df)} patients × {features_df.shape[1] - 1} features")

    print(f"[load] response ...")
    response_df = pd.read_csv(args.beataml_response)

    print(f"[load] SMILES ...")
    smiles_map = {}
    with open(args.smiles_csv) as f:
        for r in csv.DictReader(f):
            s = r.get("isomeric_smiles") or r.get("canonical_smiles") or ""
            if s:
                smiles_map[r["beataml_drug_id"]] = s
    print(f"  {len(smiles_map)} drugs with SMILES")

    drug_id_to_int = {d: i for i, d in
                       enumerate(sorted(response_df["drug_id"].unique()))}

    gene_order = [c[len("mut_"):] for c in features_df.columns
                   if c.startswith("mut_")]

    folds = patient_kfold_split(features_df, args.n_folds, args.seed)

    # Decide which arms to run
    arms_to_run = ABLATION_ARMS
    if args.arms:
        arms_to_run = [a for a in ABLATION_ARMS if a.name in args.arms]

    arm_results = {}
    for arm in arms_to_run:
        print(f"\n{'='*70}")
        print(f"ABLATION ARM: {arm.name}")
        print(f"  {arm.description}")
        print(f"  use_gin_drug={arm.use_gin_drug}  use_gat_gene={arm.use_gat_gene}")
        print(f"{'='*70}")

        dataset = V6SingleDrugDataset(
            response_df, features_df, smiles_map, drug_id_to_int,
            use_smiles=arm.use_gin_drug,
        )
        print(f"  dataset rows: {len(dataset)}")
        collate = make_collate(arm.use_gin_drug)
        # Pre-cache patient_id-as-string per row to avoid the int-vs-str
        # mismatch with the (str-keyed) kfold split sets.
        pid_str_per_row = dataset.rows["patient_id"].astype(str).values

        fold_perfs = []
        for k, (train_pids, val_pids) in enumerate(folds):
            if args.single_fold_only and k > 0:
                break
            print(f"\n--- {arm.name} fold {k+1}/{args.n_folds} ---")
            train_idx = [i for i in range(len(dataset))
                          if pid_str_per_row[i] in train_pids]
            val_idx = [i for i in range(len(dataset))
                        if pid_str_per_row[i] in val_pids]
            print(f"  train={len(train_idx)} val={len(val_idx)}")

            train_loader = DataLoader(Subset(dataset, train_idx),
                                        batch_size=args.batch_size,
                                        shuffle=True, collate_fn=collate)
            val_loader = DataLoader(Subset(dataset, val_idx),
                                      batch_size=args.batch_size,
                                      shuffle=False, collate_fn=collate)

            cfg = V6FullModelConfig(
                use_gin_drug=arm.use_gin_drug,
                use_gat_gene=arm.use_gat_gene,
                n_drugs_total=len(drug_id_to_int),
            )
            model = V6FullModel(args.ppi_json, gene_order, cfg).to(device)
            if k == 0:
                n_p = count_parameters(model)
                print(f"  model: {n_p:,} params")

            history, best = train_one_fold(
                model, train_loader, val_loader, device, args.epochs,
                args.lr, log_prefix=f"  [{arm.name}.f{k+1}] ",
            )
            fold_perfs.append(best)

            torch.save({
                "model_state": model.state_dict(),
                "arm": arm.name, "fold": k + 1, "best": best,
                "history": history, "n_params": n_p if k == 0 else None,
            }, args.out_dir / f"{arm.name}_fold_{k+1}.pt")

        pearsons = [f["pearson"] for f in fold_perfs]
        spearmans = [f["spearman"] for f in fold_perfs]
        arm_results[arm.name] = {
            "mean_pearson": float(np.mean(pearsons)),
            "std_pearson": float(np.std(pearsons)),
            "mean_spearman": float(np.mean(spearmans)),
            "std_spearman": float(np.std(spearmans)),
            "fold_perfs": fold_perfs,
            "n_params": n_p,
        }
        with open(args.out_dir / f"{arm.name}_summary.json", "w") as f:
            json.dump(arm_results[arm.name], f, indent=2)

    # Render table + decision
    table = render_table(arm_results)
    decision = decide_go_no_go(arm_results)

    out_md = (
        f"# v0.6 Ablation Results\n\n"
        f"## Per-arm performance (5-fold patient-level CV)\n\n"
        f"{table}\n\n"
        f"## GO / NO-GO Decision (per Path C)\n\n"
        f"**{decision['decision']}**\n\n"
        f"{decision['rationale']}\n\n"
        f"- v0.6.0_full Pearson r: {decision['v0.6.0_full_pearson']:.3f}\n"
        f"- v0.5_baseline Pearson r: {decision['v0.5_baseline_pearson']:.3f}\n"
        f"- Δ (full − baseline): {decision['delta_vs_baseline']:+.3f}\n"
    )
    (args.out_dir / "ablation_table.md").write_text(out_md)
    with open(args.out_dir / "go_no_go_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print(f"\n{'='*70}")
    print(out_md)
    print(f"\nFull results: {args.out_dir}/ablation_table.md")


if __name__ == "__main__":
    main()
