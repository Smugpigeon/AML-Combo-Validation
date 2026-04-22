"""Combo AUC predictor: factorized additive-plus-residual model.

Given a patient and a drug pair (d1, d2), predict the combined drug AUC using
two learned terms:

  1. SINGLE-DRUG BASELINE (frozen) — Baseline A predictions
     AUC_d1(p)   from predictions_all_patients_all_drugs.csv
     AUC_d2(p)

  2. PAIR-LEVEL SYNERGY RESIDUAL (learned here)
     synergy(d1, d2) ← small MLP trained on DrugComb ALMANAC-HL-60 pairs.
     This is patient-AGNOSTIC (we only have 186 pairs on one cell line, so the
     combo adjustment depends only on drug chemistry, not on patient biology).

Final combo AUC prediction:

    combo_auc(p, d1, d2) = 0.5 * (AUC_d1(p) + AUC_d2(p)) + k * synergy(d1, d2)

where k is a scaling constant (set to 1.0 by default — DrugComb Loewe synergy
is already on an AUC-comparable scale, roughly ±50 units).

Why this factorization:
  - Patient personalization carries through via the Baseline A per-patient AUC.
  - Drug-chemistry combo effects are captured by the residual.
  - Separating the two means ANY drug pair can be scored, even pairs never
    seen in combination (as long as both drugs exist in BeatAML's single-drug
    data) — crucial because our combo training is only 186 pairs.

Mechanism prior (Week 5 only):
  For the 20 AML-approved drugs in `knowledge/drug_mechanism_v1.csv`, we also
  score a "mechanism coverage" term that captures whether the pair jointly
  addresses the patient's driver features (FLT3+venetoclax for FLT3/BCL2
  co-deficit, etc.). This is separate from the AUC prediction here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComboConfig:
    # Architecture
    drug_emb_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.2
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    n_folds: int = 5
    random_state: int = 42
    # Synergy metric to model
    synergy_col: str = "synergy_loewe"
    # Combo AUC formula:
    #   combo_auc = 0.5 * (auc_d1 + auc_d2)
    #             + synergy_to_auc_scale * predicted_synergy
    #             - mech_prior_scale * mechanism_prior_score
    # - synergy_to_auc_scale converts Loewe synergy (~ -50..+15) to AUC units
    # - mech_prior_scale converts mechanism bonus (0..5+) to AUC reduction
    #   (lower AUC = "more cell killing"; mech-matched combos should score lower)
    synergy_to_auc_scale: float = 1.0
    mech_prior_scale: float = 30.0
    use_mech_prior: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SynergyMLP(nn.Module):
    """Symmetric drug-pair synergy predictor.

    Embeds both drugs, concats (with symmetric add to enforce commutativity),
    passes through MLP, outputs scalar synergy.
    """

    def __init__(self, n_drugs: int, drug_emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, drug_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * drug_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        e1 = self.drug_emb(d1)
        e2 = self.drug_emb(d2)
        # Symmetric features: sum + absolute-diff so that f(d1,d2) == f(d2,d1).
        sym = torch.cat([e1 + e2, (e1 - e2).abs()], dim=-1)
        return self.head(sym).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_combo_training_data(
    pairs_path: Path,
    synergy_col: str,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Load DrugComb strict pairs as (drug1_id, drug2_id, synergy) triples."""
    df = pd.read_csv(pairs_path)
    required = {"drug1_id", "drug2_id", synergy_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {pairs_path}: {missing}")
    df = df.dropna(subset=list(required)).copy()

    drugs = sorted(set(df["drug1_id"]).union(df["drug2_id"]))
    drug_to_int = {d: i for i, d in enumerate(drugs)}
    df["d1_int"] = df["drug1_id"].map(drug_to_int)
    df["d2_int"] = df["drug2_id"].map(drug_to_int)

    print(f"[combo] pairs: {len(df):,}  unique drugs: {len(drugs):,}  "
          f"synergy ({synergy_col}) mean={df[synergy_col].mean():.2f} "
          f"std={df[synergy_col].std():.2f}")
    return df, drugs, drug_to_int


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_drugs: int,
    cfg: ComboConfig,
    device: torch.device,
) -> tuple[SynergyMLP, float]:
    """Train one fold; return best-val model and its RMSE."""
    model = SynergyMLP(n_drugs, cfg.drug_emb_dim, cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_d1 = torch.tensor(train_df["d1_int"].values, dtype=torch.long, device=device)
    train_d2 = torch.tensor(train_df["d2_int"].values, dtype=torch.long, device=device)
    train_y = torch.tensor(train_df[cfg.synergy_col].values, dtype=torch.float, device=device)
    val_d1 = torch.tensor(val_df["d1_int"].values, dtype=torch.long, device=device)
    val_d2 = torch.tensor(val_df["d2_int"].values, dtype=torch.long, device=device)
    val_y = torch.tensor(val_df[cfg.synergy_col].values, dtype=torch.float, device=device)

    n = len(train_df)
    best_val_rmse = float("inf")
    best_state: dict | None = None
    epochs_since_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            pred = model(train_d1[idx], train_d2[idx])
            loss = loss_fn(pred, train_y[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_d1, val_d2)
            val_rmse = torch.sqrt(loss_fn(val_pred, val_y)).item()

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_val_rmse


def train_combo_predictor(
    pairs_path: Path = Path("data/canonical/drugcomb_aml_pairs.csv"),
    baseline_pred_path: Path = Path("runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv"),
    out_dir: Path = Path("runs/combo_predictor"),
    cfg: ComboConfig | None = None,
) -> dict:
    """Train combo residual on DrugComb strict pairs + emit combo predictions."""
    cfg = cfg or ComboConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    df, drugs, drug_to_int = load_combo_training_data(pairs_path, cfg.synergy_col)
    n_drugs = len(drugs)

    # 5-fold CV — random split since we have no patient dimension here
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    fold_rmses: list[float] = []
    all_val_preds: list[tuple[int, np.ndarray, np.ndarray]] = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        model, val_rmse = _train_fold(train_df, val_df, n_drugs, cfg, device)

        model.eval()
        with torch.no_grad():
            vd1 = torch.tensor(val_df["d1_int"].values, dtype=torch.long, device=device)
            vd2 = torch.tensor(val_df["d2_int"].values, dtype=torch.long, device=device)
            pred = model(vd1, vd2).cpu().numpy()
        all_val_preds.append((fold_i, val_df.index.values, pred))
        fold_rmses.append(val_rmse)
        print(f"[combo] fold {fold_i + 1}/{cfg.n_folds}  val RMSE = {val_rmse:.3f}  "
              f"(synergy std in data = {df[cfg.synergy_col].std():.2f})")

    # Full-data model (used for inference)
    full_model, _ = _train_fold(df, df, n_drugs, cfg, device)

    # CV predictions table
    pred_rows: list[dict] = []
    for fold_i, ids, pred in all_val_preds:
        for i, p in zip(ids, pred):
            row = df.loc[i].copy()
            row["fold"] = fold_i
            row["predicted_synergy"] = float(p)
            pred_rows.append(row.to_dict())
    cv_pred_df = pd.DataFrame(pred_rows)
    cv_pred_df.to_csv(out_dir / "cv_predictions.csv", index=False)

    # Pearson / Spearman of predicted vs observed
    pearson = float(np.corrcoef(cv_pred_df["predicted_synergy"], cv_pred_df[cfg.synergy_col])[0, 1])
    spr = float(spearmanr(cv_pred_df["predicted_synergy"], cv_pred_df[cfg.synergy_col]).correlation)
    mean_rmse = float(np.mean(fold_rmses))
    print(f"[combo] CV pooled: Pearson={pearson:.3f}  Spearman={spr:.3f}  mean RMSE={mean_rmse:.3f}")

    # --- Inference: predict synergy for ALL drug pairs in BeatAML's 165-drug vocab
    baseline_pred = pd.read_csv(baseline_pred_path, index_col=0)
    beataml_drugs = list(baseline_pred.columns)
    # Only drugs that exist in the combo-trained vocab can have a learned synergy.
    # For drugs NOT in the combo training vocab, assign synergy = 0 (no prior evidence).
    combo_vocab_set = set(drugs)
    full_model.eval()

    n_pats = baseline_pred.shape[0]
    n_beataml_drugs = len(beataml_drugs)
    print(f"[combo] predicting combo AUC for {n_pats} patients × "
          f"{n_beataml_drugs * (n_beataml_drugs - 1) // 2} unique drug pairs "
          f"({len(combo_vocab_set)} drugs have learned synergy)")

    # Precompute predicted synergy matrix (n_drugs × n_drugs), only for combo_vocab_set
    synergy_matrix = np.zeros((n_beataml_drugs, n_beataml_drugs), dtype=np.float32)
    beataml_to_int = {d: i for i, d in enumerate(beataml_drugs)}
    indices_in_combo = [
        (i, drug_to_int[d])
        for d, i in beataml_to_int.items() if d in combo_vocab_set
    ]
    if indices_in_combo:
        combo_idxs = torch.tensor([p[1] for p in indices_in_combo], dtype=torch.long, device=device)
        # Pairwise
        with torch.no_grad():
            grid = torch.cartesian_prod(combo_idxs, combo_idxs)
            pred_grid = full_model(grid[:, 0], grid[:, 1]).cpu().numpy()
        pred_grid = pred_grid.reshape(len(indices_in_combo), len(indices_in_combo))
        for r, (beat_i, _) in enumerate(indices_in_combo):
            for c, (beat_j, _) in enumerate(indices_in_combo):
                synergy_matrix[beat_i, beat_j] = pred_grid[r, c]

    # Compute combo AUC: 0.5 * (auc1 + auc2) + k * synergy - mech_scale * mech
    auc_matrix = baseline_pred.values  # (n_pats, n_drugs)
    combo_auc = (
        0.5 * (auc_matrix[:, :, None] + auc_matrix[:, None, :])
        + cfg.synergy_to_auc_scale * synergy_matrix[None, :, :]
    )  # (n_pats, n_drugs, n_drugs)

    if cfg.use_mech_prior:
        # Mechanism-driven combo prior (patient-specific, knowledge-based)
        from combo_val.combo.mechanism_prior import (
            compute_combo_mech_scores,
            diagnostics as mech_diagnostics,
        )
        # Read patient features matching the baseline_pred index order.
        pf_path = Path("data/canonical/beataml_patient_features.csv")
        pf = pd.read_csv(pf_path).set_index("patient_id")
        pf = pf.reindex(baseline_pred.index)
        mech_diag = mech_diagnostics(pf, beataml_drugs)
        print(f"[combo] mechanism prior diagnostics: {mech_diag}")
        mech_score = compute_combo_mech_scores(pf, beataml_drugs)
        # Higher mech_score = better match → lower combo_auc (more cell killing)
        combo_auc = combo_auc - cfg.mech_prior_scale * mech_score.astype(np.float32)

    # Save per-patient best-combo table (memory-friendly) rather than the full tensor
    rows: list[dict] = []
    for pi, pid in enumerate(baseline_pred.index):
        mat = combo_auc[pi]
        # Upper-triangle without diagonal
        tri = np.triu_indices(n_beataml_drugs, k=1)
        pair_aucs = mat[tri]
        pair_d1 = np.array(beataml_drugs)[tri[0]]
        pair_d2 = np.array(beataml_drugs)[tri[1]]
        # Top-5 lowest-AUC combos for this patient
        top5 = np.argsort(pair_aucs)[:5]
        for rank, tk in enumerate(top5):
            rows.append({
                "patient_id": pid,
                "rank": rank + 1,
                "drug1": pair_d1[tk],
                "drug2": pair_d2[tk],
                "combo_pred_auc": float(pair_aucs[tk]),
                "single_auc_d1": float(baseline_pred.loc[pid, pair_d1[tk]]),
                "single_auc_d2": float(baseline_pred.loc[pid, pair_d2[tk]]),
            })
    per_patient_top = pd.DataFrame(rows)
    per_patient_top.to_csv(out_dir / "per_patient_top5_combos.csv", index=False)

    # Full combo AUC tensor is too large — save a compressed npz instead.
    np.savez_compressed(
        out_dir / "combo_auc_predictions.npz",
        combo_auc=combo_auc.astype(np.float32),
        patient_ids=np.array(baseline_pred.index),
        drug_names=np.array(beataml_drugs),
    )

    manifest = {
        "cfg": asdict(cfg),
        "n_pairs_training": int(len(df)),
        "n_drugs_combo_vocab": int(len(drugs)),
        "n_drugs_beataml_vocab": int(n_beataml_drugs),
        "drugs_with_learned_synergy": int(len(combo_vocab_set)),
        "fold_val_rmses": [round(x, 4) for x in fold_rmses],
        "mean_val_rmse": round(mean_rmse, 4),
        "cv_pooled_pearson": round(pearson, 4),
        "cv_pooled_spearman": round(spr, 4),
        "synergy_col": cfg.synergy_col,
        "outputs": {
            "cv_predictions": str(out_dir / "cv_predictions.csv"),
            "per_patient_top5": str(out_dir / "per_patient_top5_combos.csv"),
            "combo_auc_npz": str(out_dir / "combo_auc_predictions.npz"),
        },
    }
    (out_dir / "combo_predictor_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", default="data/canonical/drugcomb_aml_pairs.csv")
    ap.add_argument("--baseline-pred",
                    default="runs/baseline_single_drug_mlp/predictions_all_patients_all_drugs.csv")
    ap.add_argument("--out", default="runs/combo_predictor")
    args = ap.parse_args()
    train_combo_predictor(
        pairs_path=Path(args.pairs),
        baseline_pred_path=Path(args.baseline_pred),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
