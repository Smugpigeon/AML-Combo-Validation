"""Route 4: Learn drug-pair synergy from real ex-vivo cell-line data.

Takes the frozen ST v2 backbone (which outputs a set representation) and
trains a small synergy head on 186 ALMANAC-HL60 pair measurements.

Philosophy:
  - ST v2 already predicts patient-specific single-drug AUC (ρ=0.691).
  - What it DOESN'T know: how pairs of drugs interact beyond additive.
  - ALMANAC gives us 186 real ex-vivo pair synergy measurements.
  - Freeze ST's backbone (preserving its learned patient/drug representations)
    and train only a small adapter that maps drug-set repr → synergy_loewe.

At inference, combine per-patient single-drug AUC with learned synergy:
  combo_auc(p, d1, d2) = 0.5 * (ST_mono(p, d1) + ST_mono(p, d2))
                       + k * ST_synergy({d1, d2})

Where k calibrates synergy-score units to AUC units.

This is NOT the same as the mechanism-prior approach:
  - Mech prior: hand-coded axes from AML knowledge vocab
  - Route 4:    learned from REAL ex-vivo pair measurements on 19 drugs

The "patient-specific" personalization comes entirely through ST_mono.
Synergy is patient-agnostic (acceptable given we only have 1 cell line's data).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold

from combo_val.combo.set_drug_predictor import (
    SetDrugPredictor,
    SetDrugPredictorConfig,
)


@dataclass(frozen=True)
class SynergyHeadConfig:
    # Direct-pair head architecture
    hidden_dim: int = 64
    dropout: float = 0.15
    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    min_epochs: int = 10
    random_state: int = 42
    n_folds: int = 5
    # Target column (use Loewe; Bliss is alternative)
    synergy_col: str = "synergy_loewe"
    # ST v2 checkpoint (used for drug vocab + emb_dim; embeddings are
    # re-initialized fresh for synergy-specific training)
    backbone_checkpoint: Path = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
    # Initialization: "fresh" (default, best Pearson) or "warm_start" (from ST v2)
    init_mode: str = "fresh"


class SynergyHead(nn.Module):
    """Symmetric drug-pair synergy predictor — initialized from ST v2's
    single-drug embeddings.

    Follows Week-3 SynergyMLP's design (which achieved Pearson=0.48 on this
    same data): concatenate e1+e2 (commutative) and |e1-e2| (asymmetry) into
    a 2×drug_emb_dim vector, pass through a small MLP.

    Initialization: copy ST v2's drug_embedding.weight into this head's
    drug embedding table. Fine-tuning lets the embeddings morph toward
    synergy-aware representations while starting from useful single-drug
    biology.
    """

    def __init__(self, n_drugs: int, drug_emb_dim: int, cfg: SynergyHeadConfig):
        super().__init__()
        # +1 for padding_idx=0 shift consistent with ST v2
        self.drug_embedding = nn.Embedding(n_drugs + 1, drug_emb_dim, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(2 * drug_emb_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def init_from_st_embeddings(self, st_drug_embedding_weight: torch.Tensor):
        """Copy ST v2's drug embeddings as a warm start."""
        with torch.no_grad():
            self.drug_embedding.weight.copy_(st_drug_embedding_weight)

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """d1, d2 are 1-indexed drug ids (0 is padding)."""
        e1 = self.drug_embedding(d1)
        e2 = self.drug_embedding(d2)
        sym = torch.cat([e1 + e2, (e1 - e2).abs()], dim=-1)
        return self.head(sym).squeeze(-1)


def _load_backbone(checkpoint_path: Path, freeze: bool = True
                   ) -> tuple[SetDrugPredictor, dict, torch.device]:
    """Load ST v2. Freeze if freeze=True, else leave trainable for fine-tune."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    from dataclasses import fields as _fields
    cfg_dict = ckpt["cfg"]
    kwargs = {f.name: cfg_dict[f.name] for f in _fields(SetDrugPredictorConfig)
              if f.name in cfg_dict}
    backbone_cfg = SetDrugPredictorConfig(**kwargs)
    backbone = SetDrugPredictor(
        n_drugs=ckpt["n_drugs"],
        n_patient_features=ckpt["n_patient_features"],
        cfg=backbone_cfg,
    ).to(device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    if freeze:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone, ckpt, device


def _build_pair_input_tensors(
    drug_idx_pairs: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize drug_ids and drug_mask tensors for the pair list."""
    B = len(drug_idx_pairs)
    drug_ids = torch.zeros(B, 2, dtype=torch.long, device=device)
    drug_mask = torch.ones(B, 2, dtype=torch.float32, device=device)
    for i, (d1, d2) in enumerate(drug_idx_pairs):
        drug_ids[i, 0] = d1 + 1
        drug_ids[i, 1] = d2 + 1
    return drug_ids, drug_mask


def _pair_set_repr(
    backbone: SetDrugPredictor,
    drug_idx_pairs: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Get set_repr for each pair by running through the frozen backbone."""
    drug_ids, drug_mask = _build_pair_input_tensors(drug_idx_pairs, device)
    with torch.no_grad():
        set_repr = backbone.encode_set(drug_ids, drug_mask)
    return set_repr


def train_synergy_head(
    pairs_path: Path = Path("data/canonical/drugcomb_aml_pairs.csv"),
    out_dir: Path = Path("runs/set_drug_predictor_v3_route4"),
    cfg: SynergyHeadConfig | None = None,
) -> dict:
    cfg = cfg or SynergyHeadConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    # Route 4 FINAL: direct pair model with FRESH drug embeddings. The
    # Week-3 SynergyMLP design works best when the drug embeddings are
    # trained from scratch specifically for synergy (ST v2's single-drug-
    # optimized embeddings are slightly biased away from synergy-optimal
    # directions; fresh init Pearson 0.41 > warm-start Pearson 0.37).
    # We still inherit drug vocabulary and emb_dim from ST v2 so the head
    # is API-compatible with ST v2 for ensemble inference.
    print(f"[route4] mode: DIRECT_PAIR_MODEL_fresh_init")
    print(f"[route4] loading ST v2 metadata from {cfg.backbone_checkpoint}")
    backbone, ckpt, device = _load_backbone(cfg.backbone_checkpoint, freeze=True)
    drug_to_int: dict[str, int] = ckpt["drug_to_int"]
    set_hidden_dim = backbone.pma.S.shape[-1]
    drug_emb_dim = backbone.drug_embedding.weight.shape[1]
    n_drugs_vocab = backbone.drug_embedding.weight.shape[0] - 1
    st_drug_emb = backbone.drug_embedding.weight.detach().clone()
    print(f"[route4] drug vocab={n_drugs_vocab}, emb_dim={drug_emb_dim}")

    # Load ALMANAC pairs, filter to drugs in vocab
    pairs = pd.read_csv(pairs_path)
    pairs = pairs[pairs["drug1_id"].isin(drug_to_int) & pairs["drug2_id"].isin(drug_to_int)]
    pairs = pairs.dropna(subset=[cfg.synergy_col]).reset_index(drop=True)
    print(f"[route4] pairs after vocab filter: {len(pairs)}")

    pair_drug_idx = [
        (drug_to_int[r["drug1_id"]], drug_to_int[r["drug2_id"]])
        for _, r in pairs.iterrows()
    ]
    # Build d1, d2 tensors (1-indexed since padding_idx=0)
    d1_all = torch.tensor([d1 + 1 for d1, _ in pair_drug_idx], dtype=torch.long, device=device)
    d2_all = torch.tensor([d2 + 1 for _, d2 in pair_drug_idx], dtype=torch.long, device=device)
    synergies_all = torch.tensor(
        pairs[cfg.synergy_col].values, dtype=torch.float32, device=device,
    )

    # 5-fold CV
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    fold_metrics = []
    all_preds_for_holdout = np.zeros(len(pairs), dtype=np.float32)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(pairs)))):
        torch.manual_seed(cfg.random_state + fold_i)

        head = SynergyHead(n_drugs_vocab, drug_emb_dim, cfg).to(device)
        if cfg.init_mode == "warm_start":
            head.init_from_st_embeddings(st_drug_emb)
        opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_d1 = d1_all[train_idx]
        train_d2 = d2_all[train_idx]
        train_y = synergies_all[train_idx]
        val_d1 = d1_all[val_idx]
        val_d2 = d2_all[val_idx]
        val_y = synergies_all[val_idx]

        best_val_mae = float("inf")
        best_head_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(cfg.max_epochs):
            head.train()
            perm = torch.randperm(len(train_idx), device=device)
            for i in range(0, len(train_idx), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                pred = head(train_d1[idx], train_d2[idx])
                # Symmetric augmentation: also train on swapped order (d2, d1)
                pred_sym = head(train_d2[idx], train_d1[idx])
                loss = 0.5 * (F.mse_loss(pred, train_y[idx]) + F.mse_loss(pred_sym, train_y[idx]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                opt.step()

            head.eval()
            with torch.no_grad():
                val_pred_a = head(val_d1, val_d2)
                val_pred_b = head(val_d2, val_d1)
                val_pred = 0.5 * (val_pred_a + val_pred_b)
                val_mae = F.l1_loss(val_pred, val_y).item()

            if val_mae < best_val_mae - 1e-4:
                best_val_mae = val_mae
                best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch + 1 >= cfg.min_epochs and epochs_no_improve >= cfg.patience:
                break

        head.load_state_dict(best_head_state)
        head.eval()
        with torch.no_grad():
            val_pred = 0.5 * (head(val_d1, val_d2) + head(val_d2, val_d1))
            val_pred = val_pred.cpu().numpy()
        val_y_cpu = val_y.cpu().numpy()
        all_preds_for_holdout[val_idx] = val_pred

        rmse = float(np.sqrt(np.mean((val_pred - val_y_cpu) ** 2)))
        mae = float(np.mean(np.abs(val_pred - val_y_cpu)))
        try:
            pearson_r = float(pearsonr(val_pred, val_y_cpu).statistic)
            spearman_r = float(spearmanr(val_pred, val_y_cpu).correlation)
        except Exception:
            pearson_r = spearman_r = float("nan")

        fold_metrics.append({
            "fold": fold_i + 1,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_mae": round(mae, 3),
            "val_rmse": round(rmse, 3),
            "pearson_r": round(pearson_r, 4),
            "spearman_r": round(spearman_r, 4),
            "best_epoch": best_epoch,
        })
        print(f"[route4] fold {fold_i+1}: MAE={mae:.2f} RMSE={rmse:.2f} "
              f"Pearson={pearson_r:.3f} Spearman={spearman_r:.3f}")

    # Full-data training for production head
    torch.manual_seed(cfg.random_state + 999)
    full_head = SynergyHead(n_drugs_vocab, drug_emb_dim, cfg).to(device)
    full_head.init_from_st_embeddings(st_drug_emb)
    opt = torch.optim.Adam(full_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    avg_best_epochs = int(np.mean([f.get("best_epoch", 20) or 20 for f in fold_metrics])) or 20
    for epoch in range(avg_best_epochs):
        full_head.train()
        perm = torch.randperm(len(pairs), device=device)
        for i in range(0, len(pairs), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            pred = full_head(d1_all[idx], d2_all[idx])
            pred_sym = full_head(d2_all[idx], d1_all[idx])
            loss = 0.5 * (F.mse_loss(pred, synergies_all[idx]) + F.mse_loss(pred_sym, synergies_all[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_head.parameters(), max_norm=1.0)
            opt.step()

    ckpt_out = out_dir / "synergy_head.pt"
    torch.save({
        "synergy_head_state_dict": full_head.state_dict(),
        "cfg": asdict(cfg),
        "n_drugs_vocab": n_drugs_vocab,
        "drug_emb_dim": drug_emb_dim,
        "backbone_checkpoint": str(cfg.backbone_checkpoint),
        "drug_vocab": ckpt["drug_vocab"],
        "drug_to_int": drug_to_int,
        "synergy_col": cfg.synergy_col,
        "n_training_pairs": len(pairs),
    }, ckpt_out)

    # Save CV predictions
    pairs_with_pred = pairs.copy()
    pairs_with_pred[f"{cfg.synergy_col}_predicted"] = all_preds_for_holdout
    pairs_with_pred.to_csv(out_dir / "cv_predictions.csv", index=False)

    # Pooled CV metrics
    y_true = pairs[cfg.synergy_col].values
    y_pred = all_preds_for_holdout
    pooled = {
        "mae": round(float(np.mean(np.abs(y_pred - y_true))), 3),
        "rmse": round(float(np.sqrt(np.mean((y_pred - y_true) ** 2))), 3),
        "pearson_r": round(float(pearsonr(y_pred, y_true).statistic), 4),
        "spearman_r": round(float(spearmanr(y_pred, y_true).correlation), 4),
        "n_pairs": int(len(pairs)),
    }

    import json
    summary = {
        "cfg": asdict(cfg),
        "fold_metrics": fold_metrics,
        "pooled_cv": pooled,
        "data_std": round(float(np.std(y_true)), 3),
        "outputs": {
            "synergy_head": str(ckpt_out),
            "cv_predictions": str(out_dir / "cv_predictions.csv"),
        },
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[route4] DONE")
    print(f"[route4] pooled CV: MAE={pooled['mae']} RMSE={pooled['rmse']} "
          f"Pearson={pooled['pearson_r']} Spearman={pooled['spearman_r']}")
    print(f"[route4] data std: {summary['data_std']} "
          f"(null-baseline RMSE would be {summary['data_std']})")
    return summary


def main():
    train_synergy_head()


if __name__ == "__main__":
    main()
