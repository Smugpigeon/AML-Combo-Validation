"""Path B v3 — Knowledge distillation from Path A clonal-coverage teacher.

The v2 Set Transformer (Path B) was trained on 55K single-drug AUC samples.
It predicts single-drug AUC well (ρ=0.691) but its pair rankings do NOT
align with clinical canonical combos (Gilt+Ven rank=10 for FLT3-mut, should
be ≤3 per JCO 2024 trial evidence).

Route 2 fix: distill Path A's clonal-coverage ranking into the ST's pair
predictions via a **multi-task loss**:

    L_total = α · L_single-drug-MSE     (preserves v2 single-drug capability)
            + β · L_pair-ranking-from-A  (aligns pair rankings to biology)

The pair loss uses a **differentiable Pearson correlation**: within each
batch of K sampled pairs for one patient, compute
    r = Pearson(-ST_pair_aucs, PathA_coverage_scores)
(negating ST AUC so higher = better, matching coverage direction), and
minimize `1 - r`. This imposes a SOFT ranking constraint: ST's top-k pair
picks gradually rise to agree with Path A's, without forcing exact values.

α=1.0, β=0.3 by default — single-drug task stays primary, pair ranking
added as regularizer. β can be tuned; β=0.0 recovers v2 behaviour.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from combo_val.combo.clonal_coverage import (
    CLONE_ARCHETYPES,
    build_drug_clone_coverage,
    build_patient_clone_matrix,
)
from combo_val.combo.set_drug_predictor import (
    PatientDrugSetDataset,
    SetDrugPredictor,
    SetDrugPredictorConfig,
    collate_variable_arity,
)


@dataclass(frozen=True)
class DistillConfig:
    """Config for the v2 → v3 fine-tuning pass."""
    # v2 checkpoint to initialize from
    v2_checkpoint: Path = Path("runs/set_drug_predictor_v2_stable/final_model.pt")
    out_dir: Path = Path("runs/set_drug_predictor_v3_distilled")

    # Clinical drug subset used for pair sampling (must be subset of training vocab)
    clinical_drug_filter: tuple[str, ...] = (
        "Venetoclax", "Azacytidine", "Cytarabine", "Midostaurin",
        "Quizartinib (AC220)", "Gilteritinib", "Ivosidenib", "Enasidenib",
        "Sorafenib", "Crenolanib", "Dasatinib", "Nilotinib", "Imatinib",
        "Ponatinib", "Ruxolitinib (INCB018424)", "Trametinib (GSK1120212)",
        "Selumetinib (AZD6244)", "Alisertib (MLN8237)", "Pacritinib",
        "Crizotinib (PF-2341066)",
    )

    # Multi-task loss weights — loss scales differ dramatically (single MSE
    # ~2200, pair (1-r) ~1.0), so we normalize single by a reference variance
    # before combining. After normalization, both losses live in ~[0, 2] range
    # and alpha/beta are directly interpretable as relative priorities.
    single_mse_norm_scale: float = 2200.0  # rough v2 val-MSE (divides MSE)
    alpha_single: float = 1.0   # normalized single-drug MSE weight
    beta_pair: float = 1.0      # pair ranking loss weight (equal priority)

    # Distillation training
    n_pairs_per_patient_per_batch: int = 20   # sample K pairs per patient
    lr: float = 5e-4            # lower than v2 (fine-tuning)
    weight_decay: float = 1e-5
    n_epochs: int = 8           # fine-tune, not retrain
    batch_size: int = 256        # for single-drug loader
    pair_batch_size: int = 16    # patients per pair-distill batch
    device: str = "auto"
    random_state: int = 42

    # CV
    n_folds: int = 5


# ---------------------------------------------------------------------------
# Teacher signal pre-computation
# ---------------------------------------------------------------------------


def precompute_path_a_teacher(
    patient_features: pd.DataFrame,
    clinical_drug_filter: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int]]]:
    """Precompute Path A coverage score for every (patient, pair) combination.

    Returns
    -------
    patient_clones : (n_patients, n_clones) DataFrame of clone weights.
    coverage_matrix : (n_patients, n_pairs) ndarray of A coverage scores in [0,1].
    pair_indices : list of (drug_idx_i, drug_idx_j) into clinical_drug_filter.
    """
    patient_clones = build_patient_clone_matrix(patient_features)
    drugs = list(clinical_drug_filter)
    drug_clone_cov = build_drug_clone_coverage(drugs).to_numpy(dtype=np.float64)
    clones_arr = patient_clones.to_numpy(dtype=np.float64)

    pair_indices = list(combinations(range(len(drugs)), 2))
    n_patients = clones_arr.shape[0]
    n_pairs = len(pair_indices)
    n_clones = clones_arr.shape[1]

    coverage_matrix = np.zeros((n_patients, n_pairs), dtype=np.float32)
    for pi, (i, j) in enumerate(pair_indices):
        # Bliss-IDA: 1 - (1 - c_i) * (1 - c_j) per clone
        cov_i = np.clip(drug_clone_cov[i], 0.0, 1.0)
        cov_j = np.clip(drug_clone_cov[j], 0.0, 1.0)
        per_clone_pair_cov = 1.0 - (1.0 - cov_i) * (1.0 - cov_j)  # (n_clones,)
        present = (clones_arr > 0).astype(np.float64)
        weights = clones_arr * present
        total_w = weights.sum(axis=1)
        num = (weights * per_clone_pair_cov[None, :]).sum(axis=1)
        coverage_matrix[:, pi] = np.where(total_w > 0, num / total_w, 0.0)

    return patient_clones, coverage_matrix, pair_indices


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def pair_ranking_loss_pearson(
    st_pair_aucs: torch.Tensor,      # (K,) predicted AUCs (lower = better)
    coverage_scores: torch.Tensor,   # (K,) Path A coverages (higher = better)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Pearson correlation between -ST_aucs and coverage.

    Minimize (1 - Pearson) → drive ST pair ranking to match coverage ranking.

    If all K scores are nearly constant (degenerate batch), return 0 to avoid
    divide-by-zero.
    """
    if st_pair_aucs.numel() < 2:
        return torch.zeros((), device=st_pair_aucs.device)

    x = -st_pair_aucs                                       # flip: higher = better
    y = coverage_scores
    xm = x - x.mean()
    ym = y - y.mean()
    cov = (xm * ym).sum()
    denom = torch.sqrt((xm * xm).sum() * (ym * ym).sum() + eps)
    r = cov / (denom + eps)
    return 1.0 - r                                           # in [0, 2]


# ---------------------------------------------------------------------------
# Distillation trainer
# ---------------------------------------------------------------------------


def _build_pair_tensors(
    patient_feats: torch.Tensor,          # (B, n_feat)
    sampled_pair_idx: torch.Tensor,       # (B, K) indices into pair_indices list
    pair_indices: list[tuple[int, int]],  # (d1_idx, d2_idx) in drug-vocab space
    vocab_to_st_int: list[int],           # map clinical_drug_filter idx → ST drug_to_int
    coverage_matrix: torch.Tensor,        # (n_patients_total, n_pairs_total)
    patient_row_idx: torch.Tensor,        # (B,) which patients in the batch
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand (B, K) pair samples to (B*K, 2)-size drug sets + matching teacher scores.

    Returns
    -------
    drug_ids       : (B*K, 2) long tensor, +1-shifted for padding_idx=0.
    drug_mask      : (B*K, 2) float tensor, all 1.0.
    pf_repeated    : (B*K, n_feat) float tensor.
    teacher_cov    : (B, K) float tensor of Path A coverage scores.
    """
    B, K = sampled_pair_idx.shape
    flat_pair_idx = sampled_pair_idx.view(-1)  # (B*K,)

    # drug pair (d1, d2) for each sample
    pair_arr = np.asarray(pair_indices, dtype=np.int64)  # (n_pairs, 2)
    pair_drug_idx_np = pair_arr[flat_pair_idx.cpu().numpy()]  # (B*K, 2)
    # Map clinical-filter idx → ST drug_to_int, then +1 for padding convention
    v_map = np.asarray(vocab_to_st_int, dtype=np.int64)
    st_drug_ids = v_map[pair_drug_idx_np] + 1  # (B*K, 2)

    drug_ids = torch.tensor(st_drug_ids, dtype=torch.long, device=device)
    drug_mask = torch.ones(B * K, 2, dtype=torch.float32, device=device)
    pf_repeated = patient_feats.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

    # Teacher coverage per (patient, pair) — index by patient_row_idx × sampled_pair_idx
    teacher_cov = coverage_matrix[
        patient_row_idx.unsqueeze(1).expand(B, K), sampled_pair_idx
    ]  # (B, K)

    return drug_ids, drug_mask, pf_repeated, teacher_cov


def distill_v2_to_v3(
    cfg: DistillConfig | None = None,
    patient_features_path: Path = Path("data/canonical/beataml_patient_features.csv"),
    drug_response_path: Path = Path("data/canonical/beataml_drug_response_long.csv"),
) -> dict:
    cfg = cfg or DistillConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    # Device
    if cfg.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(cfg.device)

    # ---- Load v2 checkpoint ----
    ckpt = torch.load(cfg.v2_checkpoint, weights_only=False, map_location=device)
    v2_cfg_dict = ckpt["cfg"]
    from dataclasses import fields as _fields
    v2_cfg_kwargs = {
        f.name: v2_cfg_dict[f.name] for f in _fields(SetDrugPredictorConfig)
        if f.name in v2_cfg_dict
    }
    v2_cfg = SetDrugPredictorConfig(**v2_cfg_kwargs)

    drug_vocab: list[str] = ckpt["drug_vocab"]
    drug_to_int: dict[str, int] = ckpt["drug_to_int"]
    feature_cols: list[str] = ckpt["feature_cols"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    # ---- Load data (same scaling as v2) ----
    pf = pd.read_csv(patient_features_path).set_index("patient_id")
    pf = pf[feature_cols]  # align columns
    safe_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    feat_scaled = pd.DataFrame(
        (pf.values - scaler_mean) / safe_scale,
        index=pf.index, columns=feature_cols,
    )

    long_df = pd.read_csv(drug_response_path).dropna(subset=["auc"])
    long_df["patient_id"] = long_df["patient_id"].astype(int)
    long_df = long_df[long_df["patient_id"].isin(feat_scaled.index)].reset_index(drop=True)

    print(f"[distill] patients: {feat_scaled.shape[0]}  features: {feat_scaled.shape[1]}  "
          f"drugs: {len(drug_vocab)}  samples: {len(long_df)}")
    print(f"[distill] device: {device}  alpha={cfg.alpha_single}  beta={cfg.beta_pair}")

    # ---- Precompute Path A teacher signal ----
    print("[distill] precomputing Path A coverage teacher ...")
    # Unscaled features needed for clone matrix (we'll use the ORIGINAL
    # patient_features.csv for clone detection, not the scaled one — because
    # the mut_* binaries need to be 0/1, not z-scored).
    pf_raw = pd.read_csv(patient_features_path).set_index("patient_id")
    drugs_in_filter = [d for d in cfg.clinical_drug_filter if d in drug_to_int]
    if len(drugs_in_filter) < len(cfg.clinical_drug_filter):
        missing = set(cfg.clinical_drug_filter) - set(drugs_in_filter)
        print(f"[distill] WARNING: dropping {len(missing)} drugs missing from ST vocab: {missing}")

    patient_clones, coverage_matrix, pair_indices = precompute_path_a_teacher(
        pf_raw, tuple(drugs_in_filter),
    )
    print(f"[distill] teacher: {coverage_matrix.shape[0]} patients × "
          f"{coverage_matrix.shape[1]} pairs; coverage range "
          f"[{coverage_matrix.min():.3f}, {coverage_matrix.max():.3f}]")

    # Map clinical-drug-filter → ST drug_to_int
    vocab_to_st_int = [drug_to_int[d] for d in drugs_in_filter]

    # Torch tensors
    coverage_tensor = torch.tensor(coverage_matrix, dtype=torch.float32, device=device)

    # ---- CV fine-tuning ----
    unique_patients = sorted(feat_scaled.index.tolist())
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(unique_patients)):
        train_pids = {unique_patients[i] for i in train_idx}
        val_pids = {unique_patients[i] for i in val_idx}
        train_df = long_df[long_df["patient_id"].isin(train_pids)]
        val_df = long_df[long_df["patient_id"].isin(val_pids)]

        # Re-initialize model from v2 checkpoint for each fold (fair CV)
        model = SetDrugPredictor(
            n_drugs=len(drug_vocab),
            n_patient_features=feat_scaled.shape[1],
            cfg=v2_cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        # Single-drug loader for this fold
        train_ds = PatientDrugSetDataset(train_df, feat_scaled, drug_to_int)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_variable_arity, num_workers=0,
        )

        # Training patient row indices (into feat_scaled / coverage_tensor)
        pid_to_row = {pid: i for i, pid in enumerate(feat_scaled.index)}
        train_patient_row_idx = torch.tensor(
            [pid_to_row[p] for p in sorted(train_pids)],
            dtype=torch.long, device=device,
        )

        n_total_pairs = len(pair_indices)

        # ---- Training loop ----
        t0 = time.time()
        for epoch in range(cfg.n_epochs):
            model.train()
            total_single_loss = 0.0
            total_pair_loss = 0.0
            total_samples = 0
            for batch in train_loader:
                # Single-drug MSE (preserving v2 capability)
                drug_ids = batch["drug_ids"].to(device)
                drug_mask = batch["drug_mask"].to(device)
                pf_b = batch["patient_features"].to(device)
                auc = batch["auc"].to(device)
                pred = model(drug_ids, drug_mask, pf_b)
                single_loss = F.mse_loss(pred, auc)

                # Sample a pair-distill batch (patients from training fold)
                batch_pids = train_patient_row_idx[
                    torch.randint(
                        0, len(train_patient_row_idx),
                        (cfg.pair_batch_size,), device=device,
                    )
                ]
                pair_pf = torch.tensor(
                    feat_scaled.iloc[batch_pids.cpu().numpy()].to_numpy(dtype=np.float32),
                    device=device,
                )
                sampled_pair_idx = torch.randint(
                    0, n_total_pairs,
                    (cfg.pair_batch_size, cfg.n_pairs_per_patient_per_batch),
                    device=device,
                )
                d_ids, d_mask, pf_rep, teacher = _build_pair_tensors(
                    pair_pf, sampled_pair_idx, pair_indices,
                    vocab_to_st_int, coverage_tensor, batch_pids, device,
                )
                st_pair_pred = model(d_ids, d_mask, pf_rep).view(
                    cfg.pair_batch_size, cfg.n_pairs_per_patient_per_batch,
                )
                # Per-patient Pearson loss, average over batch
                pair_loss = torch.stack([
                    pair_ranking_loss_pearson(st_pair_pred[i], teacher[i])
                    for i in range(cfg.pair_batch_size)
                ]).mean()

                # Normalize single-drug MSE to ~unit scale so pair loss has
                # comparable gradient magnitude.
                normalized_single = single_loss / cfg.single_mse_norm_scale
                loss = cfg.alpha_single * normalized_single + cfg.beta_pair * pair_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                total_single_loss += single_loss.item() * auc.shape[0]
                total_pair_loss += pair_loss.item() * auc.shape[0]
                total_samples += auc.shape[0]

            avg_single = total_single_loss / max(total_samples, 1)
            avg_pair = total_pair_loss / max(total_samples, 1)
            print(f"[distill] fold {fold_i + 1}/{cfg.n_folds} epoch {epoch + 1}/{cfg.n_epochs}  "
                  f"single_mse={avg_single:.2f}  pair_1-r={avg_pair:.4f}")

        # ---- Evaluate this fold ----
        model.eval()
        val_ds = PatientDrugSetDataset(val_df, feat_scaled, drug_to_int)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_variable_arity, num_workers=0,
        )
        preds, trues, val_df_reset = [], [], val_df.reset_index(drop=True)
        with torch.no_grad():
            for batch in val_loader:
                drug_ids = batch["drug_ids"].to(device)
                drug_mask = batch["drug_mask"].to(device)
                pf_b = batch["patient_features"].to(device)
                pred = model(drug_ids, drug_mask, pf_b)
                preds.append(pred.cpu().numpy())
                trues.append(batch["auc"].cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mae = float(np.mean(np.abs(preds - trues)))
        val_df_reset["pred"] = preds

        per_patient_r = []
        for pid, grp in val_df_reset.groupby("patient_id"):
            if len(grp) >= 3:
                r = spearmanr(grp["pred"], grp["auc"]).correlation
                if not np.isnan(r):
                    per_patient_r.append(r)
        pp_rho = float(np.mean(per_patient_r)) if per_patient_r else float("nan")

        fold_metrics.append({
            "fold": fold_i + 1,
            "val_mae": round(mae, 3),
            "mean_per_patient_spearman": round(pp_rho, 4),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[distill] fold {fold_i + 1}/{cfg.n_folds}  MAE={mae:.2f}  "
              f"per-patient ρ={pp_rho:.3f}  [{time.time() - t0:.0f}s]")

    # ---- Full-data fine-tune (average best epoch across folds) ----
    print(f"[distill] training full-data model for {cfg.n_epochs} epochs ...")
    model = SetDrugPredictor(
        n_drugs=len(drug_vocab),
        n_patient_features=feat_scaled.shape[1],
        cfg=v2_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    full_ds = PatientDrugSetDataset(long_df, feat_scaled, drug_to_int)
    full_loader = DataLoader(
        full_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_variable_arity,
    )
    pid_to_row = {pid: i for i, pid in enumerate(feat_scaled.index)}
    all_patient_row_idx = torch.tensor(
        list(range(len(feat_scaled))), dtype=torch.long, device=device,
    )
    n_total_pairs = len(pair_indices)
    for epoch in range(cfg.n_epochs):
        model.train()
        for batch in full_loader:
            drug_ids = batch["drug_ids"].to(device)
            drug_mask = batch["drug_mask"].to(device)
            pf_b = batch["patient_features"].to(device)
            auc = batch["auc"].to(device)
            pred = model(drug_ids, drug_mask, pf_b)
            single_loss = F.mse_loss(pred, auc)
            batch_pids = all_patient_row_idx[
                torch.randint(0, len(all_patient_row_idx),
                              (cfg.pair_batch_size,), device=device)
            ]
            pair_pf = torch.tensor(
                feat_scaled.iloc[batch_pids.cpu().numpy()].to_numpy(dtype=np.float32),
                device=device,
            )
            sampled_pair_idx = torch.randint(
                0, n_total_pairs,
                (cfg.pair_batch_size, cfg.n_pairs_per_patient_per_batch),
                device=device,
            )
            d_ids, d_mask, pf_rep, teacher = _build_pair_tensors(
                pair_pf, sampled_pair_idx, pair_indices,
                vocab_to_st_int, coverage_tensor, batch_pids, device,
            )
            st_pair_pred = model(d_ids, d_mask, pf_rep).view(
                cfg.pair_batch_size, cfg.n_pairs_per_patient_per_batch,
            )
            pair_loss = torch.stack([
                pair_ranking_loss_pearson(st_pair_pred[i], teacher[i])
                for i in range(cfg.pair_batch_size)
            ]).mean()
            loss = cfg.alpha_single * single_loss + cfg.beta_pair * pair_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        print(f"[distill] full-data epoch {epoch + 1}/{cfg.n_epochs}")

    # ---- Save v3 checkpoint (same schema as v2 for drop-in replacement) ----
    ckpt_path = cfg.out_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": asdict(v2_cfg),
        "distill_cfg": asdict(cfg),
        "drug_vocab": drug_vocab,
        "drug_to_int": drug_to_int,
        "feature_cols": feature_cols,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "n_patient_features": feat_scaled.shape[1],
        "n_drugs": len(drug_vocab),
        "architecture": "SetTransformer_v3_distilled",
    }, ckpt_path)

    overall = {
        "cv_folds": fold_metrics,
        "cv_mean_per_patient_spearman": round(
            float(np.mean([f["mean_per_patient_spearman"] for f in fold_metrics])), 4
        ),
        "cv_std_per_patient_spearman": round(
            float(np.std([f["mean_per_patient_spearman"] for f in fold_metrics])), 4
        ),
        "cv_mean_mae": round(
            float(np.mean([f["val_mae"] for f in fold_metrics])), 3
        ),
    }
    import json
    (cfg.out_dir / "cv_metrics.json").write_text(
        json.dumps({
            "cfg": asdict(cfg),
            "device": str(device),
            "architecture": "SetTransformer_v3_distilled",
            "overall_cv": overall,
            "output_files": {"checkpoint": str(ckpt_path)},
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[distill] DONE. CV ρ={overall['cv_mean_per_patient_spearman']:.3f} ± "
          f"{overall['cv_std_per_patient_spearman']:.3f}  "
          f"MAE={overall['cv_mean_mae']:.2f}")
    return {"overall_cv": overall, "checkpoint": str(ckpt_path)}
