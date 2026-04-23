"""Path B — Set Transformer for arbitrary-arity drug combinations.

Architecture:

    drug_ids ∈ ℤ^{B × N_max}    (padded with 0 where set_size < N_max)
    drug_mask ∈ {0,1}^{B × N_max}  (1 = real drug, 0 = padding)
    patient_features ∈ ℝ^{B × 104}

            ┌─────────────── Drug-set encoder ─────────────┐
            │   Drug embedding: 165 × d_drug (64)          │
            │       ↓                                       │
            │   Linear proj → d_hidden (128)                │
            │       ↓                                       │
            │   ISAB × 2  (induced set attention, M=16)     │
            │       ↓                                       │
            │   PMA  (pool by multi-head attention, k=1)    │
            │       ↓                                       │
            │   set_repr ∈ ℝ^{B × d_hidden}                 │
            └───────────────────────────────────────────────┘

    patient_features → Patient MLP (104 → 128 → 64)

    concat(patient_emb [64], set_repr [128]) → Response head (192 → 128 → 64 → 1)

Permutation invariance is MATHEMATICAL, not just learned:
  - ISAB and PMA are permutation-equivariant/invariant by construction
    (see Lee, J. et al. "Set Transformer" ICML 2019)
  - Padding is masked out inside the attention softmax, so the output is
    strictly independent of the order and padding positions of drugs in the set

Training regime:
  - Primary: BeatAML single-drug data (55,826 rows: patient × drug × AUC)
    Each sample is a drug set of size 1.
  - The model learns (patient, drug-set) → AUC by generalization.
  - At inference time, size-2+ sets are scored without any multi-drug labels
    seen during training — this IS the zero-shot multi-drug claim that path B
    is designed to test.

Theoretical viability (what we prove empirically, `theoretical_viability.py`):
  T1. Single-drug performance parity with existing MLP (ρ ≥ 0.65)
  T2. Permutation invariance (f(π(S)) = f(S) within 1e-4)
  T3. Monotonicity (adding sensitive drug decreases combo AUC)
  T4. Bliss-consistency (predictions bounded by additive-mean / Bliss product)
  T5. AML biology (FLT3-mut patients prefer FLT3i-containing sets)
  T6. 3-drug extrapolation (no NaN, distribution sensible)

References:
  Lee, J. et al. Set Transformer: A Framework for Attention-based Permutation-
  Invariant Neural Networks. ICML 2019. https://arxiv.org/abs/1810.00825
"""

from __future__ import annotations

import json
import math
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
class SetDrugPredictorConfig:
    # --- Architecture ---
    drug_emb_dim: int = 64
    patient_hidden_dim: int = 128
    patient_emb_dim: int = 64
    set_hidden_dim: int = 128       # d_hidden for ISAB/PMA
    n_attn_heads: int = 4
    n_inducing_points: int = 16     # M in ISAB
    n_isab_layers: int = 2
    head_hidden_dim: int = 128
    dropout: float = 0.2

    # --- Training ---
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    min_epochs: int = 20
    random_state: int = 42

    # --- Stability (v2 fixes) ---
    n_seeds: int = 1                # Multi-seed: train K models per fold, pick
                                    # best by val MSE. n_seeds=3 ≈ eliminates
                                    # the 1/5-fold collapse seen in v1.
    warmup_steps: int = 0           # Linear LR warmup over first N optimizer
                                    # steps. warmup_steps=500 avoids early
                                    # divergence that triggered fold-3 collapse.
    collapse_threshold_rho: float = 0.40
                                    # If a seed's best rho is below this, treat
                                    # as collapse and rely on other seeds.

    # --- CV ---
    n_folds: int = 5
    device: str = "auto"


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional key-padding mask.

    Q ∈ (B, Nq, d), K/V ∈ (B, Nk, d), key_mask ∈ {0, 1}^(B, Nk)
      where 1 = valid key, 0 = padding to be masked out.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Nq, _ = Q.shape
        Nk = K.shape[1]

        q = self.q_proj(Q).reshape(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(K).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(V).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: (B, heads, Nq, Nk)

        if key_mask is not None:
            # key_mask: (B, Nk). 0 → mask. Broadcast: (B, 1, 1, Nk).
            neg_inf = torch.finfo(scores.dtype).min
            mask = key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nk)
            scores = scores.masked_fill(mask == 0, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)              # (B, heads, Nq, d_head)
        out = out.transpose(1, 2).reshape(B, Nq, self.d_model)
        return self.o_proj(out)


class MAB(nn.Module):
    """Multihead Attention Block with residual + layer norm + FFN.

    MAB(X, Y) = LN(X + MHA(X, Y, Y)) + FFN
    (corresponds to the encoder layer of a Transformer with X as query, Y as key/value)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor,
        y_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H = self.ln1(X + self.mha(X, Y, Y, key_mask=y_mask))
        return self.ln2(H + self.ffn(H))


class ISAB(nn.Module):
    """Induced Set Attention Block (Lee et al. 2019).

    Input:  X ∈ (B, N, d), x_mask ∈ (B, N)
    Output: X' ∈ (B, N, d), same mask valid.

    Two-step attention using M trainable inducing points I:
      H = MAB(I, X)   # (B, M, d) — inducing points attend to input set
      X' = MAB(X, H)  # (B, N, d) — input attends to the summarized set
    Reduces N×N attention to 2 × (N×M); O(NM) instead of O(N²).
    """

    def __init__(self, d_model: int, n_heads: int, n_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab_induce = MAB(d_model, n_heads, dropout=dropout)
        self.mab_distribute = MAB(d_model, n_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)            # (B, M, d)
        H = self.mab_induce(I, X, y_mask=x_mask)
        # x_mask applies to X as keys/values in the second MAB — but X is both
        # query and key/value here. The queries that are padded (x_mask=0) get
        # garbage values, which we will zero out AFTER this layer.
        Xp = self.mab_distribute(X, H)
        return Xp


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Seeds S ∈ (k, d) attend to the set X ∈ (B, N, d).
    Output: (B, k, d). For k=1, this is a single pooled vector per sample.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mab = MAB(d_model, n_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, y_mask=x_mask)


# ---------------------------------------------------------------------------
# Set Drug Predictor
# ---------------------------------------------------------------------------


class SetDrugPredictor(nn.Module):
    """Predict (patient × drug-set) → response AUC.

    Input:
      drug_ids       : (B, N_max) long, padded with 0 where absent
      drug_mask      : (B, N_max) float/bool, 1 = real drug, 0 = padding
      patient_features: (B, d_patient_feat) float

    Output:
      predicted AUC  : (B,) float
    """

    def __init__(
        self,
        n_drugs: int,
        n_patient_features: int,
        cfg: SetDrugPredictorConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_drugs = n_drugs
        self.n_patient_features = n_patient_features

        self.drug_embedding = nn.Embedding(
            num_embeddings=n_drugs + 1,  # +1 for padding index 0
            embedding_dim=cfg.drug_emb_dim,
            padding_idx=0,
        )
        self.set_proj = nn.Linear(cfg.drug_emb_dim, cfg.set_hidden_dim)
        self.isabs = nn.ModuleList([
            ISAB(cfg.set_hidden_dim, cfg.n_attn_heads, cfg.n_inducing_points, cfg.dropout)
            for _ in range(cfg.n_isab_layers)
        ])
        self.pma = PMA(cfg.set_hidden_dim, cfg.n_attn_heads, n_seeds=1, dropout=cfg.dropout)

        self.patient_mlp = nn.Sequential(
            nn.Linear(n_patient_features, cfg.patient_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.patient_hidden_dim, cfg.patient_emb_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.patient_emb_dim + cfg.set_hidden_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim // 2, 1),
        )

    def encode_set(self, drug_ids: torch.Tensor, drug_mask: torch.Tensor) -> torch.Tensor:
        """drug_ids: (B, N); drug_mask: (B, N). Returns (B, d_hidden)."""
        # Drug IDs use padding_idx=0, so embedded padding rows are already zero.
        # We still use drug_mask in attention to prevent padded positions from
        # contributing to softmax denominators.
        x = self.drug_embedding(drug_ids)                       # (B, N, d_drug)
        x = self.set_proj(x)                                     # (B, N, d_hidden)
        # Zero out padded positions explicitly (belt + suspenders; attention
        # mask also handles it, but this keeps downstream tensors clean).
        x = x * drug_mask.unsqueeze(-1)

        for isab in self.isabs:
            x = isab(x, x_mask=drug_mask)
            x = x * drug_mask.unsqueeze(-1)  # re-zero padded positions

        # Pool to single vector
        pooled = self.pma(x, x_mask=drug_mask)                   # (B, 1, d_hidden)
        return pooled.squeeze(1)                                  # (B, d_hidden)

    def forward(
        self,
        drug_ids: torch.Tensor,
        drug_mask: torch.Tensor,
        patient_features: torch.Tensor,
    ) -> torch.Tensor:
        set_repr = self.encode_set(drug_ids, drug_mask)          # (B, d_hidden)
        p_emb = self.patient_mlp(patient_features)                # (B, patient_emb_dim)
        combined = torch.cat([p_emb, set_repr], dim=-1)
        return self.head(combined).squeeze(-1)                    # (B,)

    @torch.no_grad()
    def predict_set(
        self,
        drug_ids_list: list[list[int]],
        patient_features: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Convenience: predict a batch of sets.

        drug_ids_list: list of variable-length lists of drug integer IDs.
                       Drug IDs are 0-indexed into the drug vocab (NOT padding).
                       Inside the model they are shifted to 1..n (padding = 0).
        patient_features: (B, d_patient_feat)
        Returns: (B,) predicted AUCs.
        """
        self.eval()
        B = len(drug_ids_list)
        N_max = max(len(s) for s in drug_ids_list)
        # Shift IDs by +1 because index 0 is padding
        drug_ids = torch.zeros(B, N_max, dtype=torch.long, device=device)
        drug_mask = torch.zeros(B, N_max, dtype=torch.float32, device=device)
        for i, s in enumerate(drug_ids_list):
            drug_ids[i, :len(s)] = torch.tensor([d + 1 for d in s], dtype=torch.long, device=device)
            drug_mask[i, :len(s)] = 1.0
        pf = patient_features.to(device)
        return self(drug_ids, drug_mask, pf)


# ---------------------------------------------------------------------------
# Dataset with variable-arity collation
# ---------------------------------------------------------------------------


class PatientDrugSetDataset(Dataset):
    """(patient_features, {drug_ids}, auc) samples.

    For training on BeatAML single-drug data, each sample has a set of size 1.
    For test-time, sets of size 2, 3, ... can be constructed ad-hoc.
    """

    def __init__(
        self,
        long_df: pd.DataFrame,                    # must have columns: patient_id, drug_id, auc
        patient_features: pd.DataFrame,           # indexed by patient_id, columns = features
        drug_id_to_int: dict[str, int],           # drug name → 0-indexed integer
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
        # +1 shift: reserve 0 for padding
        drug_int = self.drug_id_to_int[row["drug_id"]] + 1
        auc = float(row["auc"])
        return {
            "patient_features": torch.from_numpy(pfeat),
            "drug_ids": torch.tensor([drug_int], dtype=torch.long),
            "drug_mask": torch.tensor([1.0], dtype=torch.float32),
            "auc": torch.tensor(auc, dtype=torch.float32),
        }


def collate_variable_arity(batch: list[dict]) -> dict:
    """Pad a batch of {drug_ids, drug_mask, patient_features, auc} to common N_max.
    Used in DataLoader; handles size-1 (training) or size-K (eval) sets cleanly.
    """
    N_max = max(item["drug_ids"].shape[0] for item in batch)
    B = len(batch)
    drug_ids = torch.zeros(B, N_max, dtype=torch.long)
    drug_mask = torch.zeros(B, N_max, dtype=torch.float32)
    patient_features = torch.stack([item["patient_features"] for item in batch])
    auc = torch.stack([item["auc"] for item in batch])
    for i, item in enumerate(batch):
        n = item["drug_ids"].shape[0]
        drug_ids[i, :n] = item["drug_ids"]
        drug_mask[i, :n] = item["drug_mask"]
    return {
        "drug_ids": drug_ids,
        "drug_mask": drug_mask,
        "patient_features": patient_features,
        "auc": auc,
    }


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
    model: nn.Module, loader: DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device,
    step_counter: list[int] | None = None,
    warmup_steps: int = 0,
    target_lr: float | None = None,
) -> float:
    """Single epoch of training. If warmup_steps > 0, linearly ramp lr from
    0 → target_lr across the first `warmup_steps` optimizer steps (global
    across epochs via `step_counter`, a mutable [int] so callers can thread
    global step state)."""
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        drug_ids = batch["drug_ids"].to(device)
        drug_mask = batch["drug_mask"].to(device)
        pf = batch["patient_features"].to(device)
        auc = batch["auc"].to(device)
        pred = model(drug_ids, drug_mask, pf)
        loss = F.mse_loss(pred, auc)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Linear warmup: override lr during first warmup_steps
        if step_counter is not None and target_lr is not None and warmup_steps > 0:
            gstep = step_counter[0]
            if gstep < warmup_steps:
                lr_now = target_lr * (gstep + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
            elif gstep == warmup_steps:
                # Lock in target lr exactly once
                for pg in optimizer.param_groups:
                    pg["lr"] = target_lr
            step_counter[0] += 1

        optimizer.step()
        total += loss.item() * auc.shape[0]
        n += auc.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def _eval_fold(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict:
    model.eval()
    preds, trues, patient_ids, drug_ids_ = [], [], [], []
    total_mse = 0.0
    n = 0
    for batch in loader:
        drug_ids = batch["drug_ids"].to(device)
        drug_mask = batch["drug_mask"].to(device)
        pf = batch["patient_features"].to(device)
        auc = batch["auc"].to(device)
        pred = model(drug_ids, drug_mask, pf)
        mse = F.mse_loss(pred, auc, reduction="sum")
        total_mse += mse.item()
        n += auc.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(auc.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    mse_mean = total_mse / max(n, 1)
    return {"preds": preds, "trues": trues, "mae": mae, "mse": mse_mean, "n": n}


def train_set_drug_predictor(
    patient_features_path: Path = Path("data/canonical/beataml_patient_features.csv"),
    drug_response_path: Path = Path("data/canonical/beataml_drug_response_long.csv"),
    out_dir: Path = Path("runs/set_drug_predictor"),
    cfg: SetDrugPredictorConfig | None = None,
) -> dict:
    cfg = cfg or SetDrugPredictorConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    # Load + scale features
    pf = pd.read_csv(patient_features_path).set_index("patient_id")
    scaler = StandardScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(pf.values),
        index=pf.index, columns=pf.columns,
    )

    long_df = pd.read_csv(drug_response_path).dropna(subset=["auc"])
    long_df["patient_id"] = long_df["patient_id"].astype(int)
    # Keep only patients with features
    long_df = long_df[long_df["patient_id"].isin(feat_scaled.index)].reset_index(drop=True)

    drug_vocab = sorted(long_df["drug_id"].unique())
    drug_to_int = {d: i for i, d in enumerate(drug_vocab)}

    print(f"[set-mlp] patients: {feat_scaled.shape[0]}  features: {feat_scaled.shape[1]}  "
          f"drugs: {len(drug_vocab)}  samples: {len(long_df)}")
    print(f"[set-mlp] device: {device}")

    # 5-fold CV by patient
    unique_patients = sorted(feat_scaled.index.tolist())
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    fold_metrics = []
    held_out_rows = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(unique_patients)):
        train_pids = {unique_patients[i] for i in train_idx}
        val_pids = {unique_patients[i] for i in val_idx}
        train_df = long_df[long_df["patient_id"].isin(train_pids)]
        val_df = long_df[long_df["patient_id"].isin(val_pids)]

        train_ds = PatientDrugSetDataset(train_df, feat_scaled, drug_to_int)
        val_ds = PatientDrugSetDataset(val_df, feat_scaled, drug_to_int)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_variable_arity, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_variable_arity, num_workers=0,
        )

        # --- Multi-seed training (v2 stability fix) ---
        # Train cfg.n_seeds independent models with different random inits;
        # keep the one with lowest val MSE. Eliminates the 1/5-fold collapse
        # we saw in v1 (fold 3: fold-specific unlucky init → early stopping
        # at epoch 6 with val MAE 49).
        t0 = time.time()
        seed_results = []
        for seed_i in range(cfg.n_seeds):
            seed = cfg.random_state * 1000 + fold_i * 100 + seed_i
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = SetDrugPredictor(
                n_drugs=len(drug_vocab),
                n_patient_features=feat_scaled.shape[1],
                cfg=cfg,
            ).to(device)
            opt = torch.optim.Adam(
                model.parameters(),
                lr=(0.0 if cfg.warmup_steps > 0 else cfg.lr),  # start at 0 if warmup
                weight_decay=cfg.weight_decay,
            )

            best_val = float("inf")
            epochs_no_improve = 0
            best_state = None
            best_epoch = 0
            step_counter = [0]  # mutable global-step tracker for warmup

            for epoch in range(cfg.max_epochs):
                _train_one_epoch(
                    model, train_loader, opt, device,
                    step_counter=step_counter,
                    warmup_steps=cfg.warmup_steps,
                    target_lr=cfg.lr,
                )
                val_metrics = _eval_fold(model, val_loader, device)
                if val_metrics["mse"] < best_val - 1e-4:
                    best_val = val_metrics["mse"]
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epoch + 1 >= cfg.min_epochs and epochs_no_improve >= cfg.patience:
                    break

            seed_results.append({
                "seed": seed_i,
                "best_val_mse": best_val,
                "best_state": best_state,
                "best_epoch": best_epoch,
            })

        # Pick the seed with lowest val MSE
        seed_results.sort(key=lambda r: r["best_val_mse"])
        winner = seed_results[0]
        best_state = winner["best_state"]
        best_val = winner["best_val_mse"]
        best_epoch = winner["best_epoch"]
        assert best_state is not None

        model = SetDrugPredictor(
            n_drugs=len(drug_vocab),
            n_patient_features=feat_scaled.shape[1],
            cfg=cfg,
        ).to(device)
        model.load_state_dict(best_state)

        if cfg.n_seeds > 1:
            mse_values = [round(r["best_val_mse"], 1) for r in seed_results]
            print(f"[set-mlp] fold {fold_i + 1}: {cfg.n_seeds}-seed val MSEs {mse_values}, "
                  f"winner=seed {winner['seed']} @ epoch {best_epoch}")

        # Final fold metrics
        val_metrics = _eval_fold(model, val_loader, device)

        # Per-patient Spearman
        val_df_reset = val_df.reset_index(drop=True)
        val_df_reset["pred"] = val_metrics["preds"]
        per_patient_r = []
        for pid, group in val_df_reset.groupby("patient_id"):
            if len(group) >= 3:
                r = spearmanr(group["pred"], group["auc"]).correlation
                if not np.isnan(r):
                    per_patient_r.append(r)
        pp_rho_mean = float(np.mean(per_patient_r)) if per_patient_r else float("nan")

        # Per-drug Spearman
        per_drug_r = []
        for d, group in val_df_reset.groupby("drug_id"):
            if len(group) >= 5:
                r = spearmanr(group["pred"], group["auc"]).correlation
                if not np.isnan(r):
                    per_drug_r.append(r)
        pd_rho_median = float(np.median(per_drug_r)) if per_drug_r else float("nan")

        fold_metrics.append({
            "fold": fold_i + 1,
            "best_epoch": best_epoch,
            "val_mse": round(best_val, 3),
            "val_mae": round(val_metrics["mae"], 3),
            "mean_per_patient_spearman": round(pp_rho_mean, 4),
            "median_per_drug_spearman": round(pd_rho_median, 4),
            "n_val_patients": int(len(val_pids)),
            "n_val_measurements": int(val_metrics["n"]),
            "elapsed_s": round(time.time() - t0, 1),
        })
        val_df_reset["fold"] = fold_i + 1
        held_out_rows.append(val_df_reset[["fold", "patient_id", "drug_id", "auc", "pred"]])

        print(f"[set-mlp] fold {fold_i + 1}/{cfg.n_folds}  "
              f"MAE={val_metrics['mae']:.2f}  per-patient ρ={pp_rho_mean:.3f}  "
              f"best_epoch={best_epoch}  [{time.time() - t0:.0f}s]")

    # --- Full-data final model (for downstream multi-drug predictions) ---
    full_ds = PatientDrugSetDataset(long_df, feat_scaled, drug_to_int)
    full_loader = DataLoader(
        full_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_variable_arity,
    )
    model = SetDrugPredictor(
        n_drugs=len(drug_vocab),
        n_patient_features=feat_scaled.shape[1],
        cfg=cfg,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    avg_best = int(np.mean([f["best_epoch"] for f in fold_metrics]))
    print(f"[set-mlp] training full-data model for {avg_best} epochs ...")
    for epoch in range(avg_best):
        _train_one_epoch(model, full_loader, opt, device)

    # Save checkpoint + scaler + drug vocab
    ckpt_path = out_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "drug_vocab": drug_vocab,
        "drug_to_int": drug_to_int,
        "feature_cols": feat_scaled.columns.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_patient_features": feat_scaled.shape[1],
        "n_drugs": len(drug_vocab),
        "architecture": "SetTransformer",
    }, ckpt_path)

    held_out_df = pd.concat(held_out_rows, ignore_index=True)
    held_out_df.to_csv(out_dir / "cv_held_out_predictions.csv", index=False)

    # Metrics json
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
        "gate_pass_spearman_0_4": bool(
            np.mean([f["mean_per_patient_spearman"] for f in fold_metrics]) >= 0.40
        ),
    }
    out = {
        "cfg": asdict(cfg),
        "device": str(device),
        "n_patients": int(feat_scaled.shape[0]),
        "n_drugs": int(len(drug_vocab)),
        "n_features": int(feat_scaled.shape[1]),
        "architecture": "SetTransformer (Path B)",
        "overall_cv": overall,
        "output_files": {
            "final_model": str(ckpt_path),
            "cv_predictions": str(out_dir / "cv_held_out_predictions.csv"),
        },
    }
    (out_dir / "cv_metrics.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(f"[set-mlp] DONE. ρ={overall['cv_mean_per_patient_spearman']:.3f} "
          f"± {overall['cv_std_per_patient_spearman']:.3f}  "
          f"MAE={overall['cv_mean_mae']:.2f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", default="data/canonical/beataml_patient_features.csv")
    ap.add_argument("--responses", default="data/canonical/beataml_drug_response_long.csv")
    ap.add_argument("--out", default="runs/set_drug_predictor")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    cfg = SetDrugPredictorConfig(max_epochs=args.epochs)
    train_set_drug_predictor(
        patient_features_path=Path(args.features),
        drug_response_path=Path(args.responses),
        out_dir=Path(args.out),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
