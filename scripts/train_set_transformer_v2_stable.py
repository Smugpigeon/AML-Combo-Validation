"""Retrain Path B Set Transformer with stability fixes (v2).

Fixes applied (vs v1 collapse where fold 3 hit val MAE 49 @ epoch 6):
  1. n_seeds=3          — per-fold multi-seed (pick lowest val MSE)
  2. warmup_steps=500   — linear LR warmup avoids early divergence
  3. min_epochs=25      — don't let patience trigger before model stabilizes
  4. patience=12        — allow recovery from transient dips

Runtime: ~15 min with 3 seeds × 5 folds on MPS.
"""
from __future__ import annotations

from pathlib import Path

from combo_val.combo.set_drug_predictor import (
    SetDrugPredictorConfig,
    train_set_drug_predictor,
)


def main():
    cfg = SetDrugPredictorConfig(
        # Architecture matches v1 (known working when training doesn't collapse)
        drug_emb_dim=48,
        patient_hidden_dim=128,
        patient_emb_dim=64,
        set_hidden_dim=96,
        n_attn_heads=4,
        n_inducing_points=8,
        n_isab_layers=1,
        head_hidden_dim=96,
        dropout=0.15,
        # Training — tightened for stability
        batch_size=512,
        lr=2e-3,
        weight_decay=1e-5,
        max_epochs=50,
        patience=12,           # was 8 → allow recovery
        min_epochs=25,         # was 15 → prevent early stop during warmup
        # v2 fixes
        n_seeds=3,             # 3 independent inits per fold
        warmup_steps=500,      # linear warmup 0 → lr across first 500 steps
        random_state=42,
        n_folds=5,
        device="auto",
    )
    train_set_drug_predictor(
        patient_features_path=Path("data/canonical/beataml_patient_features.csv"),
        drug_response_path=Path("data/canonical/beataml_drug_response_long.csv"),
        out_dir=Path("runs/set_drug_predictor_v2_stable"),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
