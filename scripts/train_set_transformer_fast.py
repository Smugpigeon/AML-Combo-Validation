"""Fast training config for Set Transformer path B.

Uses a smaller model (1 ISAB layer, 8 inducing points, 64-dim hidden) and
shorter training (max_epochs=40, patience=8) to iterate quickly. Once
theoretical viability is verified, we scale up.
"""

from __future__ import annotations

from pathlib import Path

from combo_val.combo.set_drug_predictor import (
    SetDrugPredictorConfig, train_set_drug_predictor,
)


if __name__ == "__main__":
    cfg = SetDrugPredictorConfig(
        drug_emb_dim=48,
        patient_hidden_dim=128,
        patient_emb_dim=64,
        set_hidden_dim=96,
        n_attn_heads=4,
        n_inducing_points=8,         # was 16
        n_isab_layers=1,             # was 2
        head_hidden_dim=96,
        dropout=0.15,
        batch_size=512,              # was 256
        lr=2e-3,
        weight_decay=1e-5,
        max_epochs=40,               # was 100
        patience=8,                  # was 15
        min_epochs=15,
        random_state=42,
        n_folds=5,
        device="auto",
    )
    train_set_drug_predictor(
        patient_features_path=Path("data/canonical/beataml_patient_features.csv"),
        drug_response_path=Path("data/canonical/beataml_drug_response_long.csv"),
        out_dir=Path("runs/set_drug_predictor"),
        cfg=cfg,
    )
