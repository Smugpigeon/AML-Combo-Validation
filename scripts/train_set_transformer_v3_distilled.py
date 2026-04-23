"""Fine-tune Set Transformer v2 → v3 with Path A clonal-coverage distillation.

Starts from v2 checkpoint (single-drug ρ=0.691) and adds a pair-ranking loss
that teaches ST to match Path A's clonal-coverage ordering of drug pairs.

Expected outcome:
  - Single-drug ρ stays ≥ 0.65 (we preserve the v2 capability via α=1.0 MSE)
  - FLT3i+BCL2i canonical rank moves from v2's 10 → closer to 1-3
  - Jaccard top-5 with MLP+mech-prior increases from 0.11 → ~0.4

Runtime: ~10 min (fine-tune, not retrain).
"""
from __future__ import annotations

from pathlib import Path

from combo_val.combo.set_transformer_distill import DistillConfig, distill_v2_to_v3


def main():
    cfg = DistillConfig(
        v2_checkpoint=Path("runs/set_drug_predictor_v2_stable/final_model.pt"),
        out_dir=Path("runs/set_drug_predictor_v3_distilled"),
        single_mse_norm_scale=2200.0,  # normalize MSE by ~v2 val-MSE
        alpha_single=1.0,               # anchor single-drug capability
        beta_pair=1.0,                  # equal weight for pair ranking
        n_pairs_per_patient_per_batch=20,
        lr=5e-4,
        n_epochs=8,
        batch_size=256,
        pair_batch_size=16,
        n_folds=5,
        random_state=42,
    )
    distill_v2_to_v3(cfg)


if __name__ == "__main__":
    main()
