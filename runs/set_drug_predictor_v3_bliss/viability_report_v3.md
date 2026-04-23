# Path B v3 — Bliss-IDA fine-tuned Set Transformer — viability report

Generated: 2026-04-23 20:52:32
Checkpoint: `runs/set_drug_predictor_v3_bliss/final_model.pt`

## Summary

| # | Test | Result | Gate | Pass |
|---|---|---|---|:---:|
| T1 | T1 single-drug per-patient ρ (post-hoc, 100 sampled patients) | mean ρ = 0.7091 | ≥ 0.65 | ✅ |
| T2 | T2 permutation invariance (5 patients × 3 permutations) | max_diff = 8e-06 | ≤ 0.0001 | ✅ |
| T3 | T3 Bliss triplet extrapolation (zero-shot N=3) | MAE = 51.02, Pearson = 0.7136 | MAE ≤ 30.0 | ❌ |
| T4 | T4 Gilt+Ven rank in FLT3-mut patients (top-5 gate) | top-5 = 4%, median rank = 27.0 | frac top-5 ≥ 0.55 | ❌ |
| T5 | T5 FLT3-mut canonical triplet (Gilt+Ven+Aza) preference | canonical wins = 75.0%, Δ = +14.99 | ≥ 55% | ✅ |
| T6 | T6 Jaccard top-5 vs MLP+mech-prior (FLT3-mut, n=20) |  |  | ❌ |

## Raw JSON

```json
{
  "T1": {
    "name": "T1 single-drug per-patient \u03c1 (post-hoc, 100 sampled patients)",
    "mean_rho": 0.7091,
    "median_rho": 0.7606,
    "n_patients": 77,
    "pass": true,
    "threshold": 0.65
  },
  "T2": {
    "name": "T2 permutation invariance (5 patients \u00d7 3 permutations)",
    "max_diff": 8e-06,
    "pass": true,
    "threshold": 0.0001
  },
  "T3": {
    "name": "T3 Bliss triplet extrapolation (zero-shot N=3)",
    "n": 500,
    "mae": 51.02,
    "rmse": 60.04,
    "pearson": 0.7136,
    "pass": false,
    "threshold_mae": 30.0
  },
  "T4": {
    "name": "T4 Gilt+Ven rank in FLT3-mut patients (top-5 gate)",
    "n": 50,
    "mean_rank": 25.5,
    "median_rank": 27.0,
    "frac_top_5": 0.04,
    "pass": false,
    "threshold_frac_top_5": 0.55
  },
  "T5": {
    "name": "T5 FLT3-mut canonical triplet (Gilt+Ven+Aza) preference",
    "n_patients": 100,
    "n_comparisons": 2000,
    "frac_canonical_wins": 0.75,
    "mean_delta_auc": 14.99,
    "pass": true,
    "threshold_frac": 0.55
  },
  "T6": {
    "name": "T6 Jaccard top-5 vs MLP+mech-prior (FLT3-mut, n=20)",
    "mean_jaccard": 0.018,
    "min_jaccard": 0.0,
    "max_jaccard": 0.25,
    "pass": false,
    "threshold": 0.25
  }
}
```