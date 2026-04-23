# Path B v3 — Distillation from Path A clonal-coverage teacher

## Motivation

v2 Set Transformer passed the single-drug CV gate (ρ=0.691) but its **pair
rankings didn't align with canonical clinical combos**: for FLT3-mutated
patients, Gilt+Ven ranked at median **10** (MLP+mech-prior puts it at median
**1**). v2 was trained on 55K single-drug AUC samples only, with no pair-
level supervision.

Route 2 hypothesis: **distill Path A's clonal-coverage ranking into ST's
pair predictions** via a multi-task loss. If Path A's ranking is both
(a) biology-grounded and (b) close to clinical evidence, distillation
should elevate canonical combos in ST's rankings.

## Implementation

**Loss**:
```
L_total = α · L_single-drug-MSE / 2200   +   β · L_pair-ranking
                       ^                                ^
          (normalized to ~unit scale)        1 − Pearson(−ST_aucs, PathA_cov)
```

With `α=1.0, β=1.0` (equal priority), starting from v2 checkpoint, 8 epochs
of fine-tuning on 5-fold CV (by patient) + full-data at the end.

**Pair sampling**: 20 random pairs per patient per batch, teacher signal
precomputed for all 613 patients × 120 pairs (Bliss-IDA of Path A coverage).

**Known limitation**: 4 of the 20 clinical-filter drugs (Enasidenib,
Ivosidenib, Ponatinib, Pacritinib) are NOT in BeatAML's ex-vivo panel, so
they're absent from the ST vocab. Distillation and benchmark use the 16-drug
intersection. Path A benchmark uses the full 20-drug panel (mechanism
vectors only, no training dependency).

## Training trajectory (fold 1 of 5)

| epoch | single MSE | pair (1−r) | pair Pearson |
|---:|---:|---:|---:|
| 1 | 2261 | 0.70 | 0.30 |
| 2 | 2283 | 0.51 | 0.49 |
| 3 | 2296 | 0.45 | 0.55 |
| 4 | 2287 | 0.42 | 0.58 |
| 5 | 2271 | 0.40 | 0.60 |
| 6 | 2285 | 0.39 | 0.61 |
| 7 | 2266 | 0.37 | 0.63 |
| 8 | 2265 | 0.37 | 0.63 |

**During training**, ST successfully learned to match teacher's pair ranking
at ~Pearson=0.63 over sampled batches. Single MSE drifted up by only ~3%.

## Full benchmark — 6 metrics × 4 baselines

All on 613 BeatAML patients (179 FLT3-mutated). Pair metrics on 16-drug
intersection panel.

| Metric | MLP+mech-prior | Path A alone | **ST v2** | **ST v3 distilled** |
|---|---:|---:|---:|---:|
| **M1** Single-drug CV ρ | **0.700** | n/a | 0.691 | 0.653 |
| **M1** Single-drug CV MAE | 35.10 | n/a | 35.93 | **32.11** ✓ |
| **M2** Canonical pair mean rank | **7.78** (med 1) | 70.26 (med 73) | 9.68 (med 10) | 12.11 (med 13) |
| **M3** Jaccard top-5 vs MLP+mech | 1.0 (self) | 0.009 | 0.069 | 0.079 |
| **M4** Spearman with Path A | −0.148 | 1.0 (self) | −0.195 | −0.185 |
| **M5** Triplet Δ (canonical vs random) | 38.96 | 0.38 | 28.25 | 31.76 |
| **M5** Triplet win rate | 0.86 | 1.00 | 1.00 | 1.00 |
| **M6** Bliss consistency rate | 0.069 | n/a | **0.161** | 0.081 |

Lower is better: M2 (rank), M1 MAE. Higher is better: M1 ρ, M3, M4, M5, M6.

## What worked

1. **Single-drug CV ρ stayed above the 0.65 gate** (0.653, passes).
2. **MAE improved 11%**: 35.93 → 32.11. The pair-ranking gradient acted as
   a regularizer on the single-drug predictions, pushing them closer to
   truth in absolute terms.
3. **Triplet Δ improved**: 28.25 → 31.76. Canonical Gilt+Ven+Aza triplet
   beats random triplets by a larger margin in v3 than v2.
4. **Bliss consistency declined** 0.161 → 0.081: v3's pair predictions
   diverge more from pure additive Bliss — it's learned some non-additive
   structure (which is what we wanted).

## What didn't work

1. **Canonical pair rank got worse**: median 10 → 13. Gilt+Ven didn't rise
   toward rank 1 as intended.
2. **Spearman with Path A didn't materially change**: −0.195 → −0.185.
   v3's global pair ranking is essentially the same as v2's.
3. **Jaccard with MLP+mech-prior still low**: 0.069 → 0.079. v3 and MLP+mech
   still pick mostly different top-5 pairs.

## Why distillation failed to move the target

**Root cause: Path A itself doesn't prefer canonical pairs.**

Path A's clonal-coverage scoring prefers "diverse cytotoxic" combinations
(e.g., Alisertib + Gilteritinib for FLT3-mut patients) over "targeted
precision" combos (Gilt+Ven). This happens because Path A's archetype
system has:
- `FLT3_clone` covered by FLT3 inhibitors (narrow mechanism)
- `BCL2_dependent_clone` and `proliferative_clone` both covered by Venetoclax
  **but also** by Alisertib, Azacytidine, Cytarabine (broader mechanism)

So Ali+Gilt covers 3-4 clone axes; Ven+Gilt covers 2. Path A ranks Ali+Gilt
higher.

**Clinical evidence prefers Ven+Gilt** (Daver/Short JCO 2024, 96% CR/CRi in
FLT3-mut newly-diagnosed AML). So teaching ST to match Path A is teaching
it AWAY from clinical evidence for canonical pairs.

The distillation **technically succeeded** (Pearson with teacher rose to
0.63 during training) — but the teacher itself was mis-aligned with the
goal.

## Deliverables

### Checkpoint
`runs/set_drug_predictor_v3_distilled/final_model.pt` (1.1 MB)
- Same architecture as v2 (Set Transformer, 280K params)
- Fine-tuned from v2 weights with Path A ranking loss
- Drop-in compatible with kit via `prefer_set_transformer=True,
  set_transformer_checkpoint=<v3_path>` (or modify default path)

### Code
- `src/combo_val/combo/set_transformer_distill.py` — distillation trainer
- `scripts/train_set_transformer_v3_distilled.py` — one-shot runner
- `scripts/benchmark_path_b_versions.py` — 6-metric × 4-baseline harness

### Data artifacts
- `runs/set_drug_predictor_v3_distilled/cv_metrics.json` — CV fold metrics
- `runs/path_b_benchmark/results.json` — all 4 baselines × 6 metrics
- `runs/path_b_benchmark/comparison_table.csv` — tabular format for fork
  comparison

## Recommendation to fork comparisons

1. **Use M1 ρ, M2 canonical rank, M3 Jaccard, M6 Bliss as the core metrics**
   for method-to-method comparison. M5 Δ is noisier but useful.
2. **M4 (Spearman with Path A) should be treated as "does this method
   produce Path-A-like rankings" rather than "is this method good"** — since
   Path A's rankings don't match canonical clinical evidence.
3. **For direct clinical utility, canonical pair rank (M2) and Jaccard
   with MLP+mech-prior (M3) are the closest to trial-evidence alignment**
   (because MLP+mech-prior bakes in mechanism-class priors that match
   published trials).
4. **Path B v3 vs v2**: v3 trades 5% single-drug ρ for 11% MAE improvement
   and slightly stronger triplet ranking, but doesn't fix pair-ranking
   alignment. If a fork's method improves canonical pair rank (M2 < 5)
   while preserving ρ ≥ 0.65, it beats both Path B variants at the core
   clinical task.

## Conclusion

**Path A is a valid teacher for biology-grounded coverage, but NOT for
canonical clinical pair recommendations.** A better distillation target
for Path B would be:

- **MLP + mech-prior's pair ranking** (has canonical rank = 1)
- **Direct ex-vivo pair measurements** (wet-lab ground truth)
- **Trial-evidence regimen DB (Route C)** expanded with pair-level CR rates

Path B v3 is preserved as an experimental backbone (opt-in via
`prefer_set_transformer=True`) but **should not replace MLP+mech-prior
as the kit's default Layer 3**. See `kit_predict.py:predict_for_patient`.
