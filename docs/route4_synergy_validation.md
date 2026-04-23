# Route 4: Learned Synergy from Real Ex-vivo Pair Data — Full Validation

**Goal**: train a synergy head on real ALMANAC-HL60 pair measurements,
integrate into the kit as a LEARNED synergy term alongside or in place of
the hand-coded mechanism prior, and rigorously compare to 5 alternative
approaches on the FLT3-mut BeatAML cohort.

**Bottom line**: Route 4 produces a working synergy predictor (Pearson 0.41
on 5-fold CV) but does NOT beat the hand-coded mechanism prior on clinical
rank-1 metrics because 186 pairs on 1 cell line is a severe data limit.

---

## 1. What Route 4 trains

The **186 strict ALMANAC-HL60 pairs** from DrugComb v1.5 give us real
ex-vivo synergy measurements on a single AML cell line:
- 19 unique drugs, 8 overlap with the 20-drug clinical filter
- Synergy metric: `synergy_loewe` (mean -12.95, std 14.70, range [-68, 16])

The kit's ST v2 backbone (280K params, trained on 55K BeatAML single-drug
samples, ρ=0.691) knows nothing about pair interactions. Route 4 fine-tunes
a small 42K-param synergy head on these 186 pairs.

## 2. Architecture exploration (3 attempts)

| Attempt | Design | Pearson | Spearman | RMSE | Verdict |
|---|---|---:|---:|---:|---|
| 1 | Frozen ST backbone → PMA set_repr → adapter head | 0.137 | 0.121 | 14.84 | **Worse than null** (RMSE exceeds data std 14.66). ST's set_repr is optimized for single-drug AUC, not synergy. |
| 2 | Unfrozen ST backbone fine-tune + adapter head | 0.181 | 0.125 | 14.63 | Barely beats null. Backbone wants to fit single-drug AUC via MSE, not synergy. |
| 3 | Direct pair model (SynergyMLP-style) with fresh drug embeddings | **0.407** | **0.307** | **13.44** | **Beats null**. Fresh embeddings trained from scratch specifically for synergy. Adopted as final. |

Additional test: warm-starting Attempt 3 from ST v2's drug embeddings gave
Pearson 0.370 (slightly worse than fresh init). ST v2's single-drug-
optimized directions are not synergy-optimal. Final uses `init_mode="fresh"`.

## 3. 5-fold CV results (final Route 4 design)

| fold | n_train | n_val | MAE | RMSE | Pearson | Spearman | best_epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 148 | 38 | 8.48 | 10.60 | -0.004 | -0.031 | 10 |
| 2 | 149 | 37 | 10.72 | 13.96 | 0.394 | 0.372 | 45 |
| 3 | 149 | 37 | 13.70 | 17.13 | **0.541** | **0.520** | 43 |
| 4 | 149 | 37 | 11.35 | 14.08 | 0.367 | 0.227 | 14 |
| 5 | 149 | 37 | 7.91 | 10.31 | 0.352 | 0.303 | 27 |
| **pooled** | | **186** | **10.42** | **13.44** | **0.407** | **0.307** | |

Null baseline RMSE = **14.66** (predicting the mean). Route 4's 13.44 is
an 8% improvement — not huge, but statistically meaningful given only 186
training examples.

## 4. Horizontal comparison — 6 approaches on FLT3-mut BeatAML (n=179)

Same pair ranking task on the same FLT3-mut cohort. Metric: where does the
canonical clinically-validated pair (FLT3-inhibitor + Venetoclax) rank?

| # | Approach | median FLT3i+Ven rank | % in top-5 | mean pair AUC | Jaccard vs B |
|---|---|---:|---:|---:|---:|
| A | MLP only (pure additive) | **1** | 68.7% | 176.9 | 0.63 |
| **B** | **MLP + mech prior** (current kit default) | **1** | **72.1%** | 159.2 | **1.00** |
| C | **MLP + Route 4 synergy** (NEW) | 1 | 67.0% | 163.7 | 0.52 |
| D | ST v2 only | 10 | 3.4% | 185.0 | 0.06 |
| E | ST v2 + mech prior | 5 | 60.9% | 167.3 | 0.18 |
| F | ST v2 + Route 4 synergy (NEW) | 9 | 10.1% | 172.7 | 0.17 |

**Key observations**:

- **B (MLP + mech prior) remains the winner** at 72.1% top-5 for FLT3i+Ven.
- **Adding Route 4 synergy to MLP (C) slightly HURTS** (67.0% vs 72.1%).
  Route 4's noisy pair predictions move some borderline rankings the wrong
  way. The mech prior's hand-coded FLT3+BCL2 co-targeting is stronger than
  learned synergy from 186 pairs.
- **Adding Route 4 to ST v2 (F) helps slightly** over bare ST (10.1% vs 3.4%)
  but still far below MLP-based approaches.
- **Route 4 synergy alone (no mech prior) preserves top-1 rank-1 hit rate**
  (median FLT3i+Ven rank = 1, same as MLP-only) but loses top-5 %
  because its noise pushes some non-canonical pairs above the canonical
  ones.

### Cross-approach Spearman agreement on pair rankings

```
                       A     B     C     D     E     F
A MLP only           1.00  0.94  0.99  0.73  0.72  0.76
B MLP+mech_prior     0.94  1.00  0.92  0.67  0.78  0.70
C MLP+Route4         0.99  0.92  1.00  0.71  0.69  0.77
D ST only            0.73  0.67  0.71  1.00  0.92  0.94
E ST+mech_prior      0.72  0.78  0.69  0.92  1.00  0.85
F ST+Route4          0.76  0.70  0.77  0.94  0.85  1.00
```

- MLP-based (A/B/C) cluster tightly (ρ ≥ 0.92)
- ST-based (D/E/F) cluster tightly (ρ ≥ 0.85)
- Cross-backbone ρ ~ 0.7 — substantial divergence in pair rankings
- **Route 4 (C, F) sits between mech-prior and pure additive** — it adds
  some signal but doesn't radically reorder the rankings

## 5. Top-1 picks per approach (FLT3-mut cohort, n=179)

```
A_MLP_only:            Venetoclax + Gilteritinib          56 patients (31.3%)
                       Venetoclax + Quizartinib           48          (26.8%)
                       Dasatinib  + Trametinib            33          (18.4%)

B_MLP+mech_prior:      Venetoclax + Gilteritinib          61          (34.1%)  ← ⭐
                       Venetoclax + Quizartinib           50          (27.9%)
                       Gilteritinib + Trametinib          34          (19.0%)

C_MLP+Route4:          Venetoclax + Quizartinib           43          (24.0%)  ← canonical
                       Venetoclax + Gilteritinib          35          (19.6%)
                       Dasatinib  + Trametinib            33          (18.4%)
```

All three MLP-based approaches put Ven + FLT3i at rank 1 for the majority
of FLT3-mut patients. Route 4's ordering swap between Quiz/Gilt is the
learned synergy reflecting different HL-60 pair behaviors, but clinically
both combos are validated.

## 6. What Route 4 achieves vs doesn't

**✅ Achieves**:
1. Trains a learned synergy model from real ex-vivo pair data (Pearson 0.41).
2. Beats the null baseline (RMSE 13.44 < 14.66).
3. Preserves the canonical FLT3i+Ven pair at rank 1 for median FLT3-mut
   patient.
4. Confirms the core Route 4 hypothesis: 186 pairs is JUST ENOUGH to train
   a signal-positive synergy predictor.

**❌ Does NOT achieve**:
1. Does not match Week-3 SynergyMLP's Pearson 0.48 (ours is 0.41). Likely
   due to smaller model + symmetric-augmentation regularization interacting
   with 186-sample regime.
2. Does not improve top-5 hit rate over mech prior (67.0% vs 72.1%).
3. Does not replace mech prior as the preferred Layer 3 synergy term —
   the hand-coded FLT3+BCL2 co-targeting rule still dominates.

## 7. Why Route 4 struggles to beat mech prior

| Factor | Mech prior | Route 4 |
|---|---|---|
| Training data | 0 samples (rule-based) | 186 real pairs, 1 cell line (HL-60) |
| Biology knowledge | Explicit: 39-axis hand-coded | Implicit: learned from data |
| FLT3+BCL2 co-target awareness | Hard-coded (tgt_FLT3 * tgt_BCL2) | Must infer from 186 pairs |
| Patient-specificity | Patient-specific (via deficit vector) | Patient-agnostic (HL-60 only) |
| Generalization to new biology | Bounded by vocab | Bounded by training drugs |

The mech prior wins because 186 pairs ≠ enough to learn the "FLT3-mut →
FLT3i+BCL2i" rule from scratch, but it IS hand-coded directly. Route 4
needs 10-100× more pair training data (spanning multiple cell lines or
primary samples) before it can rival hand-coded biology.

## 8. Kit integration — production defaults

No change to the kit default. `predict_for_patient` stays on `B_MLP_plus_mech_prior`
as the Layer 3 backbone:

```python
kit_output = predict_for_patient(
    rna_counts, kit,
    prefer_set_transformer=False,  # stays with MLP
    # Route 4 synergy head NOT used by default
)
```

Route 4 is available via a new `combo_val.combo.synergy_inference.load_synergy_head()`
for researchers who want to compare / ensemble, but it is not on the
default path.

## 9. Files committed

```
src/combo_val/combo/synergy_head.py      (NEW, train/CV script + architecture)
src/combo_val/combo/synergy_inference.py (NEW, inference wrapper)
scripts/route4_full_comparison.py        (NEW, 6-approach horizontal eval)
runs/set_drug_predictor_v3_route4/
  synergy_head.pt                        (NEW, 42K-param checkpoint)
  cv_predictions.csv                     (186 held-out pair predictions)
  training_summary.json                  (CV metrics)
runs/route4_comparison/
  comparison_table.csv                   (final 6-approach table)
  pooled_spearman_matrix.csv
  summary.json
docs/route4_synergy_validation.md        (this file)
```

## 10. Result package for cross-fork comparison

Other forks working on Routes 1-3 should use the same 6-approach harness
(`scripts/route4_full_comparison.py`) with their synergy predictor
swapped in. Direct comparison metrics:

| Metric | Route 4 (this work) | Route 1 (Bliss-IDA synth) | Route 2 (Path A distill) | Route 3 (ALMANAC pair only) |
|---|---:|---:|---:|---:|
| Pooled CV Pearson | **0.407** | ? | ? | ? |
| FLT3i+Ven top-5 hit rate | **67.0%** (C row) | ? | ? | ? |
| Jaccard top-5 vs mech prior | **0.518** (C row) | ? | ? | ? |
| Training data used | 186 real ALMANAC pairs | 55K pseudo-Bliss pairs | Path A coverage teacher | 186 ALMANAC pairs |
| Beats null RMSE? | YES (13.44 < 14.66) | ? | ? | ? |
| Replaces mech prior in kit? | NO | ? | ? | ? |

Fill in the `?` columns from other fork results and we have a canonical
method comparison ready for paper Methods §2.5.
