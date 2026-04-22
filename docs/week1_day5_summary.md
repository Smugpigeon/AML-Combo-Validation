# Week 1 Day 5 — End-to-end integration + Week 3 smoke test

## What ran

1. **`combo_val.integration_check`** — 20 data-layer health checks across all
   Week 1 canonical tables. Report in
   [week1_day5_integration_report.md](week1_day5_integration_report.md).
   All checks green.

2. **`combo_val.combo.combo_predictor`** — first end-to-end Week 3 run on the
   Week 1 data. Trains a 19-drug pair-residual MLP on 186 ALMANAC-HL-60 pairs,
   applies it to all 13,530 unique drug pairs for 613 BeatAML patients, saves
   the 613 × 165 × 165 combo-AUC tensor. (Formally Week-3 work; included here
   as the last smoke-test gate that the Week 1 data is complete and coherent.)

## Integration check summary

| Check | Result |
|---|---|
| Canonical files exist | 13/13 ✓ |
| BeatAML / TCGA feature-schema match | 80 cols identical ✓ |
| Drug vocab consistency | 0 orphans in strict pairs ✓ |
| Patient IDs unique per cohort | 613 BeatAML, 173 TCGA ✓ |
| AUC distribution | [0, 286], median 207.5 ✓ |
| Mutations binary | 0/1 only across 25 cols ✓ |
| ELN ordinal set | {0.0, 0.5, 1.0, 1.5, 2.0} ✓ |
| Week-3 readiness | 186 pairs, 19 drugs, synergy std 14.7 ✓ |
| Week-5 readiness | TCGA 145/173 patients with ≥1 mutation, deceased 65.9% ✓ |

## Combo predictor first run

CV performance on 186 strict pairs (patient-agnostic; 5-fold random split):

| Metric | Value |
|---|---|
| Mean fold val RMSE | **12.68** (vs null-baseline RMSE ≈ 14.70) |
| CV pooled Pearson | 0.481 |
| CV pooled Spearman | 0.384 |
| Per-fold RMSE range | 9.90 – 15.72 |

The model **beats the null-baseline** (predicting the cohort mean synergy for
every pair) by ~14% in RMSE. Pearson 0.48 is a credible signal given only 186
training pairs; the Spearman-vs-Pearson gap (0.38 vs 0.48) suggests the model
captures ranking coarsely but not tail synergy extremes — expected with this
little data.

## Top rank-1 combo picks (sanity check, not Week 4's real analysis yet)

| Pair | Times as rank-1 across 613 patients |
|---|---:|
| Elesclomol + Panobinostat | 229 |
| Elesclomol + SNS-032 | 170 |
| Elesclomol + Venetoclax | 63 |
| Elesclomol + Foretinib | 57 |
| Elesclomol + Trametinib | 16 |

Elesclomol dominates because Baseline A predicts it has a low single-drug AUC
(≈ 85-100) for many patients, and the additive `0.5 * (AUC_d1 + AUC_d2)` term
picks whichever partner best complements it. Venetoclax and Trametinib are in
the top-5 list — these are clinically meaningful AML targets (BCL2 / MEK), so
the model isn't completely ignoring biology.

**Week 3 proper will address**:
1. Calibrate `synergy_to_auc_scale` so that predicted synergies are actually
   comparable to raw AUC units (not blindly added at scale 1.0).
2. Add a mechanism-prior term for patients whose driver profile matches
   specific drug mechanisms (FLT3+BCL2 for FLT3-mutant patients, IDH2+HMA for
   IDH2-mutant patients, etc.).
3. Drop patient-agnostic synergy for a patient-modulated residual once we have
   more than 186 training pairs (or use ablation study to confirm patient
   features add value for combo adjustment).

## Tests

51/51 pass (was 46 before Day 5; added 5 combo-predictor unit tests covering
SynergyMLP symmetry, gradient flow, output shape, synthetic data loader, and
synthetic end-to-end training).

## Week 1 recap

The research-grade data layer for the AML combo-validation project is
complete in 5 days:

| Day | Deliverable | Core metric |
|---|---|---|
| 1 | Project scaffold, scratch repo, dependencies | — |
| 2 | BeatAML ETL + Baseline A (single-drug MLP) | per-patient ρ = **0.704** (gate ≥0.40) |
| 3 | DrugComb v1.5 AML subset ETL | **186** strict combo pairs + 1,603 mono screens |
| 4 | TCGA-LAML cBioPortal ETL | **173** patients, 145 with mutations, 80-dim schema |
| 5 | Integration smoke test + combo predictor end-to-end | 20/20 checks ✓; combo RMSE 12.68 |

Ready to begin Week 2 (single-drug model hardening & expansion) and Week 3
(combo model proper training + calibration).
