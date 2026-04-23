# Path B v3 — Pair Fine-tune on DrugComb ALMANAC-HL-60

**Status**: 4/6 viability gates PASS. The 2 failures are **principled and
expected**, explained below. Full results + cross-path comparison table at
the bottom.

## 1. Objective

Take the stable Path B v2 Set Transformer (CV ρ = 0.691 ± 0.011 on
BeatAML single-drug AUC) and teach it real pair behavior using the 186
strict-pair samples from DrugComb ALMANAC on HL-60. This is Route 3 from
the kit-roadmap — no wet-lab cost, uses only existing training data.

## 2. Theoretical Design

### 2.1 Data (186 pairs, 19 drugs, 1 cell line)

DrugComb ALMANAC-HL-60 strict-pair subset has 186 samples of the form
`(d1, d2, synergy_loewe)` with synergy ∈ [−68, +16], mean −12.95,
std 14.70. Only 19 unique BeatAML drugs appear across pairs.

### 2.2 HL-60 pseudo-patient features (104-dim)

| Component | Dim | Value |
|---|---:|---|
| RNA PCs | 50 | 0 (standardized BeatAML training-set mean = "average AML") |
| Mutation flags | 25 | All 0 except `mut_NRAS=1` (HL-60 documented NRAS Q61L) |
| Clinical | 29 | BeatAML training medians → ≈ 0 after StandardScaler |

Standardized NRAS = +2.66 (correct positive z for a mut=1 flag),
FLT3 = −0.64 (correct negative z for a mut=0 flag).

### 2.3 Target reconstruction

DrugComb gives `synergy_loewe` but not combo AUC. We reconstruct:

```
target_combo_auc(d1, d2) = 0.5 * (A_d1 + A_d2) + synergy_loewe(d1, d2)
```

where `A_dk` is v2's singleton AUC prediction for HL-60 × `dk`, frozen at
the start of fine-tuning. Loewe convention: negative synergy → combo
below additive mean (synergistic).

### 2.4 Joint loss (prevents catastrophic forgetting)

```
L = MSE(pair_pred, target) + 1.0 * MSE(singleton_anchors)
```

The singleton anchor keeps all 19 drugs' HL-60 predictions pinned to
their frozen v2 values.

### 2.5 Regime

- LR 1e-4 (10× lower than v2 training)
- Max 20 epochs, patience 5
- Batch size 16
- All weights trainable (gradients flow through drug embeddings +
  ISAB attention)

Converged at epoch 2, early stop at epoch 7. Total fine-tune 1 second.

## 3. Results — 6 Viability Gates

### E1 Singleton retention (v2 vs v3) → **PASS**

Correlation of v3's singleton predictions to v2's on 50 random BeatAML
patients: **Spearman ρ = 0.998**. The fine-tune did not corrupt the
singleton head.

### E2 Pair fit on 186 training targets → **PASS**

| | v2 (pre-FT) | v3 (post-FT) | Null (predict mean) |
|---|---:|---:|---:|
| MSE | 560 | **310** | 697 |
| Pearson r | 0.728 | **0.759** | 0 |

v3 cuts residual variance by 44% relative to v2 on the pair reconstruction
target. This is evidence the fine-tune IS learning pair structure.

### E3 Pair rank correlation (in-sample, flagged) → **PASS**

Spearman(v3 predicted, target) = **0.815** over all 186 pairs
(p ≈ 0, 5-fold mean 0.792). Flagged as in-sample because v3 was
trained on all 186 pairs; true OOD CV would require leave-one-out with
re-fine-tune per fold (out of scope for this pass).

### E4 Gilteritinib + Venetoclax rank on FLT3-mut patients → **FAIL**

| | v2 | v3 |
|---|---:|---:|
| Median rank (of C(20,2)=190 pairs) | 10 | 10 |
| Mean rank | 9.8 | 9.76 |
| Improved vs v2 | — | **No** |

**Why FAIL is expected**: HL-60 is FLT3-wt, NRAS-mut. The 186 pairs
contain ZERO signal about FLT3-mut-specific combo biology. The fine-
tune cannot teach a rule it was never shown. Canonical FLT3i + BCL2i
combos stay at the rank the v2 weights produce on default (≈10) because
v3's NRAS-grounded features don't transfer to FLT3-mut patients.

### E5 Aza+Ven+Gilt triplet on FLT3-mut patients → **PASS**

Win rate against 20 random non-FLT3i triplets: **v3 = 1.00 (100%)**,
same as v2 (1.00). Mean AUC delta: v2 = +30.5, v3 = +28.6. The
triplet-beats-random behavior held over from v2 — fine-tune did not
break it.

### E6 Head-to-head vs MLP+mech-prior on FLT3-mut synthetic patient → **FAIL**

Top-5 Jaccard to MLP+mech-prior (clinical-literature aligned):
**v2 = 0.111, v3 = 0.111 — unchanged**.

V3's top-5 is the same Trametinib-dominated set as v2 (HL-60-learned
biology). V3 did NOT move toward MLP's FLT3i+BCL2i-dominated set,
for the same reason as E4: no FLT3-mut data in fine-tune.

## 4. Honest interpretation

### What v3 proved
1. **Pair supervision infrastructure works**: E2+E3 show v3 accurately
   fits the pair target (Pearson 0.76, Spearman 0.82). The learning
   machinery is right.
2. **No catastrophic forgetting** (E1: ρ=0.998 vs v2 singletons).
3. **Triplet zero-shot retained** (E5: 100% win rate on canonical triplet).

### What v3 did NOT do
1. Did not learn FLT3-mut-specific pair rules (E4, E6 fail).
2. Did not move closer to clinical-literature canonical combos.

### Root cause

**The 186 pairs are the wrong training data to teach FLT3-mut rules.**
HL-60 is FLT3-wt. No amount of algorithm tuning overcomes this — the
supervision signal for "FLT3-mut patient benefits from Gilt+Ven" is not
present in the training set.

What v3 DID learn: HL-60-specific pair corrections. For example, v3
knows "on HL-60, Imatinib + Sunitinib synergizes ~20 AUC below
additive". If we ever evaluate on HL-60 (cell-line validation in wet
lab), v3 would produce better predictions than v2.

## 5. Implications for the kit

v3 should be positioned as:

| Usage | Verdict |
|---|---|
| Replace MLP+mech-prior for FLT3-mut clinical recommendations | ❌ NO |
| Layer 3 backbone for HL-60 / FLT3-wt patients | ✅ YES (learned corrections) |
| Generic single-drug AUC predictor | ✅ (matches v2, E1 retention) |
| Infrastructure for future real pair-data fine-tunes (ex-vivo / organoid) | ✅ MOST IMPORTANT |

**The v3 infrastructure is the real deliverable here.** When wet-lab
yields even 50-100 real patient × pair samples, the exact pipeline
(HL-60 swap-in → patient broadcast) can be re-run with those samples to
produce a v4 that DOES teach FLT3-mut pair rules.

## 6. Cross-path comparison (for A/B/C/D head-to-head)

Path B v3 benchmarks for the comparison table being built across forks:

| Metric | Value | Gate |
|---|---:|---|
| Single-drug CV ρ (vs v2) | 0.691 ± 0.011 | ≥ 0.65 (inherited from v2) |
| Single-drug retention ρ (v2 vs v3) | 0.998 | ≥ 0.90 |
| Pair target MSE reduction vs v2 | 560 → 310 (−45%) | v3 < v2 < null |
| Pair Pearson on 186 targets | 0.759 | > 0.50 |
| Pair Spearman on 186 targets | 0.815 | > 0.50 |
| FLT3-mut Gilt+Ven median rank (of 190) | 10 | ≤ 5 (FAIL) |
| FLT3-mut canonical triplet win rate | 100% | ≥ 55% |
| Jaccard top-5 vs MLP+mech-prior | 0.111 | > v2 (FAIL) |
| Training samples | 186 | — |
| Training time | 1 second | — |
| Architecture params | 280K | — |
| Supervision data needed | DrugComb pair data | — |
| Can recommend FLT3i+BCL2i for FLT3-mut? | No | — |
| Can recommend novel triplets in 6-arity space? | Yes | — |

### Compared to Path A (clonal-coverage), Path C (regimen retrieval), Path D (HOFM)

| Property | A | C | B-v2 | B-v3 | D |
|---|---|---|---|---|---|
| FLT3i+BCL2i for FLT3-mut → top rank? | ✅ | ✅ | ✗ (rank 10) | ✗ (rank 10) | ? |
| Training needed? | ❌ | ❌ | ✅ 55K single-drug | ✅ +186 pair | ✅ 186 pair |
| Zero-shot triplet support? | ✅ | ❌ | ✅ | ✅ | ✅ |
| 6-arity scalability? | ✅ | ❌ | ✅ | ✅ | ✅ (with softplus V) |
| Can recommend ATRA/ATO? | ✅ (mech tagged) | ✅ | ❌ (not in vocab) | ❌ | ❌ |
| Published-CR citation per output? | ❌ | ✅ | ❌ | ❌ | ❌ |
| Continuous AUC prediction (not rank)? | ❌ | ❌ | ✅ | ✅ | ✅ (on cell lines) |
| Unique contribution | Biology explanation | Clinical evidence | Continuous AUC + arity flex | Pair calibration infra | Mathematical identifiability |

## 7. Recommendation to the kit

**Do not** swap v3 into Layer 3 as the default backbone for clinical use.

**Do** keep v3 as an opt-in `prefer_set_transformer=True,
set_transformer_checkpoint=V3_CKPT` option for:
- Research exploration in 3+ arity space
- Future re-fine-tune when real patient × combo data arrives

Default kit Layer 3 stays at Baseline-A MLP + mechanism-prior factorized
combo AUC until either (a) we get multi-patient pair data covering
FLT3-mut/IDH-mut/etc., or (b) we adopt Path A's coverage score as the
pair re-ranker.
