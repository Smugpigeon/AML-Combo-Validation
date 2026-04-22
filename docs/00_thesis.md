# Thesis

## What we are testing

> For AML patients, does a mechanism-aware combination recommendation
> beat the predicted best single drug?

Formally, for each patient in a held-out validation set:

```
best_single(p) = min over drugs d of:   predicted AUC_single(p, d)
best_combo(p)  = min over pairs (d1,d2) of: predicted AUC_combo(p, d1, d2)
Δ(p) = best_single(p) − best_combo(p)     # >0 means combo is better
```

Primary test: `H0: E[Δ] ≤ 0` vs `H1: E[Δ] > 0` via bootstrap CI on mean(Δ).

## Why this is the right question

Our earlier `AML-CRAFT` work showed:

1. AML patient embeddings have ~10 intrinsic dimensions — biology is
   low-dimensional.
2. KMeans forces discrete partitions onto a mix of tight cores + a
   continuous MDS-spectrum region. Hybrid representation is needed.
3. **Single-drug AUC does not systematically vary with biology-axis
   position on the MDS-spectrum** (BH-corrected significance = 0/144 drugs).
   This negative finding is the direct motivation for this project:
   if single-drug matching can't personalize, combinations are the
   only remaining lever.

This thesis tests whether that last claim holds up under a proper
mechanism-aware combination predictor.

## Four acceptable outcomes

| Outcome | `mean(Δ)` | Paper framing |
|---|---|---|
| A. Combo clearly wins | ≥ 20% AUC reduction | New personalization framework |
| B. Combo marginally wins | 5-20% | Applies to specific subgroups |
| C. Combo equals single | ≈ 0 | SOC is optimal; personalization must come from elsewhere |
| D. Combo loses | < 0 | Model design flaw; iterate or publish methodological lesson |

All four are publishable with honest framing.

## Scope (deliberately narrow)

- **In scope**: training + evaluating two predictors, head-to-head comparison, TCGA-LAML replication.
- **Out of scope**: organoid validation (future wet-lab work), PK/PD modeling, dose optimization, multi-drug (≥4) combinations.

## Success criteria (pre-registered)

We commit to these metrics and decision rules BEFORE running the
validation, so outcome selection doesn't bias the interpretation:

1. **Data quality gate**: ≥ 450 BeatAML patients with full feature coverage
   (currently 613 ✅).
2. **Baseline quality gate**: single-drug predictor achieves Spearman ≥ 0.4
   against held-out AUC (typical DeepSynergy-level baseline).
3. **Combo predictor sanity gate**: prior-only score correlates (Spearman ≥ 0.3)
   with DrugComb Loewe synergy on held-out pairs.
4. **Primary outcome**: bootstrap 95% CI of `mean(Δ)` on held-out BeatAML patients.
5. **Replication outcome**: sign-consistency of `Δ` in TCGA-LAML.

Failure of any gate triggers model-level iteration, not metric-shopping.
