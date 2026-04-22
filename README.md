# AML Combination Pharmacology Validation

> **Core thesis**: For AML patients, do mechanism-aware drug combinations achieve better predicted response than the best single-drug recommendation?

This is a clean, focused rebuild of the question asked in the earlier `AML-CRAFT` work. The previous project accumulated a lot of supporting infrastructure (HyGReM-NC multimodal encoder, region classifier, pseudotime, purpose-driven K selection) — useful as supplementary findings, but not required to test the central hypothesis. This repo strips down to that hypothesis.

## Hypothesis framing

Let:
- `best_single(patient)` = lowest predicted AUC across all single drugs for this patient
- `best_combo(patient)` = lowest predicted (prior × residual) combination response across all legal drug pairs

Define `Δ(patient) = best_single(patient) − best_combo(patient)`.

Test H0: `mean(Δ) ≤ 0` (combinations add no value) vs H1: `mean(Δ) > 0` (combinations are better).

**Three acceptable outcomes**:

| Outcome | `mean(Δ)` | Paper framing |
|---|---|---|
| A. Combo clearly wins | ≥ 20% AUC reduction | New personalization framework |
| B. Combo marginally wins | 5–20% | Applies to specific patient subgroups |
| C. Combo equals single | ≈ 0 | SOC is optimal; personalization must come from elsewhere |
| D. Combo loses | < 0 | Model design flaw; iterate or publish methodological lesson |

## Data

| Source | Role | Size |
|---|---|---|
| BeatAML 2.0 | Training + internal validation | 805 patients, 166 drugs, ~55K AUC measurements |
| DrugComb AML subset | Combination synergy training | ~30K drug pairs × 9 AML cell lines (Week 1 Day 3) |
| TCGA-LAML | Independent cohort validation | ~200 patients with clinical outcomes (Week 5) |

## Architecture (minimal)

```
Patient features (~80 dim: RNA PCA-50 + 25-gene mutation + 5 clinical)
  │
  ├──> Baseline A: single-drug AUC predictor (multi-task MLP)
  │      → predict best_single(patient)
  │
  └──> Hypothesis B: combination predictor
        │
        ├── Mechanism-space prior (from `knowledge/mechanism_vocab.yaml`)
        └── Bayesian residual (trained on DrugComb pairs)
           → predict best_combo(patient)

           ↓
Head-to-head validation: Δ = best_single − best_combo
with bootstrap 95% CI and permutation test
```

## Directory layout

```
data/
  raw/
    BeatAML2.0/          # 4 raw files (gitignored)
    drugcomb/            # ALL_MATRIX.csv.gz (gitignored, download in Week 1 Day 3)
    tcga_laml/           # public aligned tables (gitignored, Week 5)
    h.all.v2026.1.Hs.symbols.gmt   # Hallmark pathways
  canonical/             # single-source-of-truth unified tables
    beataml_patient_features.csv
    beataml_drug_response_long.csv
    drugcomb_aml_pairs.csv
    tcga_laml_validation.csv

src/combo_val/
  data/        # ETL per source
  features/    # PCA, mutation one-hot, mechanism vectorization
  knowledge/   # mechanism_vocab.yaml + drug_mechanism_v1.csv (from AML-CRAFT)
  baselines/   # Baseline A: single-drug multi-task MLP
  combo/       # Hypothesis B: mechanism prior + Bayesian residual
  validation/  # Head-to-head Δ framework
  figures/     # Paper figure generators

configs/       # Experiment configs (YAML)
runs/          # (gitignored) Per-experiment output
tests/         # pytest
docs/          # Thesis, methods draft, ablation log
```

## Timeline and status

| Week | Goal | Status | Result |
|---|---|---|---|
| 1 | Data consolidation | ✅ done | BeatAML 613p × 165d, DrugComb 186 strict pairs, TCGA 173p |
| 2 | Baseline A | ✅ done | Single-drug MLP, per-patient ρ = 0.704 (gate ≥ 0.40) |
| 3 | Combo predictor | ✅ done | Factorized additive + synergy residual + mechanism prior |
| 4 | Head-to-head | ✅ done | **FLT3-mut: Δ = +16.67** [14.98, 18.19]; driver-neg: Δ = -14 |
| 5 | TCGA validation | ✅ done | Reproduces clinical combos; driver+ median OS 9.5 vs 12.0 mo (p=0.065) |
| 6 | Manuscript | ✅ done | `docs/manuscript.md` + Figure 1 + Figure 2 |

**Landed outcome**: **C (partial yes)** — combination prediction beats best
single drug specifically in FLT3-mutated / driver-positive AML patients,
defining the precision-medicine target population. See
[`docs/manuscript.md`](docs/manuscript.md) for full write-up and
[`docs/week4_summary.md`](docs/week4_summary.md) for per-subgroup numerics.

## Running

Data setup:
```bash
python -m combo_val.data.beataml_etl --out data/canonical/
```

Training:
```bash
python -m combo_val.baselines.single_drug_xgb --config configs/week2_baseline.yaml
python -m combo_val.combo.combo_predictor --config configs/week3_combo.yaml
```

Validation:
```bash
python -m combo_val.validation.head_to_head --config configs/week4_validation.yaml
```

## Lineage

This repo replaces the combination-pharmacology-related portions of `AML-CRAFT`. The earlier work provided three artifacts that are preserved here:

1. `mechanism_vocab.yaml` and `drug_mechanism_v1.csv` — the AML mechanism knowledge base (retained as-is pending clinical collaborator review).
2. `parse_beataml_variants.py` — BeatAML variantSummary parser (solves a real data-prep issue with the public clinical summary).
3. The ETL path from BeatAML raw files to canonical tables (simplified here; no HyGReM-NC encoder).

Everything else (multimodal encoder, region classifier, pseudotime, K selection) is left in the `AML-CRAFT` project as historical work and will only be referenced as supporting supplementary findings in the manuscript if their specific results become useful.
