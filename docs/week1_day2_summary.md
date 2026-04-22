# Week 1 Day 2 — Baseline A: single-drug MLP

## Summary

Baseline A trained successfully and **passes the pre-registered quality gate**
(per-patient Spearman ≥ 0.40) with a large margin. This is the strong baseline
the combo predictor (Week 3) will have to beat.

## Architecture

Multi-task MLP, ~50K parameters:

```
Patient features (80) → [Patient MLP 128 → 64] → patient_emb
                                                    │
Drug ID (0..164)      → [Drug Embedding 64]  → drug_emb
                                                    │
                              concat (128)          │
                                   │                │
                        [Response Head 128→64→1]    │
                                   │                │
                              predicted AUC ◄───────┘
```

## 5-fold CV by patient (no leakage)

| Fold | best epoch | MAE | per-patient ρ | per-drug median ρ | wall time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 23 | 34.15 | 0.698 | 0.278 | 137 s |
| 2 | 29 | 35.75 | 0.709 | 0.226 | 149 s |
| 3 | 20 | 36.39 | 0.681 | 0.286 | 110 s |
| 4 | 29 | 35.69 | 0.700 | 0.369 | 138 s |
| 5 | 28 | 33.21 | 0.730 | 0.354 | 133 s |
| **mean** | 26 | **35.04** | **0.704** | 0.303 | 134 s |
| std | 4 | 1.27 | **0.018** | 0.061 | |

## Pre-registered gates

| Gate | Threshold | Actual | Verdict |
|---|---|---|---|
| Per-patient Spearman | ≥ 0.40 | **0.704** | ✅ PASS (+76%) |
| MAE (AUC 0-300 scale) | < 50 | 34.6 | ✅ |
| Std across folds | < 0.05 | 0.018 | ✅ extremely stable |

## Per-drug predictability (most-predictable drugs)

Drugs with per-drug Spearman > 0.40:

| Drug | n_patients | Spearman | Biology |
|---|:---:|:---:|---|
| Venetoclax | 382 | **0.580** | BCL2i, AML SOC anchor |
| Sunitinib | 513 | 0.537 | Multi-kinase inhibitor |
| Sorafenib | 518 | 0.520 | FLT3i / RAF |
| Cabozantinib | 469 | 0.512 | VEGFR / FLT3 |
| KW-2449 | 468 | 0.506 | FLT3 / Aurora B |
| Tivozanib | 467 | 0.500 | VEGFR |
| Dasatinib | 518 | 0.500 | BCR-ABL / SRC |
| Dovitinib | 474 | 0.494 | RTK |
| Foretinib | 472 | 0.487 | MET / VEGFR |
| Selumetinib | 475 | 0.482 | MEK |

**Biological pattern**: the most-predictable drugs are kinase inhibitors
(FLT3, multi-RTK) and BCL2i — exactly the drugs whose response is tightly
coupled to the FLT3/NPM1/BCL2-status features we encoded. This is a clean
sanity check that the model is learning real biology, not memorizing
noise.

## Per-patient predictability distribution

- 485 patients have ≥ 5 drug measurements in hold-out across the 5 folds
- **304/485 (63%) have Spearman > 0.7** (excellent)
- 16/485 (3.3%) have Spearman < 0.3 (hard-to-predict outliers — likely those with sparse or atypical drug panels)

## Outputs

```
runs/baseline_single_drug_mlp/
├── final_model.pt                                  # checkpoint + scaler + drug vocab
├── cv_metrics.json                                 # fold-level + overall
├── cv_held_out_predictions.csv                     # held-out pred + true, all folds
├── per_patient_spearman.csv                        # per-patient ρ
├── per_drug_spearman.csv                           # per-drug ρ
└── predictions_all_patients_all_drugs.csv          # 613 × 165, used in Week 4
```

## Implications for Week 4

The head-to-head comparison becomes:

```
for each patient p in held-out:
    best_single(p) = min over drugs of predictions_all_patients_all_drugs[p, :]
    best_combo(p)  = min over legal pairs of combo_predictor(p, d1, d2)
    Δ(p) = best_single(p) - best_combo(p)
```

A baseline with ρ=0.70 means best_single is a genuinely strong opponent
— winning against it requires the combo predictor to carry information
beyond single-drug matching, not just learn the same patterns. This
strengthens the scientific value of whatever the head-to-head shows.

## Next (Day 3) — DrugComb AML subset

User started the download. Once `summary_v_1_5.csv` is in place:

```bash
python -m combo_val.data.drugcomb_etl \
  --input data/raw/drugcomb/summary_v_1_5.csv \
  --out data/canonical/drugcomb_aml_pairs.csv
```

I'll write the ETL to filter to AML cell lines and map drug names to the
BeatAML drug vocabulary (needed so DrugComb-trained residual can apply to
BeatAML patients' drug space).
