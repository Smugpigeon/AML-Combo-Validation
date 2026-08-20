# Split-Safe BeatAML Pilot Results

Run date: 2026-08-19

## Frozen evaluation setup

- 613 patients with 104 baseline features.
- 55,826 AUC records from 487 aligned patients and 165 single drugs.
- Fixed patient split: 341 train, 73 validation, 73 held-out test.
- RNA feature selection and PCA fitted on 379 RNA samples belonging to training
  patients only; validation and test RNA samples were projection-only.
- Three independently initialized models, 40,257 parameters each.
- Validation-fitted blend coefficient: 0.8924.

## Held-out comparison

| Predictor | MAE | RMSE | Median patient Spearman | Mean top-5 overlap |
|---|---:|---:|---:|---:|
| Drug-mean baseline | 36.719 | 47.388 | 0.758 | 0.471 |
| Neural patient residual | 33.512 | 44.296 | 0.775 | 0.515 |
| Validation-calibrated blend | 33.495 | 44.144 | 0.775 | 0.510 |

## Patient-paired bootstrap

Two thousand resamples of the 73 held-out patients were used. Positive values
favor the blended model over the drug-mean baseline.

| Increment | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| Mean patient-level MAE gain | 3.714 | 2.112 to 5.426 |
| Median within-patient Spearman gain | 0.027 | 0.020 to 0.035 |
| Mean top-5 overlap gain | 0.038 | 0.003 to 0.074 |

## Interpretation

The held-out internal data support a measurable patient-specific signal beyond
the drug-mean baseline. The ranking increment is modest, while the AUC error
reduction is clearer. This is sufficient to justify a prospective pilot and a
dose-aware extension, but not a claim of clinical utility, treatment benefit,
or a complete virtual cell.

Ensemble uncertainty has only a weak positive association with absolute error
(Spearman 0.153). It should be displayed as an audit signal, not treated as a
calibrated prediction interval.

## First held-out batch

Ten adult patients with at least 115 measured drugs were selected; relapsed
samples were prioritized. Each patient has:

- a 32-dimensional baseline latent state;
- predictions for all 165 known single drugs;
- ensemble disagreement;
- observed AUC where the assay exists;
- five nearest training states for audit.

All candidate sensitivity outputs are research-only and require prospective
functional confirmation and clinician review.

## Next model boundary

The acquired raw inhibitor table contains 555,583 quality-controlled dose-level
viability records. It can support a dose/time-aware viability head. It still
cannot supervise a CPA/STATE-style post-treatment transcriptome because BeatAML
does not provide paired perturbed expression for every patient.
