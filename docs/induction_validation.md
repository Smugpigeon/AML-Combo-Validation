# Route B — Clinical Induction-Response Validation (Strict)

## Question

Does Baseline A's predicted ex-vivo AUC carry signal about each BeatAML
patient's **real** induction-chemotherapy outcome (Complete Response vs
Refractory)?

If yes → the model has clinical face validity and the combination
predictions have plausible clinical meaning.
If no → the in-silico +17 AUC finding from Week 4 cannot be taken as
clinical evidence and the kit needs a different validation path.

## Cohort filter (strict)

| Filter | Kept |
|---|---:|
| BeatAML patients with full 80-dim features | 613 |
| Clinical: specimen at Initial Diagnosis | 471 |
| Clinical: response ∈ {Complete Response, Refractory} | 519 |
| Both filters + present in baseline predictions | **327** |

Final cohort: n=327 (222 CR, 105 Refractory, CR rate 67.9%).
323 of 327 (98.8%) received Standard Chemotherapy (7+3).

## Results

All seven tests return **no significant signal**. 95% CIs span 0.5 (chance).

| # | Test | n | ROC-AUC | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| A | Cytarabine **predicted** AUC → CR | 327 | 0.524 | [0.453, 0.594] | no signal |
| B | Best-single **predicted** AUC → CR | 327 | 0.527 | [0.462, 0.590] | no signal |
| C | Combo-Δ in FLT3-mut subgroup → CR | 99 | 0.560 | [0.434, 0.684] | underpowered, CI crosses 0.5 |
| D | Venetoclax **predicted** AUC → CR | 327 | 0.507 | [0.441, 0.575] | no signal |
| **E** | **OBSERVED** Cytarabine AUC → CR | 160 | **0.533** | [0.434, 0.638] | **oracle also no signal** |
| F | Predicted AUC → overall-survival (Spearman) | 327 | ρ ≤ 0.08 | all p > 0.17 | no signal |
| G | Best-single predicted AUC → CR (std-chemo only) | 323 | 0.530 | [0.466, 0.596] | no signal |

## Why test E (the Oracle) is the decisive finding

Test E uses the **actual measured ex-vivo cytarabine AUC** (BeatAML's real
drug-sensitivity measurement, not a model prediction) to predict CR. If
**measured** cytarabine AUC doesn't predict CR, no model fit to predict
**measured** cytarabine AUC can predict CR either. That's an upper bound.

The oracle ROC-AUC is 0.533 [0.434, 0.638]. **The ex-vivo cytarabine signal
does not predict induction CR in this cohort.** This is not a modeling
failure — it is a property of the ground truth.

Three plausible reasons:

1. **7+3 ≠ cytarabine alone**. BeatAML's panel contains cytarabine but NOT
   the anthracyclines (daunorubicin, idarubicin) that are the other half
   of 7+3. Induction CR depends on the full regimen.
2. **Ex-vivo ≠ in-vivo**. A patient's cells in a dish don't recapitulate
   in-vivo drug exposure, clearance, ADME, supportive care, or
   dose-reductions due to comorbidities.
3. **Host factors dominate binary CR**. Fitness, age, pre-existing organ
   damage, prior MDS, and treatment adherence all modulate whether a
   biochemically-sensitive leukemia achieves clinical remission.

## What this DOES NOT invalidate

- The Week 4 **model-vs-model** finding (+16.67 AUC in FLT3-mut) is
  internally consistent. It says: IF you trust the ex-vivo AUC prediction,
  combination prediction beats single-drug prediction by 17 units in FLT3-
  mut patients. It doesn't claim clinical benefit for 7+3 CR.
- The Week 5 TCGA reproducibility finding (combo recommendations align
  with FDA-approved combos) is still evidence that the mechanism prior
  encodes real biology.

## What this **does** invalidate

- Any deployment claim that says "this model predicts who will achieve CR
  on 7+3." It does not, within statistical power to detect at n=327.
- Any claim that the Week 4 +17 AUC translates directly to clinical
  benefit without further validation. The AUC metric is a proxy whose
  linkage to clinical endpoints is not demonstrated for 7+3 in this cohort.

## What clinical validation must look like instead

The right validation target is NOT 7+3 CR. It is:

1. **Prospective ex-vivo validation on fresh patient samples**: take new
   AML samples, dispense the model's top-3 recommended combinations, and
   compare measured combo AUC against the single-drug AUC predicted for
   that patient. This tests the Week 4 claim directly in its own
   measurement domain.
2. **Retrospective validation on cohorts that received targeted combos**:
   VIALE-A (Ven+Aza), LACEWING (Aza+Gilteritinib), AGILE (Aza+Ivosidenib),
   QUANTUM-First (Quizartinib+7+3), ADMIRAL (Gilteritinib). BeatAML's
   induction cohort predates these regimens.
3. **Organoid / PDX systems**: as proposed in the kit plan, use the model's
   top-3 recommendations as the set of regimens to test on each patient's
   organoid. Outcome: measured combo response in patient-derived model.
   This is exactly the validation our Route A build-out is required for.

## Implication for Routes A, B, kit, and organoid plans

| Decision | Update |
|---|---|
| Is the model "clinically validated" by BeatAML retrospective outcomes? | **No.** Do not claim this in the kit deck. |
| Should Route A (new-patient pipeline) still be built? | **Yes.** Routes the model to the correct validation target (ex-vivo + organoid). |
| Should the kit be deployed to clinicians for live decision support? | **No, not yet.** First run prospective ex-vivo validation. |
| Should organoid experiments be run with model recommendations? | **Yes — this is the proper validation.** Route A makes it possible. |

## Outputs

```
runs/induction_validation/
├── cohort.csv        # 327 × all feature / prediction / outcome columns
└── summary.json      # structured results above
```

## Conclusion

**Route B is decisive: no retrospective clinical signal exists, and the
oracle (observed ex-vivo AUC) shows the ceiling is near chance.** The next
validation step must move to ex-vivo or organoid measurement domains, not
more retrospective CR analysis. Route A (new-patient feature pipeline) is
the prerequisite enabler — it is the path to prospective, correct
validation.

We proceed with Route A while being explicit in the manuscript and the kit
plan that retrospective clinical validation on BeatAML 7+3 outcomes was
attempted and failed at the oracle level.
