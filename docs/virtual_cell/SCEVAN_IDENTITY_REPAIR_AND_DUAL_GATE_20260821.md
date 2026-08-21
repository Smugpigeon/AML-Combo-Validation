# SCEVAN Identity Repair and Dual-Gate Result

**Date:** 2026-08-21  
**Scope:** public retrospective research only; not treatment selection

## Final decision

The computational identity repair succeeded, but the preregistered dual-mode
sensitivity gate did not. No drug outcomes were unblinded, no single-drug gate
was scored, and no patient-specific combination model was trained or released.

The corrected true-SCEVAN mode passed all three retrospective RNA identity
gates. The released-code effective mode failed patient5 and patient6. Because
the two semantics disagree, the frozen workflow remains at identity validation.
The strict orthogonal identity gate also remains false in both modes.

## Defect found in the released workflow

With `all_pred=FALSE`, the released scTherapy helper runs SCEVAN but writes
`SCEVAN_output=malignant` for every cell outside the supplied T-cell normal set.
For patient5 and patient12, this effective output matched that rule for every
cell, with zero mismatches. It is not the tumour/normal classification returned
by SCEVAN.

The analysis therefore preserved two explicit modes:

- `released_code_effective`, for exact released-behavior reproduction;
- `true_scevan_corrected`, for actual SCEVAN classifications.

## Corrected SCEVAN validation

The pinned Python implementation at
`c7917cb47df9ab46465f9f94afbe801a71bd74e6` was validated against native R
SCEVAN before use on patient6. Thresholds were frozen before patient results:
agreement at least 0.95, adjusted Rand index at least 0.95, malignant Jaccard at
least 0.90, and common evaluable fraction at least 0.95.

| Patient | Common evaluable cells | Agreement | ARI | Kappa | Malignant Jaccard | Pass |
|---|---:|---:|---:|---:|---:|---|
| patient5 | 2,179 | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| patient12 | 3,281 | 1.00 | 1.00 | 1.00 | 1.00 | yes |

Corrected SCEVAN then completed all three patients:

| Patient | Total cells | Analyzed | Malignant | Healthy | Filtered |
|---|---:|---:|---:|---:|---:|
| patient5 | 2,365 | 2,179 | 1,548 | 631 | 186 |
| patient6 | 5,677 | 5,305 | 2,695 | 2,610 | 372 |
| patient12 | 3,610 | 3,281 | 2,365 | 916 | 329 |

## Full patient6 CopyKAT recovery

Two resource failures were rejected: a 2-core run lost one worker and returned
incompatible row counts, and a 1-core/6 GiB run was OOM-killed at baseline
adjustment. A 1-core/7 GiB run completed in 43.523 minutes without sampling or
threshold changes.

The accepted result contains 5,677 unique input barcodes, zero missing or extra
barcodes, 3,087 malignant calls, 2,590 healthy calls, and zero unresolved calls.

## Dual-mode patient results

| Patient | Released coverage | Released malignant | Released blast gap | Released gate | Corrected coverage | Corrected malignant | Corrected blast gap | Corrected gate |
|---|---:|---:|---:|---|---:|---:|---:|---|
| patient12 | 98.98% | 73.89% | 8.59 pp | pass | 89.89% | 71.96% | 6.66 pp | pass |
| patient5 | 85.03% | 80.31% | 25.91 pp | fail | 83.76% | 69.16% | 14.76 pp | pass |
| patient6 | 86.10% | 84.57% | 44.57 pp | fail | 90.35% | 50.73% | 10.73 pp | pass |

For patient6, the corrected mode calls VS05 predominantly malignant and VS01
and VS08 predominantly healthy. All three major states pass their coverage and
purity thresholds. The released rule biases all three states toward malignant,
causing VS08 to fail and inflating the whole-patient malignant fraction.

## Adversarial review

The strongest argument for continuing is that corrected SCEVAN has exact
cell-level parity with native R on both computable references and gives a much
more coherent blast-fraction result without changing thresholds.

The strongest argument against continuing is that the repair was triggered by
a discovered semantic defect, the two analysis semantics disagree for two of
three patients, and all retrospective algorithms still share one RNA matrix.
Unblinding drug outcomes now would make a clean challenge harder to defend.

The neutral verdict is to keep monotherapy unblinding locked, preserve the
corrected result as a validated computational repair, and collect independent
cell-identity evidence using the hospital package. A future protocol may
prospectively designate true SCEVAN as the primary method, but that rule must be
registered before looking at new drug outcomes.

## Evidence and next action

Compact evidence is in `results_20260821/identity_repair/`. Full cell-level
artifacts are frozen at:

`/lhcos-data/aml_virtual_cell/derived/virtual_cell_identity_repair_20260821`

The immediate next action is to populate
`HOSPITAL_IDENTITY_EVIDENCE_INTAKE.csv` with barcode-linked genomic and
orthogonal phenotype evidence. After the strict identity gate passes, use
`HOSPITAL_PERTURBATION_ASSAY_TEMPLATE.csv` for blinded single-drug direction
validation. Combination prediction remains locked until both prior stages and
the repeated-pair cohort criteria pass.
