# Hospital Evidence Package for Patient-Specific AML Virtual-Cell Validation

Research use only. This package is not a clinical prescription protocol.

## Why additional hospital data are required

The corrected RNA-derived identity workflow now passes all three public
retrospective patients. However, ScType, SCEVAN, and CopyKAT all use the same
RNA matrix. Their agreement is method reproduction, not independent proof that
each inferred state is malignant or normal. The strict identity gate therefore
remains locked.

Drug outcomes also remain sealed because the released-code effective and
corrected identity modes disagree for patient5 and patient6. Unblinding now
would weaken the intended prospective test.

## Collection package

For every consented baseline bone-marrow sample, collect four linked aliquots:

1. Baseline scRNA-seq with stable cell barcodes and raw counts.
2. Barcode-linked genotype or scDNA/CNV evidence for AML clone assignment.
3. Matched flow or CITE-seq immunophenotype with an AML-focused panel.
4. Ex-vivo drug assay material with vehicle, positive controls, dose, time,
   replicate, malignant-cell viability, and normal-cell toxicity.

Use study-coded identifiers only. Keep the linkage key inside the hospital.

## Minimum identity fields

Populate `HOSPITAL_IDENTITY_EVIDENCE_INTAKE.csv`. A malignant or normal cell is
considered independently validated only when it has barcode-linked genomic
evidence, or when orthogonal flow/CITE evidence agrees with an AML-specific
reference call. Bulk mutations and reported blast percentage are context, not
cell-level labels.

The frozen strict gate requires:

- at least 20% independently validated cells per patient;
- at least 10% validated cells in every state covering at least 5% of a patient;
- no more than 5% conflicting calls;
- no more than 30% unresolved calls;
- any blast discrepancy above 15 percentage points to be reconciled.

## Priority evidence for the three public archetypes

- patient5-like samples: reconcile blast fraction and link the dominant
  malignant and monocytic-like states to genotype/CNV plus matched flow.
- patient6-like samples: test whether the VS05-like state is AML while
  VS01/VS08-like states are non-malignant, using FLT3/PTPN11 or clone-linked CNV
  where available and matched immunophenotype.
- patient12-like samples: link the dominant stem/progenitor state to del(5q) or
  another patient-specific clonal alteration and an AML-specific reference.

These are evidence-acquisition targets, not treatment directions.

## Perturbation design after identity passes

Populate `HOSPITAL_PERTURBATION_ASSAY_TEMPLATE.csv`. Start with blinded
single-drug direction testing across multiple doses and at least two exposure
times. Measure malignant-cell response and normal-cell toxicity separately.
Only representative sensitive, resistant, and neutral conditions need
post-treatment scRNA/CITE-seq in the first pilot.

Freeze model predictions and their SHA-256 manifest before outcomes are joined.
Analyze by independent patient, not by treating cells or wells as independent
replicates. Compare against no-change, drug-mean, state-mean, regularized
linear, and nearest-neighbor baselines.

Combination modeling begins only after the identity and single-drug gates pass
and the repeated-pair patient coverage in `HOSPITAL_UNLOCK_CRITERIA.csv` is met.
Clinical treatment selection remains locked until independent-center validation
beats additive, drug-mean, and pair-mean baselines with calibrated abstention.

## Files to return from the hospital

- Completed `HOSPITAL_IDENTITY_EVIDENCE_INTAKE.csv`.
- Baseline raw-count matrix and cell metadata using the same coded barcodes.
- Flow/CITE gating report and antibody panel.
- Single-cell genotype/CNV output and QC report.
- Completed `HOSPITAL_PERTURBATION_ASSAY_TEMPLATE.csv`.
- Assay SOP, processing delay, fresh/frozen status, medium, dose grid, exposure
  time, batch, replicate, and QC documentation.

Do not place names, identity numbers, medical-record numbers, dates of birth,
phone numbers, or other direct identifiers in repository files.
