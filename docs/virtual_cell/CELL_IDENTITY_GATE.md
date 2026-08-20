# AML Cell Identity Gate

Research use only. This gate answers one question before any perturbation
prediction is allowed:

> Are malignant and normal cells independently identified, or are they only
> suspected from expression and a reported blast fraction?

## Evidence hierarchy

The gate treats the following as independent cell-level evidence:

1. Barcode-linked single-cell DNA or CNV calls.
2. Agreement between an orthogonal flow/CITE-seq call and an AML-specific
   malignant/normal reference call.

The following are useful context but do **not** independently prove identity:

- General marker expression or CellTypist labels.
- Expression-state clusters (`VS01` to `VS12`).
- The calibrated blast-prevalence prior.
- Bulk mutation and cytogenetic findings without barcode linkage.

This prevents a circular result in which a blast prior is used to define
malignant cells and then reported as validation of those same cells.

## Pre-registered engineering gate

A patient passes only when all conditions hold:

- At least 20% of cells have independently validated identity anchors.
- Every state containing at least 5% of the patient's cells has at least 10%
  validated identity anchors.
- Conflicting identity calls affect at most 5% of cells.
- Unresolved cells affect at most 30% of cells.
- A scRNA-versus-clinical blast difference above 15 percentage points has been
  explicitly reconciled.

These are configurable engineering thresholds, not clinical diagnostic
cutoffs. They must be frozen before a challenge run and accompanied by
sensitivity analyses.

## Current three-patient result

| Patient | Identity result | Main blocker |
|---|---|---|
| `patient5` | blocked | no barcode-linked anchors; 33.3% unresolved; blast gap 28.4 points |
| `patient6` | blocked | no barcode-linked anchors; 52.7% unresolved; blast gap 26.0 points |
| `patient12` | provisional, not proven | no barcode-linked anchors in dominant `VS01`; low state confidence |

All three patients currently have `0%` independently validated cell identity
coverage. Consequently, the drug-perturbation stage and combination stage are
locked. This does not invalidate the baseline state atlas; it limits the atlas
to descriptive use.

Machine-readable results are in `current_rr3_identity_audit/`.

## Run the audit

```bash
PYTHONPATH=src python scripts/audit_rraml_cell_identity.py \
  --cell-annotations path/to/selected_rr3_cell_state_annotations.parquet \
  --patient-summary path/to/patient_state_summary.csv \
  --evidence-manifest docs/virtual_cell/CELL_IDENTITY_EVIDENCE_TEMPLATE.csv \
  --out-dir identity_audit
```

The command writes:

- `patient_identity_gate.csv`
- `state_identity_gate.csv`
- `next_identity_evidence_plan.csv`
- `identity_gate_summary.json`

Use `--write-cell-table` only when a cell-level parquet artifact is needed.

## Evidence needed next

- `patient5`: barcode-linked CNV/genotype, AML-specific monocytic reference,
  matched flow reconciliation of the blast discrepancy.
- `patient6`: FLT3/PTPN11 or CNV clone mapping, polyploidy-aware CNV evidence,
  matched flow reconciliation for `VS05` and `VS08`.
- `patient12`: del(5q)-linked single-cell CNV or genotype evidence and an
  AML-specific stem/progenitor reference for `VS01`.

The perturbation prediction command requires a passing
`identity_gate_summary.json` and refuses to run otherwise.
