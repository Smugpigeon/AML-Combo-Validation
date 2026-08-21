# Identity repair evidence bundle

Research use only. This directory contains compact audit evidence; full
cell-level outputs remain on the controlled server.

## Decision

- Native/released semantics were reproduced and audited.
- The corrected Python SCEVAN implementation matched native R SCEVAN exactly
  on both reference patients.
- Full patient6 SCEVAN and CopyKAT classifications completed without sampling.
- The corrected identity mode passed all three retrospective RNA gates.
- The released-code effective mode failed patient5 and patient6.
- The preregistered dual-mode agreement gate therefore remained locked.
- No single-drug outcomes were unblinded and no combination model was trained.

## Files

- `released_code_semantics_audit.csv`: verifies the released writeback rule.
- `python_r_scevan_parity.csv`: cell-level parity metrics on patient5 and
  patient12.
- `scevan_semantics_and_parity_summary.json`: frozen thresholds and parity
  decision.
- `python_scevan_run_status.json`: corrected SCEVAN counts for all patients.
- `copykat/`: accepted patient6 CopyKAT status, R environment, and all attempts.
- `released_gate/`: released-code effective patient/state gate results.
- `corrected_gate/`: true-SCEVAN corrected patient/state gate results.
- `dual_gate/`: machine decision, comparison table, and adversarial review.

Full controlled-server root:

`/lhcos-data/aml_virtual_cell/derived/virtual_cell_identity_repair_20260821`
